#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_clean_h20_onnx_fp16_int8.sh [base-model-path-or-hf-id] [accuracy eval args]

Default:
  base model: charent/ChatLM-mini-Chinese

This is the clean H20 ONNX-only launcher:
  1. Start from a fresh OUTPUT_ROOT every run.
  2. Use GPUs 0-7 for 5-epoch contrastive SFT.
  3. Repair checkpoint assets from the original base model, including Hugging Face modeling*.py files.
  4. Create the contrastive 50% gradient one-shot pruned checkpoint.
  5. Export dense/pruned ONNX FP16 and dynamic ONNX INT8.
  6. Evaluate ONNX FP16 and ONNX INT8 only, sharded across the 8 GPUs.

Useful overrides:
  OUTPUT_ROOT=onnx_eval_outputs/my_clean_run
  ACCURACY_SHARD_PARALLELISM=8
  ACCURACY_SHARD_STREAM_LOGS=1
  ONNX_DISABLE_IO_BINDING=1
  NUM_BEAMS=5
  NUM_RETURN_SEQUENCES=5
  MAX_NEW_TOKENS=128
  SPLIT_EM1_EM5=1
  FAST_ACCURACY=1 uses NUM_BEAMS=1 NUM_RETURN_SEQUENCES=1 MAX_NEW_TOKENS=64.
  MAX_TRAIN_EXAMPLES=0
  MAX_BENCHMARK_EXAMPLES=200
  ACCURACY_ONLY=1 with OUTPUT_ROOT=<existing-run> reuses existing checkpoints/ONNX exports.

Final report:
  <OUTPUT_ROOT>/all_deployment_em_latency_report.json
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
if [[ $# -gt 0 ]]; then
  shift
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-onnx_eval_outputs/clean_h20_contrastive_gradient50_${RUN_ID}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
export ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-0,1,2,3,4,5,6,7}"
export ACCURACY_SHARD_PARALLELISM="${ACCURACY_SHARD_PARALLELISM:-8}"
export ACCURACY_SHARD_RETRIES="${ACCURACY_SHARD_RETRIES:-1}"
export ACCURACY_SHARD_STREAM_LOGS="${ACCURACY_SHARD_STREAM_LOGS:-1}"
export MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
export MAX_BENCHMARK_EXAMPLES="${MAX_BENCHMARK_EXAMPLES:-200}"
if truthy "${FAST_ACCURACY:-0}"; then
  export NUM_BEAMS="${NUM_BEAMS:-1}"
  export NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
else
  export NUM_BEAMS="${NUM_BEAMS:-5}"
  export NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
  export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
fi
export SPLIT_EM1_EM5="${SPLIT_EM1_EM5:-1}"

export FINETUNE_MODE="${FINETUNE_MODE:-contrastive}"
export FINETUNE_TRAIN_JSON="${FINETUNE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
export SOURCE_ASSET_DIR="${SOURCE_ASSET_DIR:-$BASE_MODEL}"
export PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
export SPARSITY="${SPARSITY:-0.5}"
export PRUNED_VARIANT="${PRUNED_VARIANT:-clean_contrastive_gradient50}"
export PRUNED_PRETTY_LABEL="${PRUNED_PRETTY_LABEL:-clean contrastive 50% gradient one-shot pruned}"

export RUN_INT8="${RUN_INT8:-1}"
export RUN_TENSORRT="${RUN_TENSORRT:-0}"
export RUN_PYTORCH_ACCURACY="${RUN_PYTORCH_ACCURACY:-0}"
export RUN_RUNTIME_BENCHMARK="${RUN_RUNTIME_BENCHMARK:-0}"
export RUN_ACCURACY_SHARDED="${RUN_ACCURACY_SHARDED:-1}"
export RUN_ACCURACY_PARALLEL="${RUN_ACCURACY_PARALLEL:-0}"

export FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CUDAExecutionProvider}"
export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"
export ONNX_DISABLE_IO_BINDING="${ONNX_DISABLE_IO_BINDING:-1}"
export ONNX_OPSET="${ONNX_OPSET:-18}"
export ALIGN_TOKENIZER_EMBEDDINGS="${ALIGN_TOKENIZER_EMBEDDINGS:-1}"

if truthy "${ACCURACY_ONLY:-0}"; then
  export FORCE_TRAIN="${FORCE_TRAIN:-0}"
  export FORCE_PRUNE="${FORCE_PRUNE:-0}"
  export FORCE_EXPORT="${FORCE_EXPORT:-0}"
  export FORCE_QUANTIZE="${FORCE_QUANTIZE:-0}"
else
  export FORCE_TRAIN="${FORCE_TRAIN:-1}"
  export FORCE_PRUNE="${FORCE_PRUNE:-1}"
  export FORCE_EXPORT="${FORCE_EXPORT:-1}"
  export FORCE_QUANTIZE="${FORCE_QUANTIZE:-1}"
fi
export FORCE_ACCURACY="${FORCE_ACCURACY:-1}"

exec bash scripts/run_onnx_precision_prune_eval.sh "$BASE_MODEL" "$@"
