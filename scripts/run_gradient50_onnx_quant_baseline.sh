#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_gradient50_onnx_quant_baseline.sh [base-model-path-or-hf-id] [accuracy eval args]

Default:
  base model: charent/ChatLM-mini-Chinese

This launcher builds the sparse quantized ASIC baseline:
  1. Fine-tune the base model for 5 epochs with contrastive SFT.
  2. Create the 50% gradient one-shot pruned checkpoint.
  3. Export dense and gradient-pruned checkpoints to ONNX FP16.
  4. Export FP32 ONNX sources and dynamic-quantize them to ONNX INT8.
  5. Evaluate ONNX FP16 and ONNX INT8 on benchmark/training EM1/EM5.
     Accuracy eval is sharded across the configured GPU ids by default.

Useful overrides:
  OUTPUT_ROOT=onnx_eval_outputs/contrastive_gradient50_asic_baseline
  NPROC_PER_NODE=8
  ACCURACY_GPU_IDS=0,1,2,3,4,5,6,7
  RUN_ACCURACY_SHARDED=1
  ACCURACY_SHARD_PARALLELISM=4
  ACCURACY_SHARD_STREAM_LOGS=1
  MAX_BENCHMARK_EXAMPLES=200
  MAX_TRAIN_EXAMPLES=0
  NUM_BEAMS=5
  NUM_RETURN_SEQUENCES=5
  MAX_NEW_TOKENS=128
  SPLIT_EM1_EM5=1
  FAST_ACCURACY=1 uses NUM_BEAMS=1 NUM_RETURN_SEQUENCES=1 MAX_NEW_TOKENS=64.
  ENFORCE_CONTRASTIVE_GRADIENT50=0 only for deliberately different ablations.
  RUN_PYTORCH_ACCURACY=1
  RUN_RUNTIME_BENCHMARK=1
  ONNX_DISABLE_IO_BINDING=1
  INT8_ONNX_PROVIDER=CUDAExecutionProvider

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

export FINETUNE_MODE="${FINETUNE_MODE:-contrastive}"
export FINETUNE_TRAIN_JSON="${FINETUNE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
export PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
export SPARSITY="${SPARSITY:-0.5}"
export PRUNED_VARIANT="${PRUNED_VARIANT:-contrastive_gradient50}"
export PRUNED_PRETTY_LABEL="${PRUNED_PRETTY_LABEL:-contrastive 50% gradient one-shot pruned}"
export ENFORCE_CONTRASTIVE_GRADIENT50="${ENFORCE_CONTRASTIVE_GRADIENT50:-1}"
export RUN_INT8="${RUN_INT8:-1}"
export RUN_TENSORRT="${RUN_TENSORRT:-0}"
export TENSORRT_SPARSITY_ENABLE="${TENSORRT_SPARSITY_ENABLE:-0}"
export RUN_PYTORCH_ACCURACY="${RUN_PYTORCH_ACCURACY:-0}"
export RUN_RUNTIME_BENCHMARK="${RUN_RUNTIME_BENCHMARK:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-$NPROC_PER_NODE}"
export RUN_ACCURACY_SHARDED="${RUN_ACCURACY_SHARDED:-1}"
export ACCURACY_SHARD_PARALLELISM="${ACCURACY_SHARD_PARALLELISM:-4}"
export ACCURACY_SHARD_RETRIES="${ACCURACY_SHARD_RETRIES:-1}"
export ACCURACY_SHARD_STREAM_LOGS="${ACCURACY_SHARD_STREAM_LOGS:-1}"
export RUN_ACCURACY_PARALLEL="${RUN_ACCURACY_PARALLEL:-0}"
export ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-0,1,2,3,4,5,6,7}"
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
export ONNX_DISABLE_IO_BINDING="${ONNX_DISABLE_IO_BINDING:-1}"
export ONNX_OPSET="${ONNX_OPSET:-18}"
export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"

exec bash scripts/run_onnx_precision_prune_eval.sh "$BASE_MODEL" "$@"
