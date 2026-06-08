#!/bin/sh
set -eu

usage() {
  cat <<'USAGE'
Usage:
  sh scripts/run_chatlm_hf_clean_onnx_fp16_int8.sh [base-model-path-or-hf-id] [accuracy eval args]

Default base model:
  charent/ChatLM-mini-Chinese

This wrapper pins SOURCE_ASSET_DIR to the base model so checkpoint repair copies
the original Hugging Face modeling_chat*.py / modeling*.py files before ONNX
FP16 and INT8 export.
USAGE
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
if [ "$#" -gt 0 ]; then
  shift
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-onnx_eval_outputs/clean_h20_contrastive_gradient50_${RUN_ID}}"

# Key provenance/repair setting: use the original base model's HF code assets.
export SOURCE_ASSET_DIR="$BASE_MODEL"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

# Lock the run to the highest-scored contrastive-SFT 50% gradient pruning path.
export ENFORCE_CONTRASTIVE_GRADIENT50="${ENFORCE_CONTRASTIVE_GRADIENT50:-1}"
export FINETUNE_MODE="${FINETUNE_MODE:-contrastive}"
export FINETUNE_TRAIN_JSON="${FINETUNE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
export PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
export SPARSITY="${SPARSITY:-0.5}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-8}"
export ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-0,1,2,3,4,5,6,7}"
export ACCURACY_SHARD_PARALLELISM="${ACCURACY_SHARD_PARALLELISM:-8}"
export ACCURACY_SHARD_STREAM_LOGS="${ACCURACY_SHARD_STREAM_LOGS:-1}"
export MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-0}"
export MAX_BENCHMARK_EXAMPLES="${MAX_BENCHMARK_EXAMPLES:-200}"

export NUM_BEAMS="${NUM_BEAMS:-5}"
export NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
export SPLIT_EM1_EM5="${SPLIT_EM1_EM5:-1}"

export ONNX_DISABLE_IO_BINDING="${ONNX_DISABLE_IO_BINDING:-1}"
export FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CUDAExecutionProvider}"
export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"
export ONNX_OPSET="${ONNX_OPSET:-18}"

exec bash scripts/run_clean_h20_onnx_fp16_int8.sh "$BASE_MODEL" "$@"
