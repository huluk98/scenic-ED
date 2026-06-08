#!/bin/sh
set -eu

usage() {
  cat <<'USAGE'
Usage:
  sh scripts/run_latency_smoke_compare.sh <completed-output-root> [base-model-path-or-hf-id]

Example:
  sh scripts/run_latency_smoke_compare.sh \
    onnx_eval_outputs/clean_h20_contrastive_gradient50_YYYYMMDD_HHMMSS \
    charent/ChatLM-mini-Chinese

Runs a quick latency comparison using existing ONNX exports:
  - 1 GPU by default: LATENCY_GPU_IDS=0
  - input length 64
  - 10 measured queries
  - 1 warmup query
  - max_new_tokens 32

Outputs a Markdown/CSV/JSON comparison table under:
  <OUTPUT_ROOT>/reports/latency_smoke_seq64_q10/
USAGE
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi

OUTPUT_ROOT="$1"
shift
BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"

export LATENCY_GPU_IDS="${LATENCY_GPU_IDS:-0}"
export LATENCY_SEQ_LENGTHS="${LATENCY_SEQ_LENGTHS:-64}"
export LATENCY_QUERIES="${LATENCY_QUERIES:-10}"
export LATENCY_WARMUP="${LATENCY_WARMUP:-1}"
export LATENCY_MAX_NEW_TOKENS="${LATENCY_MAX_NEW_TOKENS:-32}"
export LATENCY_NUM_BEAMS="${LATENCY_NUM_BEAMS:-1}"
export LATENCY_OUTPUT_DIR="${LATENCY_OUTPUT_DIR:-${OUTPUT_ROOT}/reports/latency_smoke_seq64_q10}"
export RUN_INT8="${RUN_INT8:-1}"
export RUN_PYTORCH_RUNTIME_BENCHMARK="${RUN_PYTORCH_RUNTIME_BENCHMARK:-0}"
export ONNX_DISABLE_IO_BINDING="${ONNX_DISABLE_IO_BINDING:-1}"
export FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CUDAExecutionProvider}"
export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"

exec bash scripts/run_latency_testdrive_8gpu.sh "$OUTPUT_ROOT" "$BASE_MODEL"
