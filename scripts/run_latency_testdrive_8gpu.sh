#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  OUTPUT_ROOT=onnx_eval_outputs/clean_h20_contrastive_gradient50_YYYYMMDD_HHMMSS \
    bash scripts/run_latency_testdrive_8gpu.sh [base-model-path-or-hf-id]

  bash scripts/run_latency_testdrive_8gpu.sh \
    onnx_eval_outputs/clean_h20_contrastive_gradient50_YYYYMMDD_HHMMSS \
    [base-model-path-or-hf-id]

Runs latency/TPS only, one worker per GPU, using an existing run folder.
It skips all accuracy generation and skips PyTorch runtime latency by default.

Useful overrides:
  LATENCY_GPU_IDS=0,1,2,3,4,5,6,7
  LATENCY_SEQ_LENGTHS="64"
  LATENCY_QUERIES=10
  LATENCY_WARMUP=1
  LATENCY_MAX_NEW_TOKENS=32
  ONNX_DISABLE_IO_BINDING=1
  RUN_INT8=1

Outputs:
  <OUTPUT_ROOT>/reports/latency_testdrive_8gpu/gpu_<id>_runtime_benchmark.json
  <OUTPUT_ROOT>/reports/latency_testdrive_8gpu/gpu_<id>.log
  <OUTPUT_ROOT>/reports/latency_testdrive_8gpu/latency_comparison_summary.md
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
  fi
  export OUTPUT_ROOT="$1"
  shift
fi

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
LATENCY_GPU_IDS="${LATENCY_GPU_IDS:-0,1,2,3,4,5,6,7}"
LATENCY_OUTPUT_DIR="${LATENCY_OUTPUT_DIR:-${OUTPUT_ROOT}/reports/latency_testdrive_8gpu}"
mkdir -p "$LATENCY_OUTPUT_DIR"

required_paths=(
  "${OUTPUT_ROOT}/checkpoints/sft5/config.json"
  "${OUTPUT_ROOT}/checkpoints/sft5_clean_contrastive_gradient50_pruned/config.json"
  "${OUTPUT_ROOT}/onnx/sft5_fp16_dense"
  "${OUTPUT_ROOT}/onnx/sft5_fp16_pruned"
)

if [[ "${RUN_INT8:-1}" != "0" ]]; then
  required_paths+=(
    "${OUTPUT_ROOT}/onnx/sft5_int8_dense"
    "${OUTPUT_ROOT}/onnx/sft5_int8_pruned"
  )
fi

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required existing artifact: $path" >&2
    echo "Use the full clean launcher first, or set OUTPUT_ROOT to a completed run." >&2
    exit 2
  fi
done

IFS=',' read -r -a gpu_ids <<< "${LATENCY_GPU_IDS// /,}"

echo "Latency test-drive output: $LATENCY_OUTPUT_DIR"
echo "Using existing output root: $OUTPUT_ROOT"
echo "Launching one latency-only worker per GPU: ${gpu_ids[*]}"
echo "Latency settings: seq=${LATENCY_SEQ_LENGTHS:-64}, queries=${LATENCY_QUERIES:-10}, warmup=${LATENCY_WARMUP:-1}, max_new_tokens=${LATENCY_MAX_NEW_TOKENS:-32}"

pids=()
labels=()
for gpu in "${gpu_ids[@]}"; do
  [[ -n "$gpu" ]] || continue
  log_path="${LATENCY_OUTPUT_DIR}/gpu_${gpu}.log"
  runtime_json="${LATENCY_OUTPUT_DIR}/gpu_${gpu}_runtime_benchmark.json"
  final_json="${LATENCY_OUTPUT_DIR}/gpu_${gpu}_final_report.json"
  echo "Starting latency worker on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export PYTHONUNBUFFERED=1
    export OUTPUT_ROOT="$OUTPUT_ROOT"
    export RUNTIME_BENCHMARK_JSON="$runtime_json"
    export FINAL_JSON="$final_json"

    export FORCE_TRAIN=0
    export FORCE_PRUNE=0
    export FORCE_EXPORT=0
    export FORCE_QUANTIZE=0
    export FORCE_ACCURACY=0
    export FORCE_BENCHMARK=1

    export RUN_PYTORCH_ACCURACY=0
    export RUN_ONNX_FP16_DENSE_ACCURACY=0
    export RUN_ONNX_FP16_PRUNED_ACCURACY=0
    export RUN_ONNX_INT8_DENSE_ACCURACY=0
    export RUN_ONNX_INT8_PRUNED_ACCURACY=0
    export RUN_ACCURACY_SHARDED=0
    export RUN_ACCURACY_PARALLEL=0
    export RUN_RUNTIME_BENCHMARK=1
    export RUN_PYTORCH_RUNTIME_BENCHMARK="${RUN_PYTORCH_RUNTIME_BENCHMARK:-0}"

    export RUN_INT8="${RUN_INT8:-1}"
    export RUN_TENSORRT="${RUN_TENSORRT:-0}"
    export ENFORCE_CONTRASTIVE_GRADIENT50="${ENFORCE_CONTRASTIVE_GRADIENT50:-1}"
    export FINETUNE_MODE="${FINETUNE_MODE:-contrastive}"
    export FINETUNE_TRAIN_JSON="${FINETUNE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
    export PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
    export SPARSITY="${SPARSITY:-0.5}"
    export PRUNED_VARIANT="${PRUNED_VARIANT:-clean_contrastive_gradient50}"
    export PRUNED_PRETTY_LABEL="${PRUNED_PRETTY_LABEL:-clean contrastive 50% gradient one-shot pruned}"
    export ONNX_DISABLE_IO_BINDING="${ONNX_DISABLE_IO_BINDING:-1}"
    export FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CUDAExecutionProvider}"
    export INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-CUDAExecutionProvider}"

    export LATENCY_SEQ_LENGTHS="${LATENCY_SEQ_LENGTHS:-64}"
    export LATENCY_QUERIES="${LATENCY_QUERIES:-10}"
    export LATENCY_WARMUP="${LATENCY_WARMUP:-1}"
    export LATENCY_NUM_BEAMS="${LATENCY_NUM_BEAMS:-1}"
    export LATENCY_MAX_NEW_TOKENS="${LATENCY_MAX_NEW_TOKENS:-32}"

    bash scripts/run_onnx_precision_prune_eval.sh "$BASE_MODEL"
  ) >"$log_path" 2>&1 &
  pids+=("$!")
  labels+=("gpu ${gpu}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Completed ${labels[$index]}"
  else
    echo "Failed ${labels[$index]}; tailing log:" >&2
    tail -n 60 "${LATENCY_OUTPUT_DIR}/${labels[$index]/ /_}.log" >&2 || true
    status=1
  fi
done

if [[ "$status" -eq 0 ]]; then
  echo "Latency test-drive complete."
  echo "JSON files: ${LATENCY_OUTPUT_DIR}/gpu_*_runtime_benchmark.json"
  "${PYTHON:-python}" scripts/summarize_latency_testdrive.py \
    --input-glob "${LATENCY_OUTPUT_DIR}/gpu_*_runtime_benchmark.json" \
    --output-json "${LATENCY_OUTPUT_DIR}/latency_comparison_summary.json" \
    --output-csv "${LATENCY_OUTPUT_DIR}/latency_comparison_summary.csv" \
    --output-md "${LATENCY_OUTPUT_DIR}/latency_comparison_summary.md"
  echo "Summary: ${LATENCY_OUTPUT_DIR}/latency_comparison_summary.md"
else
  echo "One or more latency workers failed. See logs in $LATENCY_OUTPUT_DIR." >&2
fi
exit "$status"
