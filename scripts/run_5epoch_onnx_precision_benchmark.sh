#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_5epoch_onnx_precision_benchmark.sh <base-model-path-or-hf-id> [accuracy eval args]

One-line SCENIC ONNX precision benchmark:
  1. Fine-tune the base model for 5 epochs on the SCENIC training data.
  2. Export dense FP16 and FP32 ONNX artifacts.
  3. Dynamic-quantize the dense FP32 ONNX artifact to INT8.
  4. Run tools/benchmark_onnx_precision.py and write CSV/Markdown/LaTeX tables.

Common overrides:
  OUTPUT_ROOT=onnx_eval_outputs/my_run
  FINETUNE_EPOCHS=5
  TRAIN_JSON=data/SCENIC_full_training_dataset.json
  BENCHMARK_JSON=generated/iot_instruction_benchmark_200.json
  ONNX_BENCHMARK_PROVIDERS="CPUExecutionProvider CUDAExecutionProvider QNNExecutionProvider NNAPIExecutionProvider CoreMLExecutionProvider OpenVINOExecutionProvider"
  DEVICE_NAME="Jetson Orin Nano"
  POWER_LOG=/path/to/power.csv
  ONNX_PRECISION_WARMUP=30
  ONNX_PRECISION_RUNS=200
  ONNX_PRECISION_CALIBRATION_SAMPLES=128
  BENCHMARK_ONNX_NAME=encoder_model.onnx

Smoke test example:
  FINETUNE_EPOCHS=1 MAX_BENCHMARK_EXAMPLES=20 MAX_TRAIN_EXAMPLES=20 ONNX_PRECISION_RUNS=20 \
  bash scripts/run_5epoch_onnx_precision_benchmark.sh charent/ChatLM-mini-Chinese
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

BASE_MODEL="$1"
shift

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SAFE_MODEL_NAME="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME%_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME:-model}"

export RUN_ID
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
export RUN_INT8="${RUN_INT8:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-onnx_eval_outputs/${SAFE_MODEL_NAME}_sft5_onnx_precision_${RUN_ID}}"

CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
ONNX_ROOT="${OUTPUT_ROOT}/onnx"
INTERMEDIATE_ROOT="${OUTPUT_ROOT}/intermediate"
REPORT_ROOT="${OUTPUT_ROOT}/reports"

FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${CHECKPOINT_ROOT}/sft5}"
FP16_DENSE_ONNX_DIR="${ONNX_ROOT}/sft5_fp16_dense"
FP32_DENSE_ONNX_DIR="${INTERMEDIATE_ROOT}/sft5_fp32_dense_for_int8"
INT8_DENSE_ONNX_DIR="${ONNX_ROOT}/sft5_int8_dense"
PRECISION_OUTPUT_DIR="${ONNX_PRECISION_OUTPUT_DIR:-${REPORT_ROOT}/onnx_precision_benchmark}"

select_onnx_file() {
  local dir="$1"
  local preferred="${2:-}"

  if [[ ! -d "$dir" ]]; then
    echo "ONNX directory does not exist: $dir" >&2
    return 1
  fi

  if [[ -n "$preferred" ]]; then
    if [[ -f "${dir}/${preferred}" ]]; then
      printf '%s\n' "${dir}/${preferred}"
      return 0
    fi
    echo "Preferred ONNX file was not found in ${dir}: ${preferred}" >&2
  fi

  local name
  for name in model.onnx encoder_model.onnx decoder_model.onnx decoder_with_past_model.onnx; do
    if [[ -f "${dir}/${name}" ]]; then
      printf '%s\n' "${dir}/${name}"
      return 0
    fi
  done

  shopt -s nullglob
  local files=("$dir"/*.onnx)
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No .onnx files found in ${dir}" >&2
    return 1
  fi

  printf '%s\n' "${files[0]}"
}

echo "Running 5-epoch SCENIC fine-tune/export pipeline for: ${BASE_MODEL}"
bash scripts/run_onnx_precision_prune_eval.sh "$BASE_MODEL" "$@"

FP32_ONNX="$(select_onnx_file "$FP32_DENSE_ONNX_DIR" "${BENCHMARK_ONNX_NAME:-}")"
ONNX_BASENAME="$(basename "$FP32_ONNX")"

FP16_ONNX=""
if [[ -f "${FP16_DENSE_ONNX_DIR}/${ONNX_BASENAME}" ]]; then
  FP16_ONNX="${FP16_DENSE_ONNX_DIR}/${ONNX_BASENAME}"
else
  FP16_ONNX="$(select_onnx_file "$FP16_DENSE_ONNX_DIR" "${BENCHMARK_ONNX_NAME:-}")"
fi

INT8_ONNX=""
if [[ -d "$INT8_DENSE_ONNX_DIR" ]]; then
  if [[ -f "${INT8_DENSE_ONNX_DIR}/${ONNX_BASENAME}" ]]; then
    INT8_ONNX="${INT8_DENSE_ONNX_DIR}/${ONNX_BASENAME}"
  else
    INT8_ONNX="$(select_onnx_file "$INT8_DENSE_ONNX_DIR" "${BENCHMARK_ONNX_NAME:-}" || true)"
  fi
fi

ONNX_BENCHMARK_PROVIDERS="${ONNX_BENCHMARK_PROVIDERS:-CPUExecutionProvider CUDAExecutionProvider QNNExecutionProvider NNAPIExecutionProvider CoreMLExecutionProvider OpenVINOExecutionProvider}"
ONNX_BENCHMARK_TABLE_FORMATS="${ONNX_BENCHMARK_TABLE_FORMATS:-csv markdown latex}"
read -r -a provider_args <<< "$ONNX_BENCHMARK_PROVIDERS"
read -r -a table_format_args <<< "$ONNX_BENCHMARK_TABLE_FORMATS"

bench_cmd=(
  "$PYTHON"
  tools/benchmark_onnx_precision.py
  --fp32-onnx "$FP32_ONNX"
  --fp16-onnx "$FP16_ONNX"
  --output-dir "$PRECISION_OUTPUT_DIR"
  --providers "${provider_args[@]}"
  --batch-size "${ONNX_PRECISION_BATCH_SIZE:-1}"
  --warmup "${ONNX_PRECISION_WARMUP:-30}"
  --runs "${ONNX_PRECISION_RUNS:-200}"
  --calibration-samples "${ONNX_PRECISION_CALIBRATION_SAMPLES:-128}"
  --quantization-mode "${ONNX_PRECISION_QUANTIZATION_MODE:-dynamic}"
  --table-formats "${table_format_args[@]}"
  --benchmark-json "${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
  --calibration-json "${FINETUNE_TRAIN_JSON:-${TRAIN_JSON:-data/SCENIC_full_training_dataset.json}}"
  --tokenizer-dir "$FINETUNE_OUTPUT_DIR"
  --max-input-len "${MAX_INPUT_LEN:-256}"
  --max-target-len "${TRAIN_MAX_TARGET_LENGTH:-128}"
  --drift-samples "${ONNX_PRECISION_DRIFT_SAMPLES:-16}"
)

if [[ -n "$INT8_ONNX" && -f "$INT8_ONNX" ]]; then
  bench_cmd+=(--int8-onnx "$INT8_ONNX")
fi
if [[ -n "${DEVICE_NAME:-}" ]]; then
  bench_cmd+=(--device-name "$DEVICE_NAME")
fi
if [[ -n "${POWER_LOG:-}" ]]; then
  bench_cmd+=(--power-log "$POWER_LOG")
  bench_cmd+=(--power-column "${POWER_COLUMN:-power_w}")
  bench_cmd+=(--timestamp-column "${TIMESTAMP_COLUMN:-timestamp_s}")
fi
if [[ -n "${NUM_THREADS:-}" ]]; then
  bench_cmd+=(--num-threads "$NUM_THREADS")
fi
if [[ "${DISABLE_IOBINDING:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  bench_cmd+=(--disable-iobinding)
fi
if [[ "${PROFILE_ORT:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  bench_cmd+=(--profile-ort)
fi
if [[ -n "${ONNX_PRECISION_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$ONNX_PRECISION_EXTRA_ARGS"
  bench_cmd+=("${extra_args[@]}")
fi

echo "Running ONNX precision table benchmark:"
printf '  %q' "${bench_cmd[@]}"
printf '\n'
"${bench_cmd[@]}"

echo "Done. ONNX precision benchmark table: ${PRECISION_OUTPUT_DIR}/onnx_precision_benchmark.md"
