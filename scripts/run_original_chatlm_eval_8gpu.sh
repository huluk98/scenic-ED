#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_original_chatlm_eval_8gpu.sh <base-model-path-or-hf-id> [extra evaluate_original_chatlm.py args]

Example:
  bash scripts/run_original_chatlm_eval_8gpu.sh charent/ChatLM-mini-Chinese

Useful env overrides:
  NPROC_PER_NODE=8          # default: auto-detect NVIDIA GPUs
  OUTPUT_ROOT=prune_eval_outputs/original_chatlm_baseline
  OUTPUT_JSON=prune_eval_outputs/original_chatlm_baseline/original_chatlm_eval_report.json
  LOCAL_BASE_MODEL_DIR=prune_eval_outputs/original_chatlm_baseline/base_model
  LOCAL_FILES_ONLY=1        # force local/offline model loading
  LOCAL_FILES_ONLY=0        # allow Hugging Face download/materialization
  BF16=1                    # default on NVIDIA GPUs
  FP16=1                    # optional alternative precision
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

BASE_MODEL="$1"
shift

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prune_eval_outputs/${SAFE_BASE}_original_chatlm_baseline_${RUN_ID}}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/original_chatlm_eval_report.json}"
LOCAL_BASE_MODEL_DIR="${LOCAL_BASE_MODEL_DIR:-${OUTPUT_ROOT}/base_model}"

detect_nproc() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$count" != "0" ]]; then
      echo "$count"
      return
    fi
  fi
  echo "1"
}

download_hf_model() {
  local model_id="$1"
  local output_dir="$2"
  if command -v hf >/dev/null 2>&1; then
    hf download "$model_id" --local-dir "$output_dir"
    return
  fi
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$model_id" --local-dir "$output_dir"
    return
  fi
  echo "The Hugging Face CLI is required to materialize HF model id '$model_id' into a local directory." >&2
  echo "Install huggingface_hub so the 'hf' command is available, or pass a local model directory instead." >&2
  exit 2
}

NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_nproc)}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "$NPROC_PER_NODE" -gt 1 ]]; then
  CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((NPROC_PER_NODE - 1)))"
  export CUDA_VISIBLE_DEVICES
fi

case "${LOCAL_FILES_ONLY:-auto}" in
  auto)
    if [[ -d "$BASE_MODEL" ]]; then
      USE_LOCAL_FILES_ONLY=1
    else
      USE_LOCAL_FILES_ONLY=0
    fi
    ;;
  1|true|TRUE|yes|YES)
    USE_LOCAL_FILES_ONLY=1
    ;;
  0|false|FALSE|no|NO)
    USE_LOCAL_FILES_ONLY=0
    ;;
  *)
    echo "LOCAL_FILES_ONLY must be auto, 1, or 0." >&2
    exit 2
    ;;
esac

EVAL_MODEL="$BASE_MODEL"
if [[ ! -d "$BASE_MODEL" && "$USE_LOCAL_FILES_ONLY" -eq 0 ]]; then
  mkdir -p "$LOCAL_BASE_MODEL_DIR"
  echo "Downloading/materializing base model '$BASE_MODEL' into: $LOCAL_BASE_MODEL_DIR"
  download_hf_model "$BASE_MODEL" "$LOCAL_BASE_MODEL_DIR"
  EVAL_MODEL="$LOCAL_BASE_MODEL_DIR"
  USE_LOCAL_FILES_ONLY=1
fi

LOCAL_ARGS=()
if [[ "$USE_LOCAL_FILES_ONLY" -eq 1 ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  LOCAL_ARGS+=(--local-files-only)
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
  LOCAL_ARGS+=(--no-local-files-only)
fi

PRECISION_ARGS=()
case "${FP16:-0}" in
  1|true|TRUE|yes|YES)
    PRECISION_ARGS+=(--fp16 --no-bf16)
    ;;
  0|false|FALSE|no|NO)
    case "${BF16:-1}" in
      1|true|TRUE|yes|YES)
        PRECISION_ARGS+=(--bf16)
        ;;
      0|false|FALSE|no|NO)
        PRECISION_ARGS+=(--no-bf16)
        ;;
      *)
        echo "BF16 must be 1 or 0." >&2
        exit 2
        ;;
    esac
    ;;
  *)
    echo "FP16 must be 1 or 0." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"

echo "Original ChatLM baseline model: $BASE_MODEL"
echo "Evaluation model path: $EVAL_MODEL"
echo "Output JSON: $OUTPUT_JSON"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "LOCAL_FILES_ONLY for eval model: $USE_LOCAL_FILES_ONLY"

COMMON_ARGS=(
  scripts/evaluate_original_chatlm.py
  "$EVAL_MODEL"
  --output-json "$OUTPUT_JSON"
  "${LOCAL_ARGS[@]}"
  "${PRECISION_ARGS[@]}"
)
COMMON_ARGS+=("$@")

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${COMMON_ARGS[@]}"
else
  "$PYTHON" "${COMMON_ARGS[@]}"
fi

echo "Done. Original ChatLM baseline report: $OUTPUT_JSON"
