#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_prune_eval_50.sh <sft-or-contrastive-sft-model-path> [extra scenic_prune_eval.py args]" >&2
  echo "Example: METHOD=wanda bash scripts/run_prune_eval_50.sh models/chatlm_scenic_triplet_sft --max-train-examples 128" >&2
  exit 2
fi

MODEL_PATH="$1"
shift

cd "$(dirname "$0")/.."

METHOD="${METHOD:-magnitude}"
SPARSITY="${SPARSITY:-0.5}"
IGNORE_SPACES="${IGNORE_SPACES:-0}"
SAFE_MODEL_NAME="$(basename "$MODEL_PATH" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME%_}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prune_eval_outputs}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${SAFE_MODEL_NAME}_${METHOD}_50}"
PRUNED_DIR="${PRUNED_DIR:-${RUN_DIR}/pruned_model}"
REPORT_JSON="${REPORT_JSON:-${RUN_DIR}/prune_eval_report.json}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$NPROC_PER_NODE" == "0" ]]; then
      NPROC_PER_NODE=1
    fi
  else
    NPROC_PER_NODE=1
  fi
fi

mkdir -p "$RUN_DIR"

USER_SET_PRUNED_DIR=0
USER_SET_REPORT_JSON=0
for arg in "$@"; do
  case "$arg" in
    --pruned-output-dir|--pruned-output-dir=*)
      USER_SET_PRUNED_DIR=1
      ;;
    --output-json|--output-json=*)
      USER_SET_REPORT_JSON=1
      ;;
  esac
done

COMMON_ARGS=(
  scripts/scenic_prune_eval.py
  --model "$MODEL_PATH"
  --method "$METHOD"
  --sparsity "$SPARSITY"
)

case "$IGNORE_SPACES" in
  1|true|TRUE|yes|YES)
    COMMON_ARGS+=(--ignore-spaces)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "IGNORE_SPACES must be 1 or 0." >&2
    exit 2
    ;;
esac

if [[ "$USER_SET_PRUNED_DIR" -eq 0 ]]; then
  COMMON_ARGS+=(--pruned-output-dir "$PRUNED_DIR")
fi

if [[ "$USER_SET_REPORT_JSON" -eq 0 ]]; then
  COMMON_ARGS+=(--output-json "$REPORT_JSON")
fi

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${COMMON_ARGS[@]}" "$@"
else
  python "${COMMON_ARGS[@]}" "$@"
fi

if [[ "$USER_SET_REPORT_JSON" -eq 0 ]]; then
  echo "Combined JSON report: $REPORT_JSON"
else
  echo "Combined JSON report path was supplied via --output-json."
fi
