#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_contrastive_5epoch_all_prune_50.sh <base-model-path-or-hf-id> [extra scenic_prune_eval.py args]

Example:
  bash scripts/run_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese

Useful env overrides:
  OUTPUT_ROOT=prune_eval_outputs/my_run
  CONTRASTIVE_OUTPUT_DIR=models/chatlm_scenic_triplet_sft_5epoch
  FINAL_JSON=prune_eval_outputs/my_run/all_pruning_em_report.json
  NPROC_PER_NODE=8
  LOCAL_BASE_MODEL_DIR=prune_eval_outputs/my_run/base_model
  LOCAL_FILES_ONLY=1        # force local/offline base model loading
  LOCAL_FILES_ONLY=0        # allow Hugging Face base model loading
  SKIP_TRAIN=1              # reuse CONTRASTIVE_OUTPUT_DIR and only prune/eval
USAGE
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

BASE_MODEL="$1"
shift

for arg in "$@"; do
  case "$arg" in
    --model|--model=*|--method|--method=*|--sparsity|--sparsity=*|--output-json|--output-json=*|--pruned-output-dir|--pruned-output-dir=*)
      echo "This launcher controls model, method, sparsity, output-json, and pruned-output-dir. Use env overrides instead of: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-5}"
SPARSITY="${SPARSITY:-0.5}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
PRUNE_METHODS="${PRUNE_METHODS:-magnitude wanda gradient nvidia24}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prune_eval_outputs/${SAFE_BASE}_contrastive5_all50_${RUN_ID}}"
LOCAL_BASE_MODEL_DIR="${LOCAL_BASE_MODEL_DIR:-${OUTPUT_ROOT}/base_model}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${OUTPUT_ROOT}/contrastive_sft_5epoch}"
FINAL_JSON="${FINAL_JSON:-${OUTPUT_ROOT}/all_pruning_em_report.json}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-5}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.03}"
TRAIN_MAX_SOURCE_LENGTH="${TRAIN_MAX_SOURCE_LENGTH:-128}"
TRAIN_MAX_TARGET_LENGTH="${TRAIN_MAX_TARGET_LENGTH:-96}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
TRAIN_ALIGNMENT_WEIGHT="${TRAIN_ALIGNMENT_WEIGHT:-0.1}"
TRAIN_MARGIN="${TRAIN_MARGIN:-0.5}"
TRAIN_NEGATIVE_FIELD="${TRAIN_NEGATIVE_FIELD:-negative}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-10}"

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
  echo "Install huggingface_hub so the 'hf' command is available, or pass a local base model directory instead." >&2
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

TRAIN_MODEL="$BASE_MODEL"
if [[ ! -d "$BASE_MODEL" && "$USE_LOCAL_FILES_ONLY" -eq 0 ]]; then
  mkdir -p "$LOCAL_BASE_MODEL_DIR"
  echo "Downloading base model '$BASE_MODEL' into: $LOCAL_BASE_MODEL_DIR"
  download_hf_model "$BASE_MODEL" "$LOCAL_BASE_MODEL_DIR"
  TRAIN_MODEL="$LOCAL_BASE_MODEL_DIR"
  USE_LOCAL_FILES_ONLY=1
fi

TRAIN_LOCAL_ARGS=()
if [[ "$USE_LOCAL_FILES_ONLY" -eq 1 ]]; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  TRAIN_LOCAL_ARGS+=(--local-files-only)
else
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
  TRAIN_LOCAL_ARGS+=(--no-local-files-only)
fi

mkdir -p "$OUTPUT_ROOT"

echo "Base model: $BASE_MODEL"
echo "Training model path: $TRAIN_MODEL"
echo "Contrastive output: $CONTRASTIVE_OUTPUT_DIR"
echo "Final all-method JSON: $FINAL_JSON"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "LOCAL_FILES_ONLY for base model: $USE_LOCAL_FILES_ONLY"

TRAIN_ARGS=(
  contrastive_sft.py
  --model "$TRAIN_MODEL"
  --train-json "$CONTRASTIVE_TRAIN_JSON"
  --output-dir "$CONTRASTIVE_OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$TRAIN_LEARNING_RATE"
  --weight-decay "$TRAIN_WEIGHT_DECAY"
  --warmup-ratio "$TRAIN_WARMUP_RATIO"
  --max-source-length "$TRAIN_MAX_SOURCE_LENGTH"
  --max-target-length "$TRAIN_MAX_TARGET_LENGTH"
  --num-workers "$TRAIN_NUM_WORKERS"
  --alignment-weight "$TRAIN_ALIGNMENT_WEIGHT"
  --margin "$TRAIN_MARGIN"
  --negative-field "$TRAIN_NEGATIVE_FIELD"
  --ddp-timeout-minutes "$DDP_TIMEOUT_MINUTES"
  --expected-gpus "$NPROC_PER_NODE"
  --no-epoch-checkpoints
  --final-save-on-cpu
  --safe-serialization
  "${TRAIN_LOCAL_ARGS[@]}"
)

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "SKIP_TRAIN=1; reusing contrastive checkpoint: $CONTRASTIVE_OUTPUT_DIR"
else
  echo "Training 5-epoch contrastive SFT..."
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${TRAIN_ARGS[@]}"
  else
    "$PYTHON" "${TRAIN_ARGS[@]}" --allow-single-gpu
  fi
fi

if [[ -d "$TRAIN_MODEL" ]]; then
  "$PYTHON" scripts/repair_checkpoint_tokenizer.py \
    --checkpoint "$CONTRASTIVE_OUTPUT_DIR" \
    --source-tokenizer "$TRAIN_MODEL"
fi

AGG_REPORT_ARGS=()
for METHOD_LABEL in $PRUNE_METHODS; do
  case "$METHOD_LABEL" in
    nvidia|nvidia24|2of4|2:4)
      METHOD_ARG="nvidia"
      METHOD_KEY="nvidia24"
      ;;
    magnitude|gradient|wanda)
      METHOD_ARG="$METHOD_LABEL"
      METHOD_KEY="$METHOD_LABEL"
      ;;
    *)
      echo "Unknown PRUNE_METHODS entry: $METHOD_LABEL" >&2
      exit 2
      ;;
  esac

  METHOD_RUN_DIR="${OUTPUT_ROOT}/${METHOD_KEY}_50"
  METHOD_REPORT_JSON="${METHOD_RUN_DIR}/prune_eval_report.json"
  METHOD_PRUNED_DIR="${METHOD_RUN_DIR}/pruned_model"
  echo "Running ${METHOD_KEY} 50% prune/eval..."
  METHOD="$METHOD_ARG" \
    SPARSITY="$SPARSITY" \
    RUN_DIR="$METHOD_RUN_DIR" \
    REPORT_JSON="$METHOD_REPORT_JSON" \
    PRUNED_DIR="$METHOD_PRUNED_DIR" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    bash scripts/run_prune_eval_50.sh "$CONTRASTIVE_OUTPUT_DIR" "$@"
  AGG_REPORT_ARGS+=(--report "${METHOD_KEY}=${METHOD_REPORT_JSON}")
done

"$PYTHON" scripts/aggregate_prune_eval_reports.py \
  --output-json "$FINAL_JSON" \
  --base-model "$BASE_MODEL" \
  --contrastive-model "$CONTRASTIVE_OUTPUT_DIR" \
  --contrastive-train-json "$CONTRASTIVE_TRAIN_JSON" \
  --epochs "$EPOCHS" \
  --sparsity "$SPARSITY" \
  "${AGG_REPORT_ARGS[@]}"

echo "Done. All EM@1/EM@5 results are in: $FINAL_JSON"
