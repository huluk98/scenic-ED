#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh <base-model-path-or-hf-id> [extra scenic_prune_eval.py args]

Example:
  bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh charent/ChatLM-mini-Chinese

Useful env overrides:
  OUTPUT_ROOT=prune_eval_outputs/my_run
  FINAL_JSON=prune_eval_outputs/my_run/all_sft_contrastive_pruning_em_report.json
  SPARSITY_CHECK_JSON=prune_eval_outputs/my_run/sparsity_check.json
  NPROC_PER_NODE=8
  LOCAL_BASE_MODEL_DIR=prune_eval_outputs/my_run/base_model
  LOCAL_FILES_ONLY=1       # force local/offline base model loading
  LOCAL_FILES_ONLY=0       # allow Hugging Face base model loading
  IGNORE_SPACES=1          # default for Chinese EM; set 0 for strict whitespace-sensitive EM
  PRUNE_SCOPE=all-linear     # default for true full-model 50% sparsity
  SPARSITY_BASIS=full-model  # default: make the whole checkpoint 50% sparse
  SKIP_SPARSITY_CHECK=1    # optional: skip final sparsity verification
  SKIP_TRAIN=1             # reuse both SFT checkpoints and only prune/eval
  SKIP_REGULAR_TRAIN=1     # reuse regular SFT checkpoint
  SKIP_CONTRASTIVE_TRAIN=1 # reuse contrastive SFT checkpoint
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
    --model|--model=*|--method|--method=*|--sparsity|--sparsity=*|--sparsity-basis|--sparsity-basis=*|--output-json|--output-json=*|--pruned-output-dir|--pruned-output-dir=*)
      echo "This launcher controls model, method, sparsity, sparsity-basis, output-json, and pruned-output-dir. Use env overrides instead of: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-5}"
SPARSITY="${SPARSITY:-0.5}"
REGULAR_TRAIN_JSON="${REGULAR_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
EVAL_TRAIN_JSON="${EVAL_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$EVAL_TRAIN_JSON}"
PRUNE_METHODS="${PRUNE_METHODS:-magnitude wanda gradient nvidia24}"
PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
SPARSITY_BASIS="${SPARSITY_BASIS:-full-model}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-prune_eval_outputs/${SAFE_BASE}_sft_contrastive5_all50_${RUN_ID}}"
LOCAL_BASE_MODEL_DIR="${LOCAL_BASE_MODEL_DIR:-${OUTPUT_ROOT}/base_model}"
REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-${OUTPUT_ROOT}/regular_sft_5epoch}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${OUTPUT_ROOT}/contrastive_sft_5epoch}"
FINAL_JSON="${FINAL_JSON:-${OUTPUT_ROOT}/all_sft_contrastive_pruning_em_report.json}"
SPARSITY_CHECK_JSON="${SPARSITY_CHECK_JSON:-${OUTPUT_ROOT}/sparsity_check.json}"
IGNORE_SPACES="${IGNORE_SPACES:-1}"

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

PRECISION_ARGS=(--bf16)

EVAL_EXTRA_ARGS=(
  --train-json "$EVAL_TRAIN_JSON"
  --benchmark-json "$BENCHMARK_JSON"
  --calibration-json "$CALIBRATION_JSON"
)
EVAL_EXTRA_ARGS+=("$@")
case "$IGNORE_SPACES" in
  1|true|TRUE|yes|YES)
    EVAL_EXTRA_ARGS+=(--ignore-spaces)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "IGNORE_SPACES must be 1 or 0." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"

echo "Base model: $BASE_MODEL"
echo "Training model path: $TRAIN_MODEL"
echo "Regular SFT output: $REGULAR_OUTPUT_DIR"
echo "Contrastive SFT output: $CONTRASTIVE_OUTPUT_DIR"
echo "Final all-model JSON: $FINAL_JSON"
echo "Sparsity check JSON: $SPARSITY_CHECK_JSON"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "LOCAL_FILES_ONLY for base model: $USE_LOCAL_FILES_ONLY"
echo "Regular train data: $REGULAR_TRAIN_JSON"
echo "Contrastive train data: $CONTRASTIVE_TRAIN_JSON"
echo "Eval train data: $EVAL_TRAIN_JSON"
echo "Benchmark data: $BENCHMARK_JSON"
echo "Ignore whitespace in Chinese exact-match eval: $IGNORE_SPACES"
echo "Prune scope: $PRUNE_SCOPE"
echo "Sparsity basis: $SPARSITY_BASIS"

REGULAR_TRAIN_ARGS=(
  scripts/scenic_train_chatlm_sft.py
  --mode regular
  --model "$TRAIN_MODEL"
  --train-json "$REGULAR_TRAIN_JSON"
  --output-dir "$REGULAR_OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$TRAIN_BATCH_SIZE"
  --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$TRAIN_LEARNING_RATE"
  --weight-decay "$TRAIN_WEIGHT_DECAY"
  --warmup-ratio "$TRAIN_WARMUP_RATIO"
  --max-source-length "$TRAIN_MAX_SOURCE_LENGTH"
  --max-target-length "$TRAIN_MAX_TARGET_LENGTH"
  --num-workers "$TRAIN_NUM_WORKERS"
  --ddp-timeout-minutes "$DDP_TIMEOUT_MINUTES"
  --no-epoch-checkpoints
  --final-save-on-cpu
  --safe-serialization
  "${PRECISION_ARGS[@]}"
  "${TRAIN_LOCAL_ARGS[@]}"
)

CONTRASTIVE_TRAIN_ARGS=(
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
  "${PRECISION_ARGS[@]}"
  "${TRAIN_LOCAL_ARGS[@]}"
)

if [[ "${SKIP_TRAIN:-0}" == "1" || "${SKIP_REGULAR_TRAIN:-0}" == "1" ]]; then
  echo "Reusing regular SFT checkpoint: $REGULAR_OUTPUT_DIR"
else
  echo "Training 5-epoch regular SFT..."
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${REGULAR_TRAIN_ARGS[@]}"
  else
    "$PYTHON" "${REGULAR_TRAIN_ARGS[@]}"
  fi
fi

if [[ "${SKIP_TRAIN:-0}" == "1" || "${SKIP_CONTRASTIVE_TRAIN:-0}" == "1" ]]; then
  echo "Reusing contrastive SFT checkpoint: $CONTRASTIVE_OUTPUT_DIR"
else
  echo "Training 5-epoch contrastive SFT..."
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${CONTRASTIVE_TRAIN_ARGS[@]}"
  else
    "$PYTHON" "${CONTRASTIVE_TRAIN_ARGS[@]}" --allow-single-gpu
  fi
fi

if [[ -d "$TRAIN_MODEL" ]]; then
  "$PYTHON" scripts/repair_checkpoint_tokenizer.py \
    --checkpoint "$REGULAR_OUTPUT_DIR" \
    --source-tokenizer "$TRAIN_MODEL"
  "$PYTHON" scripts/repair_checkpoint_tokenizer.py \
    --checkpoint "$CONTRASTIVE_OUTPUT_DIR" \
    --source-tokenizer "$TRAIN_MODEL"
fi

MULTI_AGG_ARGS=(
  --model "regular_sft=${REGULAR_OUTPUT_DIR}"
  --model "contrastive_sft=${CONTRASTIVE_OUTPUT_DIR}"
  --train-json "regular_sft=${REGULAR_TRAIN_JSON}"
  --train-json "contrastive_sft=${CONTRASTIVE_TRAIN_JSON}"
)

run_prune_suite() {
  local model_label="$1"
  local model_dir="$2"
  for METHOD_LABEL in $PRUNE_METHODS; do
    local METHOD_ARG
    local METHOD_KEY
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

    local METHOD_RUN_DIR="${OUTPUT_ROOT}/${model_label}/${METHOD_KEY}_50"
    local METHOD_REPORT_JSON="${METHOD_RUN_DIR}/prune_eval_report.json"
    local METHOD_PRUNED_DIR="${METHOD_RUN_DIR}/pruned_model"
    echo "Running ${model_label} ${METHOD_KEY} 50% prune/eval..."
    METHOD="$METHOD_ARG" \
      SPARSITY="$SPARSITY" \
      RUN_DIR="$METHOD_RUN_DIR" \
      REPORT_JSON="$METHOD_REPORT_JSON" \
      PRUNED_DIR="$METHOD_PRUNED_DIR" \
      NPROC_PER_NODE="$NPROC_PER_NODE" \
      PRUNE_SCOPE="$PRUNE_SCOPE" \
      SPARSITY_BASIS="$SPARSITY_BASIS" \
      bash scripts/run_prune_eval_50.sh "$model_dir" "${EVAL_EXTRA_ARGS[@]}"
    MULTI_AGG_ARGS+=(--report "${model_label}:${METHOD_KEY}=${METHOD_REPORT_JSON}")
  done
}

run_prune_suite "regular_sft" "$REGULAR_OUTPUT_DIR"
run_prune_suite "contrastive_sft" "$CONTRASTIVE_OUTPUT_DIR"

"$PYTHON" scripts/aggregate_multi_model_prune_eval_reports.py \
  --output-json "$FINAL_JSON" \
  --base-model "$BASE_MODEL" \
  --epochs "$EPOCHS" \
  --sparsity "$SPARSITY" \
  "${MULTI_AGG_ARGS[@]}"

if [[ "${SKIP_SPARSITY_CHECK:-0}" == "1" ]]; then
  echo "SKIP_SPARSITY_CHECK=1; skipping final sparsity verification."
else
  "$PYTHON" scripts/check_pruned_model_sparsity.py \
    --report-json "$FINAL_JSON" \
    --output-json "$SPARSITY_CHECK_JSON" \
    --expected-sparsity "$SPARSITY" \
    --default-prune-scope "$PRUNE_SCOPE" \
    --default-sparsity-basis "$SPARSITY_BASIS" \
    --fail-on-mismatch
fi

echo "Done. Regular + contrastive EM@1/EM@5 results are in: $FINAL_JSON"
echo "Sparsity verification is in: $SPARSITY_CHECK_JSON"
