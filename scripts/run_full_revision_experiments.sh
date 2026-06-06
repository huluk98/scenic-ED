#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_full_revision_experiments.sh [base-model-path-or-hf-id]

Default base model:
  charent/ChatLM-mini-Chinese

This one launcher:
  1. Fine-tunes the original model for 5 epochs with regular SFT.
  2. Fine-tunes the original model for 5 epochs with contrastive/triplet SFT.
  3. Runs the existing one-shot 50% pruning/eval suite for both checkpoints.
  4. Runs the added one-shot 30% pruning/eval suite for both checkpoints.
  5. Runs the new 0/30/50 linear sparsity matrix for both checkpoints:
       dense_0, oneshot_30, oneshot_50, progressive_30, progressive_50
     Progressive methods default to one final recovery epoch total.
  6. Runs ONNX FP32/FP16/INT8 precision benchmark tables for both checkpoints.

Common overrides:
  REVISION_OUTPUT_ROOT=results/scenic_revision_full_run
  EPOCHS=5
  SEED=42
  NPROC_PER_NODE=8
  LOCAL_FILES_ONLY=0
  RUN_LEGACY_PRUNE=1
  RUN_LEGACY_ONESHOT_30=1
  RUN_SPARSITY=1
  RUN_ONNX=1
  LEGACY_30_PRUNE_METHODS="magnitude wanda gradient"
  SPARSITY_LEVELS="0 0.3 0.5"
  SPARSITY_PRUNING_MODES="dense oneshot progressive"
  SPARSITY_RECOVERY_EPOCHS_PER_STAGE=0
  SPARSITY_FINAL_RECOVERY_EPOCHS=1
  ONNX_BENCHMARK_PROVIDERS="CPUExecutionProvider CUDAExecutionProvider"
  DEVICE_NAME="Jetson Orin Nano"
  POWER_LOG=/path/to/power.csv

Smoke test:
  EPOCHS=1 MAX_BENCHMARK_EXAMPLES=20 MAX_TRAIN_EXAMPLES=20 ONNX_PRECISION_RUNS=20 \
  bash scripts/run_full_revision_experiments.sh
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"

REVISION_OUTPUT_ROOT="${REVISION_OUTPUT_ROOT:-results/${SAFE_BASE}_full_revision_${RUN_ID}}"
LEGACY_OUTPUT_ROOT="${LEGACY_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/legacy_regular_contrastive_5epoch_prune50}"
REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-${LEGACY_OUTPUT_ROOT}/regular_sft_5epoch}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${LEGACY_OUTPUT_ROOT}/contrastive_sft_5epoch}"
LEGACY_30_OUTPUT_ROOT="${LEGACY_30_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/legacy_oneshot_30}"
SPARSITY_OUTPUT_ROOT="${SPARSITY_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/linear_sparsity_0_30_50}"
ONNX_OUTPUT_ROOT="${ONNX_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/onnx_precision}"
FINAL_MANIFEST="${FINAL_MANIFEST:-${REVISION_OUTPUT_ROOT}/full_revision_manifest.txt}"

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
REGULAR_TRAIN_JSON="${REGULAR_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
EVAL_TRAIN_JSON="${EVAL_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$EVAL_TRAIN_JSON}"
MODEL_FAMILY="${MODEL_FAMILY:-encoder_decoder}"
IGNORE_SPACES="${IGNORE_SPACES:-1}"

RUN_LEGACY_PRUNE="${RUN_LEGACY_PRUNE:-1}"
RUN_LEGACY_ONESHOT_30="${RUN_LEGACY_ONESHOT_30:-1}"
RUN_SPARSITY="${RUN_SPARSITY:-1}"
RUN_ONNX="${RUN_ONNX:-1}"
LEGACY_30_PRUNE_METHODS="${LEGACY_30_PRUNE_METHODS:-magnitude wanda gradient}"

SPARSITY_LEVELS="${SPARSITY_LEVELS:-0 0.3 0.5}"
SPARSITY_PRUNING_MODES="${SPARSITY_PRUNING_MODES:-dense oneshot progressive}"
SPARSITY_RECOVERY_EPOCHS_PER_STAGE="${SPARSITY_RECOVERY_EPOCHS_PER_STAGE:-0}"
SPARSITY_FINAL_RECOVERY_EPOCHS="${SPARSITY_FINAL_RECOVERY_EPOCHS:-1}"
SPARSITY_NUM_BEAMS="${SPARSITY_NUM_BEAMS:-5}"
SPARSITY_NUM_RETURN_SEQUENCES="${SPARSITY_NUM_RETURN_SEQUENCES:-5}"
SPARSITY_MAX_NEW_TOKENS="${SPARSITY_MAX_NEW_TOKENS:-128}"

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$REVISION_OUTPUT_ROOT"

echo "Full SCENIC revision run"
echo "Base model: $BASE_MODEL"
echo "Output root: $REVISION_OUTPUT_ROOT"
echo "Regular SFT checkpoint: $REGULAR_OUTPUT_DIR"
echo "Contrastive SFT checkpoint: $CONTRASTIVE_OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Seed: $SEED"

legacy_eval_args=()
if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
  legacy_eval_args+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
  legacy_eval_args+=(--max-benchmark-examples "$MAX_BENCHMARK_EXAMPLES")
fi

if truthy "$RUN_LEGACY_PRUNE"; then
  echo
  echo "== Step 1: regular + contrastive 5-epoch SFT and existing one-shot 50% prune/eval =="
  OUTPUT_ROOT="$LEGACY_OUTPUT_ROOT" \
  REGULAR_OUTPUT_DIR="$REGULAR_OUTPUT_DIR" \
  CONTRASTIVE_OUTPUT_DIR="$CONTRASTIVE_OUTPUT_DIR" \
  EPOCHS="$EPOCHS" \
  REGULAR_TRAIN_JSON="$REGULAR_TRAIN_JSON" \
  CONTRASTIVE_TRAIN_JSON="$CONTRASTIVE_TRAIN_JSON" \
  EVAL_TRAIN_JSON="$EVAL_TRAIN_JSON" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  CALIBRATION_JSON="$CALIBRATION_JSON" \
  IGNORE_SPACES="$IGNORE_SPACES" \
  bash scripts/run_sft_contrastive_5epoch_all_prune_50.sh "$BASE_MODEL" "${legacy_eval_args[@]}"
else
  echo "RUN_LEGACY_PRUNE=0; expecting checkpoints to already exist."
fi

if [[ ! -f "${REGULAR_OUTPUT_DIR}/config.json" ]]; then
  echo "Missing regular checkpoint after training step: ${REGULAR_OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CONTRASTIVE_OUTPUT_DIR}/config.json" ]]; then
  echo "Missing contrastive checkpoint after training step: ${CONTRASTIVE_OUTPUT_DIR}" >&2
  exit 1
fi

if truthy "$RUN_LEGACY_ONESHOT_30"; then
  echo
  echo "== Step 1b: added one-shot 30% prune/eval using existing checkpoints =="
  PYTHON="$PYTHON" \
  BASE_MODEL="$BASE_MODEL" \
  EPOCHS="$EPOCHS" \
  SEED="$SEED" \
  REVISION_OUTPUT_ROOT="$REVISION_OUTPUT_ROOT" \
  LEGACY_OUTPUT_ROOT="$LEGACY_OUTPUT_ROOT" \
  REGULAR_OUTPUT_DIR="$REGULAR_OUTPUT_DIR" \
  CONTRASTIVE_OUTPUT_DIR="$CONTRASTIVE_OUTPUT_DIR" \
  LEGACY_30_OUTPUT_ROOT="$LEGACY_30_OUTPUT_ROOT" \
  REGULAR_TRAIN_JSON="$REGULAR_TRAIN_JSON" \
  CONTRASTIVE_TRAIN_JSON="$CONTRASTIVE_TRAIN_JSON" \
  EVAL_TRAIN_JSON="$EVAL_TRAIN_JSON" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  CALIBRATION_JSON="$CALIBRATION_JSON" \
  LEGACY_30_PRUNE_METHODS="$LEGACY_30_PRUNE_METHODS" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-}" \
  PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}" \
  SPARSITY_BASIS="${SPARSITY_BASIS:-targeted-linear}" \
  PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-0}" \
  IGNORE_SPACES="$IGNORE_SPACES" \
  MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}" \
  MAX_BENCHMARK_EXAMPLES="${MAX_BENCHMARK_EXAMPLES:-}" \
  MAX_CALIBRATION_EXAMPLES="${MAX_CALIBRATION_EXAMPLES:-}" \
  RUN_LEGACY_ONESHOT_30=1 \
  RUN_SPARSITY_30=0 \
  bash scripts/run_30pct_revision_experiments.sh "$REVISION_OUTPUT_ROOT"
else
  echo "RUN_LEGACY_ONESHOT_30=0; skipping legacy one-shot 30% suite."
fi

run_sparsity_matrix() {
  local label="$1"
  local checkpoint="$2"
  local train_json="$3"
  local output_dir="${SPARSITY_OUTPUT_ROOT}/${label}"
  local extra_args=()
  if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
    extra_args+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
  fi
  if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
    extra_args+=(--max_benchmark_examples "$MAX_BENCHMARK_EXAMPLES")
  fi

  echo
  echo "== Step 2/${label}: 0/30/50 linear sparsity matrix =="
  # shellcheck disable=SC2206
  local sparsity_levels=( $SPARSITY_LEVELS )
  # shellcheck disable=SC2206
  local pruning_modes=( $SPARSITY_PRUNING_MODES )
  "$PYTHON" scripts/run_sparsity_experiments.py \
    --experiment_name "scenic_${label}_linear_sparsity_0_30_50" \
    --model_family "$MODEL_FAMILY" \
    --model_checkpoint "$checkpoint" \
    --benchmark_path "$BENCHMARK_JSON" \
    --train_path "$train_json" \
    --sparsity_levels "${sparsity_levels[@]}" \
    --pruning_modes "${pruning_modes[@]}" \
    --prune_scope linear_weights \
    --prune_method magnitude \
    --recovery_epochs_per_stage "$SPARSITY_RECOVERY_EPOCHS_PER_STAGE" \
    --final_recovery_epochs "$SPARSITY_FINAL_RECOVERY_EPOCHS" \
    --num_beams "$SPARSITY_NUM_BEAMS" \
    --num_return_sequences "$SPARSITY_NUM_RETURN_SEQUENCES" \
    --max_new_tokens "$SPARSITY_MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --output_dir "$output_dir" \
    "${extra_args[@]}"
}

if truthy "$RUN_SPARSITY"; then
  run_sparsity_matrix "regular_sft" "$REGULAR_OUTPUT_DIR" "$REGULAR_TRAIN_JSON"
  run_sparsity_matrix "contrastive_sft" "$CONTRASTIVE_OUTPUT_DIR" "$CONTRASTIVE_TRAIN_JSON"
else
  echo "RUN_SPARSITY=0; skipping linear sparsity matrix."
fi

run_onnx_precision() {
  local label="$1"
  local checkpoint="$2"
  local train_json="$3"
  local output_root="${ONNX_OUTPUT_ROOT}/${label}"

  echo
  echo "== Step 3/${label}: ONNX FP32/FP16/INT8 precision benchmark =="
  OUTPUT_ROOT="$output_root" \
  FINETUNE_OUTPUT_DIR="$checkpoint" \
  FINETUNE_TRAIN_JSON="$train_json" \
  TRAIN_JSON="$train_json" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  SKIP_TRAIN=1 \
  RUN_INT8="${RUN_INT8:-1}" \
  bash scripts/run_5epoch_onnx_precision_benchmark.sh "$BASE_MODEL"
}

if truthy "$RUN_ONNX"; then
  run_onnx_precision "regular_sft" "$REGULAR_OUTPUT_DIR" "$REGULAR_TRAIN_JSON"
  run_onnx_precision "contrastive_sft" "$CONTRASTIVE_OUTPUT_DIR" "$CONTRASTIVE_TRAIN_JSON"
else
  echo "RUN_ONNX=0; skipping ONNX precision benchmarks."
fi

cat > "$FINAL_MANIFEST" <<EOF
SCENIC full revision run
created_at=${RUN_ID}
base_model=${BASE_MODEL}
output_root=${REVISION_OUTPUT_ROOT}
legacy_report=${LEGACY_OUTPUT_ROOT}/all_sft_contrastive_pruning_em_report.json
legacy_sparsity_check=${LEGACY_OUTPUT_ROOT}/sparsity_check.json
legacy_30_report=${LEGACY_30_OUTPUT_ROOT}/all_sft_contrastive_pruning_em_report_30.json
legacy_30_sparsity_check=${LEGACY_30_OUTPUT_ROOT}/sparsity_check_30.json
regular_checkpoint=${REGULAR_OUTPUT_DIR}
contrastive_checkpoint=${CONTRASTIVE_OUTPUT_DIR}
regular_sparsity_summary=${SPARSITY_OUTPUT_ROOT}/regular_sft/summary_metrics.csv
contrastive_sparsity_summary=${SPARSITY_OUTPUT_ROOT}/contrastive_sft/summary_metrics.csv
regular_onnx_table=${ONNX_OUTPUT_ROOT}/regular_sft/reports/onnx_precision_benchmark/onnx_precision_benchmark.md
contrastive_onnx_table=${ONNX_OUTPUT_ROOT}/contrastive_sft/reports/onnx_precision_benchmark/onnx_precision_benchmark.md
EOF

echo
echo "Done. Manifest: $FINAL_MANIFEST"
