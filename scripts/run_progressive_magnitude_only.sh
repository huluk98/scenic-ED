#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run only the progressive magnitude sparsity rows for existing SCENIC checkpoints.

This skips SFT, legacy pruning, one-shot magnitude, Wanda, gradient, NVIDIA 2:4,
and ONNX. By default it runs progressive magnitude at 30% and 50% targeted
Linear sparsity for both regular_sft and contrastive_sft checkpoints.

Required checkpoint input, one of:
  SOURCE_REVISION_ROOT=results/ChatLM-mini-Chinese_full_revision_YYYY...
  REGULAR_CHECKPOINT=/path/to/regular_sft_5epoch
  CONTRASTIVE_CHECKPOINT=/path/to/contrastive_sft_5epoch

Examples:
  SOURCE_REVISION_ROOT=results/ChatLM-mini-Chinese_full_revision_20260606T132621Z \
    bash scripts/run_progressive_magnitude_only.sh

  RUN_CONTRASTIVE=0 REGULAR_CHECKPOINT=results/.../regular_sft_5epoch \
    OUTPUT_ROOT=results/rebuild_regular_progressive \
    bash scripts/run_progressive_magnitude_only.sh

Useful environment variables:
  OUTPUT_ROOT                         default: results/<base>_progressive_magnitude_<utc run id>
  SPARSITY_LEVELS                     default: "0.3 0.5"
  RUN_DENSE_BASELINE                  default: 0; set 1 to include dense 0% for retention
  RUN_REGULAR                         default: 1
  RUN_CONTRASTIVE                     default: 1
  REGULAR_TRAIN_JSON                  default: data/SCENIC_full_training_dataset.json
  CONTRASTIVE_TRAIN_JSON              default: data/SCENIC_full_anchor_positive_negative.json
  BENCHMARK_JSON                      default: generated/iot_instruction_benchmark_200.json
  SPARSITY_RECOVERY_EPOCHS_PER_STAGE  default: 1
  SPARSITY_FINAL_RECOVERY_EPOCHS      default: 1
  SPARSITY_NUM_BEAMS                  default: 5
  SPARSITY_NUM_RETURN_SEQUENCES       default: 5
  SPARSITY_MAX_NEW_TOKENS             default: 128
  SEED                                default: 42
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

BASE_MODEL="${BASE_MODEL:-charent/ChatLM-mini-Chinese}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SAFE_BASE="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_BASE="${SAFE_BASE%_}"
SAFE_BASE="${SAFE_BASE:-chatlm}"

PYTHON="${PYTHON:-python}"
MODEL_FAMILY="${MODEL_FAMILY:-encoder_decoder}"
SEED="${SEED:-42}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/${SAFE_BASE}_progressive_magnitude_${RUN_ID}}"
SOURCE_REVISION_ROOT="${SOURCE_REVISION_ROOT:-${REVISION_OUTPUT_ROOT:-}}"
LEGACY_OUTPUT_ROOT="${LEGACY_OUTPUT_ROOT:-}"

REGULAR_CHECKPOINT="${REGULAR_CHECKPOINT:-${REGULAR_OUTPUT_DIR:-}}"
CONTRASTIVE_CHECKPOINT="${CONTRASTIVE_CHECKPOINT:-${CONTRASTIVE_OUTPUT_DIR:-}}"
if [[ -z "$REGULAR_CHECKPOINT" && -n "$LEGACY_OUTPUT_ROOT" ]]; then
  REGULAR_CHECKPOINT="${LEGACY_OUTPUT_ROOT}/regular_sft_5epoch"
fi
if [[ -z "$CONTRASTIVE_CHECKPOINT" && -n "$LEGACY_OUTPUT_ROOT" ]]; then
  CONTRASTIVE_CHECKPOINT="${LEGACY_OUTPUT_ROOT}/contrastive_sft_5epoch"
fi
if [[ -z "$REGULAR_CHECKPOINT" && -n "$SOURCE_REVISION_ROOT" ]]; then
  REGULAR_CHECKPOINT="${SOURCE_REVISION_ROOT}/legacy_regular_contrastive_5epoch_prune50/regular_sft_5epoch"
fi
if [[ -z "$CONTRASTIVE_CHECKPOINT" && -n "$SOURCE_REVISION_ROOT" ]]; then
  CONTRASTIVE_CHECKPOINT="${SOURCE_REVISION_ROOT}/legacy_regular_contrastive_5epoch_prune50/contrastive_sft_5epoch"
fi

REGULAR_TRAIN_JSON="${REGULAR_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
SPARSITY_LEVELS="${SPARSITY_LEVELS:-0.3 0.5}"
RUN_DENSE_BASELINE="${RUN_DENSE_BASELINE:-0}"
RUN_REGULAR="${RUN_REGULAR:-1}"
RUN_CONTRASTIVE="${RUN_CONTRASTIVE:-1}"
SPARSITY_RECOVERY_EPOCHS_PER_STAGE="${SPARSITY_RECOVERY_EPOCHS_PER_STAGE:-1}"
SPARSITY_FINAL_RECOVERY_EPOCHS="${SPARSITY_FINAL_RECOVERY_EPOCHS:-1}"
SPARSITY_NUM_BEAMS="${SPARSITY_NUM_BEAMS:-5}"
SPARSITY_NUM_RETURN_SEQUENCES="${SPARSITY_NUM_RETURN_SEQUENCES:-5}"
SPARSITY_MAX_NEW_TOKENS="${SPARSITY_MAX_NEW_TOKENS:-128}"
SPARSITY_BATCH_SIZE="${SPARSITY_BATCH_SIZE:-4}"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-ignore_spaces}"
AUDIT_FAIL_ON_WARNING="${AUDIT_FAIL_ON_WARNING:-0}"

read -r -a sparsity_levels <<< "$SPARSITY_LEVELS"
pruning_modes=(progressive)
audit_expected_rows=()
if truthy "$RUN_DENSE_BASELINE"; then
  pruning_modes=(dense progressive)
  audit_expected_rows+=(--expected-row dense:0.0)
fi
for level in "${sparsity_levels[@]}"; do
  audit_expected_rows+=(--expected-row "progressive:${level}")
done

extra_args=()
if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
  extra_args+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
  extra_args+=(--max_benchmark_examples "$MAX_BENCHMARK_EXAMPLES")
fi
if [[ -n "${VALIDATION_JSON:-}" ]]; then
  extra_args+=(--validation_path "$VALIDATION_JSON")
fi
if [[ -n "${MAX_VALIDATION_EXAMPLES:-}" ]]; then
  extra_args+=(--max_validation_examples "$MAX_VALIDATION_EXAMPLES")
fi
if [[ -n "${DEVICE:-}" ]]; then
  extra_args+=(--device "$DEVICE")
fi
if truthy "${BF16:-0}"; then
  extra_args+=(--bf16)
fi
if truthy "${FP16:-0}"; then
  extra_args+=(--fp16)
fi
if ! truthy "${LOCAL_FILES_ONLY:-1}"; then
  extra_args+=(--no-local_files_only)
fi

run_progressive() {
  local label="$1"
  local checkpoint="$2"
  local train_json="$3"
  local output_dir="${OUTPUT_ROOT}/${label}"
  local checkpoint_hint="REGULAR_CHECKPOINT"
  if [[ "$label" == "contrastive_sft" ]]; then
    checkpoint_hint="CONTRASTIVE_CHECKPOINT"
  fi

  if [[ -z "$checkpoint" ]]; then
    echo "Missing checkpoint for ${label}. Set ${checkpoint_hint} or SOURCE_REVISION_ROOT." >&2
    exit 2
  fi
  if [[ ! -f "${checkpoint}/config.json" ]]; then
    echo "Checkpoint does not look valid for ${label}: ${checkpoint}" >&2
    echo "Expected ${checkpoint}/config.json" >&2
    exit 2
  fi
  if [[ ! -f "$train_json" ]]; then
    echo "Missing train JSON for ${label}: ${train_json}" >&2
    exit 2
  fi

  mkdir -p "$output_dir"
  echo
  echo "== Progressive magnitude only: ${label} =="
  echo "Checkpoint: ${checkpoint}"
  echo "Train JSON: ${train_json}"
  echo "Output: ${output_dir}"

  "$PYTHON" scripts/run_sparsity_experiments.py \
    --experiment_name "scenic_${label}_progressive_magnitude_only" \
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
    --normalization_mode "$NORMALIZATION_MODE" \
    --batch_size "$SPARSITY_BATCH_SIZE" \
    --seed "$SEED" \
    --output_dir "$output_dir" \
    "${extra_args[@]}"

  audit_args=(
    --summary-csv "${output_dir}/summary_metrics.csv"
    --output-json "${output_dir}/audit.json"
    "${audit_expected_rows[@]}"
  )
  if truthy "$AUDIT_FAIL_ON_WARNING"; then
    audit_args+=(--fail-on-warning)
  fi
  "$PYTHON" scripts/audit_sparsity_outputs.py "${audit_args[@]}"
}

mkdir -p "$OUTPUT_ROOT"

echo "SCENIC progressive magnitude-only run"
echo "Output root: ${OUTPUT_ROOT}"
echo "Sparsity levels: ${SPARSITY_LEVELS}"
echo "Dense baseline: ${RUN_DENSE_BASELINE}"

ran_any=0
if truthy "$RUN_REGULAR"; then
  run_progressive "regular_sft" "$REGULAR_CHECKPOINT" "$REGULAR_TRAIN_JSON"
  ran_any=1
fi
if truthy "$RUN_CONTRASTIVE"; then
  run_progressive "contrastive_sft" "$CONTRASTIVE_CHECKPOINT" "$CONTRASTIVE_TRAIN_JSON"
  ran_any=1
fi
if [[ "$ran_any" -eq 0 ]]; then
  echo "Nothing selected. Set RUN_REGULAR=1 or RUN_CONTRASTIVE=1." >&2
  exit 2
fi

cat > "${OUTPUT_ROOT}/progressive_magnitude_manifest.txt" <<EOF
SCENIC progressive magnitude-only run
created_at=${RUN_ID}
base_model=${BASE_MODEL}
output_root=${OUTPUT_ROOT}
model_family=${MODEL_FAMILY}
seed=${SEED}
sparsity_levels=${SPARSITY_LEVELS}
run_dense_baseline=${RUN_DENSE_BASELINE}
regular_checkpoint=${REGULAR_CHECKPOINT}
contrastive_checkpoint=${CONTRASTIVE_CHECKPOINT}
regular_summary=${OUTPUT_ROOT}/regular_sft/summary_metrics.csv
contrastive_summary=${OUTPUT_ROOT}/contrastive_sft/summary_metrics.csv
EOF

echo
echo "Done. Manifest: ${OUTPUT_ROOT}/progressive_magnitude_manifest.txt"
