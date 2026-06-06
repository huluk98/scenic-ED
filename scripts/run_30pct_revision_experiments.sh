#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_30pct_revision_experiments.sh [full-revision-output-root] [extra scenic_prune_eval.py args]

Default behavior:
  Reuse the regular and contrastive 5-epoch checkpoints from a completed
  run_full_revision_experiments.sh output directory, then run only the missing
  30% legacy one-shot pruning/eval methods.

If no output root is provided, the latest results/*_full_revision_* directory is used.

Common overrides:
  REVISION_OUTPUT_ROOT=results/ChatLM-mini-Chinese_full_revision_...
  LEGACY_30_PRUNE_METHODS="magnitude wanda gradient"
  INCLUDE_NVIDIA24_30=0       # nvidia24 is 2:4 and therefore effectively 50%
  FORCE_RERUN=0               # skip completed reports by default
  RUN_LEGACY_ONESHOT_30=1
  RUN_SPARSITY_30=0           # set 1 to also run only 30% linear oneshot/progressive
  SPARSITY_PRUNING_MODES_30="oneshot progressive"
  SPARSITY_RECOVERY_EPOCHS_PER_STAGE=1
  SPARSITY_FINAL_RECOVERY_EPOCHS=1

Examples:
  bash scripts/run_30pct_revision_experiments.sh results/ChatLM-mini-Chinese_full_revision_20260606T000000Z

  RUN_SPARSITY_30=1 bash scripts/run_30pct_revision_experiments.sh \
    results/ChatLM-mini-Chinese_full_revision_20260606T000000Z
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

find_latest_full_run() {
  find results -maxdepth 1 -type d -name '*_full_revision_*' 2>/dev/null | sort | tail -n 1 || true
}

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

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

REVISION_OUTPUT_ROOT="${REVISION_OUTPUT_ROOT:-}"
if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  REVISION_OUTPUT_ROOT="$1"
  shift
fi
if [[ -z "$REVISION_OUTPUT_ROOT" ]]; then
  REVISION_OUTPUT_ROOT="$(find_latest_full_run)"
fi
if [[ -z "$REVISION_OUTPUT_ROOT" || ! -d "$REVISION_OUTPUT_ROOT" ]]; then
  echo "No full revision output root found. Pass one explicitly or set REVISION_OUTPUT_ROOT." >&2
  exit 2
fi

FULL_MANIFEST="${FULL_MANIFEST:-${REVISION_OUTPUT_ROOT}/full_revision_manifest.txt}"
BASE_MODEL="${BASE_MODEL:-}"
if [[ -z "$BASE_MODEL" && -f "$FULL_MANIFEST" ]]; then
  BASE_MODEL="$(awk -F= '$1=="base_model"{print substr($0,index($0,"=")+1); exit}' "$FULL_MANIFEST")"
fi
BASE_MODEL="${BASE_MODEL:-charent/ChatLM-mini-Chinese}"

PYTHON="${PYTHON:-python}"
EPOCHS="${EPOCHS:-5}"
SEED="${SEED:-42}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
REGULAR_TRAIN_JSON="${REGULAR_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CONTRASTIVE_TRAIN_JSON="${CONTRASTIVE_TRAIN_JSON:-data/SCENIC_full_anchor_positive_negative.json}"
EVAL_TRAIN_JSON="${EVAL_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$EVAL_TRAIN_JSON}"
MODEL_FAMILY="${MODEL_FAMILY:-encoder_decoder}"

LEGACY_OUTPUT_ROOT="${LEGACY_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/legacy_regular_contrastive_5epoch_prune50}"
REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-${LEGACY_OUTPUT_ROOT}/regular_sft_5epoch}"
CONTRASTIVE_OUTPUT_DIR="${CONTRASTIVE_OUTPUT_DIR:-${LEGACY_OUTPUT_ROOT}/contrastive_sft_5epoch}"
LEGACY_30_OUTPUT_ROOT="${LEGACY_30_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/legacy_oneshot_30}"
LEGACY_30_FINAL_JSON="${LEGACY_30_FINAL_JSON:-${LEGACY_30_OUTPUT_ROOT}/all_sft_contrastive_pruning_em_report_30.json}"
LEGACY_30_SPARSITY_CHECK_JSON="${LEGACY_30_SPARSITY_CHECK_JSON:-${LEGACY_30_OUTPUT_ROOT}/sparsity_check_30.json}"
SPARSITY_30_OUTPUT_ROOT="${SPARSITY_30_OUTPUT_ROOT:-${REVISION_OUTPUT_ROOT}/linear_sparsity_30_only}"
THIRTY_MANIFEST="${THIRTY_MANIFEST:-${REVISION_OUTPUT_ROOT}/thirty_percent_manifest.txt}"

RUN_LEGACY_ONESHOT_30="${RUN_LEGACY_ONESHOT_30:-1}"
RUN_SPARSITY_30="${RUN_SPARSITY_30:-0}"
RUN_SPARSITY_CHECK_30="${RUN_SPARSITY_CHECK_30:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

LEGACY_30_PRUNE_METHODS="${LEGACY_30_PRUNE_METHODS:-${PRUNE_METHODS_30:-magnitude wanda gradient}}"
if truthy "${INCLUDE_NVIDIA24_30:-0}"; then
  LEGACY_30_PRUNE_METHODS="${LEGACY_30_PRUNE_METHODS} nvidia24"
fi
PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
SPARSITY_BASIS="${SPARSITY_BASIS:-targeted-linear}"
PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-0}"
IGNORE_SPACES="${IGNORE_SPACES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_nproc)}"

SPARSITY_PRUNING_MODES_30="${SPARSITY_PRUNING_MODES_30:-oneshot progressive}"
SPARSITY_RECOVERY_EPOCHS_PER_STAGE="${SPARSITY_RECOVERY_EPOCHS_PER_STAGE:-1}"
SPARSITY_FINAL_RECOVERY_EPOCHS="${SPARSITY_FINAL_RECOVERY_EPOCHS:-1}"
SPARSITY_NUM_BEAMS="${SPARSITY_NUM_BEAMS:-5}"
SPARSITY_NUM_RETURN_SEQUENCES="${SPARSITY_NUM_RETURN_SEQUENCES:-5}"
SPARSITY_MAX_NEW_TOKENS="${SPARSITY_MAX_NEW_TOKENS:-128}"
SPARSITY_NORMALIZATION_MODE="${SPARSITY_NORMALIZATION_MODE:-ignore_spaces}"

if [[ ! -f "${REGULAR_OUTPUT_DIR}/config.json" ]]; then
  echo "Missing regular 5-epoch checkpoint: ${REGULAR_OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${CONTRASTIVE_OUTPUT_DIR}/config.json" ]]; then
  echo "Missing contrastive 5-epoch checkpoint: ${CONTRASTIVE_OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "$LEGACY_30_OUTPUT_ROOT" "$SPARSITY_30_OUTPUT_ROOT"

echo "SCENIC 30%-only follow-up run"
echo "Full revision root: $REVISION_OUTPUT_ROOT"
echo "Base model metadata: $BASE_MODEL"
echo "Regular SFT checkpoint: $REGULAR_OUTPUT_DIR"
echo "Contrastive SFT checkpoint: $CONTRASTIVE_OUTPUT_DIR"
echo "Legacy 30% output: $LEGACY_30_OUTPUT_ROOT"
echo "Linear 30-only output: $SPARSITY_30_OUTPUT_ROOT"

eval_extra_args=(
  --train-json "$EVAL_TRAIN_JSON"
  --benchmark-json "$BENCHMARK_JSON"
  --calibration-json "$CALIBRATION_JSON"
)
if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
  eval_extra_args+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi
if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
  eval_extra_args+=(--max-benchmark-examples "$MAX_BENCHMARK_EXAMPLES")
fi
if [[ -n "${MAX_CALIBRATION_EXAMPLES:-}" ]]; then
  eval_extra_args+=(--max-calibration-examples "$MAX_CALIBRATION_EXAMPLES")
fi
case "$IGNORE_SPACES" in
  1|true|TRUE|yes|YES|y|Y|on|ON)
    eval_extra_args+=(--ignore-spaces)
    ;;
  0|false|FALSE|no|NO|n|N|off|OFF)
    ;;
  *)
    echo "IGNORE_SPACES must be 1 or 0." >&2
    exit 2
    ;;
esac
eval_extra_args+=("$@")

multi_agg_args=(
  --model "regular_sft=${REGULAR_OUTPUT_DIR}"
  --model "contrastive_sft=${CONTRASTIVE_OUTPUT_DIR}"
  --train-json "regular_sft=${REGULAR_TRAIN_JSON}"
  --train-json "contrastive_sft=${CONTRASTIVE_TRAIN_JSON}"
)
legacy_report_count=0

run_legacy_30_suite() {
  local model_label="$1"
  local model_dir="$2"
  for method_label in $LEGACY_30_PRUNE_METHODS; do
    local method_arg
    local method_key
    case "$method_label" in
      nvidia|nvidia24|2of4|2:4)
        method_arg="nvidia"
        method_key="nvidia24"
        ;;
      magnitude|gradient|wanda)
        method_arg="$method_label"
        method_key="$method_label"
        ;;
      *)
        echo "Unknown LEGACY_30_PRUNE_METHODS entry: $method_label" >&2
        exit 2
        ;;
    esac

    local method_run_dir="${LEGACY_30_OUTPUT_ROOT}/${model_label}/${method_key}_30"
    local method_report_json="${method_run_dir}/prune_eval_report.json"
    local method_pruned_dir="${method_run_dir}/pruned_model"
    if ! truthy "$FORCE_RERUN" && [[ -f "$method_report_json" && -d "$method_pruned_dir" ]]; then
      echo "Reusing existing ${model_label} ${method_key} 30% report: $method_report_json"
    else
      echo "Running ${model_label} ${method_key} 30% prune/eval..."
      METHOD="$method_arg" \
        SPARSITY=0.3 \
        RUN_DIR="$method_run_dir" \
        REPORT_JSON="$method_report_json" \
        PRUNED_DIR="$method_pruned_dir" \
        NPROC_PER_NODE="$NPROC_PER_NODE" \
        PRUNE_SCOPE="$PRUNE_SCOPE" \
        SPARSITY_BASIS="$SPARSITY_BASIS" \
        PRUNE_LM_HEAD="$PRUNE_LM_HEAD" \
        bash scripts/run_prune_eval_50.sh "$model_dir" "${eval_extra_args[@]}"
    fi
    multi_agg_args+=(--report "${model_label}:${method_key}=${method_report_json}")
    legacy_report_count=$((legacy_report_count + 1))
  done
}

if truthy "$RUN_LEGACY_ONESHOT_30"; then
  echo
  echo "== Legacy one-shot methods at 30% =="
  run_legacy_30_suite "regular_sft" "$REGULAR_OUTPUT_DIR"
  run_legacy_30_suite "contrastive_sft" "$CONTRASTIVE_OUTPUT_DIR"

  if [[ "$legacy_report_count" -eq 0 ]]; then
    echo "No legacy 30% reports were selected." >&2
    exit 2
  fi

  "$PYTHON" scripts/aggregate_multi_model_prune_eval_reports.py \
    --output-json "$LEGACY_30_FINAL_JSON" \
    --base-model "$BASE_MODEL" \
    --epochs "$EPOCHS" \
    --sparsity 0.3 \
    "${multi_agg_args[@]}"

  if truthy "$RUN_SPARSITY_CHECK_30"; then
    "$PYTHON" scripts/check_pruned_model_sparsity.py \
      --report-json "$LEGACY_30_FINAL_JSON" \
      --output-json "$LEGACY_30_SPARSITY_CHECK_JSON" \
      --expected-sparsity 0.3 \
      --default-prune-scope "$PRUNE_SCOPE" \
      --default-sparsity-basis "$SPARSITY_BASIS" \
      --fail-on-mismatch
  else
    echo "RUN_SPARSITY_CHECK_30=0; skipping legacy 30% sparsity verification."
  fi
else
  echo "RUN_LEGACY_ONESHOT_30=0; skipping legacy one-shot 30% suite."
fi

run_linear_30_matrix() {
  local label="$1"
  local checkpoint="$2"
  local train_json="$3"
  local output_dir="${SPARSITY_30_OUTPUT_ROOT}/${label}"
  local summary_csv="${output_dir}/summary_metrics.csv"
  if ! truthy "$FORCE_RERUN" && [[ -f "$summary_csv" ]]; then
    echo "Reusing existing ${label} linear 30-only summary: $summary_csv"
    return
  fi

  local extra_args=()
  if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
    extra_args+=(--max_train_examples "$MAX_TRAIN_EXAMPLES")
  fi
  if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
    extra_args+=(--max_benchmark_examples "$MAX_BENCHMARK_EXAMPLES")
  fi
  if [[ -n "${MAX_VALIDATION_EXAMPLES:-}" ]]; then
    extra_args+=(--max_validation_examples "$MAX_VALIDATION_EXAMPLES")
  fi
  if [[ -n "${BENCHMARK_DIFFICULTY_PATH:-}" ]]; then
    extra_args+=(--benchmark_difficulty_path "$BENCHMARK_DIFFICULTY_PATH")
  fi
  if [[ -n "${VALIDATION_JSON:-}" ]]; then
    extra_args+=(--validation_path "$VALIDATION_JSON")
  fi

  # shellcheck disable=SC2206
  local pruning_modes=( $SPARSITY_PRUNING_MODES_30 )
  "$PYTHON" scripts/run_sparsity_experiments.py \
    --experiment_name "scenic_${label}_linear_sparsity_30_only" \
    --model_family "$MODEL_FAMILY" \
    --model_checkpoint "$checkpoint" \
    --benchmark_path "$BENCHMARK_JSON" \
    --train_path "$train_json" \
    --sparsity_levels 0.3 \
    --pruning_modes "${pruning_modes[@]}" \
    --prune_scope linear_weights \
    --prune_method magnitude \
    --recovery_epochs_per_stage "$SPARSITY_RECOVERY_EPOCHS_PER_STAGE" \
    --final_recovery_epochs "$SPARSITY_FINAL_RECOVERY_EPOCHS" \
    --num_beams "$SPARSITY_NUM_BEAMS" \
    --num_return_sequences "$SPARSITY_NUM_RETURN_SEQUENCES" \
    --max_new_tokens "$SPARSITY_MAX_NEW_TOKENS" \
    --normalization_mode "$SPARSITY_NORMALIZATION_MODE" \
    --seed "$SEED" \
    --output_dir "$output_dir" \
    "${extra_args[@]}"
}

if truthy "$RUN_SPARSITY_30"; then
  echo
  echo "== Linear sparsity matrix at 30% only =="
  run_linear_30_matrix "regular_sft" "$REGULAR_OUTPUT_DIR" "$REGULAR_TRAIN_JSON"
  run_linear_30_matrix "contrastive_sft" "$CONTRASTIVE_OUTPUT_DIR" "$CONTRASTIVE_TRAIN_JSON"
else
  echo "RUN_SPARSITY_30=0; skipping linear sparsity 30-only matrix."
fi

cat > "$THIRTY_MANIFEST" <<EOF
SCENIC 30%-only revision follow-up
base_model=${BASE_MODEL}
full_revision_root=${REVISION_OUTPUT_ROOT}
regular_checkpoint=${REGULAR_OUTPUT_DIR}
contrastive_checkpoint=${CONTRASTIVE_OUTPUT_DIR}
legacy_30_report=${LEGACY_30_FINAL_JSON}
legacy_30_sparsity_check=${LEGACY_30_SPARSITY_CHECK_JSON}
regular_linear_30_summary=${SPARSITY_30_OUTPUT_ROOT}/regular_sft/summary_metrics.csv
contrastive_linear_30_summary=${SPARSITY_30_OUTPUT_ROOT}/contrastive_sft/summary_metrics.csv
note=nvidia24 is not included by default because 2:4 pruning is effectively 50% selected-weight sparsity.
EOF

echo
echo "Done. 30%-only manifest: $THIRTY_MANIFEST"
