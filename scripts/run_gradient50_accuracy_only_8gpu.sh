#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_gradient50_accuracy_only_8gpu.sh [base-model-path-or-hf-id]

Default:
  base model: charent/ChatLM-mini-Chinese

This launcher is for accuracy-change only:
  1. Fine-tune the base model for 5 epochs with regular SFT.
  2. Run 50% gradient one-shot pruning.
  3. Evaluate dense vs pruned PyTorch FP16 EM@1 / EM@5 on benchmark and training data.
  4. Use torchrun across 8 GPUs for training and prune/eval.
  5. Skip ONNX export, ONNX INT8, and latency/TPS benchmarking entirely.

Main outputs:
  <OUTPUT_ROOT>/gradient50_prune_eval_report.json
  <OUTPUT_ROOT>/gradient50_accuracy_delta_summary.json

Useful overrides:
  OUTPUT_ROOT=accuracy_only_outputs/my_run
  NPROC_PER_NODE=8
  MAX_BENCHMARK_EXAMPLES=200
  MAX_TRAIN_EXAMPLES=
  CALIBRATION_BATCHES=64
  LOCAL_FILES_ONLY=0
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

BASE_MODEL="${1:-charent/ChatLM-mini-Chinese}"
PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SAFE_MODEL_NAME="$(basename "$BASE_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME%_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME:-model}"

OUTPUT_ROOT="${OUTPUT_ROOT:-accuracy_only_outputs/${SAFE_MODEL_NAME}_sft5_gradient50_accuracy_${RUN_ID}}"
CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${CHECKPOINT_ROOT}/regular_sft5}"
PRUNED_OUTPUT_DIR="${PRUNED_OUTPUT_DIR:-${CHECKPOINT_ROOT}/regular_sft5_gradient50_pruned}"
REPORT_JSON="${REPORT_JSON:-${OUTPUT_ROOT}/gradient50_prune_eval_report.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_ROOT}/gradient50_accuracy_delta_summary.json}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
TRAIN_JSON="${TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$TRAIN_JSON}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-5}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.03}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-256}"
MAX_TARGET_LEN="${MAX_TARGET_LEN:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_BEAMS="${NUM_BEAMS:-5}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-64}"
SPARSITY="${SPARSITY:-0.5}"
PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
SPARSITY_BASIS="${SPARSITY_BASIS:-targeted-linear}"
PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_PRUNE_EVAL="${FORCE_PRUNE_EVAL:-0}"

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

maybe_add_limit_args() {
  if [[ -n "${MAX_TRAIN_EXAMPLES:-}" ]]; then
    printf '%s\0%s\0' --max-train-examples "$MAX_TRAIN_EXAMPLES"
  fi
  if [[ -n "${MAX_BENCHMARK_EXAMPLES:-}" ]]; then
    printf '%s\0%s\0' --max-benchmark-examples "$MAX_BENCHMARK_EXAMPLES"
  fi
  if [[ -n "${MAX_CALIBRATION_EXAMPLES:-}" ]]; then
    printf '%s\0%s\0' --max-calibration-examples "$MAX_CALIBRATION_EXAMPLES"
  fi
}

mkdir -p "$OUTPUT_ROOT" "$CHECKPOINT_ROOT"

echo "Gradient-50 accuracy-only run"
echo "Base model: $BASE_MODEL"
echo "Output root: $OUTPUT_ROOT"
echo "Fine-tuned checkpoint: $FINETUNE_OUTPUT_DIR"
echo "Pruned checkpoint: $PRUNED_OUTPUT_DIR"
echo "Final accuracy report: $REPORT_JSON"
echo "Delta summary: $SUMMARY_JSON"

if [[ -f "${FINETUNE_OUTPUT_DIR}/config.json" ]] && ! truthy "$FORCE_TRAIN"; then
  echo "Skipping SFT; found ${FINETUNE_OUTPUT_DIR}/config.json"
else
  train_cmd=(
    scripts/scenic_train_chatlm_sft.py
    --mode regular
    --model "$BASE_MODEL"
    --train-json "$TRAIN_JSON"
    --output-dir "$FINETUNE_OUTPUT_DIR"
    --epochs "$FINETUNE_EPOCHS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$TRAIN_LEARNING_RATE"
    --weight-decay "$TRAIN_WEIGHT_DECAY"
    --warmup-ratio "$TRAIN_WARMUP_RATIO"
    --max-source-length "$MAX_INPUT_LEN"
    --max-target-length "$MAX_TARGET_LEN"
    --fp16
  )
  if truthy "$LOCAL_FILES_ONLY"; then
    train_cmd+=(--local-files-only)
  fi
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_train_args <<< "$TRAIN_EXTRA_ARGS"
    train_cmd+=("${extra_train_args[@]}")
  fi

  echo "Training regular SFT for ${FINETUNE_EPOCHS} epochs with ${NPROC_PER_NODE} GPU process(es)."
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${train_cmd[@]}"
fi

repair_cmd=("$PYTHON" scripts/repair_checkpoint_tokenizer.py --checkpoint "$FINETUNE_OUTPUT_DIR" --source-tokenizer "$BASE_MODEL")
if truthy "$LOCAL_FILES_ONLY"; then
  repair_cmd+=(--local-files-only)
fi
"${repair_cmd[@]}"

if [[ -f "$REPORT_JSON" && -f "${PRUNED_OUTPUT_DIR}/config.json" ]] && ! truthy "$FORCE_PRUNE_EVAL"; then
  echo "Skipping prune/eval; found $REPORT_JSON"
else
  limit_args=()
  while IFS= read -r -d '' item; do
    limit_args+=("$item")
  done < <(maybe_add_limit_args)

  prune_eval_cmd=(
    scripts/scenic_prune_eval.py
    --model "$FINETUNE_OUTPUT_DIR"
    --method gradient
    --sparsity "$SPARSITY"
    --sparsity-basis "$SPARSITY_BASIS"
    --prune-scope "$PRUNE_SCOPE"
    --train-json "$TRAIN_JSON"
    --benchmark-json "$BENCHMARK_JSON"
    --calibration-json "$CALIBRATION_JSON"
    --output-json "$REPORT_JSON"
    --pruned-output-dir "$PRUNED_OUTPUT_DIR"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --calibration-batch-size "$CALIBRATION_BATCH_SIZE"
    --calibration-batches "$CALIBRATION_BATCHES"
    --max-input-len "$MAX_INPUT_LEN"
    --max-target-len "$MAX_TARGET_LEN"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --num-beams "$NUM_BEAMS"
    --num-return-sequences "$NUM_RETURN_SEQUENCES"
    --ignore-spaces
    "${limit_args[@]}"
  )
  if truthy "$PRUNE_LM_HEAD"; then
    prune_eval_cmd+=(--prune-lm-head)
  fi
  if truthy "$LOCAL_FILES_ONLY"; then
    prune_eval_cmd+=(--local-files-only)
  else
    prune_eval_cmd+=(--no-local-files-only)
  fi

  echo "Running distributed dense-vs-gradient50 prune/eval with ${NPROC_PER_NODE} GPU process(es)."
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${prune_eval_cmd[@]}"
fi

REPORT_JSON="$REPORT_JSON" SUMMARY_JSON="$SUMMARY_JSON" "$PYTHON" <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


report_path = Path(os.environ["REPORT_JSON"])
summary_path = Path(os.environ["SUMMARY_JSON"])
report = read_json(report_path)
summary = report.get("summary", {})
before = summary.get("original_before_prune", {})
after = summary.get("pruned_after_50_percent", {})

rows: list[dict[str, Any]] = []
for dataset_name, before_metrics in before.items():
    after_metrics = after.get(dataset_name, {})
    if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
        continue
    before_em1 = float(before_metrics.get("em1_percent", 0.0))
    before_em5 = float(before_metrics.get("em5_percent", 0.0))
    after_em1 = float(after_metrics.get("em1_percent", 0.0))
    after_em5 = float(after_metrics.get("em5_percent", 0.0))
    rows.append(
        {
            "dataset": dataset_name,
            "total": after_metrics.get("total", before_metrics.get("total", 0)),
            "dense_em1_percent": before_em1,
            "dense_em5_percent": before_em5,
            "gradient50_em1_percent": after_em1,
            "gradient50_em5_percent": after_em5,
            "delta_em1_percent": after_em1 - before_em1,
            "delta_em5_percent": after_em5 - before_em5,
            "retention_em1_percent": after_em1 / before_em1 * 100.0 if before_em1 else None,
            "retention_em5_percent": after_em5 / before_em5 * 100.0 if before_em5 else None,
        }
    )

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_report": str(report_path),
    "accuracy_delta_table": rows,
    "pruning": report.get("pruning"),
    "dense_model": report.get("original_before_prune", {}).get("model"),
    "gradient50_model": report.get("pruned_after_50_percent", {}).get("model"),
    "note": "Accuracy-only PyTorch FP16 dense-vs-gradient50 report. ONNX export, ONNX INT8, and latency are intentionally skipped.",
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote accuracy delta summary: {summary_path}")
PY

echo "Done. Accuracy delta summary: $SUMMARY_JSON"
