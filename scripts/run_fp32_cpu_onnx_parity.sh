#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_fp32_cpu_onnx_parity.sh \
    onnx_eval_outputs/clean_h20_contrastive_gradient50_YYYYMMDD_HHMMSS \
    [base-model-path-or-hf-id]

Runs the first conversion sanity check only:
  - CPU FP32 checkpoints only
  - model.eval()
  - torch pruning reparameterizations removed if present
  - ONNX Runtime CPUExecutionProvider
  - ORT graph optimizations disabled
  - raw PyTorch-vs-ONNX logits compared on the same inputs
  - no FP16, no INT8, no EM generation, no latency benchmark

Useful overrides:
  PARITY_MAX_EXAMPLES=8
  PARITY_MAX_INPUT_LEN=256
  PARITY_DECODER_LENGTH=8
  FORCE_BAKE=1
  FORCE_EXPORT=1
  FORCE_PARITY=1

Outputs:
  <OUTPUT_ROOT>/reports/fp32_cpu_onnx_logits_parity.json
USAGE
}

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
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
SOURCE_ASSET_DIR="${SOURCE_ASSET_DIR:-$BASE_MODEL}"
PYTHON="${PYTHON:-python}"
ONNX_TASK="${ONNX_TASK:-text2text-generation-with-past}"
ONNX_OPSET="${ONNX_OPSET:-18}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
PARITY_MAX_EXAMPLES="${PARITY_MAX_EXAMPLES:-8}"
PARITY_MAX_INPUT_LEN="${PARITY_MAX_INPUT_LEN:-256}"
PARITY_DECODER_LENGTH="${PARITY_DECODER_LENGTH:-8}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"

DENSE_CHECKPOINT="${OUTPUT_ROOT}/checkpoints/sft5"
PRUNED_CHECKPOINT="${OUTPUT_ROOT}/checkpoints/sft5_clean_contrastive_gradient50_pruned"
PARITY_ROOT="${OUTPUT_ROOT}/intermediate/fp32_cpu_onnx_parity"
REPORT_ROOT="${OUTPUT_ROOT}/reports"

BAKED_DENSE="${PARITY_ROOT}/baked_dense_checkpoint"
BAKED_PRUNED="${PARITY_ROOT}/baked_pruned_checkpoint"
ONNX_DENSE="${PARITY_ROOT}/onnx_dense_fp32_cpu"
ONNX_PRUNED="${PARITY_ROOT}/onnx_pruned_fp32_cpu"
DENSE_BAKE_JSON="${REPORT_ROOT}/fp32_cpu_bake_dense.json"
PRUNED_BAKE_JSON="${REPORT_ROOT}/fp32_cpu_bake_pruned.json"
DENSE_PARITY_JSON="${REPORT_ROOT}/fp32_cpu_onnx_logits_parity_dense.json"
PRUNED_PARITY_JSON="${REPORT_ROOT}/fp32_cpu_onnx_logits_parity_pruned.json"
FINAL_PARITY_JSON="${REPORT_ROOT}/fp32_cpu_onnx_logits_parity.json"

required_paths=(
  "${DENSE_CHECKPOINT}/config.json"
  "${PRUNED_CHECKPOINT}/config.json"
  "$BENCHMARK_JSON"
)

for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    echo "Point OUTPUT_ROOT at a completed clean H20 run first." >&2
    exit 2
  fi
done

mkdir -p "$PARITY_ROOT" "$REPORT_ROOT"

repair_checkpoint() {
  local checkpoint="$1"
  local label="$2"
  local cmd=(
    "$PYTHON"
    scripts/repair_checkpoint_tokenizer.py
    --checkpoint "$checkpoint"
    --source-tokenizer "$SOURCE_ASSET_DIR"
    --resize-token-embeddings
  )
  if truthy "$LOCAL_FILES_ONLY"; then
    cmd+=(--local-files-only)
  fi
  echo "Repairing ${label} assets from ${SOURCE_ASSET_DIR}"
  "${cmd[@]}"
}

bake_checkpoint() {
  local source="$1"
  local output="$2"
  local summary="$3"
  local label="$4"
  if [[ -f "${output}/config.json" && -f "$summary" ]] && ! truthy "$FORCE_BAKE"; then
    echo "Skipping ${label} bake; found ${output}"
    return
  fi
  if [[ -d "$output" ]] && truthy "$FORCE_BAKE"; then
    rm -rf "$output"
  fi
  echo "Baking ${label} checkpoint to CPU FP32 eval -> ${output}"
  local cmd=(
    "$PYTHON"
    scripts/bake_checkpoint_for_onnx_export.py
    --model "$source"
    --output-dir "$output"
    --source-tokenizer "$SOURCE_ASSET_DIR"
    --summary-json "$summary"
  )
  if truthy "$LOCAL_FILES_ONLY"; then
    cmd+=(--local-files-only)
  else
    cmd+=(--no-local-files-only)
  fi
  if truthy "$TRUST_REMOTE_CODE"; then
    cmd+=(--trust-remote-code)
  else
    cmd+=(--no-trust-remote-code)
  fi
  "${cmd[@]}"
}

export_fp32_cpu_onnx() {
  local source="$1"
  local output="$2"
  local label="$3"
  if [[ -f "${output}/.export.done" ]] && ! truthy "$FORCE_EXPORT"; then
    echo "Skipping ${label} ONNX export; found ${output}/.export.done"
    return
  fi
  if [[ -d "$output" ]] && truthy "$FORCE_EXPORT"; then
    rm -rf "$output"
  fi
  mkdir -p "$output"
  echo "Exporting ${label} ONNX FP32 on CPU -> ${output}"
  local cmd=(
    optimum-cli export onnx
    --model "$source"
    --task "$ONNX_TASK"
    --opset "$ONNX_OPSET"
    --device cpu
  )
  if truthy "$TRUST_REMOTE_CODE"; then
    cmd+=(--trust-remote-code)
  fi
  cmd+=("$output")
  "${cmd[@]}"
  shopt -s nullglob
  local onnx_files=("$output"/*.onnx)
  shopt -u nullglob
  if [[ "${#onnx_files[@]}" -eq 0 ]]; then
    echo "No ONNX files found after export: $output" >&2
    exit 1
  fi
  touch "${output}/.export.done"
}

compare_logits() {
  local label="$1"
  local pytorch_model="$2"
  local onnx_model="$3"
  local tokenizer="$4"
  local output_json="$5"
  if [[ -f "$output_json" ]] && ! truthy "$FORCE_PARITY"; then
    echo "Skipping ${label} parity; found ${output_json}"
    return
  fi
  echo "Comparing raw logits for ${label} -> ${output_json}"
  local cmd=(
    "$PYTHON"
    scripts/compare_fp32_cpu_onnx_logits.py
    --label "$label"
    --pytorch-model "$pytorch_model"
    --onnx-model "$onnx_model"
    --tokenizer "$tokenizer"
    --benchmark-json "$BENCHMARK_JSON"
    --output-json "$output_json"
    --max-examples "$PARITY_MAX_EXAMPLES"
    --max-input-len "$PARITY_MAX_INPUT_LEN"
    --decoder-length "$PARITY_DECODER_LENGTH"
  )
  if truthy "$LOCAL_FILES_ONLY"; then
    cmd+=(--local-files-only)
  else
    cmd+=(--no-local-files-only)
  fi
  if truthy "$TRUST_REMOTE_CODE"; then
    cmd+=(--trust-remote-code)
  else
    cmd+=(--no-trust-remote-code)
  fi
  "${cmd[@]}"
}

repair_checkpoint "$DENSE_CHECKPOINT" "dense checkpoint"
repair_checkpoint "$PRUNED_CHECKPOINT" "pruned checkpoint"

bake_checkpoint "$DENSE_CHECKPOINT" "$BAKED_DENSE" "$DENSE_BAKE_JSON" "dense"
bake_checkpoint "$PRUNED_CHECKPOINT" "$BAKED_PRUNED" "$PRUNED_BAKE_JSON" "pruned"

export_fp32_cpu_onnx "$BAKED_DENSE" "$ONNX_DENSE" "dense"
export_fp32_cpu_onnx "$BAKED_PRUNED" "$ONNX_PRUNED" "pruned"

compare_logits "dense_fp32_cpu" "$BAKED_DENSE" "$ONNX_DENSE" "$BAKED_DENSE" "$DENSE_PARITY_JSON"
compare_logits "pruned_fp32_cpu" "$BAKED_PRUNED" "$ONNX_PRUNED" "$BAKED_PRUNED" "$PRUNED_PARITY_JSON"

FINAL_PARITY_JSON="$FINAL_PARITY_JSON" \
DENSE_PARITY_JSON="$DENSE_PARITY_JSON" \
PRUNED_PARITY_JSON="$PRUNED_PARITY_JSON" \
DENSE_BAKE_JSON="$DENSE_BAKE_JSON" \
PRUNED_BAKE_JSON="$PRUNED_BAKE_JSON" \
"$PYTHON" <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": "fp32_cpu_onnx_raw_logits_parity",
    "purpose": "Validate conversion before debugging FP16, INT8, CUDA, EM1/EM5, or post-processing.",
    "dense": read_json(os.environ["DENSE_PARITY_JSON"]),
    "pruned": read_json(os.environ["PRUNED_PARITY_JSON"]),
    "bake_reports": {
        "dense": read_json(os.environ["DENSE_BAKE_JSON"]),
        "pruned": read_json(os.environ["PRUNED_BAKE_JSON"]),
    },
}
output = Path(os.environ["FINAL_PARITY_JSON"])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"Wrote combined FP32 CPU ONNX parity report: {output}")
PY

echo "Done. FP32 CPU raw-logit parity report: $FINAL_PARITY_JSON"
