#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_onnx_precision_prune_eval.sh <original-model-path-or-hf-id> [accuracy eval args]

End-to-end SCENIC deployment pass:
  1. Fine-tune the original model for 5 epochs with regular SFT.
  2. Create a 50% gradient one-shot pruned checkpoint from the fine-tuned model.
  3. Export dense and gradient-pruned checkpoints to ONNX FP16.
  4. Export FP32 ONNX sources and dynamic-quantize them to ONNX INT8.
  5. Run EM@1 / EM@5 on both the 200-example benchmark and training data.
  6. Benchmark batch=1 deployment latency/TPS for PyTorch FP16, ONNX FP16,
     and ONNX INT8.
     TensorRT FP16 is ready behind RUN_TENSORRT=1 when that runtime is available.

Main outputs:
  onnx_eval_outputs/<run>/all_deployment_em_latency_report.json
  onnx_eval_outputs/<run>/reports/deployment_runtime_benchmark.json
  onnx_eval_outputs/<run>/reports/*_accuracy_report.json

Common environment overrides:
  OUTPUT_ROOT=onnx_eval_outputs/my_run
  SOURCE_ASSET_DIR=/path/to/original/base_model-or-hf-id
  LOCAL_FILES_ONLY=1
  SKIP_TRAIN=1
  FORCE_TRAIN=1
  FORCE_PRUNE=1
  FORCE_EXPORT=1
  FORCE_QUANTIZE=1
  FORCE_ACCURACY=1
  FORCE_BENCHMARK=1

Training:
  FINETUNE_MODE=regular
  FINETUNE_EPOCHS=5
  FINETUNE_TRAIN_JSON=data/SCENIC_full_training_dataset.json
  FINETUNE_OUTPUT_DIR=<output-root>/checkpoints/sft5
  TRAIN_BATCH_SIZE=4
  TRAIN_GRADIENT_ACCUMULATION_STEPS=4
  TRAIN_NPROC_PER_NODE=8

Pruning:
  PRUNE_METHOD=gradient        # gradient, magnitude, wanda, or nvidia
  SPARSITY=0.5
  PRUNE_SCOPE=all-linear
  SPARSITY_BASIS=targeted-linear
  PRUNE_LM_HEAD=0
  CALIBRATION_JSON=data/SCENIC_full_training_dataset.json
  CALIBRATION_BATCH_SIZE=4
  CALIBRATION_BATCHES=64

Accuracy:
  BENCHMARK_JSON=generated/iot_instruction_benchmark_200.json
  MAX_BENCHMARK_EXAMPLES=200
  TRAIN_JSON=data/SCENIC_full_training_dataset.json
  MAX_TRAIN_EXAMPLES=    # empty means full training set
  NUM_BEAMS=5
  NUM_RETURN_SEQUENCES=5
  MAX_INPUT_LEN=256
  MAX_NEW_TOKENS=128
  RUN_ACCURACY_PARALLEL=1
  ACCURACY_GPU_IDS="0,1,2,3,4,5,6,7"

Latency:
  LATENCY_SEQ_LENGTHS="64 128"
  LATENCY_BATCH_SIZE=1
  LATENCY_QUERIES=200
  LATENCY_WARMUP=10
  LATENCY_NUM_BEAMS=1
  LATENCY_MAX_NEW_TOKENS=128

TensorRT:
  RUN_TENSORRT=0        # default: do not run TensorRT on non-TensorRT machines
  TENSORRT_REQUIRED=0   # set to 1 when TensorRT must be present
  TENSORRT_SPARSITY_ENABLE=0 # gradient sparsity is unstructured, not 2:4
  TENSORRT_ENGINE_CACHE_ROOT=<output-root>/tensorrt_engines

INT8:
  RUN_INT8=1
  INT8_ONNX_PROVIDER=CUDAExecutionProvider

Example smoke test:
  FINETUNE_EPOCHS=1 MAX_BENCHMARK_EXAMPLES=20 MAX_TRAIN_EXAMPLES=20 LATENCY_QUERIES=20 \
  bash scripts/run_onnx_precision_prune_eval.sh charent/ChatLM-mini-Chinese

Required Python/CLI packages:
  python -m pip install "optimum[onnxruntime]" onnx onnxruntime onnxruntime-gpu
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

ORIGINAL_MODEL="$1"
shift
ACCURACY_EXTRA_ARGS=("$@")

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SAFE_MODEL_NAME="$(basename "$ORIGINAL_MODEL" | tr -c 'A-Za-z0-9_.-' '_')"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME%_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME:-model}"
PRUNE_METHOD="${PRUNE_METHOD:-gradient}"
SPARSITY="${SPARSITY:-0.5}"
case "$SPARSITY" in
  0.5|.5|0.50) SPARSITY_TAG="50" ;;
  0.3|.3|0.30) SPARSITY_TAG="30" ;;
  *) SPARSITY_TAG="$(printf '%s' "$SPARSITY" | tr -c 'A-Za-z0-9_.-' '_' | sed 's/_$//')" ;;
esac
PRUNED_VARIANT="${PRUNED_VARIANT:-${PRUNE_METHOD}${SPARSITY_TAG}}"
PRUNED_PRETTY_LABEL="${PRUNED_PRETTY_LABEL:-${SPARSITY_TAG}% ${PRUNE_METHOD} pruned}"
OUTPUT_ROOT="${OUTPUT_ROOT:-onnx_eval_outputs/${SAFE_MODEL_NAME}_sft5_${PRUNED_VARIANT}_deploy_${RUN_ID}}"

CHECKPOINT_ROOT="${OUTPUT_ROOT}/checkpoints"
ONNX_ROOT="${OUTPUT_ROOT}/onnx"
INTERMEDIATE_ROOT="${OUTPUT_ROOT}/intermediate"
REPORT_ROOT="${OUTPUT_ROOT}/reports"
TENSORRT_ENGINE_CACHE_ROOT="${TENSORRT_ENGINE_CACHE_ROOT:-${OUTPUT_ROOT}/tensorrt_engines}"
FINAL_JSON="${FINAL_JSON:-${OUTPUT_ROOT}/all_deployment_em_latency_report.json}"

FINETUNE_MODE="${FINETUNE_MODE:-regular}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
FINETUNE_TRAIN_JSON="${FINETUNE_TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
FINETUNE_OUTPUT_DIR="${FINETUNE_OUTPUT_DIR:-${CHECKPOINT_ROOT}/sft5}"
SOURCE_ASSET_DIR="${SOURCE_ASSET_DIR:-$ORIGINAL_MODEL}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-5}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.03}"
TRAIN_MAX_SOURCE_LENGTH="${TRAIN_MAX_SOURCE_LENGTH:-256}"
TRAIN_MAX_TARGET_LENGTH="${TRAIN_MAX_TARGET_LENGTH:-128}"
TRAIN_NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE:-${NPROC_PER_NODE:-8}}"
TRAIN_PRECISION="${TRAIN_PRECISION:-fp16}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

PRUNE_SCOPE="${PRUNE_SCOPE:-all-linear}"
SPARSITY_BASIS="${SPARSITY_BASIS:-targeted-linear}"
PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-0}"
FORCE_PRUNE="${FORCE_PRUNE:-0}"

TRAIN_JSON="${TRAIN_JSON:-data/SCENIC_full_training_dataset.json}"
BENCHMARK_JSON="${BENCHMARK_JSON:-generated/iot_instruction_benchmark_200.json}"
CALIBRATION_JSON="${CALIBRATION_JSON:-$TRAIN_JSON}"
CALIBRATION_BATCH_SIZE="${CALIBRATION_BATCH_SIZE:-4}"
CALIBRATION_BATCHES="${CALIBRATION_BATCHES:-64}"
MAX_CALIBRATION_EXAMPLES="${MAX_CALIBRATION_EXAMPLES:-}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_BENCHMARK_EXAMPLES="${MAX_BENCHMARK_EXAMPLES:-200}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
NUM_BEAMS="${NUM_BEAMS:-5}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
MAX_INPUT_LEN="${MAX_INPUT_LEN:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
INCLUDE_PREDICTIONS="${INCLUDE_PREDICTIONS:-1}"
IGNORE_SPACES="${IGNORE_SPACES:-1}"
FORCE_ACCURACY="${FORCE_ACCURACY:-0}"
RUN_ACCURACY_PARALLEL="${RUN_ACCURACY_PARALLEL:-1}"
ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-}"
ACCURACY_PYTORCH_DEVICE="${ACCURACY_PYTORCH_DEVICE:-cuda}"

LATENCY_SEQ_LENGTHS="${LATENCY_SEQ_LENGTHS:-64 128}"
LATENCY_BATCH_SIZE="${LATENCY_BATCH_SIZE:-1}"
LATENCY_QUERIES="${LATENCY_QUERIES:-200}"
LATENCY_WARMUP="${LATENCY_WARMUP:-10}"
LATENCY_NUM_BEAMS="${LATENCY_NUM_BEAMS:-1}"
LATENCY_MAX_NEW_TOKENS="${LATENCY_MAX_NEW_TOKENS:-128}"
FORCE_BENCHMARK="${FORCE_BENCHMARK:-0}"

LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ONNX_TASK="${ONNX_TASK:-text2text-generation-with-past}"
ONNX_OPSET="${ONNX_OPSET:-17}"
FP32_EXPORT_DEVICE="${FP32_EXPORT_DEVICE:-cpu}"
RUN_INT8="${RUN_INT8:-1}"
RUN_TENSORRT="${RUN_TENSORRT:-0}"
TENSORRT_REQUIRED="${TENSORRT_REQUIRED:-0}"
if [[ "$PRUNE_METHOD" == "nvidia" ]]; then
  TENSORRT_SPARSITY_ENABLE="${TENSORRT_SPARSITY_ENABLE:-1}"
else
  TENSORRT_SPARSITY_ENABLE="${TENSORRT_SPARSITY_ENABLE:-0}"
fi
OPTIMUM_DTYPE_MODE="${OPTIMUM_DTYPE_MODE:-auto}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
FORCE_QUANTIZE="${FORCE_QUANTIZE:-0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  EXPORT_DEVICE="${EXPORT_DEVICE:-cuda}"
  PYTORCH_DEVICE="${PYTORCH_DEVICE:-cuda}"
  FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CUDAExecutionProvider}"
else
  EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"
  PYTORCH_DEVICE="${PYTORCH_DEVICE:-cpu}"
  FP16_ONNX_PROVIDER="${FP16_ONNX_PROVIDER:-CPUExecutionProvider}"
fi
INT8_ONNX_PROVIDER="${INT8_ONNX_PROVIDER:-$FP16_ONNX_PROVIDER}"
TENSORRT_ONNX_PROVIDER="${TENSORRT_ONNX_PROVIDER:-TensorrtExecutionProvider}"

FINETUNED_CHECKPOINT_DIR="$FINETUNE_OUTPUT_DIR"
PRUNED_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/sft5_${PRUNED_VARIANT}_pruned"
PRUNED_SUMMARY_JSON="${CHECKPOINT_ROOT}/sft5_${PRUNED_VARIANT}_pruning_summary.json"

FP16_DENSE_ONNX="${ONNX_ROOT}/sft5_fp16_dense"
FP16_PRUNED_ONNX="${ONNX_ROOT}/sft5_fp16_pruned"
FP32_DENSE_ONNX="${INTERMEDIATE_ROOT}/sft5_fp32_dense_for_int8"
FP32_PRUNED_ONNX="${INTERMEDIATE_ROOT}/sft5_fp32_pruned_for_int8"
INT8_DENSE_ONNX="${ONNX_ROOT}/sft5_int8_dense"
INT8_PRUNED_ONNX="${ONNX_ROOT}/sft5_int8_pruned"
RUNTIME_BENCHMARK_JSON="${REPORT_ROOT}/deployment_runtime_benchmark.json"

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

detect_accuracy_gpu_ids() {
  if [[ -n "$ACCURACY_GPU_IDS" ]]; then
    printf '%s\n' "$ACCURACY_GPU_IDS"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L | wc -l | tr -d ' ')"
    if [[ "$count" != "0" ]]; then
      seq -s, 0 $((count - 1))
      return
    fi
  fi
  printf '\n'
}

reset_output_dir() {
  local dir="$1"
  local force_flag="$2"
  if truthy "$force_flag" && [[ -d "$dir" ]]; then
    case "$dir" in
      "$OUTPUT_ROOT"/*)
        rm -rf "$dir"
        ;;
      *)
        echo "Refusing to remove directory outside OUTPUT_ROOT: $dir" >&2
        exit 2
        ;;
    esac
  fi
}

check_python_module() {
  local module="$1"
  "$PYTHON" - "$module" <<'PY'
import importlib
import sys

module = sys.argv[1]
try:
    importlib.import_module(module)
except Exception as exc:
    raise SystemExit(f"Missing Python module {module!r}: {exc}")
PY
}

check_tensorrt_provider() {
  "$PYTHON" <<'PY'
import onnxruntime as ort
import sys

providers = ort.get_available_providers()
print("Available ONNX Runtime providers:", ", ".join(providers))
sys.exit(0 if "TensorrtExecutionProvider" in providers else 1)
PY
}

check_onnx_provider_available() {
  local provider="$1"
  local label="$2"
  if [[ -z "$provider" ]]; then
    return
  fi
  "$PYTHON" - "$provider" "$label" <<'PY'
from __future__ import annotations

import sys

import onnxruntime as ort

provider = sys.argv[1]
label = sys.argv[2]
available = ort.get_available_providers()
if provider in available:
    raise SystemExit(0)

print(
    f"Requested {provider!r} for {label}, but ONNX Runtime only exposes: "
    f"{', '.join(available) or '<none>'}.",
    file=sys.stderr,
)
print(
    "Install onnxruntime-gpu in the active environment, set the provider to an available "
    "provider, or run the accuracy-only GPU path: "
    "NPROC_PER_NODE=8 bash scripts/run_gradient50_accuracy_only_8gpu.sh charent/ChatLM-mini-Chinese",
    file=sys.stderr,
)
raise SystemExit(2)
PY
}

check_tools() {
  check_python_module torch
  check_python_module transformers
  if ! command -v optimum-cli >/dev/null 2>&1; then
    echo "Missing optimum-cli. Install Optimum ONNX support first:" >&2
    echo '  python -m pip install "optimum[onnxruntime]" onnx onnxruntime onnxruntime-gpu' >&2
    exit 2
  fi
  check_python_module optimum.onnxruntime
  check_python_module onnxruntime
  check_python_module onnxruntime.quantization
  check_onnx_provider_available "$FP16_ONNX_PROVIDER" "ONNX FP16 export/eval"
  if truthy "$RUN_INT8"; then
    check_onnx_provider_available "$INT8_ONNX_PROVIDER" "ONNX INT8 eval"
  fi

  if truthy "$RUN_TENSORRT"; then
    if ! check_tensorrt_provider; then
      if truthy "$TENSORRT_REQUIRED"; then
        echo "RUN_TENSORRT=1 but ONNX Runtime does not expose TensorrtExecutionProvider." >&2
        echo "Install a GPU ONNX Runtime/TensorRT stack, or set RUN_TENSORRT=0." >&2
        exit 2
      fi
      echo "TensorRT provider is unavailable; disabling TensorRT benchmark."
      RUN_TENSORRT=0
    fi
  fi
}

detect_optimum_dtype_mode() {
  if [[ "$OPTIMUM_DTYPE_MODE" != "auto" ]]; then
    return
  fi

  local help_text
  help_text="$(optimum-cli export onnx --help 2>&1 || true)"
  if [[ "$help_text" == *"--dtype"* ]]; then
    OPTIMUM_DTYPE_MODE="dtype"
  else
    OPTIMUM_DTYPE_MODE="fp16-flag"
  fi
}

export_offline_env() {
  if truthy "$LOCAL_FILES_ONLY"; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  fi
}

run_finetune() {
  if truthy "$SKIP_TRAIN"; then
    if [[ ! -f "${FINETUNED_CHECKPOINT_DIR}/config.json" ]]; then
      echo "SKIP_TRAIN=1 but fine-tuned checkpoint is missing: ${FINETUNED_CHECKPOINT_DIR}" >&2
      exit 2
    fi
    echo "Skipping 5-epoch fine-tune; using ${FINETUNED_CHECKPOINT_DIR}"
    return
  fi

  if [[ -f "${FINETUNED_CHECKPOINT_DIR}/config.json" ]] && ! truthy "$FORCE_TRAIN"; then
    echo "Skipping fine-tune; found ${FINETUNED_CHECKPOINT_DIR}/config.json"
    return
  fi

  reset_output_dir "$FINETUNED_CHECKPOINT_DIR" "$FORCE_TRAIN"
  mkdir -p "$FINETUNED_CHECKPOINT_DIR"

  local train_cmd=(
    scripts/scenic_train_chatlm_sft.py
    --mode "$FINETUNE_MODE"
    --model "$ORIGINAL_MODEL"
    --train-json "$FINETUNE_TRAIN_JSON"
    --output-dir "$FINETUNED_CHECKPOINT_DIR"
    --epochs "$FINETUNE_EPOCHS"
    --batch-size "$TRAIN_BATCH_SIZE"
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$TRAIN_LEARNING_RATE"
    --weight-decay "$TRAIN_WEIGHT_DECAY"
    --warmup-ratio "$TRAIN_WARMUP_RATIO"
    --max-source-length "$TRAIN_MAX_SOURCE_LENGTH"
    --max-target-length "$TRAIN_MAX_TARGET_LENGTH"
  )

  case "$TRAIN_PRECISION" in
    fp16|FP16)
      train_cmd+=(--fp16)
      ;;
    bf16|BF16)
      train_cmd+=(--bf16)
      ;;
    fp32|FP32)
      train_cmd+=(--no-fp16)
      ;;
    *)
      echo "TRAIN_PRECISION must be fp16, bf16, or fp32." >&2
      exit 2
      ;;
  esac

  if truthy "$LOCAL_FILES_ONLY"; then
    train_cmd+=(--local-files-only)
  fi
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_train_args <<< "$TRAIN_EXTRA_ARGS"
    train_cmd+=("${extra_train_args[@]}")
  fi

  echo "Fine-tuning original model for ${FINETUNE_EPOCHS} epoch(s) -> ${FINETUNED_CHECKPOINT_DIR}"
  if [[ "$TRAIN_NPROC_PER_NODE" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$TRAIN_NPROC_PER_NODE" "${train_cmd[@]}"
  else
    "$PYTHON" "${train_cmd[@]}"
  fi
}

repair_checkpoint_assets() {
  local checkpoint_dir="$1"
  local source_ref="$2"
  local label="$3"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "Cannot repair ${label}; checkpoint directory is missing: ${checkpoint_dir}" >&2
    exit 2
  fi

  local repair_cmd=(
    "$PYTHON"
    scripts/repair_checkpoint_tokenizer.py
    --checkpoint "$checkpoint_dir"
  )
  if [[ -n "$source_ref" ]]; then
    repair_cmd+=(--source-tokenizer "$source_ref")
  fi
  if truthy "$LOCAL_FILES_ONLY"; then
    repair_cmd+=(--local-files-only)
  fi

  if [[ -d "$source_ref" ]]; then
    echo "Repairing ${label} tokenizer/custom-code assets from local source: ${source_ref}"
  elif [[ -n "$source_ref" ]]; then
    echo "Repairing ${label} tokenizer/custom-code assets from Hugging Face/cache source: ${source_ref}"
  else
    echo "Repairing ${label} tokenizer/custom-code assets using checkpoint metadata."
    echo "If custom modeling files are still missing, rerun with SOURCE_ASSET_DIR=/path/to/original/base_model-or-hf-id."
  fi
  "${repair_cmd[@]}"

  if [[ -d "$source_ref" && -f "${source_ref}/modeling_chat_model.py" && ! -f "${checkpoint_dir}/modeling_chat_model.py" ]]; then
    echo "Expected modeling_chat_model.py to be copied into ${checkpoint_dir}, but it is still missing." >&2
    exit 1
  fi
}

create_pruned_checkpoint() {
  if [[ -f "${PRUNED_CHECKPOINT_DIR}/config.json" && -f "$PRUNED_SUMMARY_JSON" ]] && ! truthy "$FORCE_PRUNE"; then
    echo "Skipping ${PRUNED_PRETTY_LABEL}; found ${PRUNED_CHECKPOINT_DIR}"
    return
  fi

  reset_output_dir "$PRUNED_CHECKPOINT_DIR" "$FORCE_PRUNE"
  mkdir -p "$PRUNED_CHECKPOINT_DIR"
  mkdir -p "$(dirname "$PRUNED_SUMMARY_JSON")"

  echo "Creating ${PRUNED_PRETTY_LABEL} checkpoint from fine-tuned model -> ${PRUNED_CHECKPOINT_DIR}"
  MODEL_PATH="$FINETUNED_CHECKPOINT_DIR" \
  PRUNED_OUTPUT_DIR="$PRUNED_CHECKPOINT_DIR" \
  PRUNED_SUMMARY_JSON="$PRUNED_SUMMARY_JSON" \
  PRUNE_METHOD="$PRUNE_METHOD" \
  SPARSITY="$SPARSITY" \
  PRUNE_SCOPE="$PRUNE_SCOPE" \
  SPARSITY_BASIS="$SPARSITY_BASIS" \
  PRUNE_LM_HEAD="$PRUNE_LM_HEAD" \
  CALIBRATION_JSON="$CALIBRATION_JSON" \
  CALIBRATION_BATCH_SIZE="$CALIBRATION_BATCH_SIZE" \
  CALIBRATION_BATCHES="$CALIBRATION_BATCHES" \
  MAX_CALIBRATION_EXAMPLES="$MAX_CALIBRATION_EXAMPLES" \
  MAX_INPUT_LEN="$MAX_INPUT_LEN" \
  TRAIN_MAX_TARGET_LENGTH="$TRAIN_MAX_TARGET_LENGTH" \
  LOCAL_FILES_ONLY=1 \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  PYTORCH_DEVICE="$PYTORCH_DEVICE" \
  "$PYTHON" <<'PY'
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import (  # noqa: E402
    DistributedState,
    load_model_and_tokenizer,
    read_records,
    run_pruning,
    save_pruned_model,
    summarize_model,
    truncate_records,
    write_json,
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def resolve_device() -> torch.device:
    requested = os.environ.get("PYTORCH_DEVICE", "auto")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


model_path = os.environ["MODEL_PATH"]
out_dir = Path(os.environ["PRUNED_OUTPUT_DIR"]).expanduser()
summary_json = Path(os.environ["PRUNED_SUMMARY_JSON"]).expanduser()
calibration_json = Path(os.environ["CALIBRATION_JSON"]).expanduser()
max_calibration_examples_raw = os.environ.get("MAX_CALIBRATION_EXAMPLES", "")
max_calibration_examples = int(max_calibration_examples_raw) if max_calibration_examples_raw else None
device = resolve_device()
state = DistributedState(enabled=False, rank=0, local_rank=0, world_size=1, device=device)
args = argparse.Namespace(
    trust_remote_code=env_bool("TRUST_REMOTE_CODE", True),
    local_files_only=env_bool("LOCAL_FILES_ONLY", True),
    bf16=False,
    fp16=True,
    method=os.environ.get("PRUNE_METHOD", "gradient"),
    sparsity=float(os.environ.get("SPARSITY", "0.5")),
    sparsity_basis=os.environ.get("SPARSITY_BASIS", "targeted-linear"),
    prune_scope=os.environ.get("PRUNE_SCOPE", "all-linear"),
    prune_lm_head=env_bool("PRUNE_LM_HEAD", False),
    full_model_correction=True,
    calibration_batch_size=int(os.environ.get("CALIBRATION_BATCH_SIZE", "4")),
    calibration_batches=int(os.environ.get("CALIBRATION_BATCHES", "64")),
    max_input_len=int(os.environ.get("MAX_INPUT_LEN", "256")),
    max_target_len=int(os.environ.get("TRAIN_MAX_TARGET_LENGTH", "128")),
)

calibration_records = truncate_records(read_records(calibration_json), max_calibration_examples)
tokenizer, model = load_model_and_tokenizer(args, model_path, state)
before = summarize_model(model, model_path)
pruning = run_pruning(model, tokenizer, calibration_records, args, state)
after = summarize_model(model, str(out_dir))
save_pruned_model(model, tokenizer, out_dir)
write_json(
    summary_json,
    {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_model_path": model_path,
        "pruned_model_path": str(out_dir),
        "device": str(device),
        "calibration_json": str(calibration_json),
        "calibration_examples_loaded": len(calibration_records),
        "model_before_prune": before,
        "model_after_prune": after,
        "pruning": pruning,
    },
)
print(f"Wrote pruning summary: {summary_json}")
PY
}

require_onnx_files() {
  local dir="$1"
  shopt -s nullglob
  local files=("$dir"/*.onnx)
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No .onnx files found in $dir" >&2
    exit 1
  fi
}

export_onnx() {
  local source_model="$1"
  local out_dir="$2"
  local label="$3"
  local precision="$4"
  local device="$5"

  if [[ -f "${out_dir}/.export.done" ]] && ! truthy "$FORCE_EXPORT"; then
    echo "Skipping ${label} export; found ${out_dir}/.export.done"
    require_onnx_files "$out_dir"
    return
  fi

  reset_output_dir "$out_dir" "$FORCE_EXPORT"
  mkdir -p "$out_dir"

  local cmd=(
    optimum-cli export onnx
    --model "$source_model"
    --task "$ONNX_TASK"
    --opset "$ONNX_OPSET"
    --device "$device"
  )
  if truthy "$TRUST_REMOTE_CODE"; then
    cmd+=(--trust-remote-code)
  fi
  if [[ "$precision" == "fp16" ]]; then
    case "$OPTIMUM_DTYPE_MODE" in
      dtype)
        cmd+=(--dtype fp16)
        ;;
      fp16-flag)
        cmd+=(--fp16)
        ;;
      *)
        echo "OPTIMUM_DTYPE_MODE must be auto, dtype, or fp16-flag." >&2
        exit 2
        ;;
    esac
  fi
  if [[ -n "${ONNX_EXPORT_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_export_args <<< "$ONNX_EXPORT_EXTRA_ARGS"
    cmd+=("${extra_export_args[@]}")
  fi
  cmd+=("$out_dir")

  echo "Exporting ${label} ONNX (${precision}, device=${device}) -> ${out_dir}"
  "${cmd[@]}"
  require_onnx_files "$out_dir"
  touch "${out_dir}/.export.done"
}

quantize_onnx_dynamic_int8() {
  local source_dir="$1"
  local out_dir="$2"
  local label="$3"

  if ! truthy "$RUN_INT8"; then
    echo "RUN_INT8=0; skipping ${label} INT8 quantization."
    return
  fi

  if [[ -f "${out_dir}/.quantize.done" ]] && ! truthy "$FORCE_QUANTIZE"; then
    echo "Skipping ${label} INT8 quantization; found ${out_dir}/.quantize.done"
    require_onnx_files "$out_dir"
    return
  fi

  reset_output_dir "$out_dir" "$FORCE_QUANTIZE"
  mkdir -p "$out_dir"
  echo "Quantizing ${label} ONNX to dynamic INT8 -> ${out_dir}"
  SOURCE_ONNX_DIR="$source_dir" \
  INT8_ONNX_DIR="$out_dir" \
  "$PYTHON" <<'PY'
from __future__ import annotations

import os
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

source_dir = Path(os.environ["SOURCE_ONNX_DIR"]).expanduser()
out_dir = Path(os.environ["INT8_ONNX_DIR"]).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

onnx_files = sorted(source_dir.glob("*.onnx"))
if not onnx_files:
    raise SystemExit(f"No .onnx files found in {source_dir}")

for item in source_dir.iterdir():
    if item.suffix == ".onnx" or item.name.startswith("."):
        continue
    target = out_dir / item.name
    if item.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)

for onnx_path in onnx_files:
    output_path = out_dir / onnx_path.name
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"Quantized {onnx_path.name} -> {output_path}")
PY
  require_onnx_files "$out_dir"
  touch "${out_dir}/.quantize.done"
}

evaluate_accuracy_variant() {
  local label="$1"
  local pretty_label="$2"
  local runtime="$3"
  local source_path="$4"
  local tokenizer_fallback="$5"
  local provider="$6"
  local precision="$7"
  local sparsity_kind="$8"
  local report_json="$9"

  if [[ -f "$report_json" ]] && ! truthy "$FORCE_ACCURACY"; then
    echo "Skipping ${pretty_label} accuracy; found ${report_json}"
    return
  fi

  mkdir -p "$(dirname "$report_json")"
  echo "Evaluating ${pretty_label} accuracy -> ${report_json}"
  VARIANT_LABEL="$label" \
  VARIANT_PRETTY_LABEL="$pretty_label" \
  RUNTIME="$runtime" \
  SOURCE_PATH="$source_path" \
  TOKENIZER_FALLBACK="$tokenizer_fallback" \
  ONNX_PROVIDER="$provider" \
  VARIANT_PRECISION="$precision" \
  VARIANT_SPARSITY_KIND="$sparsity_kind" \
  REPORT_JSON="$report_json" \
  TRAIN_JSON="$TRAIN_JSON" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  MAX_TRAIN_EXAMPLES="$MAX_TRAIN_EXAMPLES" \
  MAX_BENCHMARK_EXAMPLES="$MAX_BENCHMARK_EXAMPLES" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  NUM_BEAMS="$NUM_BEAMS" \
  NUM_RETURN_SEQUENCES="$NUM_RETURN_SEQUENCES" \
  MAX_INPUT_LEN="$MAX_INPUT_LEN" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  IGNORE_SPACES="$IGNORE_SPACES" \
  INCLUDE_PREDICTIONS="$INCLUDE_PREDICTIONS" \
  LOCAL_FILES_ONLY=1 \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  PYTORCH_DEVICE="$PYTORCH_DEVICE" \
  TENSORRT_ENGINE_CACHE_ROOT="$TENSORRT_ENGINE_CACHE_ROOT" \
  TENSORRT_SPARSITY_ENABLE="$TENSORRT_SPARSITY_ENABLE" \
  "$PYTHON" - "${ACCURACY_EXTRA_ARGS[@]}" <<'PY'
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import (  # noqa: E402
    DistributedState,
    compact_metrics,
    evaluate_all,
    read_records,
    summarize_model,
    truncate_records,
    write_json,
)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def env_int_or_none(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return int(value)


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tokenizer_source(source_path: Path, fallback: str) -> str:
    tokenizer_markers = (
        "tokenizer_config.json",
        "tokenizer.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "vocab.txt",
        "vocab.json",
    )
    if source_path.is_dir() and any((source_path / marker).exists() for marker in tokenizer_markers):
        return str(source_path)
    return fallback


class GenerateAdapter:
    def __init__(self, model: Any, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device

    def eval(self) -> "GenerateAdapter":
        if hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def generate(self, **kwargs: Any) -> Any:
        if self.device is not None:
            kwargs = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in kwargs.items()
            }
        return self.model.generate(**kwargs)


parser = argparse.ArgumentParser(description="Evaluate one deployed SCENIC model.")
parser.add_argument("--train-json", default=os.environ["TRAIN_JSON"])
parser.add_argument("--benchmark-json", default=os.environ["BENCHMARK_JSON"])
parser.add_argument("--max-train-examples", type=int, default=env_int_or_none("MAX_TRAIN_EXAMPLES"))
parser.add_argument("--max-benchmark-examples", type=int, default=env_int_or_none("MAX_BENCHMARK_EXAMPLES"))
parser.add_argument("--eval-batch-size", type=int, default=int(os.environ["EVAL_BATCH_SIZE"]))
parser.add_argument("--max-input-len", type=int, default=int(os.environ["MAX_INPUT_LEN"]))
parser.add_argument("--max-new-tokens", type=int, default=int(os.environ["MAX_NEW_TOKENS"]))
parser.add_argument("--num-beams", type=int, default=int(os.environ["NUM_BEAMS"]))
parser.add_argument("--num-return-sequences", type=int, default=int(os.environ["NUM_RETURN_SEQUENCES"]))
parser.add_argument("--ignore-spaces", action=argparse.BooleanOptionalAction, default=env_bool("IGNORE_SPACES", True))
parser.add_argument(
    "--include-predictions",
    action=argparse.BooleanOptionalAction,
    default=env_bool("INCLUDE_PREDICTIONS", True),
)
parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=env_bool("TRUST_REMOTE_CODE", True))
parser.add_argument("--device", default=os.environ.get("PYTORCH_DEVICE", "auto"))
parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--fp16", action="store_true", default=True)
args, unknown_args = parser.parse_known_args()
if unknown_args:
    print(f"Ignoring accuracy args not used by this wrapper: {' '.join(unknown_args)}")
if args.num_return_sequences > args.num_beams:
    raise ValueError("--num-return-sequences cannot exceed --num-beams for beam search.")

runtime = os.environ["RUNTIME"]
source_path = Path(os.environ["SOURCE_PATH"]).expanduser()
report_json = Path(os.environ["REPORT_JSON"]).expanduser()
tokenizer_path = tokenizer_source(source_path, os.environ["TOKENIZER_FALLBACK"])
provider = os.environ["ONNX_PROVIDER"]

if args.device != "auto":
    device = torch.device(args.device)
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
state_device = device if runtime == "pytorch" else torch.device("cpu")
state = DistributedState(enabled=False, rank=0, local_rank=0, world_size=1, device=state_device)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    trust_remote_code=args.trust_remote_code,
    local_files_only=True,
    use_fast=False,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

model_summary: dict[str, Any] = {}
engine_cache_dir: Path | None = None
provider_options: dict[str, Any] | None = None
if runtime == "pytorch":
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": True,
    }
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(str(source_path), **load_kwargs)
    if hasattr(torch_model.config, "use_cache"):
        torch_model.config.use_cache = True
    torch_model.to(device)
    model_summary = summarize_model(torch_model, str(source_path))
    model = GenerateAdapter(torch_model, device=device)
elif runtime in {"onnx", "tensorrt"}:
    if runtime == "tensorrt":
        trt_sparse_enabled = env_bool("TENSORRT_SPARSITY_ENABLE", True)
        engine_cache_dir = (
            Path(os.environ["TENSORRT_ENGINE_CACHE_ROOT"])
            / os.environ["VARIANT_LABEL"]
            / ("sparse" if trt_sparse_enabled else "dense_tactics")
            / "accuracy"
        )
        engine_cache_dir.mkdir(parents=True, exist_ok=True)
        provider_options = {
            "trt_fp16_enable": "1",
            "trt_sparsity_enable": "1" if trt_sparse_enabled else "0",
            "trt_engine_cache_enable": "1",
            "trt_engine_cache_path": str(engine_cache_dir),
        }
    ort_model = ORTModelForSeq2SeqLM.from_pretrained(
        str(source_path),
        provider=provider,
        provider_options=provider_options,
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    model = GenerateAdapter(ort_model)
else:
    raise ValueError(f"Unknown runtime: {runtime}")

train_records = truncate_records(read_records(Path(args.train_json).expanduser()), args.max_train_examples)
benchmark_records = truncate_records(
    read_records(Path(args.benchmark_json).expanduser()),
    args.max_benchmark_examples,
)
datasets = {
    "benchmark": benchmark_records,
    "training": train_records,
}
results = evaluate_all(model, tokenizer, datasets, args, state, label=os.environ["VARIANT_LABEL"])
engine_size_bytes = path_size_bytes(engine_cache_dir) if engine_cache_dir is not None else 0
report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "variant": os.environ["VARIANT_LABEL"],
    "variant_label": os.environ["VARIANT_PRETTY_LABEL"],
    "runtime": runtime,
    "precision": os.environ["VARIANT_PRECISION"],
    "sparsity_kind": os.environ["VARIANT_SPARSITY_KIND"],
    "source_path": str(source_path),
    "tokenizer_path": tokenizer_path,
    "onnx_provider": provider if runtime != "pytorch" else None,
    "onnx_provider_options": provider_options,
    "model_or_engine_size_bytes": engine_size_bytes or path_size_bytes(source_path),
    "model_or_engine_size_mb": (engine_size_bytes or path_size_bytes(source_path)) / 1_000_000,
    "tensorrt_engine_cache_dir": str(engine_cache_dir) if engine_cache_dir is not None else None,
    "tensorrt_engine_cache_size_bytes": engine_size_bytes,
    "generation": {
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "max_new_tokens": args.max_new_tokens,
        "ignore_spaces": args.ignore_spaces,
        "batch_size": args.eval_batch_size,
    },
    "datasets": {
        "benchmark": {"path": str(Path(args.benchmark_json).expanduser()), "total": len(benchmark_records)},
        "training": {"path": str(Path(args.train_json).expanduser()), "total": len(train_records)},
    },
    "model": model_summary,
    "summary": {
        "accuracy_definition": "accuracy is exact-match@1 / EM@1",
        os.environ["VARIANT_LABEL"]: compact_metrics(results),
    },
    "evaluations": results,
}
write_json(report_json, report)
print(f"Wrote accuracy report: {report_json}")
PY
}

accuracy_gpu_ids=()
accuracy_job_pids=()
accuracy_job_labels=()
accuracy_gpu_index=0

prepare_accuracy_gpus() {
  accuracy_gpu_ids=()
  local gpu_csv
  gpu_csv="$(detect_accuracy_gpu_ids)"
  gpu_csv="${gpu_csv// /,}"
  if [[ -n "$gpu_csv" ]]; then
    IFS=',' read -r -a accuracy_gpu_ids <<< "$gpu_csv"
  fi
}

run_accuracy_eval() {
  local label="$1"
  local pretty_label="$2"

  if truthy "$RUN_ACCURACY_PARALLEL" && [[ "${#accuracy_gpu_ids[@]}" -gt 1 ]]; then
    local gpu="${accuracy_gpu_ids[$((accuracy_gpu_index % ${#accuracy_gpu_ids[@]}))]}"
    accuracy_gpu_index=$((accuracy_gpu_index + 1))
    echo "Starting accuracy eval on GPU ${gpu}: ${pretty_label}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      export PYTORCH_DEVICE="$ACCURACY_PYTORCH_DEVICE"
      evaluate_accuracy_variant "$@"
    ) &
    accuracy_job_pids+=("$!")
    accuracy_job_labels+=("$pretty_label")
    return
  fi

  evaluate_accuracy_variant "$@"
}

wait_accuracy_evals() {
  local status=0
  local index
  for index in "${!accuracy_job_pids[@]}"; do
    if ! wait "${accuracy_job_pids[$index]}"; then
      echo "Accuracy eval failed: ${accuracy_job_labels[$index]}" >&2
      status=1
    fi
  done
  return "$status"
}

benchmark_runtime() {
  if [[ -f "$RUNTIME_BENCHMARK_JSON" ]] && ! truthy "$FORCE_BENCHMARK"; then
    echo "Skipping runtime benchmark; found ${RUNTIME_BENCHMARK_JSON}"
    return
  fi

  mkdir -p "$REPORT_ROOT" "$TENSORRT_ENGINE_CACHE_ROOT"
  echo "Benchmarking batch=1 deployment latency -> ${RUNTIME_BENCHMARK_JSON}"
  FINETUNED_CHECKPOINT_DIR="$FINETUNED_CHECKPOINT_DIR" \
  PRUNED_CHECKPOINT_DIR="$PRUNED_CHECKPOINT_DIR" \
  FP16_DENSE_ONNX="$FP16_DENSE_ONNX" \
  FP16_PRUNED_ONNX="$FP16_PRUNED_ONNX" \
  INT8_DENSE_ONNX="$INT8_DENSE_ONNX" \
  INT8_PRUNED_ONNX="$INT8_PRUNED_ONNX" \
  BENCHMARK_JSON="$BENCHMARK_JSON" \
  LATENCY_SEQ_LENGTHS="$LATENCY_SEQ_LENGTHS" \
  LATENCY_BATCH_SIZE="$LATENCY_BATCH_SIZE" \
  LATENCY_QUERIES="$LATENCY_QUERIES" \
  LATENCY_WARMUP="$LATENCY_WARMUP" \
  LATENCY_NUM_BEAMS="$LATENCY_NUM_BEAMS" \
  LATENCY_MAX_NEW_TOKENS="$LATENCY_MAX_NEW_TOKENS" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  PYTORCH_DEVICE="$PYTORCH_DEVICE" \
  FP16_ONNX_PROVIDER="$FP16_ONNX_PROVIDER" \
  INT8_ONNX_PROVIDER="$INT8_ONNX_PROVIDER" \
  RUN_INT8="$RUN_INT8" \
  RUN_TENSORRT="$RUN_TENSORRT" \
  TENSORRT_ONNX_PROVIDER="$TENSORRT_ONNX_PROVIDER" \
  TENSORRT_SPARSITY_ENABLE="$TENSORRT_SPARSITY_ENABLE" \
  TENSORRT_ENGINE_CACHE_ROOT="$TENSORRT_ENGINE_CACHE_ROOT" \
  RUNTIME_BENCHMARK_JSON="$RUNTIME_BENCHMARK_JSON" \
  "$PYTHON" <<'PY'
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import extract_prompt_response, read_records  # noqa: E402


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def _sample_once(self) -> int | None:
        if not self.available:
            return None
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        values: list[int] = []
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(int(line.split()[0]))
            except ValueError:
                continue
        return max(values) if values else None

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._sample_once()
            if value is not None:
                self.samples.append(value)
            time.sleep(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        if self.available:
            first = self._sample_once()
            if first is not None:
                self.samples.append(first)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        last = self._sample_once()
        if last is not None:
            self.samples.append(last)

    @property
    def peak_mb(self) -> int | None:
        return max(self.samples) if self.samples else None


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tokenizer_for(path: Path, fallback: Path | None = None) -> Any:
    source = path
    if fallback is not None and not (path / "tokenizer_config.json").exists():
        source = fallback
    tokenizer = AutoTokenizer.from_pretrained(
        str(source),
        trust_remote_code=env_bool("TRUST_REMOTE_CODE", True),
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class RuntimeModel:
    def __init__(self, runtime: str, source: Path, tokenizer_fallback: Path | None, provider: str | None, cache_dir: Path | None) -> None:
        self.runtime = runtime
        self.source = source
        self.provider = provider
        self.cache_dir = cache_dir
        self.device: torch.device | None = None
        self.tokenizer = tokenizer_for(source, tokenizer_fallback)

        if runtime == "pytorch":
            requested = os.environ.get("PYTORCH_DEVICE", "auto")
            if requested != "auto":
                self.device = torch.device(requested)
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            load_kwargs: dict[str, Any] = {
                "trust_remote_code": env_bool("TRUST_REMOTE_CODE", True),
                "local_files_only": True,
            }
            if self.device.type == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(source), **load_kwargs)
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = True
            self.model.to(self.device)
            self.model.eval()
        elif runtime in {"onnx", "tensorrt"}:
            provider_options: dict[str, Any] | None = None
            if runtime == "tensorrt":
                assert cache_dir is not None
                cache_dir.mkdir(parents=True, exist_ok=True)
                provider_options = {
                    "trt_fp16_enable": "1",
                    "trt_sparsity_enable": "1" if env_bool("TENSORRT_SPARSITY_ENABLE", True) else "0",
                    "trt_engine_cache_enable": "1",
                    "trt_engine_cache_path": str(cache_dir),
                }
            self.model = ORTModelForSeq2SeqLM.from_pretrained(
                str(source),
                provider=provider,
                provider_options=provider_options,
                local_files_only=True,
                trust_remote_code=env_bool("TRUST_REMOTE_CODE", True),
            )
        else:
            raise ValueError(f"Unknown runtime {runtime}")

    def generate_once(self, prompt: str, seq_len: int, max_new_tokens: int, num_beams: int) -> None:
        encoded = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        encoded.pop("token_type_ids", None)
        if self.device is not None:
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
            )


benchmark_records = read_records(Path(os.environ["BENCHMARK_JSON"]).expanduser())
prompts = [extract_prompt_response(record)[0] for record in benchmark_records]
queries = max(1, int(os.environ["LATENCY_QUERIES"]))
warmup = max(0, int(os.environ["LATENCY_WARMUP"]))
batch_size = int(os.environ["LATENCY_BATCH_SIZE"])
if batch_size != 1:
    raise ValueError("This deployment benchmark is intentionally fixed to LATENCY_BATCH_SIZE=1.")
seq_lengths = [int(item) for item in os.environ["LATENCY_SEQ_LENGTHS"].split() if item.strip()]
max_new_tokens = int(os.environ["LATENCY_MAX_NEW_TOKENS"])
num_beams = int(os.environ["LATENCY_NUM_BEAMS"])

dense_checkpoint = Path(os.environ["FINETUNED_CHECKPOINT_DIR"]).expanduser()
pruned_checkpoint = Path(os.environ["PRUNED_CHECKPOINT_DIR"]).expanduser()
dense_onnx = Path(os.environ["FP16_DENSE_ONNX"]).expanduser()
pruned_onnx = Path(os.environ["FP16_PRUNED_ONNX"]).expanduser()
int8_dense_onnx = Path(os.environ["INT8_DENSE_ONNX"]).expanduser()
int8_pruned_onnx = Path(os.environ["INT8_PRUNED_ONNX"]).expanduser()
trt_cache_root = Path(os.environ["TENSORRT_ENGINE_CACHE_ROOT"]).expanduser()
run_int8 = env_bool("RUN_INT8", True)
run_tensorrt = env_bool("RUN_TENSORRT", False)
trt_sparse_enabled = env_bool("TENSORRT_SPARSITY_ENABLE", True)
trt_cache_suffix = "sparse" if trt_sparse_enabled else "dense_tactics"

variants = [
    {
        "model_variant": "dense",
        "runtime": "pytorch",
        "runtime_label": "PyTorch FP16",
        "precision": "fp16",
        "source": dense_checkpoint,
        "tokenizer_fallback": None,
        "provider": None,
        "cache_dir": None,
    },
    {
        "model_variant": "pruned",
        "runtime": "pytorch",
        "runtime_label": "PyTorch FP16",
        "precision": "fp16",
        "source": pruned_checkpoint,
        "tokenizer_fallback": None,
        "provider": None,
        "cache_dir": None,
    },
    {
        "model_variant": "dense",
        "runtime": "onnx",
        "runtime_label": "ONNX FP16",
        "precision": "fp16",
        "source": dense_onnx,
        "tokenizer_fallback": dense_checkpoint,
        "provider": os.environ["FP16_ONNX_PROVIDER"],
        "cache_dir": None,
    },
    {
        "model_variant": "pruned",
        "runtime": "onnx",
        "runtime_label": "ONNX FP16",
        "precision": "fp16",
        "source": pruned_onnx,
        "tokenizer_fallback": pruned_checkpoint,
        "provider": os.environ["FP16_ONNX_PROVIDER"],
        "cache_dir": None,
    },
]
if run_int8:
    variants.extend(
        [
            {
                "model_variant": "dense",
                "runtime": "onnx",
                "runtime_label": "ONNX INT8",
                "precision": "int8",
                "source": int8_dense_onnx,
                "tokenizer_fallback": dense_checkpoint,
                "provider": os.environ["INT8_ONNX_PROVIDER"],
                "cache_dir": None,
            },
            {
                "model_variant": "pruned",
                "runtime": "onnx",
                "runtime_label": "ONNX INT8",
                "precision": "int8",
                "source": int8_pruned_onnx,
                "tokenizer_fallback": pruned_checkpoint,
                "provider": os.environ["INT8_ONNX_PROVIDER"],
                "cache_dir": None,
            },
        ]
    )
if run_tensorrt:
    variants.extend(
        [
            {
                "model_variant": "dense",
                "runtime": "tensorrt",
                "runtime_label": "TensorRT FP16",
                "precision": "fp16",
                "source": dense_onnx,
                "tokenizer_fallback": dense_checkpoint,
                "provider": os.environ["TENSORRT_ONNX_PROVIDER"],
                "cache_dir": trt_cache_root / "dense" / trt_cache_suffix,
            },
            {
                "model_variant": "pruned",
                "runtime": "tensorrt",
                "runtime_label": "TensorRT FP16",
                "precision": "fp16",
                "source": pruned_onnx,
                "tokenizer_fallback": pruned_checkpoint,
                "provider": os.environ["TENSORRT_ONNX_PROVIDER"],
                "cache_dir": trt_cache_root / "pruned" / trt_cache_suffix,
            },
        ]
    )

rows: list[dict[str, Any]] = []
for spec in variants:
    print(f"Loading {spec['model_variant']} {spec['runtime_label']} for latency benchmark...")
    runtime_model = RuntimeModel(
        runtime=spec["runtime"],
        source=spec["source"],
        tokenizer_fallback=spec["tokenizer_fallback"],
        provider=spec["provider"],
        cache_dir=spec["cache_dir"],
    )
    source_size = path_size_bytes(spec["source"])

    for seq_len in seq_lengths:
        selected_prompts = [prompts[index % len(prompts)] for index in range(queries)]
        warmup_prompts = [prompts[index % len(prompts)] for index in range(warmup)]
        for prompt in warmup_prompts:
            runtime_model.generate_once(prompt, seq_len, max_new_tokens, num_beams)
        cuda_sync()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        latencies_ms: list[float] = []
        with MemorySampler() as memory_sampler:
            for prompt in selected_prompts:
                cuda_sync()
                started = time.perf_counter()
                runtime_model.generate_once(prompt, seq_len, max_new_tokens, num_beams)
                cuda_sync()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

        mean_ms = mean(latencies_ms) if latencies_ms else 0.0
        p95_ms = percentile(latencies_ms, 0.95)
        throughput_qps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
        torch_peak_allocated = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        )
        torch_peak_reserved = (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None
        )
        engine_size = path_size_bytes(spec["cache_dir"]) if spec["cache_dir"] is not None else 0
        rows.append(
            {
                "model_variant": spec["model_variant"],
                "runtime": spec["runtime"],
                "runtime_label": spec["runtime_label"],
                "precision": spec["precision"],
                "batch_size": batch_size,
                "input_length": seq_len,
                "queries": len(latencies_ms),
                "warmup_queries": warmup,
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
                "mean_latency_ms_per_query": mean_ms,
                "p95_latency_ms_per_query": p95_ms,
                "throughput_queries_per_second": throughput_qps,
                "peak_gpu_memory_mb_nvidia_smi": memory_sampler.peak_mb,
                "peak_torch_allocated_bytes": torch_peak_allocated,
                "peak_torch_reserved_bytes": torch_peak_reserved,
                "model_size_bytes": source_size,
                "model_size_mb": source_size / 1_000_000,
                "tensorrt_engine_cache_dir": str(spec["cache_dir"]) if spec["cache_dir"] is not None else None,
                "tensorrt_sparsity_enable": trt_sparse_enabled if spec["runtime"] == "tensorrt" else None,
                "tensorrt_engine_size_bytes": engine_size,
                "tensorrt_engine_size_mb": engine_size / 1_000_000,
            }
        )
        print(
            f"{spec['model_variant']} {spec['runtime_label']} seq={seq_len}: "
            f"mean={mean_ms:.2f} ms, p95={p95_ms:.2f} ms, qps={throughput_qps:.2f}"
        )

output_json = Path(os.environ["RUNTIME_BENCHMARK_JSON"]).expanduser()
output_json.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "benchmark_json": os.environ["BENCHMARK_JSON"],
    "metric_contract": {
        "runtime": "PyTorch FP16 and ONNX FP16 by default; optional TensorRT FP16 with RUN_TENSORRT=1",
        "precision": "FP16 and INT8 ONNX deployment benchmark rows; optional TensorRT FP16 with RUN_TENSORRT=1",
        "latency": "mean latency, ms/query, batch size 1",
        "p95_latency": "95th percentile latency, ms/query",
        "throughput": "queries/second",
        "peak_memory": "peak GPU memory sampled with nvidia-smi when available",
        "engine_model_size": "model directory or TensorRT engine cache size in MB",
        "input_length": seq_lengths,
        "batch_size": batch_size,
    },
    "rows": rows,
}
with output_json.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"Wrote runtime benchmark report: {output_json}")
PY
}

write_final_report() {
  FINAL_JSON="$FINAL_JSON" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  ORIGINAL_MODEL="$ORIGINAL_MODEL" \
  FINETUNED_CHECKPOINT_DIR="$FINETUNED_CHECKPOINT_DIR" \
  PRUNED_CHECKPOINT_DIR="$PRUNED_CHECKPOINT_DIR" \
  PRUNED_SUMMARY_JSON="$PRUNED_SUMMARY_JSON" \
  PRUNED_PRETTY_LABEL="$PRUNED_PRETTY_LABEL" \
  PRUNE_METHOD="$PRUNE_METHOD" \
  SPARSITY="$SPARSITY" \
  RUNTIME_BENCHMARK_JSON="$RUNTIME_BENCHMARK_JSON" \
  REPORT_ROOT="$REPORT_ROOT" \
  RUN_INT8="$RUN_INT8" \
  RUN_TENSORRT="$RUN_TENSORRT" \
  "$PYTHON" <<'PY'
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


def maybe_read(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


report_root = Path(os.environ["REPORT_ROOT"])
pruned_label = os.environ["PRUNED_PRETTY_LABEL"]
accuracy_specs = [
    ("pytorch_fp16_dense", "PyTorch FP16 dense", report_root / "pytorch_fp16_dense_accuracy_report.json"),
    ("pytorch_fp16_pruned", f"PyTorch FP16 {pruned_label}", report_root / "pytorch_fp16_pruned_accuracy_report.json"),
    ("onnx_fp16_dense", "ONNX FP16 dense", report_root / "onnx_fp16_dense_accuracy_report.json"),
    ("onnx_fp16_pruned", f"ONNX FP16 {pruned_label}", report_root / "onnx_fp16_pruned_accuracy_report.json"),
]
if env_bool("RUN_TENSORRT", False):
    accuracy_specs.extend(
        [
            ("tensorrt_fp16_dense", "TensorRT FP16 dense", report_root / "tensorrt_fp16_dense_accuracy_report.json"),
            (
                "tensorrt_fp16_pruned",
                f"TensorRT FP16 {pruned_label}",
                report_root / "tensorrt_fp16_pruned_accuracy_report.json",
            ),
        ]
    )
if env_bool("RUN_INT8", True):
    accuracy_specs.extend(
        [
            ("onnx_int8_dense", "ONNX INT8 dense", report_root / "onnx_int8_dense_accuracy_report.json"),
            ("onnx_int8_pruned", f"ONNX INT8 {pruned_label}", report_root / "onnx_int8_pruned_accuracy_report.json"),
        ]
    )

accuracy_reports: dict[str, Any] = {}
accuracy_table: list[dict[str, Any]] = []
for key, label, path in accuracy_specs:
    report = maybe_read(path)
    if report is None:
        continue
    summary = report.get("summary", {}).get(key, {})
    accuracy_reports[key] = {
        "label": label,
        "report_json": str(path),
        "runtime": report.get("runtime"),
        "precision": report.get("precision"),
        "source_path": report.get("source_path"),
        "model_or_engine_size_mb": report.get("model_or_engine_size_mb"),
        "summary": summary,
    }
    for dataset, metrics in summary.items():
        accuracy_table.append(
            {
                "variant": key,
                "label": label,
                "dataset": dataset,
                "total": metrics.get("total", 0),
                "em1": metrics.get("em1", 0.0),
                "em5": metrics.get("em5", 0.0),
                "em1_percent": metrics.get("em1_percent", 0.0),
                "em5_percent": metrics.get("em5_percent", 0.0),
                "accuracy": metrics.get("accuracy", metrics.get("em1", 0.0)),
                "accuracy_percent": metrics.get("accuracy_percent", metrics.get("em1_percent", 0.0)),
            }
        )

baseline_by_dataset = {
    row["dataset"]: row
    for row in accuracy_table
    if row["variant"] == "pytorch_fp16_dense"
}
accuracy_delta_table: list[dict[str, Any]] = []
for row in accuracy_table:
    baseline = baseline_by_dataset.get(row["dataset"])
    if baseline is None:
        continue
    accuracy_delta_table.append(
        {
            **row,
            "baseline_variant": "pytorch_fp16_dense",
            "delta_em1_percent": row["em1_percent"] - baseline["em1_percent"],
            "delta_em5_percent": row["em5_percent"] - baseline["em5_percent"],
            "retention_em1_percent": (
                row["em1_percent"] / baseline["em1_percent"] * 100.0
                if baseline["em1_percent"]
                else None
            ),
            "retention_em5_percent": (
                row["em5_percent"] / baseline["em5_percent"] * 100.0
                if baseline["em5_percent"]
                else None
            ),
        }
    )

model_size_table = [
    {
        "variant": key,
        "label": report["label"],
        "runtime": report["runtime"],
        "precision": report["precision"],
        "model_or_engine_size_mb": report["model_or_engine_size_mb"],
        "source_path": report["source_path"],
    }
    for key, report in accuracy_reports.items()
]

runtime_report = maybe_read(Path(os.environ["RUNTIME_BENCHMARK_JSON"]))
output_json = Path(os.environ["FINAL_JSON"])
output_json.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "original_model": os.environ["ORIGINAL_MODEL"],
    "fine_tuned_checkpoint_path": os.environ["FINETUNED_CHECKPOINT_DIR"],
    "pruned_checkpoint_path": os.environ["PRUNED_CHECKPOINT_DIR"],
    "pruning_summary_json": os.environ["PRUNED_SUMMARY_JSON"],
    "pruning_method": os.environ["PRUNE_METHOD"],
    "requested_sparsity": float(os.environ["SPARSITY"]),
    "pruned_variant_label": pruned_label,
    "output_root": os.environ["OUTPUT_ROOT"],
    "metric_contract": {
        "runtime": "PyTorch FP16, ONNX FP16, and ONNX INT8 by default; optional TensorRT FP16 with RUN_TENSORRT=1",
        "precision": "FP16 and INT8 comparison after 50% gradient pruning; this is a sparse quantized baseline for ASIC comparison, not a full edge deployment claim",
        "latency": "mean latency, ms/query, batch size 1",
        "p95_latency": "95th percentile latency, ms/query",
        "throughput": "queries/second",
        "peak_memory": "GPU/edge memory during inference",
        "engine_model_size": "MB of ONNX model directory or TensorRT engine cache",
        "accuracy_after_export": "EM@1 / EM@5 on both benchmark and training data",
        "batch_size": 1,
    },
    "accuracy_reports": accuracy_reports,
    "accuracy_table": accuracy_table,
    "accuracy_delta_table": accuracy_delta_table,
    "model_size_table": model_size_table,
    "runtime_benchmark": runtime_report,
    "pruning_summary": maybe_read(Path(os.environ["PRUNED_SUMMARY_JSON"])),
}
with output_json.open("w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
print(f"Wrote final deployment report: {output_json}")
PY
}

check_tools
detect_optimum_dtype_mode
export_offline_env
mkdir -p "$CHECKPOINT_ROOT" "$ONNX_ROOT" "$INTERMEDIATE_ROOT" "$REPORT_ROOT" "$TENSORRT_ENGINE_CACHE_ROOT"

echo "Output root: $OUTPUT_ROOT"
echo "Original model: $ORIGINAL_MODEL"
echo "Fine-tune: ${FINETUNE_MODE}, ${FINETUNE_EPOCHS} epoch(s), train=${FINETUNE_TRAIN_JSON}"
echo "Benchmark accuracy data: $BENCHMARK_JSON"
echo "Deployment benchmark seq lengths: $LATENCY_SEQ_LENGTHS, batch=1"

run_finetune
repair_checkpoint_assets "$FINETUNED_CHECKPOINT_DIR" "$SOURCE_ASSET_DIR" "fine-tuned checkpoint"
create_pruned_checkpoint
repair_checkpoint_assets "$PRUNED_CHECKPOINT_DIR" "$FINETUNED_CHECKPOINT_DIR" "${PRUNED_PRETTY_LABEL} checkpoint"

export_onnx "$FINETUNED_CHECKPOINT_DIR" "$FP16_DENSE_ONNX" "fine-tuned dense FP16" "fp16" "$EXPORT_DEVICE"
export_onnx "$PRUNED_CHECKPOINT_DIR" "$FP16_PRUNED_ONNX" "fine-tuned ${PRUNED_PRETTY_LABEL} FP16" "fp16" "$EXPORT_DEVICE"

if truthy "$RUN_INT8"; then
  export_onnx "$FINETUNED_CHECKPOINT_DIR" "$FP32_DENSE_ONNX" "fine-tuned dense FP32 INT8 source" "fp32" "$FP32_EXPORT_DEVICE"
  export_onnx "$PRUNED_CHECKPOINT_DIR" "$FP32_PRUNED_ONNX" "fine-tuned ${PRUNED_PRETTY_LABEL} FP32 INT8 source" "fp32" "$FP32_EXPORT_DEVICE"
  quantize_onnx_dynamic_int8 "$FP32_DENSE_ONNX" "$INT8_DENSE_ONNX" "fine-tuned dense"
  quantize_onnx_dynamic_int8 "$FP32_PRUNED_ONNX" "$INT8_PRUNED_ONNX" "fine-tuned ${PRUNED_PRETTY_LABEL}"
fi

prepare_accuracy_gpus
if truthy "$RUN_ACCURACY_PARALLEL" && [[ "${#accuracy_gpu_ids[@]}" -gt 1 ]]; then
  echo "Accuracy evals will run in parallel across GPUs: ${accuracy_gpu_ids[*]}"
else
  echo "Accuracy evals will run sequentially."
fi

run_accuracy_eval \
  "pytorch_fp16_dense" \
  "PyTorch FP16 dense" \
  "pytorch" \
  "$FINETUNED_CHECKPOINT_DIR" \
  "$FINETUNED_CHECKPOINT_DIR" \
  "" \
  "fp16" \
  "dense" \
  "$REPORT_ROOT/pytorch_fp16_dense_accuracy_report.json"

run_accuracy_eval \
  "pytorch_fp16_pruned" \
  "PyTorch FP16 ${PRUNED_PRETTY_LABEL}" \
  "pytorch" \
  "$PRUNED_CHECKPOINT_DIR" \
  "$PRUNED_CHECKPOINT_DIR" \
  "" \
  "fp16" \
  "pruned" \
  "$REPORT_ROOT/pytorch_fp16_pruned_accuracy_report.json"

run_accuracy_eval \
  "onnx_fp16_dense" \
  "ONNX FP16 dense" \
  "onnx" \
  "$FP16_DENSE_ONNX" \
  "$FINETUNED_CHECKPOINT_DIR" \
  "$FP16_ONNX_PROVIDER" \
  "fp16" \
  "dense" \
  "$REPORT_ROOT/onnx_fp16_dense_accuracy_report.json"

run_accuracy_eval \
  "onnx_fp16_pruned" \
  "ONNX FP16 ${PRUNED_PRETTY_LABEL}" \
  "onnx" \
  "$FP16_PRUNED_ONNX" \
  "$PRUNED_CHECKPOINT_DIR" \
  "$FP16_ONNX_PROVIDER" \
  "fp16" \
  "pruned" \
  "$REPORT_ROOT/onnx_fp16_pruned_accuracy_report.json"

if truthy "$RUN_TENSORRT"; then
  run_accuracy_eval \
    "tensorrt_fp16_dense" \
    "TensorRT FP16 dense" \
    "tensorrt" \
    "$FP16_DENSE_ONNX" \
    "$FINETUNED_CHECKPOINT_DIR" \
    "$TENSORRT_ONNX_PROVIDER" \
    "fp16" \
    "dense" \
    "$REPORT_ROOT/tensorrt_fp16_dense_accuracy_report.json"

  run_accuracy_eval \
    "tensorrt_fp16_pruned" \
    "TensorRT FP16 ${PRUNED_PRETTY_LABEL}" \
    "tensorrt" \
    "$FP16_PRUNED_ONNX" \
    "$PRUNED_CHECKPOINT_DIR" \
    "$TENSORRT_ONNX_PROVIDER" \
    "fp16" \
    "pruned" \
    "$REPORT_ROOT/tensorrt_fp16_pruned_accuracy_report.json"
fi

if truthy "$RUN_INT8"; then
  run_accuracy_eval \
    "onnx_int8_dense" \
    "ONNX INT8 dense" \
    "onnx" \
    "$INT8_DENSE_ONNX" \
    "$FINETUNED_CHECKPOINT_DIR" \
    "$INT8_ONNX_PROVIDER" \
    "int8" \
    "dense" \
    "$REPORT_ROOT/onnx_int8_dense_accuracy_report.json"

  run_accuracy_eval \
    "onnx_int8_pruned" \
    "ONNX INT8 ${PRUNED_PRETTY_LABEL}" \
    "onnx" \
    "$INT8_PRUNED_ONNX" \
    "$PRUNED_CHECKPOINT_DIR" \
    "$INT8_ONNX_PROVIDER" \
    "int8" \
    "pruned" \
    "$REPORT_ROOT/onnx_int8_pruned_accuracy_report.json"
fi

wait_accuracy_evals
benchmark_runtime
write_final_report

echo "Done. Final deployment report: $FINAL_JSON"
