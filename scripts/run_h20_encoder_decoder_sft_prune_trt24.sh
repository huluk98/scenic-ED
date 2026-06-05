#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage:
  bash scripts/run_h20_encoder_decoder_sft_prune_trt24.sh --base_model /PATH/TO/BASE_MODEL_OR_HF_ID

Only --base_model is required. All other paths default to repo-local values:
  --train_jsonl       data/SCENIC_full_training_dataset.json
  --iot200_jsonl      generated/iot_instruction_benchmark_200.json
  --output_dir        runs/h20_encoder_decoder_trt24_<UTC timestamp>

Optional overrides:
  --gpus              default: keep CUDA_VISIBLE_DEVICES, else auto-detect
  --epochs            default: 5
  --source_seq_len    default: 64
  --target_seq_len    default: 64
  --seq_len           default: 64
  --batch_size        default: 8 per GPU
  --eval_batch_size   default: 8 per GPU
  --measure_iters     default: 1000
  --warmup_iters      default: 100

Useful env overrides:
  LOCAL_FILES_ONLY    default: auto; local dirs stay offline, HF ids are downloaded first
  LOCAL_BASE_MODEL_DIR default: <output_dir>/base_model for downloaded HF snapshots
  HF_TOKEN            optional token for private Hugging Face models

Example:
  bash scripts/run_h20_encoder_decoder_sft_prune_trt24.sh \
    --base_model charent/ChatLM-mini-Chinese
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

BASE_MODEL="${BASE_MODEL:-}"
TRAIN_JSONL="${TRAIN_JSONL:-data/SCENIC_full_training_dataset.json}"
IOT200_JSONL="${IOT200_JSONL:-generated/iot_instruction_benchmark_200.json}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/h20_encoder_decoder_trt24_${RUN_ID}}"
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
GPUS="${GPUS:-${ORIGINAL_CUDA_VISIBLE_DEVICES}}"
EPOCHS="${EPOCHS:-5}"
SOURCE_SEQ_LEN="${SOURCE_SEQ_LEN:-64}"
TARGET_SEQ_LEN="${TARGET_SEQ_LEN:-64}"
SEQ_LEN="${SEQ_LEN:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MEASURE_ITERS="${MEASURE_ITERS:-1000}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"

TRAIN_GRADIENT_ACCUMULATION_STEPS="${TRAIN_GRADIENT_ACCUMULATION_STEPS:-1}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-5e-5}"
TRAIN_WEIGHT_DECAY="${TRAIN_WEIGHT_DECAY:-0.01}"
TRAIN_WARMUP_RATIO="${TRAIN_WARMUP_RATIO:-0.03}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
DDP_TIMEOUT_MINUTES="${DDP_TIMEOUT_MINUTES:-10}"
IGNORE_SPACES="${IGNORE_SPACES:-1}"
PRUNE_LM_HEAD="${PRUNE_LM_HEAD:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-auto}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"
FORCE_TRT="${FORCE_TRT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PRUNE_EVAL="${SKIP_PRUNE_EVAL:-0}"
SKIP_ONNX="${SKIP_ONNX:-0}"
SKIP_TRT="${SKIP_TRT:-0}"
SKIP_LATENCY="${SKIP_LATENCY:-0}"

detect_gpu_mask() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    if ! count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"; then
      count="0"
    fi
    if [[ "${count}" =~ ^[0-9]+$ && "${count}" -gt 0 ]]; then
      seq -s, 0 $((count - 1))
      return
    fi
  fi
  echo "0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)
      BASE_MODEL="$2"
      shift 2
      ;;
    --base_model=*)
      BASE_MODEL="${1#*=}"
      shift
      ;;
    --train_jsonl)
      TRAIN_JSONL="$2"
      shift 2
      ;;
    --train_jsonl=*)
      TRAIN_JSONL="${1#*=}"
      shift
      ;;
    --iot200_jsonl)
      IOT200_JSONL="$2"
      shift 2
      ;;
    --iot200_jsonl=*)
      IOT200_JSONL="${1#*=}"
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output_dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS="${1#*=}"
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      shift
      ;;
    --source_seq_len)
      SOURCE_SEQ_LEN="$2"
      shift 2
      ;;
    --source_seq_len=*)
      SOURCE_SEQ_LEN="${1#*=}"
      shift
      ;;
    --target_seq_len)
      TARGET_SEQ_LEN="$2"
      shift 2
      ;;
    --target_seq_len=*)
      TARGET_SEQ_LEN="${1#*=}"
      shift
      ;;
    --seq_len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --seq_len=*)
      SEQ_LEN="${1#*=}"
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --eval_batch_size)
      EVAL_BATCH_SIZE="$2"
      shift 2
      ;;
    --eval_batch_size=*)
      EVAL_BATCH_SIZE="${1#*=}"
      shift
      ;;
    --measure_iters)
      MEASURE_ITERS="$2"
      shift 2
      ;;
    --measure_iters=*)
      MEASURE_ITERS="${1#*=}"
      shift
      ;;
    --warmup_iters)
      WARMUP_ITERS="$2"
      shift 2
      ;;
    --warmup_iters=*)
      WARMUP_ITERS="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$BASE_MODEL" ]]; then
  echo "Missing required --base_model /PATH/TO/BASE_MODEL_OR_HF_ID." >&2
  usage
  exit 2
fi

if [[ -z "$GPUS" ]]; then
  GPUS="$(detect_gpu_mask)"
fi
if [[ -n "$GPUS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
  IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
else
  GPU_ARRAY=(0)
fi
NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_ARRAY[@]}}"
export NPROC_PER_NODE

ENV_DIR="${OUTPUT_DIR}/env"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
EVAL_DIR="${OUTPUT_DIR}/eval"
ONNX_DIR="${OUTPUT_DIR}/onnx"
ENGINE_DIR="${OUTPUT_DIR}/engines"
REPORT_DIR="${OUTPUT_DIR}/reports"
LOG_DIR="${OUTPUT_DIR}/logs"
RESULT_DIR="${OUTPUT_DIR}/results"
HELPER_DIR="${OUTPUT_DIR}/helpers"

BASE_MODEL_SOURCE="$BASE_MODEL"
LOCAL_BASE_MODEL_DIR="${LOCAL_BASE_MODEL_DIR:-${OUTPUT_DIR}/base_model}"
TRAIN_MODEL="$BASE_MODEL"
USE_LOCAL_FILES_ONLY=0

DENSE_CKPT="${CHECKPOINT_DIR}/dense_sft_fp16"
PRUNED_CKPT="${CHECKPOINT_DIR}/nvidia_2_4_sft_fp16"
PRUNE_EVAL_REPORT="${REPORT_DIR}/dense_and_nvidia_2_4_prune_eval_report.json"
SPARSITY_REPORT="${REPORT_DIR}/sparsity_2_4_report.json"
DENSE_ONNX_DIR="${ONNX_DIR}/dense_sft_fp16"
PRUNED_ONNX_DIR="${ONNX_DIR}/nvidia_2_4_sft_fp16"
DENSE_ONNX="${DENSE_ONNX_DIR}/model.onnx"
PRUNED_ONNX="${PRUNED_ONNX_DIR}/model.onnx"
DENSE_ENGINE="${ENGINE_DIR}/dense_sft_fp16_seq${SEQ_LEN}.plan"
PRUNED_ENGINE="${ENGINE_DIR}/nvidia_2_4_sft_fp16_seq${SEQ_LEN}.plan"

mkdir -p \
  "$ENV_DIR" "$CHECKPOINT_DIR" "$EVAL_DIR" "$ONNX_DIR" "$ENGINE_DIR" \
  "$REPORT_DIR" "$LOG_DIR" "$RESULT_DIR" "$HELPER_DIR"

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

normalize_base_model_source() {
  case "$LOCAL_FILES_ONLY" in
    auto)
      if [[ -d "$BASE_MODEL" ]]; then
        USE_LOCAL_FILES_ONLY=1
        TRAIN_MODEL="$BASE_MODEL"
      else
        USE_LOCAL_FILES_ONLY=0
        TRAIN_MODEL="$LOCAL_BASE_MODEL_DIR"
      fi
      ;;
    1|true|TRUE|yes|YES|y|Y|on|ON)
      USE_LOCAL_FILES_ONLY=1
      TRAIN_MODEL="$BASE_MODEL"
      ;;
    0|false|FALSE|no|NO|n|N|off|OFF)
      if [[ -d "$BASE_MODEL" ]]; then
        USE_LOCAL_FILES_ONLY=1
        TRAIN_MODEL="$BASE_MODEL"
      else
        USE_LOCAL_FILES_ONLY=0
        TRAIN_MODEL="$LOCAL_BASE_MODEL_DIR"
      fi
      ;;
    *)
      echo "LOCAL_FILES_ONLY must be auto, 1, or 0." >&2
      exit 2
      ;;
  esac
}

materialize_hf_base_model() {
  if [[ "$USE_LOCAL_FILES_ONLY" -eq 1 ]]; then
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
    return
  fi

  if [[ -d "$TRAIN_MODEL" && -f "${TRAIN_MODEL}/config.json" ]]; then
    echo "Reusing local Hugging Face base model snapshot: $TRAIN_MODEL"
  else
    echo "Downloading Hugging Face base model '${BASE_MODEL_SOURCE}' to: $TRAIN_MODEL"
    BASE_MODEL_SOURCE="$BASE_MODEL_SOURCE" \
    LOCAL_BASE_MODEL_DIR="$TRAIN_MODEL" \
    "$PYTHON" <<'PY'
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

repo_id = os.environ["BASE_MODEL_SOURCE"]
local_dir = Path(os.environ["LOCAL_BASE_MODEL_DIR"]).expanduser()
local_dir.mkdir(parents=True, exist_ok=True)
kwargs = {
    "repo_id": repo_id,
    "local_dir": str(local_dir),
    "token": os.environ.get("HF_TOKEN") or None,
}
try:
    snapshot_download(**kwargs, local_dir_use_symlinks=False)
except TypeError:
    snapshot_download(**kwargs)
print(f"Downloaded {repo_id} to {local_dir}")
PY
  fi

  TRAIN_MODEL="$LOCAL_BASE_MODEL_DIR"
  USE_LOCAL_FILES_ONLY=1
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
}

write_helpers() {
  cat > "${HELPER_DIR}/env_report.py" <<'PY'
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

out_txt = Path(os.environ["ENV_REPORT_TXT"])
out_json = Path(os.environ["ENV_REPORT_JSON"])
out_txt.parent.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str]) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command, "returncode": None, "stdout": "", "stderr": "not found"}
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "available": True,
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def import_version(module: str) -> str | None:
    try:
        imported = importlib.import_module(module)
    except Exception:
        return None
    return str(getattr(imported, "__version__", "unknown"))


report: dict[str, Any] = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "nproc_per_node": os.environ.get("NPROC_PER_NODE"),
    },
    "python": {
        "executable": sys.executable,
        "version": sys.version,
        "platform": platform.platform(),
    },
    "commands": {
        "nvidia_smi": run_command(["nvidia-smi"]),
        "nvcc": run_command(["nvcc", "--version"]),
        "trtexec": run_command(["trtexec", "--version"]),
    },
    "python_modules": {
        "tensorrt": import_version("tensorrt"),
        "onnx": import_version("onnx"),
        "onnxruntime": import_version("onnxruntime"),
    },
}

try:
    import torch

    torch_info: dict[str, Any] = {
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            torch_info["cuda_devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    report["torch"] = torch_info
except Exception as exc:
    report["torch"] = {"error": repr(exc), "cuda_is_available": False, "cuda_device_count": 0, "cuda_devices": []}

try:
    import onnxruntime as ort

    report["onnxruntime"] = {
        "version": ort.__version__,
        "available_providers": ort.get_available_providers(),
    }
except Exception as exc:
    report["onnxruntime"] = {"error": repr(exc), "available_providers": []}

try:
    import tensorrt as trt

    report["tensorrt"] = {"version": trt.__version__}
except Exception as exc:
    report["tensorrt"] = {"error": repr(exc), "version": None}

lines: list[str] = []
lines.append(f"created_at: {report['created_at']}")
lines.append(f"python: {report['python']['version'].splitlines()[0]}")
lines.append(f"CUDA_VISIBLE_DEVICES: {report['environment'].get('cuda_visible_devices')}")
lines.append(f"NVIDIA_VISIBLE_DEVICES: {report['environment'].get('nvidia_visible_devices')}")
lines.append(f"NPROC_PER_NODE: {report['environment'].get('nproc_per_node')}")
lines.append("")
for name, key in (("nvidia-smi", "nvidia_smi"), ("nvcc --version", "nvcc"), ("trtexec --version", "trtexec")):
    command_report = report["commands"][key]
    lines.append(f"## {name}")
    if not command_report["available"]:
        lines.append("not found")
    else:
        if command_report["stdout"]:
            lines.append(command_report["stdout"].rstrip())
        if command_report["stderr"]:
            lines.append(command_report["stderr"].rstrip())
        lines.append(f"returncode: {command_report['returncode']}")
    lines.append("")

torch_info = report["torch"]
lines.append("## torch")
for key in ("version", "cuda_version", "cuda_is_available", "cuda_device_count"):
    lines.append(f"{key}: {torch_info.get(key)}")
for device in torch_info.get("cuda_devices", []):
    lines.append(f"gpu {device['index']}: {device['name']} capability={device['capability']}")
lines.append("")

ort_info = report.get("onnxruntime", {})
lines.append("## onnxruntime")
lines.append(f"version: {ort_info.get('version')}")
lines.append(f"providers: {ort_info.get('available_providers')}")
lines.append("")

trt_info = report.get("tensorrt", {})
lines.append("## tensorrt")
lines.append(f"version: {trt_info.get('version')}")
if trt_info.get("error"):
    lines.append(f"error: {trt_info['error']}")

out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\n".join(lines))

if not report["torch"].get("cuda_is_available"):
    raise SystemExit("CUDA is not available according to torch; aborting H20 GPU benchmark.")
if int(report["torch"].get("cuda_device_count") or 0) <= 0:
    raise SystemExit("No CUDA GPU is visible according to torch; aborting H20 GPU benchmark.")
PY

  cat > "${HELPER_DIR}/split_prune_eval_report.py" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

report_path = Path(os.environ["PRUNE_EVAL_REPORT"])
eval_dir = Path(os.environ["EVAL_DIR"])
report = json.loads(report_path.read_text(encoding="utf-8"))

mapping = [
    ("dense_sft_fp16", "original_before_prune"),
    ("nvidia_2_4_sft_fp16", "pruned_after_50_percent"),
]
dataset_mapping = [
    ("iot200", "benchmark"),
    ("train", "training"),
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


for out_label, section_name in mapping:
    section = report.get(section_name, {})
    evaluations = section.get("evaluations", {})
    for out_dataset, dataset_key in dataset_mapping:
        result = evaluations.get(dataset_key, {})
        outputs = list(result.get("outputs", []))
        metrics = {key: value for key, value in result.items() if key != "outputs"}
        metrics.update(
            {
                "source_report": str(report_path),
                "section": section_name,
                "dataset_key": dataset_key,
                "prediction_count": len(outputs),
            }
        )
        write_jsonl(eval_dir / out_label / f"{out_dataset}_predictions.jsonl", outputs)
        write_json(eval_dir / out_label / f"{out_dataset}_metrics.json", metrics)
print(f"Wrote split predictions and metrics under {eval_dir}")
PY

  cat > "${HELPER_DIR}/verify_nvidia24_sparsity.py" <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

model_dir = Path(os.environ["MODEL_DIR"]).expanduser()
output_json = Path(os.environ["OUTPUT_JSON"]).expanduser()
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}

model = AutoModelForSeq2SeqLM.from_pretrained(
    str(model_dir),
    trust_remote_code=trust_remote_code,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map=None,
)
model.to("cpu")
model.eval()

per_layer: list[dict[str, Any]] = []
total_checked_weights = 0
total_blocks = 0
exact_2_zero_blocks = 0
at_least_2_zero_blocks = 0
total_zero_count = 0
non_compliant_layers: list[dict[str, Any]] = []

for name, module in model.named_modules():
    if not isinstance(module, nn.Linear):
        continue
    weight = module.weight.detach().cpu()
    out_features, in_features = weight.shape
    zero_count = int(weight.eq(0).sum().item())
    layer_total = int(weight.numel())
    layer: dict[str, Any] = {
        "name": name,
        "shape": [int(out_features), int(in_features)],
        "checked": False,
        "weight_count": layer_total,
        "zero_count": zero_count,
        "sparsity_pct": (zero_count / layer_total * 100.0) if layer_total else 0.0,
    }
    if in_features % 4 != 0:
        layer.update(
            {
                "reason": "in_features_not_divisible_by_4",
                "total_blocks": 0,
                "exact_2_zero_blocks": 0,
                "at_least_2_zero_blocks": 0,
                "exact_2_zero_block_pct": 0.0,
                "tensorrt_eligible_block_pct": 0.0,
            }
        )
        non_compliant_layers.append({"name": name, "reason": layer["reason"], "shape": layer["shape"]})
        per_layer.append(layer)
        continue

    grouped = weight.reshape(out_features, in_features // 4, 4)
    zeros_per_block = grouped.eq(0).sum(dim=2)
    blocks = int(zeros_per_block.numel())
    exact = int(zeros_per_block.eq(2).sum().item())
    eligible = int(zeros_per_block.ge(2).sum().item())
    total_checked_weights += layer_total
    total_blocks += blocks
    exact_2_zero_blocks += exact
    at_least_2_zero_blocks += eligible
    total_zero_count += zero_count
    layer.update(
        {
            "checked": True,
            "total_blocks": blocks,
            "exact_2_zero_blocks": exact,
            "at_least_2_zero_blocks": eligible,
            "exact_2_zero_block_pct": (exact / blocks * 100.0) if blocks else 0.0,
            "tensorrt_eligible_block_pct": (eligible / blocks * 100.0) if blocks else 0.0,
        }
    )
    if exact != blocks:
        non_compliant_layers.append(
            {
                "name": name,
                "reason": "not_all_blocks_have_exactly_2_zeros",
                "shape": layer["shape"],
                "exact_2_zero_block_pct": layer["exact_2_zero_block_pct"],
            }
        )
    per_layer.append(layer)

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model_dir": str(model_dir),
    "total_checked_weights": total_checked_weights,
    "total_blocks": total_blocks,
    "exact_2_zero_blocks": exact_2_zero_blocks,
    "at_least_2_zero_blocks": at_least_2_zero_blocks,
    "exact_2_zero_block_pct": (exact_2_zero_blocks / total_blocks * 100.0) if total_blocks else 0.0,
    "tensorrt_eligible_block_pct": (at_least_2_zero_blocks / total_blocks * 100.0) if total_blocks else 0.0,
    "total_zero_count": total_zero_count,
    "total_sparsity_pct": (total_zero_count / total_checked_weights * 100.0) if total_checked_weights else 0.0,
    "per_layer": per_layer,
    "non_compliant_layers": non_compliant_layers,
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote NVIDIA 2:4 sparsity report: {output_json}")
if non_compliant_layers:
    print(f"WARNING: {len(non_compliant_layers)} linear layers are not exact 2:4 compliant.")
PY

  cat > "${HELPER_DIR}/export_inspect_onnx.py" <<'PY'
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    import onnx
    from onnx import TensorProto
except Exception as exc:
    raise SystemExit(f"Missing ONNX package. Install onnx and onnxscript. Original error: {exc}")

model_dir = Path(os.environ["MODEL_DIR"]).expanduser()
out_dir = Path(os.environ["ONNX_OUT_DIR"]).expanduser()
report_json = Path(os.environ["ONNX_REPORT_JSON"]).expanduser()
source_seq_len = int(os.environ["SOURCE_SEQ_LEN"])
target_seq_len = int(os.environ["TARGET_SEQ_LEN"])
force = os.environ.get("FORCE_EXPORT", "0").lower() in {"1", "true", "yes", "on"}
trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}

onnx_path = out_dir / "model.onnx"
if force and out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir),
    trust_remote_code=trust_remote_code,
    local_files_only=True,
    use_fast=False,
)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(model_dir),
    trust_remote_code=trust_remote_code,
    local_files_only=True,
    torch_dtype=dtype,
)
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False
model.to(device)
model.eval()

decoder_start_token_id = getattr(model.config, "decoder_start_token_id", None)
if decoder_start_token_id is None:
    decoder_start_token_id = getattr(tokenizer, "bos_token_id", None)
if decoder_start_token_id is None:
    decoder_start_token_id = getattr(tokenizer, "eos_token_id", None)
if decoder_start_token_id is None:
    decoder_start_token_id = getattr(tokenizer, "pad_token_id", None)
if decoder_start_token_id is None:
    decoder_start_token_id = 0


class LogitsOnlyWrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


wrapper = LogitsOnlyWrapper(model).eval()
input_ids = torch.full((1, source_seq_len), int(tokenizer.pad_token_id or 0), dtype=torch.long, device=device)
attention_mask = torch.ones((1, source_seq_len), dtype=torch.long, device=device)
decoder_input_ids = torch.full((1, target_seq_len), int(decoder_start_token_id), dtype=torch.long, device=device)

if not onnx_path.exists() or force:
    print(f"Exporting logits-only fixed-shape ONNX: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, decoder_input_ids),
            str(onnx_path),
            input_names=["input_ids", "attention_mask", "decoder_input_ids"],
            output_names=["logits"],
            opset_version=17,
            do_constant_folding=True,
            dynamic_axes=None,
        )
else:
    print(f"Reusing existing ONNX: {onnx_path}")

model_onnx = onnx.load(str(onnx_path))
onnx.checker.check_model(model_onnx)

dtype_names = {value: key for key, value in TensorProto.DataType.items()}
initializers = []
initializer_dtypes: dict[str, int] = {}
floating_initializer_count = 0
fp16_initializer_count = 0
for initializer in model_onnx.graph.initializer:
    dtype_name = dtype_names.get(initializer.data_type, str(initializer.data_type))
    initializer_dtypes[dtype_name] = initializer_dtypes.get(dtype_name, 0) + 1
    if initializer.data_type in {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE}:
        floating_initializer_count += 1
        if initializer.data_type == TensorProto.FLOAT16:
            fp16_initializer_count += 1
    initializers.append(
        {
            "name": initializer.name,
            "dtype": dtype_name,
            "shape": [int(dim) for dim in initializer.dims],
        }
    )


def value_info_to_dict(value: Any) -> dict[str, Any]:
    tensor_type = value.type.tensor_type
    shape = []
    dynamic_axes = []
    for axis, dim in enumerate(tensor_type.shape.dim):
        if dim.dim_param:
            shape.append(dim.dim_param)
            dynamic_axes.append({"axis": axis, "name": dim.dim_param})
        elif dim.dim_value:
            shape.append(int(dim.dim_value))
        else:
            shape.append(None)
            dynamic_axes.append({"axis": axis, "name": None})
    return {
        "name": value.name,
        "elem_type": dtype_names.get(tensor_type.elem_type, str(tensor_type.elem_type)),
        "shape": shape,
        "dynamic_axes": dynamic_axes,
    }


external_files = [item for item in out_dir.iterdir() if item.is_file() and item.name != "model.onnx"]
model_size_bytes = onnx_path.stat().st_size + sum(item.stat().st_size for item in external_files)
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "onnx_path": str(onnx_path),
    "onnx_path_exists": onnx_path.exists(),
    "export_kind": "logits-only fixed source/target sequence graph",
    "batch_size": 1,
    "source_seq_len": source_seq_len,
    "target_seq_len": target_seq_len,
    "input_names": [item.name for item in model_onnx.graph.input],
    "output_names": [item.name for item in model_onnx.graph.output],
    "input_shapes": [value_info_to_dict(item) for item in model_onnx.graph.input],
    "output_shapes": [value_info_to_dict(item) for item in model_onnx.graph.output],
    "dynamic_axes": {
        item.name: value_info_to_dict(item)["dynamic_axes"]
        for item in list(model_onnx.graph.input) + list(model_onnx.graph.output)
        if value_info_to_dict(item)["dynamic_axes"]
    },
    "initializer_dtypes": initializer_dtypes,
    "floating_initializer_count": floating_initializer_count,
    "fp16_initializer_count": fp16_initializer_count,
    "weights_are_fp16": floating_initializer_count > 0 and floating_initializer_count == fp16_initializer_count,
    "onnx_model_size_bytes": model_size_bytes,
    "onnx_model_size_mb": model_size_bytes / 1_000_000,
    "initializers_preview": initializers[:100],
}
report_json.parent.mkdir(parents=True, exist_ok=True)
report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote ONNX inspection report: {report_json}")
PY

  cat > "${HELPER_DIR}/trtexec_shape_arg.py" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import onnx

onnx_path = Path(sys.argv[1])
source_seq_len = int(sys.argv[2])
target_seq_len = int(sys.argv[3])
model = onnx.load(str(onnx_path), load_external_data=False)

items = []
for value in model.graph.input:
    name = value.name
    dims = value.type.tensor_type.shape.dim
    if len(dims) != 2:
        continue
    length = target_seq_len if "decoder" in name.lower() else source_seq_len
    items.append(f"{name}:1x{length}")

print(",".join(items))
PY

  cat > "${HELPER_DIR}/parse_trt_sparse_log.py" <<'PY'
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

log_path = Path(os.environ["SPARSE_TRT_LOG"])
output_json = Path(os.environ["OUTPUT_JSON"])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
lines = [line for line in text.splitlines() if re.search(r"spars|tactic|choose|chose", line, re.I)]
eligible = bool(re.search(r"eligible.*sparse|sparse.*eligible", text, re.I))
selected = bool(re.search(r"(chose|choose|select|selected).{0,120}sparse", text, re.I))
if re.search(r"sparse tactics[^0-9]*[1-9]", text, re.I):
    selected = True
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "log_path": str(log_path),
    "sparse_tactics_eligible": eligible,
    "sparse_tactics_selected": selected,
    "matching_log_lines": lines[:500],
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote TensorRT sparse tactics report: {output_json}")
PY

  cat > "${HELPER_DIR}/benchmark_pytorch_generation.py" <<'PY'
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from scenic_prune_eval import extract_prompt_response, read_records  # noqa: E402


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


class MemorySampler:
    def __init__(self) -> None:
        self.samples: list[int] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def sample_once(self) -> int | None:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return max(int(item.strip()) for item in output.splitlines() if item.strip())
        except Exception:
            return None

    def run(self) -> None:
        while not self.stop.is_set():
            value = self.sample_once()
            if value is not None:
                self.samples.append(value)
            time.sleep(0.05)

    def __enter__(self) -> "MemorySampler":
        first = self.sample_once()
        if first is not None:
            self.samples.append(first)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        last = self.sample_once()
        if last is not None:
            self.samples.append(last)

    @property
    def peak_mb(self) -> int | None:
        return max(self.samples) if self.samples else None


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def benchmark(model_dir: Path, model_name: str, sparsity: str) -> dict[str, Any]:
    trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}
    source_seq_len = int(os.environ["SOURCE_SEQ_LEN"])
    target_seq_len = int(os.environ["TARGET_SEQ_LEN"])
    warmup_iters = int(os.environ["WARMUP_ITERS"])
    measure_iters = int(os.environ["MEASURE_ITERS"])
    benchmark_json = Path(os.environ["IOT200_JSONL"]).expanduser()
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=torch.float16,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.to(device)
    model.eval()

    records = read_records(benchmark_json)
    prompts = [extract_prompt_response(record)[0] for record in records]
    if not prompts:
        raise ValueError(f"No prompts found in {benchmark_json}")

    def generate(prompt: str) -> None:
        encoded = tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=source_seq_len,
        )
        encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
        with torch.no_grad():
            model.generate(
                **encoded,
                max_new_tokens=target_seq_len,
                num_beams=1,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=True,
            )

    for index in range(warmup_iters):
        generate(prompts[index % len(prompts)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies: list[float] = []
    with MemorySampler() as memory:
        for index in range(measure_iters):
            torch.cuda.synchronize()
            started = time.perf_counter()
            generate(prompts[index % len(prompts)])
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.mean(latencies) if latencies else None
    median_ms = statistics.median(latencies) if latencies else None
    return {
        "Model": model_name,
        "Architecture": "encoder-decoder",
        "Runtime": "PyTorch",
        "Precision": "FP16",
        "Sparsity": sparsity,
        "Seq. Len.": source_seq_len,
        "Batch Size": 1,
        "Latency": mean_ms,
        "Median Lat.": median_ms,
        "P95 Lat.": percentile(latencies, 0.95),
        "P99 Lat.": percentile(latencies, 0.99),
        "Throughput QPS": (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None,
        "Memory": memory.peak_mb,
        "ONNX MB": None,
        "Engine MB": None,
        "GPU": torch.cuda.get_device_name(0),
        "Provider": "torch.cuda",
        "Sparse Tactics Selected": None,
        "Speedup vs Dense TRT FP16": None,
        "runtime_note": "end-to-end greedy generation",
        "model_size_mb": path_size_bytes(model_dir) / 1_000_000,
    }


rows = [
    benchmark(Path(os.environ["DENSE_CKPT"]), "dense_sft_fp16", "dense"),
    benchmark(Path(os.environ["PRUNED_CKPT"]), "nvidia_2_4_sft_fp16", "NVIDIA 2:4"),
]
output_json = Path(os.environ["OUTPUT_JSON"])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "rows": rows,
    "warmup_iters": int(os.environ["WARMUP_ITERS"]),
    "measure_iters": int(os.environ["MEASURE_ITERS"]),
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote PyTorch latency report: {output_json}")
PY

  cat > "${HELPER_DIR}/benchmark_onnx_logits.py" <<'PY'
from __future__ import annotations

import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import sys

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from scenic_prune_eval import extract_prompt_response, read_records  # noqa: E402


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def benchmark(onnx_path: Path, tokenizer_dir: Path, model_name: str, sparsity: str, provider: str) -> dict[str, Any] | None:
    available = ort.get_available_providers()
    if provider not in available:
        return None
    trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}
    source_seq_len = int(os.environ["SOURCE_SEQ_LEN"])
    target_seq_len = int(os.environ["TARGET_SEQ_LEN"])
    warmup_iters = int(os.environ["WARMUP_ITERS"])
    measure_iters = int(os.environ["MEASURE_ITERS"])
    benchmark_json = Path(os.environ["IOT200_JSONL"]).expanduser()

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    provider_options: list[dict[str, str]] | None = None
    if provider == "TensorrtExecutionProvider":
        cache_dir = Path(os.environ["ORT_TRT_CACHE_ROOT"]) / model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        provider_options = [
            {
                "trt_fp16_enable": "1",
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": str(cache_dir),
                "trt_sparsity_enable": "1" if "2_4" in model_name else "0",
            }
        ]
    session = ort.InferenceSession(str(onnx_path), providers=[provider], provider_options=provider_options)

    records = read_records(benchmark_json)
    prompts = [extract_prompt_response(record)[0] for record in records]
    if not prompts:
        raise ValueError(f"No prompts found in {benchmark_json}")

    def make_inputs(prompt: str) -> dict[str, np.ndarray]:
        encoded = tokenizer(
            [prompt],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=source_seq_len,
        )
        return {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "decoder_input_ids": np.full((1, target_seq_len), int(decoder_start_token_id), dtype=np.int64),
        }

    input_names = {item.name for item in session.get_inputs()}
    prepared = [
        {key: value for key, value in make_inputs(prompts[index % len(prompts)]).items() if key in input_names}
        for index in range(max(warmup_iters, measure_iters))
    ]

    for index in range(warmup_iters):
        session.run(None, prepared[index % len(prepared)])

    latencies: list[float] = []
    for index in range(measure_iters):
        started = time.perf_counter()
        session.run(None, prepared[index % len(prepared)])
        latencies.append((time.perf_counter() - started) * 1000.0)

    mean_ms = statistics.mean(latencies) if latencies else None
    return {
        "Model": model_name,
        "Architecture": "encoder-decoder",
        "Runtime": "ONNX Runtime TensorRT" if provider == "TensorrtExecutionProvider" else "ONNX Runtime CUDA",
        "Precision": "FP16",
        "Sparsity": sparsity,
        "Seq. Len.": source_seq_len,
        "Batch Size": 1,
        "Latency": mean_ms,
        "Median Lat.": statistics.median(latencies) if latencies else None,
        "P95 Lat.": percentile(latencies, 0.95),
        "P99 Lat.": percentile(latencies, 0.99),
        "Throughput QPS": (1000.0 / mean_ms) if mean_ms and mean_ms > 0 else None,
        "Memory": None,
        "ONNX MB": path_size_bytes(onnx_path) / 1_000_000,
        "Engine MB": None,
        "GPU": None,
        "Provider": provider,
        "Sparse Tactics Selected": None,
        "Speedup vs Dense TRT FP16": None,
        "runtime_note": "logits-only fixed-shape graph, not autoregressive generation",
    }


rows: list[dict[str, Any]] = []
for provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
    dense = benchmark(Path(os.environ["DENSE_ONNX"]), Path(os.environ["DENSE_CKPT"]), "dense_sft_fp16", "dense", provider)
    pruned = benchmark(
        Path(os.environ["PRUNED_ONNX"]),
        Path(os.environ["PRUNED_CKPT"]),
        "nvidia_2_4_sft_fp16",
        "NVIDIA 2:4",
        provider,
    )
    if dense is not None:
        rows.append(dense)
    if pruned is not None:
        rows.append(pruned)

output_json = Path(os.environ["OUTPUT_JSON"])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "available_providers": ort.get_available_providers(),
    "rows": rows,
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote ONNX Runtime latency report: {output_json}")
PY

  cat > "${HELPER_DIR}/compare_onnx_logits.py" <<'PY'
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import sys

PROJECT_ROOT = Path.cwd()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from scenic_prune_eval import extract_prompt_response, read_records  # noqa: E402


def run_one(model_dir: Path, onnx_path: Path, label: str, provider: str) -> dict[str, Any]:
    available = ort.get_available_providers()
    if provider not in available:
        return {
            "label": label,
            "provider": provider,
            "available": False,
            "reason": f"{provider} is unavailable in ONNX Runtime providers: {available}",
        }
    trust_remote_code = os.environ.get("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes", "on"}
    source_seq_len = int(os.environ["SOURCE_SEQ_LEN"])
    target_seq_len = int(os.environ["TARGET_SEQ_LEN"])
    records = read_records(Path(os.environ["IOT200_JSONL"]).expanduser())[: int(os.environ.get("LOGIT_COMPARE_LIMIT", "200"))]
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder_start_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id or tokenizer.pad_token_id or 0

    torch_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        trust_remote_code=trust_remote_code,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("cuda:0")
    torch_model.eval()
    if hasattr(torch_model.config, "use_cache"):
        torch_model.config.use_cache = False

    provider_options: list[dict[str, str]] | None = None
    if provider == "TensorrtExecutionProvider":
        cache_dir = Path(os.environ["ORT_TRT_CACHE_ROOT"]) / f"{label}_compare"
        cache_dir.mkdir(parents=True, exist_ok=True)
        provider_options = [
            {
                "trt_fp16_enable": "1",
                "trt_engine_cache_enable": "1",
                "trt_engine_cache_path": str(cache_dir),
                "trt_sparsity_enable": "1" if "nvidia" in label else "0",
            }
        ]
    session = ort.InferenceSession(str(onnx_path), providers=[provider], provider_options=provider_options)
    input_names = {item.name for item in session.get_inputs()}

    total = 0
    top1_agree = 0
    top5_agree = 0
    max_abs_diff = 0.0
    for record in records:
        prompt, _ = extract_prompt_response(record)
        encoded_np = tokenizer(
            [prompt],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=source_seq_len,
        )
        decoder_np = np.full((1, target_seq_len), int(decoder_start_token_id), dtype=np.int64)
        ort_inputs = {
            "input_ids": encoded_np["input_ids"].astype(np.int64),
            "attention_mask": encoded_np["attention_mask"].astype(np.int64),
            "decoder_input_ids": decoder_np,
        }
        ort_inputs = {key: value for key, value in ort_inputs.items() if key in input_names}
        ort_logits = session.run(None, ort_inputs)[0]

        torch_inputs = {
            "input_ids": torch.from_numpy(encoded_np["input_ids"].astype(np.int64)).to("cuda:0"),
            "attention_mask": torch.from_numpy(encoded_np["attention_mask"].astype(np.int64)).to("cuda:0"),
            "decoder_input_ids": torch.from_numpy(decoder_np).to("cuda:0"),
        }
        with torch.no_grad():
            torch_logits = torch_model(**torch_inputs, use_cache=False, return_dict=True).logits.detach().float().cpu().numpy()

        max_abs_diff = max(max_abs_diff, float(np.max(np.abs(torch_logits - ort_logits))))
        torch_last = torch_logits[0, -1]
        ort_last = ort_logits[0, -1]
        torch_top5 = set(np.argsort(torch_last)[-5:].tolist())
        ort_top5 = set(np.argsort(ort_last)[-5:].tolist())
        top1_agree += int(int(np.argmax(torch_last)) == int(np.argmax(ort_last)))
        top5_agree += int(bool(torch_top5 & ort_top5))
        total += 1

    return {
        "label": label,
        "provider": provider,
        "available": True,
        "total": total,
        "top1_agreement": top1_agree / total if total else 0.0,
        "top5_overlap_agreement": top5_agree / total if total else 0.0,
        "max_abs_diff": max_abs_diff,
        "limitation": "Compares logits from the fixed-shape ONNX graph, not TensorRT autoregressive generation.",
    }


provider = "TensorrtExecutionProvider" if "TensorrtExecutionProvider" in ort.get_available_providers() else "CUDAExecutionProvider"
rows = [
    run_one(Path(os.environ["DENSE_CKPT"]), Path(os.environ["DENSE_ONNX"]), "dense_sft_fp16", provider),
    run_one(Path(os.environ["PRUNED_CKPT"]), Path(os.environ["PRUNED_ONNX"]), "nvidia_2_4_sft_fp16", provider),
]
output_json = Path(os.environ["OUTPUT_JSON"])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "provider_used": provider,
    "available_providers": ort.get_available_providers(),
    "rows": rows,
}
output_json.parent.mkdir(parents=True, exist_ok=True)
output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote ONNX/TensorRT logits consistency report: {output_json}")
PY

  cat > "${HELPER_DIR}/compose_final_results.py" <<'PY'
from __future__ import annotations

import csv
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

output_dir = Path(os.environ["OUTPUT_DIR"])
env_json = output_dir / "env" / "env_report.json"
result_dir = output_dir / "results"
report_dir = output_dir / "reports"
engine_dir = output_dir / "engines"
log_dir = output_dir / "logs"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_size / 1_000_000
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / 1_000_000


def metric_value(model_label: str, dataset: str, key: str) -> Any:
    path = output_dir / "eval" / model_label / f"{dataset}_metrics.json"
    payload = read_json(path, {})
    return payload.get(key)


def parse_times_json(path: Path) -> dict[str, float | None]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    values: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float)) and re.search(r"(latency|time|compute|duration)", str(key), re.I):
                    number = float(item)
                    if 0.0 < number < 1_000_000.0:
                        values.append(number)
                else:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(payload)
    if not values:
        return {}
    values.sort()
    mean = sum(values) / len(values)

    def pct(p: float) -> float:
        index = (len(values) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return values[int(index)]
        return values[lower] * (upper - index) + values[upper] * (index - lower)

    return {
        "Latency": mean,
        "Median Lat.": pct(0.5),
        "P95 Lat.": pct(0.95),
        "P99 Lat.": pct(0.99),
        "Throughput QPS": 1000.0 / mean if mean > 0 else None,
    }


def parse_trtexec_log(path: Path) -> dict[str, float | None]:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    out: dict[str, float | None] = {}
    mean_match = re.search(r"mean\s*=\s*([0-9.]+)\s*ms", text, re.I)
    median_match = re.search(r"median\s*=\s*([0-9.]+)\s*ms", text, re.I)
    p95_match = re.search(r"percentile\(95%?\)\s*=\s*([0-9.]+)\s*ms", text, re.I)
    p99_match = re.search(r"percentile\(99%?\)\s*=\s*([0-9.]+)\s*ms", text, re.I)
    qps_match = re.search(r"Throughput:\s*([0-9.]+)\s*qps", text, re.I)
    if mean_match:
        out["Latency"] = float(mean_match.group(1))
    if median_match:
        out["Median Lat."] = float(median_match.group(1))
    if p95_match:
        out["P95 Lat."] = float(p95_match.group(1))
    if p99_match:
        out["P99 Lat."] = float(p99_match.group(1))
    if qps_match:
        out["Throughput QPS"] = float(qps_match.group(1))
    elif out.get("Latency"):
        out["Throughput QPS"] = 1000.0 / float(out["Latency"])
    return out


def base_row(model: str, runtime: str, precision: str, sparsity: str) -> dict[str, Any]:
    return {
        "Model": model,
        "Architecture": "encoder-decoder",
        "Runtime": runtime,
        "Precision": precision,
        "Sparsity": sparsity,
        "Seq. Len.": int(os.environ["SEQ_LEN"]),
        "Batch Size": 1,
        "Latency": None,
        "Median Lat.": None,
        "P95 Lat.": None,
        "P99 Lat.": None,
        "Throughput QPS": None,
        "Memory": None,
        "ONNX MB": None,
        "Engine MB": None,
        "EM@1 IoT200": metric_value(model, "iot200", "em1"),
        "EM@5 IoT200": metric_value(model, "iot200", "em5"),
        "EM@1 Train": metric_value(model, "train", "em1"),
        "EM@5 Train": metric_value(model, "train", "em5"),
        "GPU": None,
        "Provider": None,
        "Sparse Tactics Selected": None,
        "Speedup vs Dense TRT FP16": None,
    }


rows: list[dict[str, Any]] = []
env = read_json(env_json, {})
gpu_name = None
try:
    gpu_name = env["torch"]["cuda_devices"][0]["name"]
except Exception:
    gpu_name = None

for latency_path in (report_dir / "pytorch_latency_report.json", report_dir / "onnxruntime_latency_report.json"):
    payload = read_json(latency_path, {})
    for row in payload.get("rows", []):
        merged = base_row(row["Model"], row["Runtime"], row["Precision"], row["Sparsity"])
        merged.update(row)
        merged["GPU"] = row.get("GPU") or gpu_name
        rows.append(merged)

sparse_tactics = read_json(report_dir / "trt_sparse_tactics_report.json", {})
trt_specs = [
    (
        "dense_sft_fp16",
        "dense",
        engine_dir / f"dense_sft_fp16_seq{os.environ['SEQ_LEN']}.plan",
        log_dir / "build_dense_fp16.log",
        report_dir / "dense_trt_times.json",
        False,
    ),
    (
        "nvidia_2_4_sft_fp16",
        "NVIDIA 2:4",
        engine_dir / f"nvidia_2_4_sft_fp16_seq{os.environ['SEQ_LEN']}.plan",
        log_dir / "build_nvidia_2_4_sparse_fp16.log",
        report_dir / "nvidia_2_4_trt_times.json",
        bool(sparse_tactics.get("sparse_tactics_selected")),
    ),
]
for model, sparsity, engine, log, times_json, sparse_selected in trt_specs:
    if not engine.exists():
        continue
    row = base_row(model, "TensorRT native", "FP16", sparsity)
    parsed = parse_times_json(times_json)
    if not parsed:
        parsed = parse_trtexec_log(log)
    row.update(parsed)
    row["GPU"] = gpu_name
    row["Provider"] = "trtexec"
    row["Engine MB"] = size_mb(engine)
    onnx_report = report_dir / (
        "onnx_inspection_dense_sft_fp16.json" if model == "dense_sft_fp16" else "onnx_inspection_nvidia_2_4_sft_fp16.json"
    )
    row["ONNX MB"] = read_json(onnx_report, {}).get("onnx_model_size_mb")
    row["Sparse Tactics Selected"] = sparse_selected if model == "nvidia_2_4_sft_fp16" else False
    rows.append(row)

dense_trt = next((row for row in rows if row["Model"] == "dense_sft_fp16" and row["Runtime"] == "TensorRT native"), None)
sparse_trt = next((row for row in rows if row["Model"] == "nvidia_2_4_sft_fp16" and row["Runtime"] == "TensorRT native"), None)
main_speedup = None
throughput_gain = None
if dense_trt and sparse_trt and dense_trt.get("Latency") and sparse_trt.get("Latency"):
    main_speedup = float(dense_trt["Latency"]) / float(sparse_trt["Latency"])
    throughput_gain = (
        float(sparse_trt["Throughput QPS"]) / float(dense_trt["Throughput QPS"])
        if sparse_trt.get("Throughput QPS") and dense_trt.get("Throughput QPS")
        else None
    )
    sparse_trt["Speedup vs Dense TRT FP16"] = main_speedup

int8_status = read_json(report_dir / "int8_status.json", {})
logit_consistency = read_json(report_dir / "trt_logits_consistency.json", {})
final = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "metric_contract": {
        "main_speedup": "dense FP16 native TensorRT mean latency / NVIDIA 2:4 FP16 native TensorRT mean latency",
        "throughput_gain": "NVIDIA 2:4 FP16 native TensorRT QPS / dense FP16 native TensorRT QPS",
        "cpu_onnx_excluded": True,
    },
    "main_speedup_vs_dense_trt_fp16": main_speedup,
    "throughput_gain_vs_dense_trt_fp16": throughput_gain,
    "rows": rows,
    "environment": env,
    "int8_status": int8_status,
    "sparse_tactics": sparse_tactics,
    "logit_consistency": logit_consistency,
}

result_dir.mkdir(parents=True, exist_ok=True)
(result_dir / "final_metrics.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

columns = [
    "Model",
    "Architecture",
    "Runtime",
    "Precision",
    "Sparsity",
    "Seq. Len.",
    "Batch Size",
    "Latency",
    "Median Lat.",
    "P95 Lat.",
    "P99 Lat.",
    "Throughput QPS",
    "Memory",
    "ONNX MB",
    "Engine MB",
    "EM@1 IoT200",
    "EM@5 IoT200",
    "EM@1 Train",
    "EM@5 Train",
    "GPU",
    "Provider",
    "Sparse Tactics Selected",
    "Speedup vs Dense TRT FP16",
]
with (result_dir / "final_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
print(f"Wrote final metrics JSON/CSV under {result_dir}")
PY

  cat > "${HELPER_DIR}/write_summary.py" <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

output_dir = Path(os.environ["OUTPUT_DIR"])


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


env = read_json(output_dir / "env" / "env_report.json", {})
final = read_json(output_dir / "results" / "final_metrics.json", {})
int8 = read_json(output_dir / "reports" / "int8_status.json", {})
sparse = read_json(output_dir / "reports" / "trt_sparse_tactics_report.json", {})

torch_info = env.get("torch", {})
ort_info = env.get("onnxruntime", {})
providers = ort_info.get("available_providers", [])
gpu_ran = bool(torch_info.get("cuda_is_available")) and int(torch_info.get("cuda_device_count") or 0) > 0
cpu_onnx_fallback = providers == ["CPUExecutionProvider"] or providers == ["AzureExecutionProvider", "CPUExecutionProvider"]
speedup = final.get("main_speedup_vs_dense_trt_fp16")
rows = final.get("rows", [])


def metric(model: str, key: str) -> Any:
    for row in rows:
        if row.get("Model") == model:
            return row.get(key)
    return None


lines = [
    "# H20 Encoder-Decoder SFT + NVIDIA 2:4 + TensorRT Summary",
    "",
    f"- Benchmark ran on GPU: {gpu_ran}",
    f"- Visible CUDA GPUs: {torch_info.get('cuda_device_count')}",
    f"- GPU 0: {(torch_info.get('cuda_devices') or [{}])[0].get('name') if torch_info.get('cuda_devices') else None}",
    f"- ONNX Runtime providers: {providers}",
    f"- CPU ONNX fallback happened: {cpu_onnx_fallback}",
    f"- TensorRT sparse tactics selected: {sparse.get('sparse_tactics_selected')}",
    f"- NVIDIA 2:4 real TensorRT speedup: {speedup}",
    f"- INT8 status: {int8.get('status')}",
    "",
    "## Accuracy",
    "",
    f"- Dense SFT FP16 IoT200 EM@1/EM@5: {metric('dense_sft_fp16', 'EM@1 IoT200')} / {metric('dense_sft_fp16', 'EM@5 IoT200')}",
    f"- Dense SFT FP16 train EM@1/EM@5: {metric('dense_sft_fp16', 'EM@1 Train')} / {metric('dense_sft_fp16', 'EM@5 Train')}",
    f"- NVIDIA 2:4 FP16 IoT200 EM@1/EM@5: {metric('nvidia_2_4_sft_fp16', 'EM@1 IoT200')} / {metric('nvidia_2_4_sft_fp16', 'EM@5 IoT200')}",
    f"- NVIDIA 2:4 FP16 train EM@1/EM@5: {metric('nvidia_2_4_sft_fp16', 'EM@1 Train')} / {metric('nvidia_2_4_sft_fp16', 'EM@5 Train')}",
    "",
    "## Limitations",
    "",
    "- The ONNX export is a fixed-shape logits-only encoder-decoder graph with batch size 1, source length 64, and target length 64.",
    "- Native TensorRT benchmarking uses that logits-only graph, so the TensorRT latency row is not full autoregressive generation latency.",
    "- PyTorch latency rows are end-to-end greedy generation and are therefore not directly equivalent to logits-only TensorRT rows.",
    "- CPU ONNX rows are excluded from final speedup calculations.",
    "- INT8 is skipped unless a real Q/DQ export or calibration path is added; random INT8 ranges are not used.",
    "",
    "## Files",
    "",
    f"- Final metrics JSON: {output_dir / 'results' / 'final_metrics.json'}",
    f"- Final metrics CSV: {output_dir / 'results' / 'final_metrics.csv'}",
    f"- Sparse report: {output_dir / 'reports' / 'sparsity_2_4_report.json'}",
]
(output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote summary: {output_dir / 'SUMMARY.md'}")
PY
}

run_env_report() {
  echo "== Environment verification =="
  ENV_REPORT_TXT="${ENV_DIR}/env_report.txt" \
  ENV_REPORT_JSON="${ENV_DIR}/env_report.json" \
  "$PYTHON" "${HELPER_DIR}/env_report.py"

  "$PYTHON" - "${ENV_DIR}/env_report.json" <<'PY'
import json
import sys

path = sys.argv[1]
payload = json.load(open(path, encoding="utf-8"))
providers = payload.get("onnxruntime", {}).get("available_providers", [])
if providers == ["CPUExecutionProvider"] or providers == ["AzureExecutionProvider", "CPUExecutionProvider"]:
    print("WARNING: ONNX Runtime only reports CPUExecutionProvider. CPU ONNX will not be used for speedup claims.")
if "TensorrtExecutionProvider" not in providers:
    print("ONNX Runtime TensorRT provider is unavailable; native trtexec will be used when available.")
PY
}

run_training() {
  if truthy "$SKIP_TRAIN" && [[ -d "$DENSE_CKPT" ]]; then
    echo "== Reusing dense SFT checkpoint: $DENSE_CKPT =="
    return
  fi

  echo "== Training dense SFT FP16 checkpoint =="
  TRAIN_ARGS=(
    scripts/scenic_train_chatlm_sft.py
    --mode regular
    --model "$TRAIN_MODEL"
    --train-json "$TRAIN_JSONL"
    --output-dir "$DENSE_CKPT"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --gradient-accumulation-steps "$TRAIN_GRADIENT_ACCUMULATION_STEPS"
    --learning-rate "$TRAIN_LEARNING_RATE"
    --weight-decay "$TRAIN_WEIGHT_DECAY"
    --warmup-ratio "$TRAIN_WARMUP_RATIO"
    --max-source-length "$SOURCE_SEQ_LEN"
    --max-target-length "$TARGET_SEQ_LEN"
    --num-workers "$TRAIN_NUM_WORKERS"
    --ddp-timeout-minutes "$DDP_TIMEOUT_MINUTES"
    --no-epoch-checkpoints
    --final-save-on-cpu
    --safe-serialization
    --fp16
  )
  if [[ "$USE_LOCAL_FILES_ONLY" -eq 1 ]]; then
    TRAIN_ARGS+=(--local-files-only)
    export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
  fi

  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/train_dense_sft_fp16.log"
}

run_prune_eval() {
  if truthy "$SKIP_PRUNE_EVAL" && [[ -d "$PRUNED_CKPT" && -f "$PRUNE_EVAL_REPORT" ]]; then
    echo "== Reusing prune/eval report and pruned checkpoint =="
  else
    echo "== Evaluating dense checkpoint, applying NVIDIA 2:4 pruning, and evaluating pruned checkpoint =="
    PRUNE_ARGS=(
      scripts/scenic_prune_eval.py
      --model "$DENSE_CKPT"
      --method nvidia
      --sparsity 0.5
      --sparsity-basis targeted-linear
      --prune-scope all-linear
      --pruned-output-dir "$PRUNED_CKPT"
      --train-json "$TRAIN_JSONL"
      --benchmark-json "$IOT200_JSONL"
      --output-json "$PRUNE_EVAL_REPORT"
      --eval-batch-size "$EVAL_BATCH_SIZE"
      --max-input-len "$SOURCE_SEQ_LEN"
      --max-target-len "$TARGET_SEQ_LEN"
      --max-new-tokens "$TARGET_SEQ_LEN"
      --num-beams 5
      --num-return-sequences 5
      --local-files-only
      --trust-remote-code
      --no-bf16
      --fp16
      --include-predictions
    )
    if truthy "$PRUNE_LM_HEAD"; then
      PRUNE_ARGS+=(--prune-lm-head)
    fi
    if truthy "$IGNORE_SPACES"; then
      PRUNE_ARGS+=(--ignore-spaces)
    fi
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "${PRUNE_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/prune_eval_nvidia_2_4.log"
  fi

  PRUNE_EVAL_REPORT="$PRUNE_EVAL_REPORT" \
  EVAL_DIR="$EVAL_DIR" \
  "$PYTHON" "${HELPER_DIR}/split_prune_eval_report.py"

  MODEL_DIR="$PRUNED_CKPT" \
  OUTPUT_JSON="$SPARSITY_REPORT" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  "$PYTHON" "${HELPER_DIR}/verify_nvidia24_sparsity.py"
}

run_onnx_export() {
  if truthy "$SKIP_ONNX"; then
    echo "== SKIP_ONNX=1; skipping ONNX export =="
    return
  fi

  echo "== Exporting dense FP16 ONNX =="
  MODEL_DIR="$DENSE_CKPT" \
  ONNX_OUT_DIR="$DENSE_ONNX_DIR" \
  ONNX_REPORT_JSON="${REPORT_DIR}/onnx_inspection_dense_sft_fp16.json" \
  SOURCE_SEQ_LEN="$SOURCE_SEQ_LEN" \
  TARGET_SEQ_LEN="$TARGET_SEQ_LEN" \
  FORCE_EXPORT="$FORCE_EXPORT" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  "$PYTHON" "${HELPER_DIR}/export_inspect_onnx.py" 2>&1 | tee "${LOG_DIR}/export_dense_onnx.log"

  echo "== Exporting NVIDIA 2:4 FP16 ONNX =="
  MODEL_DIR="$PRUNED_CKPT" \
  ONNX_OUT_DIR="$PRUNED_ONNX_DIR" \
  ONNX_REPORT_JSON="${REPORT_DIR}/onnx_inspection_nvidia_2_4_sft_fp16.json" \
  SOURCE_SEQ_LEN="$SOURCE_SEQ_LEN" \
  TARGET_SEQ_LEN="$TARGET_SEQ_LEN" \
  FORCE_EXPORT="$FORCE_EXPORT" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  "$PYTHON" "${HELPER_DIR}/export_inspect_onnx.py" 2>&1 | tee "${LOG_DIR}/export_nvidia_2_4_onnx.log"
}

write_int8_status() {
  "$PYTHON" - "${REPORT_DIR}/int8_status.json" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "skipped",
    "reason": (
        "No repo-native Q/DQ ONNX export, calibration cache, or TensorRT calibration dataloader "
        "is available in this script. INT8 engines and INT8 EM claims are intentionally skipped."
    ),
    "random_int8_ranges_used": False,
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote INT8 status: {path}")
PY
}

run_tensorrt_builds() {
  if truthy "$SKIP_TRT"; then
    echo "== SKIP_TRT=1; skipping native TensorRT engine build =="
    return
  fi
  if ! command -v trtexec >/dev/null 2>&1; then
    echo "trtexec is unavailable; native TensorRT engine build is skipped." | tee "${LOG_DIR}/trtexec_unavailable.log"
    "$PYTHON" - "${REPORT_DIR}/trt_sparse_tactics_report.json" <<'PY'
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "sparse_tactics_eligible": False,
    "sparse_tactics_selected": False,
    "reason": "trtexec not found",
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
    return
  fi

  local dense_shapes
  local sparse_shapes
  dense_shapes="$("$PYTHON" "${HELPER_DIR}/trtexec_shape_arg.py" "$DENSE_ONNX" "$SOURCE_SEQ_LEN" "$TARGET_SEQ_LEN")"
  sparse_shapes="$("$PYTHON" "${HELPER_DIR}/trtexec_shape_arg.py" "$PRUNED_ONNX" "$SOURCE_SEQ_LEN" "$TARGET_SEQ_LEN")"

  echo "== Building and benchmarking dense FP16 TensorRT engine =="
  if [[ ! -f "$DENSE_ENGINE" || "$FORCE_TRT" == "1" ]]; then
    trtexec \
      --onnx="$DENSE_ONNX" \
      --saveEngine="$DENSE_ENGINE" \
      --fp16 \
      --sparsity=disable \
      --shapes="$dense_shapes" \
      --iterations="$MEASURE_ITERS" \
      --warmUp="$WARMUP_ITERS" \
      --useCudaGraph \
      --noDataTransfers \
      --exportTimes="${REPORT_DIR}/dense_trt_times.json" \
      --separateProfileRun \
      2>&1 | tee "${LOG_DIR}/build_dense_fp16.log"
  else
    echo "Reusing existing dense TensorRT engine: $DENSE_ENGINE"
  fi

  echo "== Building and benchmarking NVIDIA 2:4 sparse FP16 TensorRT engine =="
  if [[ ! -f "$PRUNED_ENGINE" || "$FORCE_TRT" == "1" ]]; then
    trtexec \
      --onnx="$PRUNED_ONNX" \
      --saveEngine="$PRUNED_ENGINE" \
      --fp16 \
      --sparsity=enable \
      --verbose \
      --profilingVerbosity=detailed \
      --shapes="$sparse_shapes" \
      --iterations="$MEASURE_ITERS" \
      --warmUp="$WARMUP_ITERS" \
      --useCudaGraph \
      --noDataTransfers \
      --exportTimes="${REPORT_DIR}/nvidia_2_4_trt_times.json" \
      --separateProfileRun \
      2>&1 | tee "${LOG_DIR}/build_nvidia_2_4_sparse_fp16.log"
  else
    echo "Reusing existing NVIDIA 2:4 TensorRT engine: $PRUNED_ENGINE"
  fi

  SPARSE_TRT_LOG="${LOG_DIR}/build_nvidia_2_4_sparse_fp16.log" \
  OUTPUT_JSON="${REPORT_DIR}/trt_sparse_tactics_report.json" \
  "$PYTHON" "${HELPER_DIR}/parse_trt_sparse_log.py"
}

run_latency_reports() {
  if truthy "$SKIP_LATENCY"; then
    echo "== SKIP_LATENCY=1; skipping PyTorch/ONNX latency helpers =="
    return
  fi

  echo "== Measuring PyTorch FP16 generation latency on one GPU =="
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  DENSE_CKPT="$DENSE_CKPT" \
  PRUNED_CKPT="$PRUNED_CKPT" \
  IOT200_JSONL="$IOT200_JSONL" \
  SOURCE_SEQ_LEN="$SOURCE_SEQ_LEN" \
  TARGET_SEQ_LEN="$TARGET_SEQ_LEN" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  MEASURE_ITERS="$MEASURE_ITERS" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  OUTPUT_JSON="${REPORT_DIR}/pytorch_latency_report.json" \
  "$PYTHON" "${HELPER_DIR}/benchmark_pytorch_generation.py" 2>&1 | tee "${LOG_DIR}/benchmark_pytorch_generation.log"

  echo "== Measuring ONNX Runtime logits latency when GPU providers are available =="
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  DENSE_ONNX="$DENSE_ONNX" \
  PRUNED_ONNX="$PRUNED_ONNX" \
  DENSE_CKPT="$DENSE_CKPT" \
  PRUNED_CKPT="$PRUNED_CKPT" \
  IOT200_JSONL="$IOT200_JSONL" \
  SOURCE_SEQ_LEN="$SOURCE_SEQ_LEN" \
  TARGET_SEQ_LEN="$TARGET_SEQ_LEN" \
  WARMUP_ITERS="$WARMUP_ITERS" \
  MEASURE_ITERS="$MEASURE_ITERS" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  ORT_TRT_CACHE_ROOT="${OUTPUT_DIR}/ort_trt_cache" \
  OUTPUT_JSON="${REPORT_DIR}/onnxruntime_latency_report.json" \
  "$PYTHON" "${HELPER_DIR}/benchmark_onnx_logits.py" 2>&1 | tee "${LOG_DIR}/benchmark_onnxruntime_logits.log"

  echo "== Comparing ONNX/TensorRT logits against PyTorch when possible =="
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
  DENSE_ONNX="$DENSE_ONNX" \
  PRUNED_ONNX="$PRUNED_ONNX" \
  DENSE_CKPT="$DENSE_CKPT" \
  PRUNED_CKPT="$PRUNED_CKPT" \
  IOT200_JSONL="$IOT200_JSONL" \
  SOURCE_SEQ_LEN="$SOURCE_SEQ_LEN" \
  TARGET_SEQ_LEN="$TARGET_SEQ_LEN" \
  TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
  ORT_TRT_CACHE_ROOT="${OUTPUT_DIR}/ort_trt_cache" \
  OUTPUT_JSON="${REPORT_DIR}/trt_logits_consistency.json" \
  "$PYTHON" "${HELPER_DIR}/compare_onnx_logits.py" 2>&1 | tee "${LOG_DIR}/compare_onnx_logits.log"
}

compose_results() {
  OUTPUT_DIR="$OUTPUT_DIR" \
  SEQ_LEN="$SEQ_LEN" \
  "$PYTHON" "${HELPER_DIR}/compose_final_results.py"

  OUTPUT_DIR="$OUTPUT_DIR" \
  "$PYTHON" "${HELPER_DIR}/write_summary.py"
}

echo "Run configuration:"
normalize_base_model_source
echo "  base_model_source: $BASE_MODEL_SOURCE"
echo "  training_model: $TRAIN_MODEL"
echo "  local_files_only_mode: $LOCAL_FILES_ONLY"
echo "  train_jsonl: $TRAIN_JSONL"
echo "  iot200_jsonl: $IOT200_JSONL"
echo "  output_dir: $OUTPUT_DIR"
echo "  gpus: $GPUS"
echo "  nproc_per_node: $NPROC_PER_NODE"
echo "  epochs: $EPOCHS"
echo "  source_seq_len: $SOURCE_SEQ_LEN"
echo "  target_seq_len: $TARGET_SEQ_LEN"
echo "  batch_size: $BATCH_SIZE"
echo "  eval_batch_size: $EVAL_BATCH_SIZE"
echo "  warmup_iters: $WARMUP_ITERS"
echo "  measure_iters: $MEASURE_ITERS"

write_helpers
run_env_report
materialize_hf_base_model
run_training
run_prune_eval
run_onnx_export
write_int8_status
run_tensorrt_builds
run_latency_reports
compose_results

echo "Done."
echo "Shell script: scripts/run_h20_encoder_decoder_sft_prune_trt24.sh"
echo "Summary: ${OUTPUT_DIR}/SUMMARY.md"
echo "Final metrics JSON: ${OUTPUT_DIR}/results/final_metrics.json"
echo "Final metrics CSV: ${OUTPUT_DIR}/results/final_metrics.csv"
echo
echo "Example command:"
echo "  bash scripts/run_h20_encoder_decoder_sft_prune_trt24.sh \\"
echo "    --base_model charent/ChatLM-mini-Chinese"
