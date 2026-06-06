#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

np: Any = None
onnx: Any = None
ort: Any = None
psutil: Any = None


INSTALL_HINT = (
    "Install benchmark dependencies with: "
    "python -m pip install onnx onnxruntime onnxconverter-common psutil numpy pandas tabulate"
)

SCENIC_PROMPT_RESPONSE_AVAILABLE = False
try:
    from scenic_prune_eval import (  # type: ignore
        DEFAULT_BENCHMARK_JSON,
        DEFAULT_TRAIN_JSON,
        extract_prompt_response,
        read_records,
    )

    SCENIC_PROMPT_RESPONSE_AVAILABLE = True
except Exception:
    DEFAULT_BENCHMARK_JSON = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200.json"
    DEFAULT_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"

    def read_records(path: Path) -> list[dict[str, Any]]:  # type: ignore
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
        if isinstance(value, dict):
            for key in ("records", "data", "examples", "items"):
                if isinstance(value.get(key), list):
                    return [row for row in value[key] if isinstance(row, dict)]
            return [value]
        raise ValueError(f"{path} must contain JSON objects.")

    def extract_prompt_response(record: dict[str, Any]) -> tuple[str, str]:  # type: ignore
        prompt = str(record.get("prompt") or record.get("instruction") or record.get("input") or "").strip()
        response = str(record.get("response") or record.get("output") or record.get("answer") or "").strip()
        if not prompt or not response:
            raise ValueError("Record is missing prompt/response fields.")
        return prompt, response


@dataclass
class InputSpec:
    name: str
    elem_type: int
    elem_type_name: str
    shape: list[int | str | None]


@dataclass
class InputBatch:
    values: dict[str, Any]
    source: str


@dataclass
class InputSource:
    batches: list[InputBatch]
    source_kind: str
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def batch(self, index: int) -> dict[str, Any]:
        if not self.batches:
            raise RuntimeError("InputSource has no input batches.")
        return self.batches[index % len(self.batches)].values


@dataclass
class ModelVariant:
    precision: str
    path: Path
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PowerResult:
    avg_power_w: float | None
    energy_per_inference_mj: float | None
    note: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def eprint(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def require_imports() -> None:
    global np, onnx, ort, psutil
    missing: list[str] = []
    try:
        import numpy as numpy_mod

        np = numpy_mod
    except Exception:
        missing.append("numpy")
    try:
        import onnx as onnx_mod

        onnx = onnx_mod
    except Exception:
        missing.append("onnx")
    try:
        import onnxruntime as ort_mod

        ort = ort_mod
    except Exception:
        missing.append("onnxruntime or onnxruntime-gpu")
    try:
        import psutil as psutil_mod

        psutil = psutil_mod
    except Exception:
        psutil = None
        eprint("Warning: psutil is not installed; peak host RSS will be reported as N/A. " + INSTALL_HINT)

    if missing:
        raise SystemExit(f"Missing required package(s): {', '.join(missing)}. {INSTALL_HINT}")


def warn_optional_packages(args: argparse.Namespace) -> None:
    if args.fp16_onnx is None and not args.skip_fp16_conversion:
        try:
            import onnxconverter_common.float16  # noqa: F401
        except Exception:
            eprint(
                "Warning: onnxconverter-common is not installed; automatic FP16 conversion will be skipped. "
                + INSTALL_HINT
            )
    try:
        import pandas  # noqa: F401
    except Exception:
        if args.power_log:
            eprint("Warning: pandas is not installed; power logs will be read with the stdlib CSV parser.")
    try:
        import tabulate  # noqa: F401
    except Exception:
        eprint("Warning: tabulate is not installed; Markdown/LaTeX tables will use the built-in formatter.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FP32, FP16, and INT8 ONNX Runtime inference for SCENIC-style ONNX models "
            "and write paper-ready precision/provider comparison tables."
        )
    )
    parser.add_argument("--fp32-onnx", required=True, help="Path to the FP32 ONNX model.")
    parser.add_argument("--fp16-onnx", default=None, help="Path to an existing FP16 ONNX model.")
    parser.add_argument("--int8-onnx", default=None, help="Path to an existing INT8 ONNX model.")
    parser.add_argument("--output-dir", required=True, help="Directory for tables, metadata, profiles, and summary.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers to try, in separate benchmark rows.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--skip-fp16-conversion", action="store_true")
    parser.add_argument("--skip-int8-quantization", action="store_true")
    parser.add_argument("--quantization-mode", choices=("static", "dynamic"), default="static")
    parser.add_argument("--quant-format", choices=("qdq", "qoperator"), default="qdq")
    parser.add_argument("--power-log", default=None, help="CSV with timestamp and power columns.")
    parser.add_argument("--power-column", default="power_w")
    parser.add_argument("--timestamp-column", default="timestamp_s")
    parser.add_argument("--device-name", default=None, help="Human-readable hardware/device name.")
    parser.add_argument(
        "--input-shape",
        action="append",
        default=[],
        help=(
            "Fallback shape override. Repeat as needed. Use 'name:1,256' or '1,256' "
            "for the first ONNX input."
        ),
    )
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--disable-iobinding", action="store_true")
    parser.add_argument("--profile-ort", action="store_true")
    parser.add_argument(
        "--table-formats",
        nargs="+",
        choices=("csv", "markdown", "latex"),
        default=["csv", "markdown", "latex"],
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="SCENIC prompt/response JSON for drift and benchmark inputs. Defaults to generated benchmark data.",
    )
    parser.add_argument(
        "--calibration-json",
        default=None,
        help="SCENIC prompt/response JSON for static INT8 calibration. Defaults to training data when available.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Tokenizer directory for real SCENIC prompt preprocessing. If omitted, the script searches near --fp32-onnx.",
    )
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=128)
    parser.add_argument("--drift-samples", type=int, default=16)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "item"


def short_error(exc: Exception, limit: int = 240) -> str:
    text = repr(exc)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_shape_items(items: list[str]) -> tuple[dict[str, list[int]], list[int] | None]:
    named: dict[str, list[int]] = {}
    first_shape: list[int] | None = None
    for item in items:
        if ":" in item:
            name, shape_text = item.split(":", 1)
            named[name.strip()] = parse_shape(shape_text)
        else:
            first_shape = parse_shape(item)
    return named, first_shape


def parse_shape(shape_text: str) -> list[int]:
    dims: list[int] = []
    for token in shape_text.replace("x", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token in {"?", "-1", "None", "none", "dynamic"}:
            dims.append(-1)
        else:
            dims.append(int(token))
    if not dims:
        raise ValueError(f"Empty input shape override: {shape_text!r}")
    return dims


def inspect_onnx_inputs(path: Path) -> list[InputSpec]:
    model = onnx.load(str(path), load_external_data=False)
    dtype_names = {value: key for key, value in onnx.TensorProto.DataType.items()}
    initializer_names = {item.name for item in model.graph.initializer}
    specs: list[InputSpec] = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        tensor_type = value.type.tensor_type
        shape: list[int | str | None] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(int(dim.dim_value))
            else:
                shape.append(None)
        specs.append(
            InputSpec(
                name=value.name,
                elem_type=int(tensor_type.elem_type),
                elem_type_name=dtype_names.get(int(tensor_type.elem_type), str(int(tensor_type.elem_type))),
                shape=shape,
            )
        )
    return specs


def inspect_onnx_summary(path: Path) -> dict[str, Any]:
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    dtype_names = {value: key for key, value in onnx.TensorProto.DataType.items()}
    initializer_dtypes: dict[str, int] = {}
    node_op_types: dict[str, int] = {}
    for initializer in model.graph.initializer:
        dtype_name = dtype_names.get(initializer.data_type, str(initializer.data_type))
        initializer_dtypes[dtype_name] = initializer_dtypes.get(dtype_name, 0) + 1
    for node in model.graph.node:
        node_op_types[node.op_type] = node_op_types.get(node.op_type, 0) + 1
    return {
        "onnx_path": str(path),
        "onnx_file_size_bytes": path.stat().st_size,
        "onnx_file_size_mb": path.stat().st_size / 1_000_000,
        "input_names": [item.name for item in model.graph.input],
        "output_names": [item.name for item in model.graph.output],
        "initializer_dtypes": initializer_dtypes,
        "node_op_types": node_op_types,
        "opset_imports": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
    }


def numpy_dtype_for_elem_type(elem_type: int) -> Any:
    tensor = onnx.TensorProto
    mapping = {
        tensor.FLOAT: np.float32,
        tensor.UINT8: np.uint8,
        tensor.INT8: np.int8,
        tensor.UINT16: np.uint16,
        tensor.INT16: np.int16,
        tensor.INT32: np.int32,
        tensor.INT64: np.int64,
        tensor.BOOL: np.bool_,
        tensor.FLOAT16: np.float16,
        tensor.DOUBLE: np.float64,
        tensor.UINT32: np.uint32,
        tensor.UINT64: np.uint64,
    }
    return mapping.get(elem_type, np.float32)


def resolve_input_shapes(
    input_specs: list[InputSpec],
    args: argparse.Namespace,
    notes: list[str],
) -> dict[str, list[int]]:
    named_shapes, first_shape = parse_shape_items(args.input_shape)
    shapes: dict[str, list[int]] = {}
    for index, spec in enumerate(input_specs):
        if spec.name in named_shapes:
            shapes[spec.name] = [args.batch_size if dim == -1 else int(dim) for dim in named_shapes[spec.name]]
            continue
        if index == 0 and first_shape is not None:
            shapes[spec.name] = [args.batch_size if dim == -1 else int(dim) for dim in first_shape]
            continue

        resolved: list[int] = []
        for axis, dim in enumerate(spec.shape):
            if isinstance(dim, int) and dim > 0:
                if axis == 0 and dim != args.batch_size:
                    notes.append(
                        f"{spec.name} has fixed batch dimension {dim}; requested --batch-size {args.batch_size} "
                        "cannot be applied for this input."
                    )
                resolved.append(dim)
                continue
            if axis == 0:
                resolved.append(max(1, args.batch_size))
            elif "decoder" in spec.name.lower():
                resolved.append(max(1, args.max_target_len))
                notes.append(f"{spec.name} axis {axis} is dynamic; using --max-target-len {args.max_target_len}.")
            elif any(key in spec.name.lower() for key in ("input", "attention", "token")):
                resolved.append(max(1, args.max_input_len))
                notes.append(f"{spec.name} axis {axis} is dynamic; using --max-input-len {args.max_input_len}.")
            else:
                resolved.append(1)
                notes.append(f"{spec.name} axis {axis} is dynamic; using fallback dimension 1.")
        if not resolved:
            resolved = [max(1, args.batch_size)]
            notes.append(f"{spec.name} has scalar/unknown shape metadata; using fallback shape {resolved}.")
        shapes[spec.name] = resolved
    return shapes


def dummy_array_for(spec: InputSpec, shape: list[int], rng: Any) -> Any:
    dtype = numpy_dtype_for_elem_type(spec.elem_type)
    lower_name = spec.name.lower()
    if np.issubdtype(dtype, np.floating):
        return rng.normal(loc=0.0, scale=0.02, size=shape).astype(dtype)
    if dtype == np.bool_:
        return np.ones(shape, dtype=dtype)
    if "mask" in lower_name:
        return np.ones(shape, dtype=dtype)
    if "segment" in lower_name or "token_type" in lower_name:
        return np.zeros(shape, dtype=dtype)
    # Keep token-like values small and non-negative to avoid indexing surprises.
    return rng.integers(low=0, high=64, size=shape, endpoint=False).astype(dtype)


def make_dummy_input_source(
    input_specs: list[InputSpec],
    shapes: dict[str, list[int]],
    args: argparse.Namespace,
    reason: str,
) -> InputSource:
    rng = np.random.default_rng(20240606)
    batch_count = max(1, args.calibration_samples, args.drift_samples)
    batches: list[InputBatch] = []
    for _ in range(batch_count):
        values = {spec.name: dummy_array_for(spec, shapes[spec.name], rng) for spec in input_specs}
        batches.append(InputBatch(values=values, source="dummy"))
    return InputSource(
        batches=batches,
        source_kind="dummy_inputs",
        notes=[
            reason,
            "Accuracy is reported as N/A_dummy_inputs because deterministic metadata-shaped inputs were used.",
        ],
        metadata={"batch_count": batch_count, "input_shapes": shapes},
    )


def find_tokenizer_dir(fp32_onnx: Path, args: argparse.Namespace) -> Path | None:
    if args.tokenizer_dir:
        path = Path(args.tokenizer_dir).expanduser()
        return path if path.exists() else None
    candidates = [
        fp32_onnx.parent,
        fp32_onnx.parent.parent,
        fp32_onnx.parent.parent / "checkpoint",
        fp32_onnx.parent.parent / "checkpoints",
    ]
    tokenizer_files = {"tokenizer_config.json", "tokenizer.json", "spiece.model", "sentencepiece.bpe.model"}
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        if any((candidate / name).exists() for name in tokenizer_files):
            return candidate
    return None


def load_tokenizer(tokenizer_dir: Path, notes: list[str]) -> Any | None:
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        notes.append(f"transformers is unavailable; cannot tokenize SCENIC records: {short_error(exc)}")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_dir),
            trust_remote_code=True,
            local_files_only=True,
            use_fast=False,
        )
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        notes.append(f"Failed to load tokenizer from {tokenizer_dir}: {short_error(exc)}")
        return None


def decoder_start_token_id(tokenizer: Any) -> int:
    for attr in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0


def cast_tokens(values: Any, spec: InputSpec) -> Any:
    dtype = numpy_dtype_for_elem_type(spec.elem_type)
    return values.astype(dtype, copy=False)


def make_scenic_input_source(
    input_specs: list[InputSpec],
    shapes: dict[str, list[int]],
    args: argparse.Namespace,
    fp32_onnx: Path,
    records_json: Path,
    role: str,
) -> InputSource | None:
    notes: list[str] = []
    tokenizer_dir = find_tokenizer_dir(fp32_onnx, args)
    if tokenizer_dir is None:
        return None
    tokenizer = load_tokenizer(tokenizer_dir, notes)
    if tokenizer is None:
        return None

    if not records_json.exists():
        notes.append(f"{role} JSON not found: {records_json}")
        return None
    try:
        records = read_records(records_json)
    except Exception as exc:
        notes.append(f"Failed to read SCENIC {role} JSON {records_json}: {short_error(exc)}")
        return None
    if not records:
        notes.append(f"{role} JSON contains no records: {records_json}")
        return None

    source_len = args.max_input_len
    target_len = args.max_target_len
    for name, shape in shapes.items():
        lower = name.lower()
        if len(shape) >= 2 and "decoder" in lower:
            target_len = int(shape[1])
        elif len(shape) >= 2 and any(key in lower for key in ("input", "attention")):
            source_len = int(shape[1])

    main_batch = max(1, next(iter(shapes.values()))[0])
    batch_count = max(1, args.calibration_samples, args.drift_samples)
    start_id = decoder_start_token_id(tokenizer)
    batches: list[InputBatch] = []
    for batch_index in range(batch_count):
        prompts: list[str] = []
        for offset in range(main_batch):
            record = records[(batch_index * main_batch + offset) % len(records)]
            prompt, _ = extract_prompt_response(record)
            prompts.append(prompt)
        encoded = tokenizer(
            prompts,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=source_len,
        )
        values: dict[str, Any] = {}
        for spec in input_specs:
            lower = spec.name.lower()
            shape = shapes[spec.name]
            if spec.name in encoded:
                values[spec.name] = cast_tokens(encoded[spec.name], spec)
            elif "input_ids" == lower or lower.endswith("input_ids"):
                values[spec.name] = cast_tokens(encoded["input_ids"], spec)
            elif "attention_mask" == lower or lower.endswith("attention_mask"):
                values[spec.name] = cast_tokens(encoded["attention_mask"], spec)
            elif "decoder_input_ids" == lower or lower.endswith("decoder_input_ids"):
                values[spec.name] = np.full(shape, start_id, dtype=numpy_dtype_for_elem_type(spec.elem_type))
            elif "decoder_attention_mask" == lower or lower.endswith("decoder_attention_mask"):
                values[spec.name] = np.ones(shape, dtype=numpy_dtype_for_elem_type(spec.elem_type))
            elif "token_type" in lower:
                values[spec.name] = np.zeros(shape, dtype=numpy_dtype_for_elem_type(spec.elem_type))
            else:
                values[spec.name] = dummy_array_for(spec, shape, np.random.default_rng(20240606 + batch_index))
                notes.append(f"{spec.name} is not a standard tokenizer input; filled with deterministic fallback values.")
        batches.append(InputBatch(values=values, source=str(records_json)))

    notes.insert(0, f"Using SCENIC {role} records from {records_json} with tokenizer {tokenizer_dir}.")
    if not SCENIC_PROMPT_RESPONSE_AVAILABLE:
        notes.append("Using local prompt/response fallback parser because scenic_prune_eval helpers were unavailable.")
    return InputSource(
        batches=batches,
        source_kind="scenic_records",
        notes=notes,
        metadata={
            "role": role,
            "records_json": str(records_json),
            "tokenizer_dir": str(tokenizer_dir),
            "record_count": len(records),
            "batch_count": batch_count,
            "source_len": source_len,
            "target_len": target_len,
            "input_shapes": shapes,
        },
    )


def resolve_records_json(args: argparse.Namespace, role: str) -> Path:
    if role == "calibration":
        if args.calibration_json:
            return Path(args.calibration_json).expanduser()
        if Path(DEFAULT_TRAIN_JSON).exists():
            return Path(DEFAULT_TRAIN_JSON)
        return Path(DEFAULT_BENCHMARK_JSON)
    if args.benchmark_json:
        return Path(args.benchmark_json).expanduser()
    return Path(DEFAULT_BENCHMARK_JSON)


def build_input_source(
    fp32_onnx: Path,
    input_specs: list[InputSpec],
    args: argparse.Namespace,
    role: str,
) -> InputSource:
    notes: list[str] = []
    shapes = resolve_input_shapes(input_specs, args, notes)
    records_json = resolve_records_json(args, role)
    source = make_scenic_input_source(input_specs, shapes, args, fp32_onnx, records_json, role)
    if source is not None:
        source.notes = notes + source.notes
        return source
    return make_dummy_input_source(
        input_specs,
        shapes,
        args,
        f"No usable SCENIC tokenizer/dataset preprocessing path was found for {role} inputs.",
    )


def convert_fp16(fp32_path: Path, output_dir: Path, failures: list[dict[str, Any]]) -> ModelVariant | None:
    output_path = output_dir / "model_fp16.onnx"
    notes: list[str] = []
    try:
        from onnxconverter_common.float16 import convert_float_to_float16
    except Exception as exc:
        failures.append(
            {
                "precision": "FP16",
                "stage": "conversion",
                "reason": f"onnxconverter-common unavailable: {short_error(exc)}",
            }
        )
        return None

    try:
        model = onnx.load(str(fp32_path))
        try:
            converted = convert_float_to_float16(model, keep_io_types=True)
            notes.append("FP16 conversion used keep_io_types=True.")
        except Exception as exc:
            notes.append(f"keep_io_types=True failed; retried with keep_io_types=False: {short_error(exc)}")
            converted = convert_float_to_float16(model, keep_io_types=False)
        onnx.checker.check_model(converted)
        onnx.save_model(converted, str(output_path))
        summary = inspect_onnx_summary(output_path)
        fp32_initializers = summary["initializer_dtypes"].get("FLOAT", 0)
        if fp32_initializers:
            notes.append(
                f"{fp32_initializers} FLOAT initializers remain after conversion; these may be blocked/unsupported ops."
            )
        return ModelVariant(precision="FP16", path=output_path, notes=notes, metadata=summary)
    except Exception as exc:
        failures.append({"precision": "FP16", "stage": "conversion", "reason": short_error(exc)})
        return None


class StaticCalibrationReaderBase:
    def __init__(self, batches: list[dict[str, Any]], limit: int) -> None:
        self.batches = batches
        self.limit = max(1, limit)
        self.index = 0

    def get_next(self) -> dict[str, Any] | None:
        if self.index >= self.limit:
            return None
        batch = self.batches[self.index % len(self.batches)]
        self.index += 1
        return batch

    def rewind(self) -> None:
        self.index = 0


def quantize_int8(
    fp32_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    input_source: InputSource,
    failures: list[dict[str, Any]],
) -> ModelVariant | None:
    output_path = output_dir / "model_int8.onnx"
    notes: list[str] = []
    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static
        from onnxruntime.quantization.calibrate import CalibrationDataReader
    except Exception as exc:
        failures.append(
            {
                "precision": "INT8",
                "stage": "quantization",
                "reason": f"ONNX Runtime quantization support unavailable: {short_error(exc)}",
            }
        )
        return None

    quant_metadata: dict[str, Any] = {
        "quantization_mode": args.quantization_mode,
        "quant_format": args.quant_format,
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "calibration_samples_requested": args.calibration_samples,
        "calibration_input_source": input_source.source_kind,
    }
    try:
        if args.quantization_mode == "dynamic":
            quantize_dynamic(
                model_input=str(fp32_path),
                model_output=str(output_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            notes.append("INT8 dynamic quantization used QInt8 weights with per_channel=True.")
        else:
            source_for_quant = fp32_path
            preprocess_path = output_dir / "model_fp32_quant_preprocess.onnx"
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process

                quant_pre_process(str(fp32_path), str(preprocess_path))
                source_for_quant = preprocess_path
                notes.append(f"Ran quant_pre_process before static quantization: {preprocess_path}.")
            except Exception as exc:
                notes.append(f"quant_pre_process unavailable or failed; quantizing original FP32 model: {short_error(exc)}")
            if input_source.source_kind == "dummy_inputs":
                notes.append(
                    "Static INT8 calibration used deterministic dummy inputs; do not treat INT8 accuracy/scale quality "
                    "as paper-grade without --tokenizer-dir and --calibration-json/--benchmark-json."
                )

            class NumpyCalibrationDataReader(StaticCalibrationReaderBase, CalibrationDataReader):  # type: ignore[misc]
                pass

            calibration_batches = [input_source.batch(i) for i in range(max(1, args.calibration_samples))]
            quant_format = QuantFormat.QDQ if args.quant_format == "qdq" else QuantFormat.QOperator
            quantize_static(
                model_input=str(source_for_quant),
                model_output=str(output_path),
                calibration_data_reader=NumpyCalibrationDataReader(calibration_batches, args.calibration_samples),
                quant_format=quant_format,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False,
            )
            notes.append(
                f"INT8 static quantization used {args.calibration_samples} calibration batches "
                f"from {input_source.source_kind}."
            )
        model = onnx.load(str(output_path), load_external_data=False)
        onnx.checker.check_model(model)
        summary = inspect_onnx_summary(output_path)
        return ModelVariant(precision="INT8", path=output_path, notes=notes, metadata={**summary, **quant_metadata})
    except Exception as exc:
        failures.append({"precision": "INT8", "stage": "quantization", "reason": short_error(exc)})
        return None


class HostMemorySampler:
    def __init__(self, interval_s: float = 0.01) -> None:
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: Any | None = None

    def __enter__(self) -> "HostMemorySampler":
        if psutil is None:
            return self
        self._process = psutil.Process(os.getpid())
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._sample()

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            self.samples.append(int(self._process.memory_info().rss))
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)

    @property
    def peak_mb(self) -> float | None:
        if not self.samples:
            return None
        return max(self.samples) / 1_000_000


class DeviceMemorySampler:
    def __init__(self, provider: str, interval_s: float = 0.02) -> None:
        self.provider = provider
        self.interval_s = interval_s
        self.samples_mb: list[float] = []
        self.note = "N/A"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml: Any | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "DeviceMemorySampler":
        if "CUDA" not in self.provider and "TensorRT" not in self.provider:
            self.note = "device memory sampler only attempted for CUDA/TensorRT providers"
            return self
        try:
            import pynvml

            self._nvml = pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(index) for index in range(count)]
            self.note = "pynvml total device memory used; not process-isolated"
            self._sample()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        except Exception as exc:
            self.note = f"pynvml unavailable or failed: {short_error(exc)}"
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._sample()
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _sample(self) -> None:
        if self._nvml is None:
            return
        try:
            used = [self._nvml.nvmlDeviceGetMemoryInfo(handle).used / 1_000_000 for handle in self._handles]
            if used:
                self.samples_mb.append(max(used))
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval_s)

    @property
    def peak_mb(self) -> float | None:
        return max(self.samples_mb) if self.samples_mb else None


class PowerLog:
    def __init__(self, path: Path, timestamp_column: str, power_column: str) -> None:
        self.path = path
        self.timestamp_column = timestamp_column
        self.power_column = power_column
        self.rows = self._read_rows()

    def _read_rows(self) -> list[tuple[float, float]]:
        rows: list[tuple[float, float]] = []
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if self.timestamp_column not in (reader.fieldnames or []):
                raise ValueError(f"Power log missing timestamp column {self.timestamp_column!r}.")
            if self.power_column not in (reader.fieldnames or []):
                raise ValueError(f"Power log missing power column {self.power_column!r}.")
            for row in reader:
                try:
                    timestamp = float(row[self.timestamp_column])
                    power = float(row[self.power_column])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(timestamp) and math.isfinite(power):
                    rows.append((timestamp, power))
        rows.sort()
        if len(rows) < 2:
            raise ValueError("Power log needs at least two valid timestamp/power rows.")
        return rows

    def integrate(self, start_wall_s: float, end_wall_s: float, inferences: int) -> PowerResult:
        if inferences <= 0:
            return PowerResult(None, None, "energy unavailable: zero benchmark inferences")
        duration = max(0.0, end_wall_s - start_wall_s)
        if duration <= 0:
            return PowerResult(None, None, "energy unavailable: zero benchmark duration")
        timestamps = [row[0] for row in self.rows]
        if min(timestamps) <= start_wall_s and max(timestamps) >= end_wall_s:
            window_start = start_wall_s
            window_end = end_wall_s
            mode_note = "absolute wall-clock timestamps"
        elif min(timestamps) <= 0.0 and max(timestamps) >= duration:
            window_start = 0.0
            window_end = duration
            mode_note = "relative timestamps aligned to benchmark start"
        else:
            return PowerResult(
                None,
                None,
                "energy unavailable: power log timestamps do not cover the benchmark window",
            )

        points: list[tuple[float, float]] = []
        for t in (window_start, window_end):
            power = self._interpolate(t)
            if power is None:
                return PowerResult(None, None, "energy unavailable: could not interpolate power window")
            points.append((t, power))
        for timestamp, power in self.rows:
            if window_start < timestamp < window_end:
                points.append((timestamp, power))
        points.sort()
        energy_j = 0.0
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            energy_j += ((p0 + p1) / 2.0) * (t1 - t0)
        avg_power = energy_j / (window_end - window_start)
        return PowerResult(avg_power, (energy_j / inferences) * 1000.0, f"integrated from {mode_note}")

    def _interpolate(self, timestamp: float) -> float | None:
        timestamps = [row[0] for row in self.rows]
        index = bisect_left(timestamps, timestamp)
        if index < len(self.rows) and self.rows[index][0] == timestamp:
            return self.rows[index][1]
        if index == 0 or index >= len(self.rows):
            return None
        t0, p0 = self.rows[index - 1]
        t1, p1 = self.rows[index]
        if t1 == t0:
            return p0
        alpha = (timestamp - t0) / (t1 - t0)
        return p0 + alpha * (p1 - p0)


class SessionRunner:
    def __init__(self, session: Any, provider: str, use_iobinding: bool) -> None:
        self.session = session
        self.provider = provider
        self.use_iobinding = use_iobinding
        self.notes: list[str] = []

    def run(self, inputs: dict[str, Any]) -> list[Any]:
        if not self.use_iobinding:
            return self.session.run(None, inputs)
        try:
            binding = self.session.io_binding()
            for name, value in inputs.items():
                binding.bind_cpu_input(name, np.ascontiguousarray(value))
            for output in self.session.get_outputs():
                binding.bind_output(output.name)
            self.session.run_with_iobinding(binding)
            return binding.copy_outputs_to_cpu()
        except Exception as exc:
            self.use_iobinding = False
            self.notes.append(f"I/O binding failed and regular session.run was used: {short_error(exc)}")
            return self.session.run(None, inputs)


def sync_device(provider: str) -> None:
    if "CUDA" not in provider and "TensorRT" not in provider:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def make_session(
    model_path: Path,
    precision: str,
    provider: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Any, list[str]]:
    options = ort.SessionOptions()
    if args.num_threads is not None:
        options.intra_op_num_threads = int(args.num_threads)
        options.inter_op_num_threads = int(args.num_threads)
    if args.profile_ort:
        options.enable_profiling = True
        options.profile_file_prefix = str(output_dir / f"ort_profile_{precision.lower()}_{sanitize_name(provider)}")

    available = set(ort.get_available_providers())
    if provider not in available:
        raise RuntimeError(
            f"Requested provider {provider!r} is not available. Available providers: {sorted(available)}"
        )
    providers = [provider]
    notes: list[str] = []
    if provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
        notes.append("CPUExecutionProvider registered as fallback; provider profiling can reveal op-level fallback.")

    session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
    actual = session.get_providers()
    if provider not in actual:
        notes.append(f"Provider fallback detected: requested {provider}, session providers are {actual}.")
    return session, notes


def flatten_numeric(outputs: list[Any]) -> Any | None:
    chunks: list[Any] = []
    for output in outputs:
        array = np.asarray(output)
        if not np.issubdtype(array.dtype, np.number):
            continue
        chunks.append(array.astype(np.float64, copy=False).ravel())
    if not chunks:
        return None
    return np.concatenate(chunks)


def compare_outputs(reference_outputs: list[list[Any]], candidate_outputs: list[list[Any]]) -> dict[str, Any]:
    abs_sums = 0.0
    count = 0
    max_abs = 0.0
    dot = 0.0
    ref_norm_sq = 0.0
    cand_norm_sq = 0.0
    shape_mismatches = 0
    for reference, candidate in zip(reference_outputs, candidate_outputs):
        ref_flat = flatten_numeric(reference)
        cand_flat = flatten_numeric(candidate)
        if ref_flat is None or cand_flat is None:
            continue
        if ref_flat.shape != cand_flat.shape:
            shape_mismatches += 1
            limit = min(ref_flat.size, cand_flat.size)
            ref_flat = ref_flat[:limit]
            cand_flat = cand_flat[:limit]
        diff = np.abs(ref_flat - cand_flat)
        abs_sums += float(diff.sum())
        count += int(diff.size)
        max_abs = max(max_abs, float(diff.max()) if diff.size else 0.0)
        dot += float(np.dot(ref_flat, cand_flat))
        ref_norm_sq += float(np.dot(ref_flat, ref_flat))
        cand_norm_sq += float(np.dot(cand_flat, cand_flat))
    if count == 0:
        return {
            "mean_abs_error": None,
            "max_abs_error": None,
            "cosine_similarity": None,
            "shape_mismatches": shape_mismatches,
        }
    cosine = None
    if ref_norm_sq > 0 and cand_norm_sq > 0:
        cosine = dot / math.sqrt(ref_norm_sq * cand_norm_sq)
    return {
        "mean_abs_error": abs_sums / count,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine,
        "shape_mismatches": shape_mismatches,
    }


def drift_reference_outputs(session: Any, input_source: InputSource, sample_count: int) -> list[list[Any]]:
    return [session.run(None, input_source.batch(index)) for index in range(max(1, sample_count))]


def benchmark_session(
    variant: ModelVariant,
    provider: str,
    session: Any,
    input_source: InputSource,
    args: argparse.Namespace,
    power_log: PowerLog | None,
    hardware: dict[str, Any],
    session_notes: list[str],
    reference_outputs: list[list[Any]] | None,
) -> dict[str, Any]:
    use_iobinding = provider != "CPUExecutionProvider" and not args.disable_iobinding
    runner = SessionRunner(session, provider, use_iobinding)
    for index in range(max(0, args.warmup)):
        runner.run(input_source.batch(index))
    sync_device(provider)

    latencies_ms: list[float] = []
    start_wall_s = time.time()
    start_perf_ns = time.perf_counter_ns()
    with HostMemorySampler() as host_memory, DeviceMemorySampler(provider) as device_memory:
        for index in range(max(1, args.runs)):
            inputs = input_source.batch(index)
            sync_device(provider)
            started = time.perf_counter_ns()
            runner.run(inputs)
            sync_device(provider)
            elapsed_ns = time.perf_counter_ns() - started
            latencies_ms.append(elapsed_ns / 1_000_000)
    end_perf_ns = time.perf_counter_ns()
    end_wall_s = time.time()

    profile_path = None
    profile_provider_counts: dict[str, int] = {}
    if args.profile_ort:
        try:
            profile_path = session.end_profiling()
            profile_provider_counts = parse_profile_provider_counts(Path(profile_path))
        except Exception as exc:
            runner.notes.append(f"Failed to finalize or parse ORT profile: {short_error(exc)}")

    actual_batch = effective_batch_size(input_source.batch(0))
    total_samples = max(1, args.runs) * actual_batch
    elapsed_s = (end_perf_ns - start_perf_ns) / 1_000_000_000
    power_result = (
        power_log.integrate(start_wall_s, end_wall_s, total_samples)
        if power_log is not None
        else PowerResult(None, None, "energy unavailable: no --power-log supplied")
    )

    drift: dict[str, Any]
    if variant.precision == "FP32":
        drift = {
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "cosine_similarity": 1.0,
            "shape_mismatches": 0,
        }
    elif reference_outputs is None:
        drift = {
            "mean_abs_error": None,
            "max_abs_error": None,
            "cosine_similarity": None,
            "shape_mismatches": 0,
        }
    else:
        candidate_outputs = drift_reference_outputs(session, input_source, len(reference_outputs))
        drift = compare_outputs(reference_outputs, candidate_outputs)

    notes = list(variant.notes) + list(session_notes) + list(runner.notes)
    if input_source.source_kind == "dummy_inputs":
        notes.append("Accuracy metric is N/A_dummy_inputs; only numerical drift on dummy inputs is available.")
    else:
        notes.append("Raw ONNX tensor benchmark reports output drift; run the SCENIC generation eval pipeline for EM/F1-style task metrics.")
    if profile_provider_counts.get("CPUExecutionProvider", 0) > 0 and provider != "CPUExecutionProvider":
        notes.append(f"Provider fallback detected in ORT profile: {profile_provider_counts}.")
    if power_result.avg_power_w is None:
        notes.append(power_result.note)
    if device_memory.peak_mb is None:
        notes.append(f"device memory N/A: {device_memory.note}")

    session_providers = session.get_providers()
    row = {
        "precision": variant.precision,
        "onnx_model_path": str(variant.path),
        "onnx_file_size_mb": variant.path.stat().st_size / 1_000_000,
        "execution_provider": provider,
        "execution_provider_used": " + ".join(session_providers),
        "hardware_device": hardware.get("device_name") or hardware.get("platform") or "unknown",
        "hardware_class": hardware.get("hardware_class", "unknown"),
        "batch_size": actual_batch,
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "mean_latency_ms": statistics.fmean(latencies_ms),
        "median_latency_ms": statistics.median(latencies_ms),
        "p90_latency_ms": percentile(latencies_ms, 0.90),
        "p95_latency_ms": percentile(latencies_ms, 0.95),
        "throughput_samples_per_sec": total_samples / elapsed_s if elapsed_s > 0 else None,
        "peak_host_memory_rss_mb": host_memory.peak_mb,
        "device_memory_mb": device_memory.peak_mb,
        "average_power_w": power_result.avg_power_w,
        "energy_per_inference_mj": power_result.energy_per_inference_mj,
        "mean_abs_error_vs_fp32": drift["mean_abs_error"],
        "max_abs_error_vs_fp32": drift["max_abs_error"],
        "cosine_similarity_vs_fp32": drift["cosine_similarity"],
        "shape_mismatches_vs_fp32": drift["shape_mismatches"],
        "accuracy_or_drift_metric_vs_fp32": drift_metric_text(input_source, variant.precision, drift),
        "speedup_vs_fp32": None,
        "size_reduction_vs_fp32": None,
        "session_providers": session_providers,
        "profile_path": profile_path,
        "profile_provider_counts": profile_provider_counts,
        "notes": " | ".join(deduplicate_notes(notes)),
    }
    return row


def parse_profile_provider_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    counts: dict[str, int] = {}
    if not isinstance(events, list):
        return counts
    for event in events:
        if not isinstance(event, dict):
            continue
        args = event.get("args")
        if not isinstance(args, dict):
            continue
        provider = args.get("provider")
        if isinstance(provider, str) and provider:
            counts[provider] = counts.get(provider, 0) + 1
    return counts


def effective_batch_size(inputs: dict[str, Any]) -> int:
    for value in inputs.values():
        array = np.asarray(value)
        if array.ndim > 0:
            return int(array.shape[0])
    return 1


def drift_metric_text(input_source: InputSource, precision: str, drift: dict[str, Any]) -> str:
    if precision == "FP32":
        return "FP32 baseline"
    if drift.get("mean_abs_error") is None:
        return "N/A"
    prefix = "N/A_dummy_inputs; " if input_source.source_kind == "dummy_inputs" else "output drift; "
    cosine = drift.get("cosine_similarity")
    cosine_text = "N/A" if cosine is None else f"{cosine:.6g}"
    return (
        f"{prefix}MAE={drift['mean_abs_error']:.6g}, "
        f"max={drift['max_abs_error']:.6g}, cosine={cosine_text}"
    )


def deduplicate_notes(notes: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for note in notes:
        note = str(note).strip()
        if not note or note in seen:
            continue
        seen.add(note)
        result.append(note)
    return result


def collect_hardware_metadata(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "created_at": utc_now(),
        "device_name": args.device_name,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "onnxruntime_version": getattr(ort, "__version__", None),
        "available_onnxruntime_providers": ort.get_available_providers(),
    }
    if psutil is not None:
        try:
            payload["host_memory_total_mb"] = psutil.virtual_memory().total / 1_000_000
        except Exception:
            pass
    try:
        import torch

        payload["torch_version"] = getattr(torch, "__version__", None)
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["torch_cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if torch.cuda.is_available():
            payload["torch_cuda_devices"] = [
                {"index": index, "name": torch.cuda.get_device_name(index)}
                for index in range(torch.cuda.device_count())
            ]
    except Exception:
        payload["torch"] = "unavailable"
    try:
        import pynvml

        pynvml.nvmlInit()
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            raw_name = pynvml.nvmlDeviceGetName(handle)
            name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append({"index": index, "name": name, "memory_total_mb": memory.total / 1_000_000})
        payload["nvml_devices"] = devices
        pynvml.nvmlShutdown()
    except Exception:
        payload["nvml_devices"] = "unavailable"
    payload["hardware_class"] = classify_hardware(args.device_name, args.providers)
    return payload


def classify_hardware(device_name: str | None, providers: list[str]) -> str:
    text = " ".join([device_name or "", *providers]).lower()
    edge_terms = (
        "raspberry",
        "jetson",
        "orin",
        "snapdragon",
        "qualcomm",
        "android",
        "nnapi",
        "qnn",
        "npu",
        "coreml",
        "edge",
    )
    if any(term in text for term in edge_terms):
        return "edge_or_embedded_candidate_from_metadata"
    if any("cuda" in provider.lower() or "tensorrt" in provider.lower() for provider in providers):
        return "desktop_or_server_gpu_unless_device_name_indicates_edge"
    if any("openvino" in provider.lower() for provider in providers):
        return "openvino_host_or_edge_accelerator_check_device_name"
    return "host_cpu_or_unknown"


def enrich_speedup_and_size(rows: list[dict[str, Any]]) -> None:
    fp32_size = None
    for row in rows:
        if row["precision"] == "FP32":
            fp32_size = row["onnx_file_size_mb"]
            break
    baseline_by_provider = {
        row["execution_provider"]: row for row in rows if row["precision"] == "FP32" and row.get("mean_latency_ms")
    }
    for row in rows:
        baseline = baseline_by_provider.get(row["execution_provider"])
        if baseline and row.get("mean_latency_ms"):
            row["speedup_vs_fp32"] = baseline["mean_latency_ms"] / row["mean_latency_ms"]
        if fp32_size and row.get("onnx_file_size_mb") is not None:
            row["size_reduction_vs_fp32"] = (1.0 - (row["onnx_file_size_mb"] / fp32_size)) * 100.0


TABLE_COLUMNS = [
    ("precision", "Precision"),
    ("onnx_model_path", "ONNX model path"),
    ("onnx_file_size_mb", "ONNX file size (MB)"),
    ("execution_provider_used", "Execution Provider used"),
    ("hardware_device", "Hardware/device metadata"),
    ("mean_latency_ms", "Mean latency (ms)"),
    ("median_latency_ms", "Median latency (ms)"),
    ("p90_latency_ms", "p90 latency (ms)"),
    ("p95_latency_ms", "p95 latency (ms)"),
    ("throughput_samples_per_sec", "Throughput (samples/sec)"),
    ("peak_host_memory_rss_mb", "Peak host RSS (MB)"),
    ("device_memory_mb", "Device memory (MB)"),
    ("average_power_w", "Average power (W)"),
    ("energy_per_inference_mj", "Energy/inference (mJ)"),
    ("accuracy_or_drift_metric_vs_fp32", "Accuracy or drift vs FP32"),
    ("speedup_vs_fp32", "Speedup vs FP32"),
    ("size_reduction_vs_fp32", "Size reduction vs FP32 (%)"),
    ("notes", "Notes/warnings"),
]


def format_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if math.isnan(value):
            return "N/A"
        if abs(value) >= 100:
            return f"{value:.2f}"
        if abs(value) >= 10:
            return f"{value:.3f}"
        return f"{value:.4f}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_csv_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key, _ in TABLE_COLUMNS])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key, _ in TABLE_COLUMNS})


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [label for _, label in TABLE_COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [markdown_escape(format_cell(row.get(key))) for key, _ in TABLE_COLUMNS]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value).replace("\n", " ")


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = [latex_escape(label) for _, label in TABLE_COLUMNS]
    col_spec = "l" * len(headers)
    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        values = [latex_escape(format_cell(row.get(key))) for key, _ in TABLE_COLUMNS]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tables(output_dir: Path, rows: list[dict[str, Any]], formats: list[str]) -> None:
    if "csv" in formats:
        write_csv_table(output_dir / "onnx_precision_benchmark.csv", rows)
    if "markdown" in formats:
        write_markdown_table(output_dir / "onnx_precision_benchmark.md", rows)
    if "latex" in formats:
        write_latex_table(output_dir / "onnx_precision_benchmark.tex", rows)


def write_summary(output_dir: Path, rows: list[dict[str, Any]], hardware: dict[str, Any], failures: list[dict[str, Any]]) -> None:
    best_latency = min(rows, key=lambda row: row["mean_latency_ms"]) if rows else None
    smallest = min(rows, key=lambda row: row["onnx_file_size_mb"]) if rows else None
    fp16_rows = [row for row in rows if row["precision"] == "FP16" and row.get("speedup_vs_fp32") is not None]
    int8_rows = [row for row in rows if row["precision"] == "INT8" and row.get("speedup_vs_fp32") is not None]
    energy_measured = any(row.get("energy_per_inference_mj") is not None for row in rows)
    edge_statement = edge_measurement_statement(hardware, rows)
    lines = [
        "ONNX Precision Benchmark Summary",
        "",
        f"Generated at: {utc_now()}",
        f"Hardware used: {hardware.get('device_name') or hardware.get('platform') or 'unknown'}",
        f"Hardware class: {hardware.get('hardware_class', 'unknown')}",
        f"Providers tested: {', '.join(sorted({row['execution_provider'] for row in rows})) if rows else 'none'}",
        f"Actual edge hardware measured: {edge_statement}",
    ]
    if best_latency:
        lines.append(
            "Best latency configuration: "
            f"{best_latency['precision']} / {best_latency['execution_provider_used']} "
            f"at {best_latency['mean_latency_ms']:.3f} ms mean."
        )
    if smallest:
        lines.append(
            "Smallest model configuration: "
            f"{smallest['precision']} at {smallest['onnx_file_size_mb']:.3f} MB."
        )
    lines.append(f"FP16 improved latency: {improvement_text(fp16_rows)}")
    lines.append(f"INT8 improved latency: {improvement_text(int8_rows)}")
    lines.append("Energy measured: yes" if energy_measured else "Energy measured: no; energy and power are N/A without --power-log.")
    if failures:
        lines.append("")
        lines.append("Failures or skipped rows:")
        for failure in failures:
            lines.append(f"- {failure.get('precision', 'unknown')} {failure.get('provider', failure.get('stage', ''))}: {failure.get('reason')}")
    output_dir.joinpath("onnx_precision_benchmark_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def improvement_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "N/A (no comparable rows)"
    improved = [row for row in rows if (row.get("speedup_vs_fp32") or 0) > 1.0]
    best = max(rows, key=lambda row: row.get("speedup_vs_fp32") or 0)
    return (
        f"yes, {len(improved)}/{len(rows)} comparable rows"
        if improved
        else f"no, best speedup was {(best.get('speedup_vs_fp32') or 0):.3f}x"
    )


def edge_measurement_statement(hardware: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    hardware_class = str(hardware.get("hardware_class", "unknown"))
    if "edge_or_embedded" in hardware_class:
        return "likely yes from provider/device metadata; verify the named hardware before citing."
    if not hardware.get("device_name"):
        return "unknown; no --device-name was supplied."
    if rows:
        return "no clear edge provider/device signal; treat as host/desktop measurement unless hardware metadata proves otherwise."
    return "N/A; no rows ran."


def prepare_variants(
    args: argparse.Namespace,
    calibration_source: InputSource,
    failures: list[dict[str, Any]],
) -> list[ModelVariant]:
    fp32_path = Path(args.fp32_onnx).expanduser()
    variants = [ModelVariant(precision="FP32", path=fp32_path, notes=[], metadata=inspect_onnx_summary(fp32_path))]
    output_dir = Path(args.output_dir).expanduser()
    if args.fp16_onnx:
        fp16_path = Path(args.fp16_onnx).expanduser()
        if fp16_path.exists():
            variants.append(ModelVariant(precision="FP16", path=fp16_path, metadata=inspect_onnx_summary(fp16_path)))
        else:
            failures.append({"precision": "FP16", "stage": "load", "reason": f"--fp16-onnx not found: {fp16_path}"})
    elif not args.skip_fp16_conversion:
        variant = convert_fp16(fp32_path, output_dir, failures)
        if variant is not None:
            variants.append(variant)

    if args.int8_onnx:
        int8_path = Path(args.int8_onnx).expanduser()
        if int8_path.exists():
            variants.append(ModelVariant(precision="INT8", path=int8_path, metadata=inspect_onnx_summary(int8_path)))
        else:
            failures.append({"precision": "INT8", "stage": "load", "reason": f"--int8-onnx not found: {int8_path}"})
    elif not args.skip_int8_quantization:
        variant = quantize_int8(fp32_path, output_dir, args, calibration_source, failures)
        if variant is not None:
            variants.append(variant)
    return variants


def load_power_log(args: argparse.Namespace, failures: list[dict[str, Any]]) -> PowerLog | None:
    if not args.power_log:
        return None
    path = Path(args.power_log).expanduser()
    try:
        return PowerLog(path, args.timestamp_column, args.power_column)
    except Exception as exc:
        failures.append({"stage": "power_log", "reason": short_error(exc), "path": str(path)})
        return None


def run_benchmarks(
    variants: list[ModelVariant],
    input_source: InputSource,
    args: argparse.Namespace,
    hardware: dict[str, Any],
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    power_log = load_power_log(args, failures)
    for provider in args.providers:
        reference_outputs: list[list[Any]] | None = None
        for variant in variants:
            eprint(f"Benchmarking {variant.precision} with {provider}...")
            try:
                session, session_notes = make_session(variant.path, variant.precision, provider, args, Path(args.output_dir))
                if variant.precision == "FP32":
                    reference_outputs = drift_reference_outputs(session, input_source, args.drift_samples)
                row = benchmark_session(
                    variant,
                    provider,
                    session,
                    input_source,
                    args,
                    power_log,
                    hardware,
                    session_notes,
                    reference_outputs,
                )
                rows.append(row)
            except Exception as exc:
                failures.append(
                    {
                        "precision": variant.precision,
                        "provider": provider,
                        "stage": "benchmark",
                        "reason": short_error(exc),
                    }
                )
                eprint(f"Warning: skipped {variant.precision} / {provider}: {short_error(exc)}")
    enrich_speedup_and_size(rows)
    return rows


def write_config(
    output_dir: Path,
    args: argparse.Namespace,
    benchmark_input_source: InputSource,
    calibration_input_source: InputSource,
    variants: list[ModelVariant],
) -> None:
    payload = {
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "args": vars(args),
        "benchmark_input_source": {
            "source_kind": benchmark_input_source.source_kind,
            "notes": benchmark_input_source.notes,
            "metadata": benchmark_input_source.metadata,
        },
        "calibration_input_source": {
            "source_kind": calibration_input_source.source_kind,
            "notes": calibration_input_source.notes,
            "metadata": calibration_input_source.metadata,
        },
        "model_variants": [
            {
                "precision": variant.precision,
                "path": str(variant.path),
                "notes": variant.notes,
                "metadata": variant.metadata,
            }
            for variant in variants
        ],
    }
    write_json(output_dir / "benchmark_config.json", payload)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    require_imports()
    warn_optional_packages(args)

    fp32_path = Path(args.fp32_onnx).expanduser()
    if not fp32_path.exists():
        raise SystemExit(f"--fp32-onnx does not exist: {fp32_path}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1.")

    failures: list[dict[str, Any]] = []
    hardware = collect_hardware_metadata(args)
    write_json(output_dir / "hardware_metadata.json", hardware)

    input_specs = inspect_onnx_inputs(fp32_path)
    if not input_specs:
        raise SystemExit(f"No runtime graph inputs found in {fp32_path}.")
    input_source = build_input_source(fp32_path, input_specs, args, role="benchmark")
    for note in input_source.notes:
        eprint(f"Benchmark input note: {note}")

    if args.quantization_mode == "static" and args.int8_onnx is None and not args.skip_int8_quantization:
        calibration_source = build_input_source(fp32_path, input_specs, args, role="calibration")
    else:
        calibration_source = input_source
    for note in calibration_source.notes:
        eprint(f"Calibration input note: {note}")

    variants = prepare_variants(args, calibration_source, failures)
    if not variants:
        raise SystemExit("No ONNX model variants are available to benchmark.")
    write_config(output_dir, args, input_source, calibration_source, variants)

    rows = run_benchmarks(variants, input_source, args, hardware, failures)
    write_tables(output_dir, rows, args.table_formats)
    write_json(output_dir / "onnx_precision_benchmark_failures.json", failures)
    write_summary(output_dir, rows, hardware, failures)

    if not rows:
        raise SystemExit(
            "No benchmark rows ran successfully. Check onnx_precision_benchmark_failures.json for details."
        )
    eprint(f"Wrote ONNX precision benchmark outputs to {output_dir}")


if __name__ == "__main__":
    main()
