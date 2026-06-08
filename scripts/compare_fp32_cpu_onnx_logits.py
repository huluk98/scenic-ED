#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import extract_prompt_response, read_records, truncate_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw PyTorch FP32 logits with ONNX Runtime FP32 CPU logits."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--pytorch-model", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--decoder-length", type=int, default=8)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def extract_logits(outputs: Any) -> np.ndarray:
    if hasattr(outputs, "logits"):
        return tensor_to_numpy(outputs.logits)
    if isinstance(outputs, dict) and "logits" in outputs:
        return tensor_to_numpy(outputs["logits"])
    if isinstance(outputs, (tuple, list)) and outputs:
        return tensor_to_numpy(outputs[0])
    raise TypeError(f"Cannot extract logits from output type {type(outputs)!r}")


def decoder_start_token_id(model: Any, tokenizer: Any) -> int:
    config = getattr(model, "config", None)
    for name in ("decoder_start_token_id", "bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value >= 0:
            return value
    for value in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id):
        if isinstance(value, int) and value >= 0:
            return value
    raise ValueError("Could not resolve decoder start token id.")


def make_decoder_input_ids(response: str, tokenizer: Any, start_id: int, decoder_length: int) -> torch.Tensor:
    decoder_length = max(1, decoder_length)
    start = torch.tensor([[start_id]], dtype=torch.long)
    if decoder_length == 1:
        return start
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else start_id
    encoded = tokenizer(
        response,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=decoder_length - 1,
    )["input_ids"].long()
    if encoded.size(1) < decoder_length - 1:
        pad = encoded.new_full((1, decoder_length - 1 - encoded.size(1)), pad_id)
        encoded = torch.cat((encoded, pad), dim=1)
    return torch.cat((start, encoded[:, : decoder_length - 1]), dim=1)


def collect_ort_session_providers(model: Any) -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    seen: set[int] = set()
    child_attrs = (
        "encoder",
        "decoder",
        "decoder_with_past",
        "encoder_model",
        "decoder_model",
        "decoder_with_past_model",
        "model",
        "_encoder",
        "_decoder",
        "_decoder_with_past",
    )

    def visit(name: str, obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 5:
            return
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)
        session = None
        if hasattr(obj, "get_providers"):
            session = obj
        else:
            try:
                maybe_session = getattr(obj, "session")
            except Exception:
                maybe_session = None
            if hasattr(maybe_session, "get_providers"):
                session = maybe_session
        if session is not None:
            try:
                provider_list = list(session.get_providers())
            except Exception:
                provider_list = []
            providers.append({"path": name, "providers": provider_list})
        for child_attr in child_attrs:
            try:
                child = getattr(obj, child_attr)
            except Exception:
                continue
            visit(f"{name}.{child_attr}", child, depth + 1)

    visit("ort_model", model)
    return providers


def run_ort_forward(model: Any, encoded: dict[str, torch.Tensor], decoder_input_ids: torch.Tensor) -> Any:
    kwargs = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded.get("attention_mask"),
        "decoder_input_ids": decoder_input_ids,
        "use_cache": False,
    }
    try:
        return model(**kwargs)
    except TypeError:
        kwargs.pop("use_cache", None)
        return model(**kwargs)


def main() -> None:
    args = parse_args()
    pt_path = Path(args.pytorch_model).expanduser()
    onnx_path = Path(args.onnx_model).expanduser()
    tokenizer_path = Path(args.tokenizer).expanduser()

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=False, **load_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pt_model = AutoModelForSeq2SeqLM.from_pretrained(
        str(pt_path),
        torch_dtype=torch.float32,
        **load_kwargs,
    )
    if hasattr(pt_model.config, "use_cache"):
        pt_model.config.use_cache = False
    pt_model.cpu()
    pt_model.eval()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort_load_kwargs = {
        "provider": "CPUExecutionProvider",
        "session_options": session_options,
        "use_io_binding": False,
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(str(onnx_path), **ort_load_kwargs)
    except TypeError as exc:
        message = str(exc)
        if "use_io_binding" not in message and "unexpected keyword" not in message:
            raise
        ort_load_kwargs.pop("use_io_binding", None)
        ort_model = ORTModelForSeq2SeqLM.from_pretrained(str(onnx_path), **ort_load_kwargs)
    if hasattr(ort_model.config, "use_cache"):
        ort_model.config.use_cache = False

    records = truncate_records(read_records(Path(args.benchmark_json).expanduser()), args.max_examples)
    start_id = decoder_start_token_id(pt_model, tokenizer)
    rows: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    mean_abs_diffs: list[float] = []
    pass_count = 0
    top1_match_count = 0
    top5_overlap_values: list[float] = []

    for index, record in enumerate(records):
        prompt, response = extract_prompt_response(record)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.max_input_len,
        )
        encoded.pop("token_type_ids", None)
        decoder_input_ids = make_decoder_input_ids(response, tokenizer, start_id, args.decoder_length)
        with torch.no_grad():
            pt_logits = extract_logits(
                pt_model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    decoder_input_ids=decoder_input_ids,
                    use_cache=False,
                    return_dict=True,
                )
            )
            ort_logits = extract_logits(run_ort_forward(ort_model, encoded, decoder_input_ids))

        diff = np.abs(pt_logits - ort_logits)
        row_max = float(np.max(diff))
        row_mean = float(np.mean(diff))
        reference_scale = float(np.max(np.abs(pt_logits))) if pt_logits.size else 0.0
        allowed = args.atol + args.rtol * reference_scale
        row_pass = row_max <= allowed
        max_abs_diff = max(max_abs_diff, row_max)
        mean_abs_diffs.append(row_mean)
        pass_count += int(row_pass)

        pt_last = pt_logits[0, -1]
        ort_last = ort_logits[0, -1]
        pt_top = np.argsort(-pt_last)[:5].tolist()
        ort_top = np.argsort(-ort_last)[:5].tolist()
        top1_match = bool(pt_top and ort_top and pt_top[0] == ort_top[0])
        top1_match_count += int(top1_match)
        overlap = len(set(pt_top).intersection(ort_top)) / max(1, min(len(pt_top), len(ort_top)))
        top5_overlap_values.append(float(overlap))

        rows.append(
            {
                "index": index,
                "prompt_preview": prompt[:160],
                "response_preview": response[:160],
                "shape": list(pt_logits.shape),
                "max_abs_diff": row_max,
                "mean_abs_diff": row_mean,
                "allowed_abs_diff": allowed,
                "passes_tolerance": row_pass,
                "pt_last_top5_token_ids": pt_top,
                "ort_last_top5_token_ids": ort_top,
                "last_token_top1_match": top1_match,
                "last_token_top5_overlap": overlap,
            }
        )

    total = len(rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "mode": "fp32_cpu_raw_logits_parity",
        "pytorch_model": str(pt_path),
        "onnx_model": str(onnx_path),
        "tokenizer": str(tokenizer_path),
        "benchmark_json": args.benchmark_json,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "provider": "CPUExecutionProvider",
        "model_eval": not pt_model.training,
        "max_examples": args.max_examples,
        "examples_compared": total,
        "max_input_len": args.max_input_len,
        "decoder_length": args.decoder_length,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(np.mean(mean_abs_diffs)) if mean_abs_diffs else 0.0,
        "pass_count": pass_count,
        "pass_rate": pass_count / total if total else 0.0,
        "last_token_top1_match_rate": top1_match_count / total if total else 0.0,
        "last_token_top5_overlap_mean": float(np.mean(top5_overlap_values)) if top5_overlap_values else 0.0,
        "onnx_session_providers": collect_ort_session_providers(ort_model),
        "rows": rows,
    }
    write_json(Path(args.output_json).expanduser(), payload)
    print(f"Wrote FP32 CPU ONNX logits parity report: {args.output_json}")


if __name__ == "__main__":
    main()
