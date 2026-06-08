#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import summarize_model  # noqa: E402
from scenic_train_chatlm_sft import (  # noqa: E402
    repair_checkpoint_for_auto_load,
    repair_tokenizer_files_for_auto_load,
    sanitize_model_for_save,
    save_tokenizer_for_auto_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a checkpoint on CPU FP32, set eval mode, remove torch pruning "
            "reparameterizations if present, and save a baked checkpoint for ONNX export."
        )
    )
    parser.add_argument("--model", required=True, help="Checkpoint path to bake.")
    parser.add_argument("--output-dir", required=True, help="Baked checkpoint output directory.")
    parser.add_argument("--source-tokenizer", default="", help="Optional original model/tokenizer path.")
    parser.add_argument("--summary-json", default="", help="Optional JSON summary path.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def tokenizer_source(model_path: Path, source_tokenizer: str) -> str:
    source_path = Path(source_tokenizer).expanduser() if source_tokenizer else None
    if source_path is not None and source_path.is_dir():
        return str(source_path)
    return str(model_path)


def remove_pruning_reparameterizations(model: torch.nn.Module) -> list[dict[str, str]]:
    try:
        import torch.nn.utils.prune as torch_prune
    except Exception:
        return []

    removed: list[dict[str, str]] = []
    for module_name, module in model.named_modules():
        for tensor_name in ("weight", "bias"):
            if not hasattr(module, f"{tensor_name}_orig") or not hasattr(module, f"{tensor_name}_mask"):
                continue
            try:
                torch_prune.remove(module, tensor_name)
            except Exception as exc:
                removed.append(
                    {
                        "module": module_name,
                        "tensor": tensor_name,
                        "status": "error",
                        "error": str(exc),
                    }
                )
            else:
                removed.append(
                    {
                        "module": module_name,
                        "tensor": tensor_name,
                        "status": "removed",
                    }
                )
    return removed


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    source_path = Path(args.source_tokenizer).expanduser() if args.source_tokenizer else None

    if model_path.is_dir():
        repair_tokenizer_files_for_auto_load(model_path, source_dir=source_path if source_path and source_path.is_dir() else None)

    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source(model_path, args.source_tokenizer),
        use_fast=False,
        **load_kwargs,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float32,
        **load_kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.cpu()
    model.eval()

    before = summarize_model(model, str(model_path))
    removed = remove_pruning_reparameterizations(model)
    after = summarize_model(model, str(model_path))

    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_for_auto_load(tokenizer, output_dir)
    sanitize_model_for_save(model).save_pretrained(output_dir, safe_serialization=True)
    repair_checkpoint_for_auto_load(output_dir, tokenizer=tokenizer)
    if source_path is not None and source_path.is_dir():
        repair_tokenizer_files_for_auto_load(output_dir, source_dir=source_path)

    if args.summary_json:
        write_json(
            Path(args.summary_json).expanduser(),
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_model_path": str(model_path),
                "output_dir": str(output_dir),
                "source_tokenizer": args.source_tokenizer,
                "mode": "cpu_fp32_eval_baked_for_onnx_export",
                "model_eval": not model.training,
                "removed_pruning_reparameterizations": removed,
                "removed_pruning_reparameterization_count": sum(1 for item in removed if item.get("status") == "removed"),
                "model_before_bake": before,
                "model_after_bake": after,
            },
        )


if __name__ == "__main__":
    main()
