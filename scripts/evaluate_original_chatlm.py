#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_prune_eval import (  # noqa: E402
    DEFAULT_BENCHMARK_JSON,
    DEFAULT_TRAIN_JSON,
    compact_metrics,
    cleanup_distributed,
    evaluate_all,
    load_model_and_tokenizer,
    rank0_print,
    read_records,
    release_cuda_memory,
    setup_distributed,
    summarize_model,
    truncate_records,
    write_json,
)


DEFAULT_MODEL = "charent/ChatLM-mini-Chinese"
DEFAULT_OUTPUT_JSON = PROJECT_ROOT / "prune_eval_outputs" / "original_chatlm_eval_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the original Hugging Face ChatLM model on SCENIC benchmark and training data."
    )
    parser.add_argument(
        "model_path_or_hf_id",
        nargs="?",
        default=None,
        help=f"Original Hugging Face model id or local path. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Compatibility alias for the original Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--source-model-id",
        default=None,
        help="Optional original Hugging Face id to record when evaluating a local materialized copy.",
    )
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--benchmark-json", default=str(DEFAULT_BENCHMARK_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-benchmark-examples", type=int, default=None)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument(
        "--ignore-spaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignore whitespace for exact match. Defaults to true for Chinese SCENIC scoring.",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--include-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ensure-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy tokenizer EOS into model/generation config when missing or mismatched.",
    )
    parser.add_argument(
        "--eos-sample-size",
        type=int,
        default=16,
        help="Number of benchmark prompts used to verify generated sequences terminate with EOS.",
    )
    args = parser.parse_args(argv)
    if args.model_path_or_hf_id and args.model:
        parser.error("Provide the model once, either as the positional argument or with --model.")
    args.model = normalize_model_arg(args.model or args.model_path_or_hf_id or DEFAULT_MODEL)
    return args


def normalize_model_arg(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def eos_values(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    result: list[int] = []
    for item in values:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def eos_contains(value: Any, eos_token_id: int | None) -> bool:
    if eos_token_id is None:
        return False
    return int(eos_token_id) in eos_values(value)


def eos_with_token(value: Any, eos_token_id: int) -> int | list[int]:
    values = eos_values(value)
    if not values:
        return int(eos_token_id)
    if int(eos_token_id) not in values:
        values.append(int(eos_token_id))
    return values[0] if len(values) == 1 else values


def eos_diagnostics(tokenizer: Any, model: Any, *, ensure_eos: bool) -> dict[str, Any]:
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    tokenizer_eos_token = getattr(tokenizer, "eos_token", None)
    model_config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    model_eos_before = getattr(model_config, "eos_token_id", None) if model_config is not None else None
    generation_eos_before = (
        getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    )

    changed_model_config = False
    changed_generation_config = False
    if ensure_eos and tokenizer_eos is not None:
        if model_config is not None and not eos_contains(model_eos_before, tokenizer_eos):
            model_config.eos_token_id = eos_with_token(model_eos_before, int(tokenizer_eos))
            changed_model_config = True
        if generation_config is not None and not eos_contains(generation_eos_before, tokenizer_eos):
            generation_config.eos_token_id = eos_with_token(generation_eos_before, int(tokenizer_eos))
            changed_generation_config = True

    model_eos_after = getattr(model_config, "eos_token_id", None) if model_config is not None else None
    generation_eos_after = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    return {
        "ensure_eos": ensure_eos,
        "tokenizer_eos_token": tokenizer_eos_token,
        "tokenizer_eos_token_id": tokenizer_eos,
        "model_config_eos_token_id_before": model_eos_before,
        "generation_config_eos_token_id_before": generation_eos_before,
        "model_config_eos_token_id_after": model_eos_after,
        "generation_config_eos_token_id_after": generation_eos_after,
        "model_config_contains_tokenizer_eos": eos_contains(model_eos_after, tokenizer_eos),
        "generation_config_contains_tokenizer_eos": eos_contains(generation_eos_after, tokenizer_eos),
        "changed_model_config": changed_model_config,
        "changed_generation_config": changed_generation_config,
    }


def move_encoded_to_device(encoded: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in encoded.items()
        if key != "token_type_ids"
    }


def generated_sequence_ended_with_eos(sequence: torch.Tensor, eos_token_id: int, pad_token_id: int | None) -> bool:
    if pad_token_id is None:
        end_position = int(sequence.numel()) - 1
    else:
        non_pad_positions = torch.nonzero(sequence.ne(int(pad_token_id)), as_tuple=False).flatten()
        if non_pad_positions.numel() == 0:
            return False
        end_position = int(non_pad_positions[-1].item())
    if end_position < 0:
        return False
    return int(sequence[end_position].item()) == int(eos_token_id)


@torch.no_grad()
def sample_generation_eos(
    model: Any,
    tokenizer: Any,
    benchmark_records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return {
            "sample_size": 0,
            "status": "missing_tokenizer_eos",
            "ended_with_eos_count": 0,
            "ended_with_eos_rate": None,
        }
    sample_records = benchmark_records[: max(0, args.eos_sample_size)]
    if not sample_records:
        return {
            "sample_size": 0,
            "status": "empty_sample",
            "ended_with_eos_count": 0,
            "ended_with_eos_rate": None,
        }

    from scenic_prune_eval import extract_prompt_response

    prompts = [extract_prompt_response(record)[0] for record in sample_records]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
    )
    generated = model.generate(
        **move_encoded_to_device(encoded, device),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        do_sample=False,
        early_stopping=True,
    )
    generated = generated.detach().cpu()
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    ended = [
        generated_sequence_ended_with_eos(sequence, int(eos_token_id), pad_token_id)
        for sequence in generated
    ]
    missing_examples = []
    for row_index, did_end in enumerate(ended):
        if did_end:
            continue
        prompt_index = row_index // args.num_return_sequences
        missing_examples.append(
            {
                "prompt_index": prompt_index,
                "beam_index": row_index % args.num_return_sequences,
                "prompt": prompts[prompt_index],
                "decoded": tokenizer.decode(
                    generated[row_index],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ),
            }
        )
        if len(missing_examples) >= 5:
            break

    return {
        "sample_size": len(sample_records),
        "generated_sequence_count": int(generated.size(0)),
        "status": "ok",
        "ended_with_eos_count": int(sum(ended)),
        "ended_with_eos_rate": float(sum(ended) / len(ended)) if ended else None,
        "missing_eos_examples": missing_examples,
    }


def evaluate_original_chatlm(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences cannot exceed --num-beams for beam search.")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    train_json = Path(args.train_json).expanduser()
    benchmark_json = Path(args.benchmark_json).expanduser()
    output_json = Path(args.output_json).expanduser()

    state = setup_distributed(args.device)
    try:
        train_records = truncate_records(read_records(train_json), args.max_train_examples)
        benchmark_records = truncate_records(read_records(benchmark_json), args.max_benchmark_examples)
        datasets = {
            "benchmark": benchmark_records,
            "training": train_records,
        }

        if state.is_main:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            rank0_print(state, f"Original model: {args.model}")
            if args.source_model_id:
                rank0_print(state, f"Source Hugging Face id: {args.source_model_id}")
            rank0_print(state, f"Single JSON report: {output_json}")

        tokenizer, model = load_model_and_tokenizer(args, args.model, state)
        eos = eos_diagnostics(tokenizer, model, ensure_eos=args.ensure_eos)
        model_summary = summarize_model(model, args.model) if state.is_main else None
        eos_sample = (
            sample_generation_eos(model, tokenizer, benchmark_records, args, state.device)
            if state.is_main
            else None
        )
        results = evaluate_all(model, tokenizer, datasets, args, state, label="original_chatlm")
        del model
        release_cuda_memory()

        if not state.is_main:
            return None

        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_path": args.model,
            "source_model_id": args.source_model_id,
            "output_json": str(output_json),
            "original_huggingface_model": True,
            "eos": {
                **eos,
                "generation_sample": eos_sample,
            },
            "distributed": {
                "enabled": state.enabled,
                "world_size": state.world_size,
            },
            "generation": {
                "num_beams": args.num_beams,
                "num_return_sequences": args.num_return_sequences,
                "max_new_tokens": args.max_new_tokens,
                "ignore_spaces": args.ignore_spaces,
            },
            "datasets": {
                "benchmark": {"path": str(benchmark_json), "total": len(benchmark_records)},
                "training": {"path": str(train_json), "total": len(train_records)},
            },
            "model": model_summary,
            "summary": {
                "accuracy_definition": "accuracy is exact-match@1 / EM@1",
                "original_chatlm": compact_metrics(results),
            },
            "evaluations": results,
        }
        write_json(output_json, report)
        rank0_print(state, f"Wrote original ChatLM eval report: {output_json}")
        return report
    finally:
        cleanup_distributed(state)


def main() -> None:
    evaluate_original_chatlm(parse_args())


if __name__ == "__main__":
    main()
