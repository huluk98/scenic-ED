#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
DEFAULT_BENCHMARK_JSON = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200.json"

PROMPT_FIELDS = ("prompt", "anchor", "instruction", "question", "input", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")
DIFFICULTY_FIELDS = ("difficulty", "complexity", "level")
ID_FIELDS = ("id", "sample_id", "uid", "index")
HEAD_NAME_PATTERNS = (
    "lm_head",
    "classifier",
    "classification_head",
    "score",
    "output_head",
    "response_head",
    "response_projection",
    "final_projection",
    "final_logits_bias",
)


@dataclass
class EvalExample:
    sample_id: str
    prompt: str
    target: str
    difficulty: str
    record: dict[str, Any]


@dataclass
class EvalResult:
    rows: list[dict[str, Any]]
    metrics: dict[str, Any]


@dataclass
class PruningResult:
    target_sparsity: float
    targeted_linear_sparsity_actual: float
    whole_model_sparsity_actual: float
    mask_path: str
    pruning_summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run SCENIC linear-weight sparsity experiments with dense, one-shot, "
            "and progressive recovery pruning at 0%, 30%, and 50%."
        )
    )
    parser.add_argument("--experiment_name", required=True)
    parser.add_argument("--model_family", choices=("encoder_only", "decoder_only", "encoder_decoder"), required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--sparsity_levels", nargs="+", type=float, default=[0.0, 0.3, 0.5])
    parser.add_argument("--pruning_modes", nargs="+", choices=("dense", "oneshot", "progressive"), default=["dense", "oneshot", "progressive"])
    parser.add_argument("--pruning_mode", nargs="+", choices=("dense", "oneshot", "progressive"), default=None)
    parser.add_argument("--prune_scope", choices=("linear_weights", "all-linear", "encoder-linear", "decoder-linear"), default="linear_weights")
    parser.add_argument("--prune_method", choices=("magnitude",), default="magnitude")
    parser.add_argument("--progressive_schedule", choices=("staged",), default="staged")
    parser.add_argument(
        "--recovery_epochs_per_stage",
        type=int,
        default=1,
        help="Recovery epochs after each progressive mask stage. Default: 1.",
    )
    parser.add_argument(
        "--final_recovery_epochs",
        type=int,
        default=1,
        help="Final recovery epochs after progressive mask staging. Default: 1.",
    )
    parser.add_argument("--prune_output_heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--global_pruning", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--regrowth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--benchmark_path", default=str(DEFAULT_BENCHMARK_JSON))
    parser.add_argument("--benchmark_difficulty_path", default=None)
    parser.add_argument("--train_path", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--validation_path", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--normalization_mode", choices=("standard", "ignore_spaces"), default="ignore_spaces")
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--length_penalty", type=float, default=None)
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_benchmark_examples", type=int, default=None)
    parser.add_argument("--max_validation_examples", type=int, default=None)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must be a JSON object.")
                rows.append(value)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = None
        for key in ("records", "data", "examples", "items"):
            if isinstance(value.get(key), list):
                records = value[key]
                break
        if records is None:
            records = [value]
    else:
        raise ValueError(f"{path} must be JSON list, JSON object, or JSONL.")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains non-object records.")
    return list(records)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, torch.dtype):
            return str(value)
    except Exception:
        pass
    return str(value)


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def sample_id_for(record: dict[str, Any], index: int) -> str:
    for field in ID_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(index)


def extract_prompt_response(record: dict[str, Any]) -> tuple[str, str]:
    prompt = first_text(record, PROMPT_FIELDS)
    response = first_text(record, RESPONSE_FIELDS)
    if not prompt or not response:
        raise ValueError(f"Record is missing prompt/response fields: {sorted(record)}")
    return prompt, response


def normalize_text(text: str, mode: str = "ignore_spaces") -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    punctuation_map = {
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "：": ":",
        "；": ";",
        "（": "(",
        "）": ")",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    text = "".join(punctuation_map.get(ch, ch) for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    if mode == "ignore_spaces":
        text = "".join(text.split())
    return text


def normalize_difficulty(value: Any) -> str:
    label = str(value).strip().lower()
    aliases = {"simple": "easy", "moderate": "medium", "med": "medium", "complex": "hard"}
    label = aliases.get(label, label)
    if label not in {"easy", "medium", "hard"}:
        raise ValueError(f"Difficulty must be easy, medium, or hard; got {value!r}.")
    return label


def difficulty_from_record(record: dict[str, Any]) -> str | None:
    for field in DIFFICULTY_FIELDS:
        if record.get(field) not in (None, ""):
            return normalize_difficulty(record[field])
    return None


def read_difficulty_file(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    id_map: dict[str, str] = {}
    input_map: dict[str, str] = {}
    if path.suffix.lower() == ".jsonl":
        rows = read_records(path)
    elif path.suffix.lower() == ".json":
        rows = read_records(path)
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("difficulty") in (None, ""):
            continue
        difficulty = normalize_difficulty(row["difficulty"])
        for field in ("id", "sample_id"):
            if row.get(field) not in (None, ""):
                id_map[str(row[field]).strip()] = difficulty
        input_value = row.get("input") or row.get("prompt")
        if input_value not in (None, ""):
            input_map[str(input_value).strip()] = difficulty
    return id_map, input_map


def build_examples(records: list[dict[str, Any]], difficulty_path: Path | None = None) -> list[EvalExample]:
    id_difficulty: dict[str, str] = {}
    input_difficulty: dict[str, str] = {}
    if difficulty_path is not None:
        id_difficulty, input_difficulty = read_difficulty_file(difficulty_path)

    examples: list[EvalExample] = []
    missing: list[str] = []
    for index, record in enumerate(records):
        prompt, target = extract_prompt_response(record)
        sample_id = sample_id_for(record, index)
        difficulty = difficulty_from_record(record)
        if difficulty is None:
            difficulty = id_difficulty.get(sample_id)
        if difficulty is None:
            difficulty = input_difficulty.get(prompt)
        if difficulty is None:
            missing.append(sample_id)
            continue
        examples.append(EvalExample(sample_id=sample_id, prompt=prompt, target=target, difficulty=difficulty, record=record))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Missing difficulty labels for benchmark samples. "
            f"Provide --benchmark_difficulty_path with id/sample_id/input joins. First missing ids: {preview}"
        )
    return examples


def compute_em_for_predictions(predictions: list[str], target: str, normalization_mode: str) -> tuple[bool, bool]:
    normalized_target = normalize_text(target, normalization_mode)
    normalized_predictions = [normalize_text(prediction, normalization_mode) for prediction in predictions[:5]]
    em1 = bool(normalized_predictions and normalized_predictions[0] == normalized_target)
    em5 = normalized_target in normalized_predictions
    return em1, em5


def bootstrap_ci(values: list[int], resamples: int = 1000, seed: int = 42) -> tuple[float | str, float | str]:
    if len(values) < 20:
        return "insufficient_n", "insufficient_n"
    rng = random.Random(seed)
    means: list[float] = []
    count = len(values)
    for _ in range(resamples):
        sample_total = sum(values[rng.randrange(count)] for _ in range(count))
        means.append(sample_total / count)
    means.sort()
    low_index = int(math.floor(0.025 * (len(means) - 1)))
    high_index = int(math.ceil(0.975 * (len(means) - 1)))
    return means[low_index], means[high_index]


def summarize_prediction_rows(rows: list[dict[str, Any]], bootstrap_resamples: int = 1000, seed: int = 42) -> dict[str, Any]:
    summary: dict[str, Any] = {"count_total": len(rows)}
    for group in ("overall", "easy", "medium", "hard"):
        group_rows = rows if group == "overall" else [row for row in rows if row["difficulty"] == group]
        em1_values = [int(row["em1"]) for row in group_rows]
        em5_values = [int(row["em5"]) for row in group_rows]
        suffix = "overall" if group == "overall" else group
        summary[f"count_{suffix}"] = len(group_rows)
        summary[f"em1_{suffix}"] = sum(em1_values) / len(em1_values) if em1_values else 0.0
        summary[f"em5_{suffix}"] = sum(em5_values) / len(em5_values) if em5_values else 0.0
        low, high = bootstrap_ci(em1_values, bootstrap_resamples, seed)
        summary[f"em1_{suffix}_ci_low"] = low
        summary[f"em1_{suffix}_ci_high"] = high
        low, high = bootstrap_ci(em5_values, bootstrap_resamples, seed + 17)
        summary[f"em5_{suffix}_ci_low"] = low
        summary[f"em5_{suffix}_ci_high"] = high
    return summary


def compute_retention(summary: dict[str, Any], dense_summary: dict[str, Any] | None) -> None:
    for group in ("overall", "easy", "medium", "hard"):
        for metric in ("em1", "em5"):
            key = f"{metric}_{group}"
            retention_key = f"{metric}_retention_{group}"
            if dense_summary is None or dense_summary.get(key, 0) in (0, None):
                summary[retention_key] = None
            else:
                summary[retention_key] = summary.get(key, 0.0) / dense_summary[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("This experiment requires torch. Install the project requirements first.") from exc


def resolve_device(device_name: str) -> Any:
    torch = import_torch()
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def module_name_is_output_head(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    dotted = normalized.replace("_", ".")
    return any(pattern in normalized or pattern.replace("_", ".") in dotted for pattern in HEAD_NAME_PATTERNS)


def scope_allows_module(name: str, prune_scope: str) -> bool:
    if prune_scope in {"linear_weights", "all-linear"}:
        return True
    normalized = name.lower().replace("_", ".")
    parts = normalized.split(".")
    is_encoder = "encoder" in parts
    is_decoder = "decoder" in parts
    if prune_scope == "encoder-linear":
        return is_encoder and not is_decoder
    if prune_scope == "decoder-linear":
        return is_decoder
    raise ValueError(f"Unknown prune scope: {prune_scope}")


def collect_prunable_linear_modules(
    model: Any,
    prune_scope: str = "linear_weights",
    prune_output_heads: bool = False,
) -> list[tuple[str, Any]]:
    torch = import_torch()
    modules: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not scope_allows_module(name, prune_scope):
            continue
        if not prune_output_heads and module_name_is_output_head(name):
            continue
        modules.append((name, module))
    return modules


def count_zeros_in_tensors(tensors: list[Any]) -> tuple[int, int]:
    total = 0
    zero = 0
    seen: set[int] = set()
    for tensor in tensors:
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        data = tensor.detach()
        total += int(data.numel())
        zero += int(data.eq(0).sum().item())
    return total, zero


def sparsity_summary(model: Any, modules: list[tuple[str, Any]]) -> tuple[float, float]:
    targeted_total, targeted_zero = count_zeros_in_tensors([module.weight for _, module in modules])
    whole_total, whole_zero = count_zeros_in_tensors([parameter for parameter in model.parameters()])
    targeted = targeted_zero / targeted_total if targeted_total else 0.0
    whole = whole_zero / whole_total if whole_total else 0.0
    return targeted, whole


def mask_key(name: str) -> str:
    return f"{name}.weight"


def make_initial_masks(modules: list[tuple[str, Any]]) -> dict[str, Any]:
    torch = import_torch()
    return {mask_key(name): torch.ones_like(module.weight.detach(), dtype=torch.bool) for name, module in modules}


def apply_masks(modules: list[tuple[str, Any]], masks: dict[str, Any]) -> None:
    with import_torch().no_grad():
        for name, module in modules:
            key = mask_key(name)
            if key in masks:
                module.weight.data.mul_(masks[key].to(module.weight.device))


def apply_magnitude_masks(
    modules: list[tuple[str, Any]],
    target_sparsity: float,
    masks: dict[str, Any] | None = None,
    global_pruning: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = import_torch()
    target_sparsity = max(0.0, min(1.0, float(target_sparsity)))
    masks = make_initial_masks(modules) if masks is None else {key: value.clone() for key, value in masks.items()}
    layer_summaries: list[dict[str, Any]] = []

    if global_pruning:
        total = sum(int(module.weight.numel()) for _, module in modules)
        current_zero = sum(int((~masks[mask_key(name)].to(module.weight.device)).sum().item()) for name, module in modules)
        target_zero = int(round(target_sparsity * total))
        additional = max(0, target_zero - current_zero)
        candidates: list[Any] = []
        for name, module in modules:
            active = masks[mask_key(name)].to(module.weight.device)
            if int(active.sum().item()) > 0:
                candidates.append(module.weight.detach().abs().float()[active].cpu())
        if additional > 0 and candidates:
            scores = torch.cat(candidates)
            additional = min(additional, int(scores.numel()))
            threshold = torch.kthvalue(scores, additional).values.item()
            pruned = 0
            with torch.no_grad():
                for name, module in modules:
                    key = mask_key(name)
                    active = masks[key].to(module.weight.device)
                    score = module.weight.detach().abs().float()
                    prune_mask = active & (score < threshold)
                    masks[key] = (active & ~prune_mask).detach().cpu()
                    pruned += int(prune_mask.sum().item())
                remaining = additional - pruned
                if remaining > 0:
                    for name, module in modules:
                        if remaining <= 0:
                            break
                        key = mask_key(name)
                        active = masks[key].to(module.weight.device)
                        score = module.weight.detach().abs().float()
                        tie_indices = torch.nonzero((active & (score == threshold)).reshape(-1), as_tuple=False).flatten()
                        take = min(remaining, int(tie_indices.numel()))
                        if take:
                            flat_mask = active.reshape(-1).clone()
                            flat_mask[tie_indices[:take]] = False
                            masks[key] = flat_mask.reshape_as(active).detach().cpu()
                            remaining -= take
        apply_masks(modules, masks)
    else:
        with torch.no_grad():
            for name, module in modules:
                key = mask_key(name)
                active = masks[key].to(module.weight.device)
                total = int(module.weight.numel())
                current_zero = int((~active).sum().item())
                target_zero = int(round(target_sparsity * total))
                additional = max(0, target_zero - current_zero)
                if additional > 0:
                    scores = module.weight.detach().abs().float()[active].cpu()
                    additional = min(additional, int(scores.numel()))
                    threshold = torch.kthvalue(scores, additional).values.item()
                    score = module.weight.detach().abs().float()
                    prune_mask = active & (score < threshold)
                    new_mask = active & ~prune_mask
                    pruned = int(prune_mask.sum().item())
                    remaining = additional - pruned
                    if remaining > 0:
                        tie_indices = torch.nonzero((new_mask & (score == threshold)).reshape(-1), as_tuple=False).flatten()
                        take = min(remaining, int(tie_indices.numel()))
                        if take:
                            flat_mask = new_mask.reshape(-1).clone()
                            flat_mask[tie_indices[:take]] = False
                            new_mask = flat_mask.reshape_as(new_mask)
                    masks[key] = new_mask.detach().cpu()
                zero_after = int((~masks[key]).sum().item())
                layer_summaries.append(
                    {
                        "name": name,
                        "target_sparsity": target_sparsity,
                        "weight_count": total,
                        "masked_zero_count": zero_after,
                        "masked_sparsity": zero_after / total if total else 0.0,
                    }
                )
        apply_masks(modules, masks)

    return masks, {
        "target_sparsity": target_sparsity,
        "global_pruning": global_pruning,
        "layer_summaries": layer_summaries,
    }


def save_masks(path: Path, masks: dict[str, Any], metadata: dict[str, Any]) -> None:
    torch = import_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"masks": masks, "metadata": metadata}, path)


def save_model_checkpoint(model: Any, tokenizer: Any, checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)


def progressive_schedule(target_sparsity: float) -> list[float]:
    if abs(target_sparsity - 0.3) <= 1e-9:
        return [0.1, 0.2, 0.3]
    if abs(target_sparsity - 0.5) <= 1e-9:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    stages: list[float] = []
    current = 0.1
    while current < target_sparsity:
        stages.append(round(current, 4))
        current += 0.1
    if not stages or abs(stages[-1] - target_sparsity) > 1e-9:
        stages.append(target_sparsity)
    return stages


def load_model_and_tokenizer(args: argparse.Namespace, device: Any) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("transformers is required for model loading.") from exc
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=False, **load_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = dict(load_kwargs)
    try:
        torch = import_torch()
        if device.type == "cuda":
            if args.bf16:
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif args.fp16:
                model_kwargs["torch_dtype"] = torch.float16
    except Exception:
        pass
    if args.model_family == "encoder_decoder":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint, **model_kwargs)
    elif args.model_family == "decoder_only":
        model = AutoModelForCausalLM.from_pretrained(args.model_checkpoint, **model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_checkpoint, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.to(device)
    return tokenizer, model


def prepare_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "num_return_sequences": max(args.num_return_sequences, 5),
        "do_sample": False,
        "early_stopping": args.early_stopping,
    }
    if args.length_penalty is not None:
        kwargs["length_penalty"] = args.length_penalty
    return kwargs


def generate_predictions(model: Any, tokenizer: Any, examples: list[EvalExample], args: argparse.Namespace, device: Any) -> list[dict[str, Any]]:
    torch = import_torch()
    model.eval()
    rows: list[dict[str, Any]] = []
    generation_kwargs = prepare_generation_kwargs(args)
    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            prompts = [example.prompt for example in batch]
            if args.model_family == "encoder_only":
                predictions = encoder_only_predictions(model, tokenizer, prompts, args, device)
            elif args.model_family == "encoder_decoder":
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_len,
                )
                encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
                generated = model.generate(**encoded, **generation_kwargs)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                predictions = []
                n = generation_kwargs["num_return_sequences"]
                for offset in range(len(batch)):
                    predictions.append(decoded[offset * n : (offset + 1) * n])
            else:
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_len,
                )
                encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
                input_lengths = encoded.get("attention_mask").sum(dim=1).detach().cpu().tolist()
                generated = model.generate(**encoded, **generation_kwargs)
                n = generation_kwargs["num_return_sequences"]
                predictions = []
                for offset in range(len(batch)):
                    item_predictions: list[str] = []
                    for candidate_index in range(n):
                        generated_index = offset * n + candidate_index
                        prompt_len = int(input_lengths[offset])
                        new_tokens = generated[generated_index][prompt_len:]
                        item_predictions.append(
                            tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        )
                    predictions.append(item_predictions)
            for example, top_predictions in zip(batch, predictions):
                em1, em5 = compute_em_for_predictions(top_predictions, example.target, args.normalization_mode)
                rows.append(
                    {
                        "sample_id": example.sample_id,
                        "input": example.prompt,
                        "target": example.target,
                        "difficulty": example.difficulty,
                        "top1_prediction": top_predictions[0] if top_predictions else "",
                        "top5_predictions": json.dumps(top_predictions[:5], ensure_ascii=False),
                        "em1": int(em1),
                        "em5": int(em5),
                    }
                )
    return rows


def encoder_only_predictions(model: Any, tokenizer: Any, prompts: list[str], args: argparse.Namespace, device: Any) -> list[list[str]]:
    torch = import_torch()
    id2label = getattr(model.config, "id2label", None)
    if not id2label:
        raise ValueError(
            "encoder_only evaluation requires model.config.id2label so logits can map to canonical responses."
        )
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_input_len)
    encoded = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
    logits = model(**encoded).logits
    topk = torch.topk(logits, k=min(5, logits.shape[-1]), dim=-1).indices.detach().cpu().tolist()
    return [[str(id2label[int(index)]) for index in row] for row in topk]


def training_batches(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[start : start + batch_size] for start in range(0, len(records), batch_size)]


def make_encoder_decoder_training_tensors(
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    args: argparse.Namespace,
    device: Any,
) -> tuple[dict[str, Any], Any]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
    )
    try:
        labels = tokenizer(
            text_target=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_target_len,
        )["input_ids"]
    except TypeError:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_target_len,
            )["input_ids"]
    labels = labels.to(device)
    if tokenizer.pad_token_id is not None:
        labels = labels.clone()
        labels[labels == tokenizer.pad_token_id] = -100
    inputs = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
    return inputs, labels


def make_decoder_only_training_tensors(
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    args: argparse.Namespace,
    device: Any,
) -> tuple[dict[str, Any], Any]:
    torch = import_torch()
    separator = tokenizer.eos_token or "\n"
    full_texts = [f"{prompt}{separator}{target}" for prompt, target in zip(prompts, targets)]
    prompt_texts = [f"{prompt}{separator}" for prompt in prompts]
    encoded = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len + args.max_target_len,
    )
    prompt_encoded = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len + args.max_target_len,
    )
    labels = encoded["input_ids"].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    prompt_lengths = prompt_encoded["attention_mask"].sum(dim=1)
    for row, prompt_length in enumerate(prompt_lengths):
        labels[row, : int(prompt_length.item())] = -100
    inputs = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
    return inputs, labels.to(device)


def make_encoder_only_training_tensors(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    targets: list[str],
    args: argparse.Namespace,
    device: Any,
) -> tuple[dict[str, Any], Any]:
    torch = import_torch()
    label2id = getattr(model.config, "label2id", None) or {}
    if not label2id:
        id2label = getattr(model.config, "id2label", None) or {}
        label2id = {str(label): int(index) for index, label in id2label.items()}
    labels: list[int] = []
    for target in targets:
        if target not in label2id:
            raise ValueError(
                "encoder_only recovery fine-tuning requires target responses to exist in model.config.label2id."
            )
        labels.append(int(label2id[target]))
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_len,
    )
    inputs = {key: value.to(device) for key, value in encoded.items() if key != "token_type_ids"}
    return inputs, torch.tensor(labels, dtype=torch.long, device=device)


def recovery_finetune(
    model: Any,
    tokenizer: Any,
    train_records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: Any,
    modules: list[tuple[str, Any]],
    masks: dict[str, Any],
    epochs: int,
) -> float:
    torch = import_torch()
    if epochs <= 0:
        return 0.0
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    losses: list[float] = []
    batches = training_batches(train_records, args.batch_size)
    for _epoch in range(epochs):
        random.shuffle(batches)
        for batch_records in batches:
            prompts: list[str] = []
            targets: list[str] = []
            for record in batch_records:
                prompt, target = extract_prompt_response(record)
                prompts.append(prompt)
                targets.append(target)
            if args.model_family == "encoder_decoder":
                inputs, labels = make_encoder_decoder_training_tensors(tokenizer, prompts, targets, args, device)
                outputs = model(**inputs, labels=labels)
            elif args.model_family == "decoder_only":
                inputs, labels = make_decoder_only_training_tensors(tokenizer, prompts, targets, args, device)
                outputs = model(**inputs, labels=labels)
            else:
                inputs, labels = make_encoder_only_training_tensors(model, tokenizer, prompts, targets, args, device)
                outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if not args.regrowth:
                apply_masks(modules, masks)
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses) if losses else 0.0


def evaluate_model(model: Any, tokenizer: Any, examples: list[EvalExample], args: argparse.Namespace, device: Any) -> EvalResult:
    rows = generate_predictions(model, tokenizer, examples, args, device)
    metrics = summarize_prediction_rows(rows, args.bootstrap_resamples, args.seed)
    return EvalResult(rows=rows, metrics=metrics)


def filename_sparsity(sparsity: float) -> str:
    return str(sparsity).replace(".", "p")


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    fieldnames = [
        "sample_id",
        "input",
        "target",
        "difficulty",
        "top1_prediction",
        "top5_predictions",
        "em1",
        "em5",
        "model_family",
        "pruning_mode",
        "pruning_method",
        "target_sparsity",
        "targeted_linear_sparsity_actual",
        "whole_model_sparsity_actual",
        "seed",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, metadata.get(key, "")) for key in fieldnames})


def append_metadata_to_rows(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**row, **metadata} for row in rows]


SUMMARY_COLUMNS = [
    "experiment_name",
    "model_family",
    "pruning_mode",
    "pruning_method",
    "target_sparsity",
    "targeted_linear_sparsity_actual",
    "whole_model_sparsity_actual",
    "seed",
    "em1_overall",
    "em1_overall_ci_low",
    "em1_overall_ci_high",
    "em5_overall",
    "em5_overall_ci_low",
    "em5_overall_ci_high",
    "em1_easy",
    "em1_easy_ci_low",
    "em1_easy_ci_high",
    "em5_easy",
    "em5_easy_ci_low",
    "em5_easy_ci_high",
    "count_easy",
    "em1_medium",
    "em1_medium_ci_low",
    "em1_medium_ci_high",
    "em5_medium",
    "em5_medium_ci_low",
    "em5_medium_ci_high",
    "count_medium",
    "em1_hard",
    "em1_hard_ci_low",
    "em1_hard_ci_high",
    "em5_hard",
    "em5_hard_ci_low",
    "em5_hard_ci_high",
    "count_hard",
    "count_total",
    "em1_retention_overall",
    "em5_retention_overall",
    "em1_retention_easy",
    "em5_retention_easy",
    "em1_retention_medium",
    "em5_retention_medium",
    "em1_retention_hard",
    "em5_retention_hard",
    "decoding_config_json",
    "training_config_json",
    "pruning_config_json",
    "checkpoint_path",
    "mask_path",
    "prediction_path",
    "progressive_log_path",
]


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_paper_table(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "model_family",
        "pruning_mode",
        "target_sparsity",
        "overall EM@1",
        "overall EM@5",
        "easy EM@1",
        "easy EM@5",
        "medium EM@1",
        "medium EM@5",
        "hard EM@1",
        "hard EM@5",
        "targeted linear sparsity",
        "whole-model sparsity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_family": row["model_family"],
                    "pruning_mode": row["pruning_mode"],
                    "target_sparsity": row["target_sparsity"],
                    "overall EM@1": row["em1_overall"],
                    "overall EM@5": row["em5_overall"],
                    "easy EM@1": row["em1_easy"],
                    "easy EM@5": row["em5_easy"],
                    "medium EM@1": row["em1_medium"],
                    "medium EM@5": row["em5_medium"],
                    "hard EM@1": row["em1_hard"],
                    "hard EM@5": row["em5_hard"],
                    "targeted linear sparsity": row["targeted_linear_sparsity_actual"],
                    "whole-model sparsity": row["whole_model_sparsity_actual"],
                }
            )


def config_jsons(args: argparse.Namespace) -> tuple[str, str, str]:
    decoding = {
        "num_beams": args.num_beams,
        "num_return_sequences": max(args.num_return_sequences, 5),
        "max_new_tokens": args.max_new_tokens,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
        "normalization_mode": args.normalization_mode,
    }
    training = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "recovery_epochs_per_stage": args.recovery_epochs_per_stage,
        "final_recovery_epochs": args.final_recovery_epochs,
        "seed": args.seed,
        "max_grad_norm": args.max_grad_norm,
        "train_path": args.train_path,
    }
    pruning = {
        "prune_scope": args.prune_scope,
        "prune_method": args.prune_method,
        "prune_output_heads": args.prune_output_heads,
        "global_pruning": args.global_pruning,
        "regrowth": args.regrowth,
        "progressive_schedule": args.progressive_schedule,
    }
    return (
        json.dumps(decoding, ensure_ascii=False, sort_keys=True),
        json.dumps(training, ensure_ascii=False, sort_keys=True),
        json.dumps(pruning, ensure_ascii=False, sort_keys=True),
    )


def build_run_plan(args: argparse.Namespace) -> list[tuple[str, float]]:
    modes = args.pruning_mode if args.pruning_mode is not None else args.pruning_modes
    levels = sorted({round(float(level), 6) for level in args.sparsity_levels})
    plan: list[tuple[str, float]] = []
    if "dense" in modes:
        plan.append(("dense", 0.0))
    for mode in modes:
        if mode == "dense":
            continue
        for level in levels:
            if level <= 0:
                continue
            plan.append((mode, level))
    return plan


def evaluate_validation_if_available(
    model: Any,
    tokenizer: Any,
    validation_examples: list[EvalExample] | None,
    args: argparse.Namespace,
    device: Any,
) -> tuple[Any, Any]:
    if not validation_examples:
        return "N/A", "N/A"
    result = evaluate_model(model, tokenizer, validation_examples, args, device)
    return result.metrics["em1_overall"], result.metrics["em5_overall"]


def run_single_experiment(
    args: argparse.Namespace,
    mode: str,
    target_sparsity: float,
    benchmark_examples: list[EvalExample],
    train_records: list[dict[str, Any]],
    validation_examples: list[EvalExample] | None,
    dense_summaries: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    torch = import_torch()
    device = resolve_device(args.device)
    tokenizer, model = load_model_and_tokenizer(args, device)
    modules = collect_prunable_linear_modules(model, args.prune_scope, args.prune_output_heads)
    if not modules:
        raise ValueError("No prunable torch.nn.Linear.weight modules found for this pruning scope.")
    masks = make_initial_masks(modules)
    pruning_summary: dict[str, Any] = {"mode": mode, "target_sparsity": target_sparsity}
    mask_path = ""
    checkpoint_path = str(args.model_checkpoint)
    prediction_path = output_dir / f"predictions_{args.model_family}_{mode}_{filename_sparsity(target_sparsity)}_{args.seed}.csv"
    progressive_log_path = ""
    progressive_log_rows: list[dict[str, Any]] = []

    if mode == "oneshot":
        masks, pruning_summary = apply_magnitude_masks(modules, target_sparsity, masks, args.global_pruning)
        checkpoint_dir = output_dir / "checkpoints" / f"{args.model_family}_{mode}_{filename_sparsity(target_sparsity)}_seed{args.seed}"
        mask_path = str(checkpoint_dir / "linear_weight_masks.pt")
        targeted, whole = sparsity_summary(model, modules)
        save_masks(Path(mask_path), masks, {"target_sparsity": target_sparsity, "mode": mode, "summary": pruning_summary})
        save_model_checkpoint(model, tokenizer, checkpoint_dir)
        checkpoint_path = str(checkpoint_dir)
    elif mode == "progressive":
        stages = progressive_schedule(target_sparsity)
        for stage_index, stage_sparsity in enumerate(stages, start=1):
            masks, pruning_summary = apply_magnitude_masks(modules, stage_sparsity, masks, args.global_pruning)
            if args.recovery_epochs_per_stage == 0:
                targeted, whole = sparsity_summary(model, modules)
                val_em1, val_em5 = evaluate_validation_if_available(model, tokenizer, validation_examples, args, device)
                progressive_log_rows.append(
                    {
                        "stage": stage_index,
                        "stage_target_sparsity": stage_sparsity,
                        "targeted_linear_sparsity_actual": targeted,
                        "whole_model_sparsity_actual": whole,
                        "recovery_epoch": 0,
                        "train_loss": "N/A",
                        "val_em1": val_em1,
                        "val_em5": val_em5,
                    }
                )
            for recovery_epoch in range(1, args.recovery_epochs_per_stage + 1):
                train_loss = recovery_finetune(
                    model, tokenizer, train_records, args, device, modules, masks, epochs=1
                )
                targeted, whole = sparsity_summary(model, modules)
                val_em1, val_em5 = evaluate_validation_if_available(model, tokenizer, validation_examples, args, device)
                progressive_log_rows.append(
                    {
                        "stage": stage_index,
                        "stage_target_sparsity": stage_sparsity,
                        "targeted_linear_sparsity_actual": targeted,
                        "whole_model_sparsity_actual": whole,
                        "recovery_epoch": recovery_epoch,
                        "train_loss": train_loss,
                        "val_em1": val_em1,
                        "val_em5": val_em5,
                    }
                )
        for final_epoch in range(1, args.final_recovery_epochs + 1):
            train_loss = recovery_finetune(model, tokenizer, train_records, args, device, modules, masks, epochs=1)
            targeted, whole = sparsity_summary(model, modules)
            val_em1, val_em5 = evaluate_validation_if_available(model, tokenizer, validation_examples, args, device)
            progressive_log_rows.append(
                {
                    "stage": "final",
                    "stage_target_sparsity": target_sparsity,
                    "targeted_linear_sparsity_actual": targeted,
                    "whole_model_sparsity_actual": whole,
                    "recovery_epoch": final_epoch,
                    "train_loss": train_loss,
                    "val_em1": val_em1,
                    "val_em5": val_em5,
                }
            )
        checkpoint_dir = output_dir / "checkpoints" / f"{args.model_family}_{mode}_{filename_sparsity(target_sparsity)}_seed{args.seed}"
        mask_path = str(checkpoint_dir / "linear_weight_masks.pt")
        save_masks(Path(mask_path), masks, {"target_sparsity": target_sparsity, "mode": mode, "summary": pruning_summary})
        save_model_checkpoint(model, tokenizer, checkpoint_dir)
        checkpoint_path = str(checkpoint_dir)
        progressive_log_path = str(
            output_dir / f"progressive_logs_{args.model_family}_{filename_sparsity(target_sparsity)}_{args.seed}.csv"
        )
        write_progressive_log(Path(progressive_log_path), progressive_log_rows)
    else:
        apply_masks(modules, masks)

    targeted, whole = sparsity_summary(model, modules)
    eval_result = evaluate_model(model, tokenizer, benchmark_examples, args, device)
    decoding_json, training_json, pruning_json = config_jsons(args)
    dense_key = args.model_family
    summary = {
        "experiment_name": args.experiment_name,
        "model_family": args.model_family,
        "pruning_mode": mode,
        "pruning_method": args.prune_method if mode != "dense" else "none",
        "target_sparsity": target_sparsity,
        "targeted_linear_sparsity_actual": targeted,
        "whole_model_sparsity_actual": whole,
        "seed": args.seed,
        **eval_result.metrics,
        "decoding_config_json": decoding_json,
        "training_config_json": training_json,
        "pruning_config_json": pruning_json,
        "checkpoint_path": checkpoint_path,
        "mask_path": mask_path,
        "prediction_path": str(prediction_path),
        "progressive_log_path": progressive_log_path,
    }
    if mode == "dense" and abs(target_sparsity) <= 1e-9:
        dense_summaries[dense_key] = summary
    compute_retention(summary, dense_summaries.get(dense_key))
    row_metadata = {
        "model_family": args.model_family,
        "pruning_mode": mode,
        "pruning_method": summary["pruning_method"],
        "target_sparsity": target_sparsity,
        "targeted_linear_sparsity_actual": targeted,
        "whole_model_sparsity_actual": whole,
        "seed": args.seed,
    }
    prediction_rows = append_metadata_to_rows(eval_result.rows, row_metadata)
    write_prediction_csv(prediction_path, prediction_rows, row_metadata)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def write_progressive_log(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "stage",
        "stage_target_sparsity",
        "targeted_linear_sparsity_actual",
        "whole_model_sparsity_actual",
        "recovery_epoch",
        "train_loss",
        "val_em1",
        "val_em5",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def copy_source_config(args: argparse.Namespace, output_dir: Path) -> None:
    write_json(output_dir / "experiment_config.json", vars(args))


def main() -> None:
    args = parse_args()
    if args.num_return_sequences < 5:
        print("num_return_sequences < 5; using 5 for EM@5 candidate generation.", file=sys.stderr)
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_source_config(args, output_dir)

    benchmark_records = read_records(Path(args.benchmark_path))
    if args.max_benchmark_examples is not None:
        benchmark_records = benchmark_records[: args.max_benchmark_examples]
    benchmark_examples = build_examples(
        benchmark_records,
        Path(args.benchmark_difficulty_path) if args.benchmark_difficulty_path else None,
    )
    train_records = read_records(Path(args.train_path))
    if args.max_train_examples is not None:
        train_records = train_records[: args.max_train_examples]
    validation_examples = None
    if args.validation_path:
        validation_records = read_records(Path(args.validation_path))
        if args.max_validation_examples is not None:
            validation_records = validation_records[: args.max_validation_examples]
        validation_examples = build_examples(
            validation_records,
            Path(args.benchmark_difficulty_path) if args.benchmark_difficulty_path else None,
        )

    summaries: list[dict[str, Any]] = []
    dense_summaries: dict[str, dict[str, Any]] = {}
    for mode, sparsity in build_run_plan(args):
        print(f"Running {args.model_family} {mode} sparsity={sparsity} seed={args.seed}", flush=True)
        summary = run_single_experiment(
            args,
            mode,
            sparsity,
            benchmark_examples,
            train_records,
            validation_examples,
            dense_summaries,
            output_dir,
        )
        summaries.append(summary)
        write_summary_csv(output_dir / "summary_metrics.csv", summaries)
        write_paper_table(output_dir / "paper_table_sparsity_difficulty.csv", summaries)

    print("\nSCENIC linear sparsity experiment summary")
    for summary in summaries:
        print(
            f"{summary['model_family']} {summary['pruning_mode']} {summary['target_sparsity']}: "
            f"EM@1={summary['em1_overall']:.4f}, EM@5={summary['em5_overall']:.4f}, "
            f"targeted_sparsity={summary['targeted_linear_sparsity_actual']:.4f}, "
            f"whole_model_sparsity={summary['whole_model_sparsity_actual']:.4f}"
        )
    print(f"\nWrote results to: {output_dir}")


if __name__ == "__main__":
    main()
