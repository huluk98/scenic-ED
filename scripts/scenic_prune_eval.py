#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import (  # noqa: E402
    append_eos_to_targets,
    repair_checkpoint_for_auto_load,
    repair_tokenizer_files_for_auto_load,
    sanitize_model_for_save,
    save_tokenizer_for_auto_load,
)


DEFAULT_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
DEFAULT_BENCHMARK_JSON = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "prune_eval_outputs"

PROMPT_FIELDS = ("prompt", "anchor", "instruction", "question", "input", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")


@dataclass
class DistributedState:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class CalibrationDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        max_input_len: int,
        max_target_len: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prompt, response = extract_prompt_response(self.records[index])
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
        )
        labels, label_attention_mask = tokenize_targets(
            self.tokenizer,
            [response],
            max_length=self.max_target_len,
        )
        labels = mask_pad_tokens(labels, self.tokenizer.pad_token_id, label_attention_mask)
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a SCENIC SFT checkpoint, prune it at 50 percent sparsity, "
            "evaluate again, and write one JSON report."
        )
    )
    parser.add_argument("--model", required=True, help="Regular SFT or contrastive SFT model path.")
    parser.add_argument("--method", choices=("magnitude", "gradient", "wanda", "nvidia"), default="magnitude")
    parser.add_argument("--sparsity", type=float, default=0.5, help="Unstructured sparsity for magnitude/gradient/WANDA.")
    parser.add_argument("--train-json", default=str(DEFAULT_TRAIN_JSON))
    parser.add_argument("--benchmark-json", default=str(DEFAULT_BENCHMARK_JSON))
    parser.add_argument("--calibration-json", default=None, help="Defaults to --train-json.")
    parser.add_argument("--output-json", default=None, help="Single JSON report path.")
    parser.add_argument("--pruned-output-dir", default=None, help="Directory where the pruned model is saved.")
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--calibration-batch-size", type=int, default=4)
    parser.add_argument("--calibration-batches", type=int, default=64)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-benchmark-examples", type=int, default=None)
    parser.add_argument("--max-calibration-examples", type=int, default=None)
    parser.add_argument("--max-input-len", type=int, default=256)
    parser.add_argument("--max-target-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--ignore-spaces", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps for single-process runs.")
    parser.add_argument("--include-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and write a plan without loading a model.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = Path(value).name or "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    sparsity_tag = int(round(args.sparsity * 100))
    default_dir = DEFAULT_OUTPUT_ROOT / f"{safe_name(args.model)}_{args.method}_{sparsity_tag}"
    output_json = Path(args.output_json).expanduser() if args.output_json else default_dir / "prune_eval_report.json"
    pruned_output_dir = (
        Path(args.pruned_output_dir).expanduser() if args.pruned_output_dir else output_json.parent / "pruned_model"
    )
    return output_json, pruned_output_dir


def setup_distributed(device_arg: str) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed prune/eval requires CUDA.")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistributedState(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
        )

    if device_arg != "auto":
        device = torch.device(device_arg)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return DistributedState(enabled=False, rank=0, local_rank=0, world_size=1, device=device)


def cleanup_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def sync_distributed(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier()


def release_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def rank0_print(state: DistributedState, message: str) -> None:
    if state.is_main:
        print(message, flush=True)


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain one JSON object.")
                records.append(value)
        return records

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        for key in ("records", "data", "examples", "items"):
            if isinstance(value.get(key), list):
                records = value[key]
                break
        else:
            records = [value]
    else:
        raise ValueError(f"{path} must contain a JSON list, object, or JSONL rows.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains at least one non-object row.")
    return list(records)


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = str(record.get(field, "")).strip()
        if value:
            return value
    return ""


def extract_prompt_response(record: dict[str, Any]) -> tuple[str, str]:
    prompt = first_text(record, PROMPT_FIELDS)
    response = first_text(record, RESPONSE_FIELDS)
    if not prompt or not response:
        raise ValueError(f"Record is missing prompt/response fields: {sorted(record)}")
    return prompt, response


def normalize_text(text: str, ignore_spaces: bool) -> str:
    text = str(text).strip()
    if ignore_spaces:
        text = "".join(text.split())
    return text


def truncate_records(records: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return records
    return records[:limit]


def tokenize_targets(tokenizer: Any, targets: list[str], max_length: int) -> tuple[Any, Any | None]:
    try:
        labels = tokenizer(
            text_target=targets,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
    return append_eos_to_targets(
        labels["input_ids"],
        labels.get("attention_mask"),
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        max_length,
    )


def mask_pad_tokens(labels: Any, pad_token_id: int | None, attention_mask: Any | None = None) -> Any:
    if attention_mask is not None:
        labels = labels.clone()
        labels[attention_mask == 0] = -100
        return labels
    if pad_token_id is None:
        return labels
    labels = labels.clone()
    labels[labels == pad_token_id] = -100
    return labels


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def load_model_and_tokenizer(args: argparse.Namespace, model_path: str, state: DistributedState) -> tuple[Any, Any]:
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    tokenizer_kwargs = {**load_kwargs, "use_fast": False}
    model_kwargs = dict(load_kwargs)
    if state.device.type == "cuda":
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

    checkpoint_dir = Path(model_path).expanduser()
    if checkpoint_dir.is_dir():
        repair_tokenizer_files_for_auto_load(checkpoint_dir)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    except ValueError as exc:
        if "tokenizersbackend" not in str(exc).replace("_", "").replace("-", "").lower():
            raise
        if checkpoint_dir.is_dir():
            repair_tokenizer_files_for_auto_load(checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.to(state.device)
    return tokenizer, model


def iter_linear_modules(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def summarize_model(model: nn.Module, model_path: str) -> dict[str, Any]:
    total_params = 0
    trainable_params = 0
    linear_total = 0
    linear_zero = 0
    linear_layers = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    for _, module in iter_linear_modules(model):
        linear_layers += 1
        weight = module.weight.detach()
        linear_total += weight.numel()
        linear_zero += int(weight.eq(0).sum().item())
    config = getattr(model, "config", None)
    return {
        "model_path": model_path,
        "architectures": getattr(config, "architectures", None),
        "model_type": getattr(config, "model_type", None),
        "parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "linear_layer_count": linear_layers,
        "linear_weight_count": linear_total,
        "linear_zero_weight_count": linear_zero,
        "linear_sparsity": linear_zero / linear_total if linear_total else 0.0,
    }


@torch.no_grad()
def evaluate_local_records(
    model: nn.Module,
    tokenizer: Any,
    indexed_records: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
    state: DistributedState,
    dataset_name: str,
) -> dict[str, Any]:
    model.eval()
    outputs: list[dict[str, Any]] = []
    em1_correct = 0
    em5_correct = 0
    total = 0

    progress = tqdm(
        range(0, len(indexed_records), args.eval_batch_size),
        desc=f"{dataset_name} eval",
        disable=not state.is_main,
    )
    for start in progress:
        batch = indexed_records[start : start + args.eval_batch_size]
        prompts: list[str] = []
        targets: list[str] = []
        indices: list[int] = []
        for index, record in batch:
            prompt, target = extract_prompt_response(record)
            indices.append(index)
            prompts.append(prompt)
            targets.append(target)

        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_len,
        )
        encoded = {key: value.to(state.device) for key, value in encoded.items() if key != "token_type_ids"}
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            do_sample=False,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        for offset, prompt in enumerate(prompts):
            predictions = decoded[
                offset * args.num_return_sequences : (offset + 1) * args.num_return_sequences
            ]
            gold = normalize_text(targets[offset], args.ignore_spaces)
            normalized_predictions = [normalize_text(prediction, args.ignore_spaces) for prediction in predictions]
            em1 = bool(normalized_predictions and normalized_predictions[0] == gold)
            em5 = gold in normalized_predictions
            em1_correct += int(em1)
            em5_correct += int(em5)
            total += 1
            if args.include_predictions:
                outputs.append(
                    {
                        "index": indices[offset],
                        "prompt": prompt,
                        "target": targets[offset],
                        "em1_prediction": predictions[0] if predictions else "",
                        "em5_predictions": predictions,
                        "em1_correct": em1,
                        "em5_correct": em5,
                    }
                )

    return {
        "total": total,
        "em1_correct": em1_correct,
        "em5_correct": em5_correct,
        "outputs": outputs,
    }


def merge_eval_results(local_result: dict[str, Any], state: DistributedState) -> dict[str, Any] | None:
    if not state.enabled:
        return finalize_eval_result(local_result)

    gathered: list[dict[str, Any] | None] = [None for _ in range(state.world_size)]
    dist.all_gather_object(gathered, local_result)
    if not state.is_main:
        return None

    merged = {
        "total": 0,
        "em1_correct": 0,
        "em5_correct": 0,
        "outputs": [],
    }
    for result in gathered:
        if result is None:
            continue
        merged["total"] += int(result["total"])
        merged["em1_correct"] += int(result["em1_correct"])
        merged["em5_correct"] += int(result["em5_correct"])
        merged["outputs"].extend(result.get("outputs", []))
    merged["outputs"].sort(key=lambda item: item["index"])
    return finalize_eval_result(merged)


def finalize_eval_result(result: dict[str, Any]) -> dict[str, Any]:
    total = int(result["total"])
    em1 = result["em1_correct"] / total if total else 0.0
    em5 = result["em5_correct"] / total if total else 0.0
    return {
        "total": total,
        "em1_correct": int(result["em1_correct"]),
        "em5_correct": int(result["em5_correct"]),
        "em1": em1,
        "em5": em5,
        "em1_percent": em1 * 100.0,
        "em5_percent": em5 * 100.0,
        "accuracy": em1,
        "accuracy_percent": em1 * 100.0,
        "accuracy_definition": "accuracy is exact-match@1 / EM@1",
        "outputs": result.get("outputs", []),
    }


def evaluate_dataset(
    model: nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    state: DistributedState,
    dataset_name: str,
) -> dict[str, Any] | None:
    indexed = list(enumerate(records))
    local_indexed = indexed[state.rank :: state.world_size]
    local_result = evaluate_local_records(model, tokenizer, local_indexed, args, state, dataset_name)
    return merge_eval_results(local_result, state)


def prune_by_score(weight: torch.Tensor, score: torch.Tensor, sparsity: float, per_row: bool) -> None:
    with torch.no_grad():
        if per_row:
            keep = int(weight.shape[1] * (1.0 - sparsity))
            if keep <= 0:
                weight.zero_()
                return
            if keep >= weight.shape[1]:
                return
            topk = torch.topk(score.float(), keep, dim=1, largest=True).indices
            mask = torch.zeros_like(weight, dtype=torch.bool)
            mask.scatter_(1, topk, True)
            weight.mul_(mask)
            return

        flat_score = score.float().reshape(-1)
        keep = int(flat_score.numel() * (1.0 - sparsity))
        if keep <= 0:
            weight.zero_()
            return
        if keep >= flat_score.numel():
            return
        threshold = torch.topk(flat_score, keep, largest=True).values.min()
        weight.mul_(score >= threshold)


def magnitude_prune(model: nn.Module, sparsity: float) -> dict[str, Any]:
    pruned_layers = 0
    for _, module in iter_linear_modules(model):
        prune_by_score(module.weight.data, module.weight.data.abs(), sparsity=sparsity, per_row=False)
        pruned_layers += 1
    return {"method": "magnitude", "sparsity": sparsity, "pruned_linear_layers": pruned_layers}


def nvidia_2_4_prune(model: nn.Module) -> dict[str, Any]:
    pruned_layers = 0
    skipped_layers = 0
    with torch.no_grad():
        for _, module in iter_linear_modules(model):
            weight = module.weight.data
            if weight.shape[1] % 4 != 0:
                skipped_layers += 1
                continue
            grouped = weight.view(weight.shape[0], -1, 4)
            _, prune_indices = torch.topk(grouped.abs(), 2, dim=2, largest=False)
            mask = torch.ones_like(grouped, dtype=torch.bool)
            mask.scatter_(2, prune_indices, False)
            weight.copy_((grouped * mask).view_as(weight))
            pruned_layers += 1
    return {
        "method": "nvidia",
        "pattern": "2:4",
        "effective_sparsity": 0.5,
        "pruned_linear_layers": pruned_layers,
        "skipped_linear_layers": skipped_layers,
    }


def make_calibration_loader(
    records: list[dict[str, Any]],
    tokenizer: Any,
    args: argparse.Namespace,
) -> DataLoader:
    dataset = CalibrationDataset(
        records=records,
        tokenizer=tokenizer,
        max_input_len=args.max_input_len,
        max_target_len=args.max_target_len,
    )
    return DataLoader(dataset, batch_size=args.calibration_batch_size, shuffle=False)


def gradient_prune(
    model: nn.Module,
    tokenizer: Any,
    calibration_records: list[dict[str, Any]],
    args: argparse.Namespace,
    state: DistributedState,
) -> dict[str, Any]:
    model.train()
    saliency = {
        module: torch.zeros_like(module.weight.data, dtype=torch.float32, device=module.weight.device)
        for _, module in iter_linear_modules(model)
    }
    loader = make_calibration_loader(calibration_records, tokenizer, args)
    used_batches = 0

    for batch_index, batch in enumerate(tqdm(loader, desc="gradient calibration", disable=not state.is_main)):
        if batch_index >= args.calibration_batches:
            break
        batch = move_batch_to_device(batch, state.device)
        model.zero_grad(set_to_none=True)
        outputs = model(**batch, return_dict=True)
        outputs.loss.backward()
        for module, score in saliency.items():
            if module.weight.grad is not None:
                score.add_((module.weight.detach().float() * module.weight.grad.detach().float()).abs())
        used_batches += 1

    model.zero_grad(set_to_none=True)
    for module, score in saliency.items():
        prune_by_score(module.weight.data, score, sparsity=args.sparsity, per_row=False)
    return {
        "method": "gradient",
        "sparsity": args.sparsity,
        "calibration_examples": min(len(calibration_records), used_batches * args.calibration_batch_size),
        "calibration_batches": used_batches,
        "pruned_linear_layers": len(saliency),
    }


def wanda_prune(
    model: nn.Module,
    tokenizer: Any,
    calibration_records: list[dict[str, Any]],
    args: argparse.Namespace,
    state: DistributedState,
) -> dict[str, Any]:
    activation_stats: dict[nn.Linear, dict[str, Any]] = {}
    module_names = {module: name for name, module in iter_linear_modules(model)}
    skipped_non_tensor: set[nn.Linear] = set()
    skipped_shape_mismatch: set[nn.Linear] = set()
    handles = []

    def hook(module: nn.Linear, inputs: tuple[Any, ...], _output: Any) -> None:
        if not inputs:
            return
        if not torch.is_tensor(inputs[0]):
            skipped_non_tensor.add(module)
            return
        activation = inputs[0].detach()
        if activation.numel() == 0:
            return
        if activation.shape[-1] != module.weight.shape[1]:
            skipped_shape_mismatch.add(module)
            return
        activation = activation.reshape(-1, activation.shape[-1]).float()
        stats = activation_stats.setdefault(
            module,
            {
                "sum_sq": torch.zeros(activation.shape[-1], device=activation.device, dtype=torch.float32),
                "count": 0,
            },
        )
        stats["sum_sq"].add_((activation * activation).sum(dim=0))
        stats["count"] += activation.shape[0]

    for _, module in iter_linear_modules(model):
        handles.append(module.register_forward_hook(hook))

    loader = make_calibration_loader(calibration_records, tokenizer, args)
    used_batches = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(tqdm(loader, desc="wanda calibration", disable=not state.is_main)):
                if batch_index >= args.calibration_batches:
                    break
                batch = move_batch_to_device(batch, state.device)
                model(**batch, return_dict=True)
                used_batches += 1
    finally:
        for handle in handles:
            handle.remove()

    skipped_prune_shape_mismatch: set[nn.Linear] = set()
    pruned_layers = 0
    for module, stats in activation_stats.items():
        count = max(1, int(stats["count"]))
        activation_norm = torch.sqrt(stats["sum_sq"] / count)
        activation_norm = torch.nan_to_num(activation_norm, nan=0.0, posinf=0.0, neginf=0.0)
        if activation_norm.numel() != module.weight.shape[1]:
            skipped_prune_shape_mismatch.add(module)
            continue
        activation_norm = activation_norm.to(module.weight.device)
        score = module.weight.data.abs().float() * activation_norm.unsqueeze(0)
        prune_by_score(module.weight.data, score, sparsity=args.sparsity, per_row=True)
        pruned_layers += 1

    skipped_modules = skipped_non_tensor | skipped_shape_mismatch | skipped_prune_shape_mismatch
    return {
        "method": "wanda",
        "sparsity": args.sparsity,
        "calibration_examples": min(len(calibration_records), used_batches * args.calibration_batch_size),
        "calibration_batches": used_batches,
        "pruned_linear_layers": pruned_layers,
        "skipped_linear_layers": len(skipped_modules),
        "skipped_non_tensor_activation_layers": sorted(module_names.get(module, "<unnamed>") for module in skipped_non_tensor),
        "skipped_shape_mismatch_layers": sorted(
            module_names.get(module, "<unnamed>") for module in (skipped_shape_mismatch | skipped_prune_shape_mismatch)
        ),
    }


def run_pruning(
    model: nn.Module,
    tokenizer: Any,
    calibration_records: list[dict[str, Any]],
    args: argparse.Namespace,
    state: DistributedState,
) -> dict[str, Any]:
    if args.method == "magnitude":
        return magnitude_prune(model, args.sparsity)
    if args.method == "gradient":
        return gradient_prune(model, tokenizer, calibration_records, args, state)
    if args.method == "wanda":
        return wanda_prune(model, tokenizer, calibration_records, args, state)
    if args.method == "nvidia":
        return nvidia_2_4_prune(model)
    raise ValueError(f"Unknown pruning method: {args.method}")


def save_pruned_model(model: nn.Module, tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_for_auto_load(tokenizer, output_dir)
    sanitize_model_for_save(model).save_pretrained(output_dir, safe_serialization=True)
    repair_checkpoint_for_auto_load(output_dir, tokenizer=tokenizer)


def evaluate_all(
    model: nn.Module,
    tokenizer: Any,
    datasets: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    state: DistributedState,
    label: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for dataset_name, records in datasets.items():
        rank0_print(state, f"Evaluating {label} on {dataset_name}: {len(records)} rows")
        result = evaluate_dataset(model, tokenizer, records, args, state, f"{label}/{dataset_name}")
        if state.is_main and result is not None:
            results[dataset_name] = result
    return results


def compact_metrics(results: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for dataset_name, result in results.items():
        compact[dataset_name] = {
            "total": result["total"],
            "em1": result["em1"],
            "em5": result["em5"],
            "em1_percent": result["em1_percent"],
            "em5_percent": result["em5_percent"],
            "accuracy": result["accuracy"],
            "accuracy_percent": result["accuracy_percent"],
        }
    return compact


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def make_plan_report(
    args: argparse.Namespace,
    output_json: Path,
    pruned_output_dir: Path,
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_model_path": args.model,
        "pruned_model_path": str(pruned_output_dir),
        "output_json": str(output_json),
        "pruning": {
            "method": args.method,
            "target_sparsity": args.sparsity,
        },
        "datasets": {name: {"total": len(records)} for name, records in datasets.items()},
    }


def run_pruning_on_rank0_and_sync(
    model: nn.Module,
    tokenizer: Any,
    calibration_records: list[dict[str, Any]],
    args: argparse.Namespace,
    state: DistributedState,
    pruned_output_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pruning_summary = None
    pruned_model_summary = None
    status: dict[str, Any] = {"ok": True, "error": None}

    if state.is_main:
        try:
            rank0_print(state, f"Running {args.method} pruning at target sparsity {args.sparsity:.2f}")
            pruning_summary = run_pruning(model, tokenizer, calibration_records, args, state)
            pruned_model_summary = summarize_model(model, str(pruned_output_dir))
            rank0_print(state, f"Saving pruned model to: {pruned_output_dir}")
            save_pruned_model(model, tokenizer, pruned_output_dir)
        except Exception:
            status = {"ok": False, "error": traceback.format_exc()}

    if state.enabled:
        status_list = [status]
        dist.broadcast_object_list(status_list, src=0)
        status = status_list[0]

    if not status["ok"]:
        raise RuntimeError(f"Rank 0 failed during {args.method} pruning:\n{status['error']}")

    return pruning_summary, pruned_model_summary


def main() -> None:
    args = parse_args()
    if not 0.0 < args.sparsity < 1.0:
        raise ValueError("--sparsity must be between 0 and 1.")
    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences cannot exceed --num-beams for beam search.")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    output_json, pruned_output_dir = resolve_output_paths(args)
    train_json = Path(args.train_json).expanduser()
    benchmark_json = Path(args.benchmark_json).expanduser()
    calibration_json = Path(args.calibration_json).expanduser() if args.calibration_json else train_json

    state = setup_distributed(args.device)
    try:
        train_records = truncate_records(read_records(train_json), args.max_train_examples)
        benchmark_records = truncate_records(read_records(benchmark_json), args.max_benchmark_examples)
        calibration_records = truncate_records(read_records(calibration_json), args.max_calibration_examples)
        datasets = {
            "benchmark": benchmark_records,
            "training": train_records,
        }

        if state.is_main:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            rank0_print(state, f"Input model: {args.model}")
            rank0_print(state, f"Pruned model will be saved to: {pruned_output_dir}")
            rank0_print(state, f"Single JSON report: {output_json}")

        if args.dry_run:
            if state.is_main:
                write_json(output_json, make_plan_report(args, output_json, pruned_output_dir, datasets))
            return

        tokenizer, model = load_model_and_tokenizer(args, args.model, state)
        original_model_summary = summarize_model(model, args.model) if state.is_main else None
        original_results = evaluate_all(model, tokenizer, datasets, args, state, label="original")

        sync_distributed(state)
        pruning_summary, pruned_model_summary = run_pruning_on_rank0_and_sync(
            model,
            tokenizer,
            calibration_records,
            args,
            state,
            pruned_output_dir,
        )

        del model
        release_cuda_memory()
        sync_distributed(state)

        pruned_tokenizer, pruned_model = load_model_and_tokenizer(args, str(pruned_output_dir), state)
        pruned_results = evaluate_all(pruned_model, pruned_tokenizer, datasets, args, state, label="pruned")
        del pruned_model
        release_cuda_memory()

        if state.is_main:
            report = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "input_model_path": args.model,
                "pruned_model_path": str(pruned_output_dir),
                "output_json": str(output_json),
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
                "summary": {
                    "accuracy_definition": "accuracy is exact-match@1 / EM@1",
                    "original_before_prune": compact_metrics(original_results),
                    "pruned_after_50_percent": compact_metrics(pruned_results),
                },
                "datasets": {
                    "benchmark": {"path": str(benchmark_json), "total": len(benchmark_records)},
                    "training": {"path": str(train_json), "total": len(train_records)},
                    "calibration": {"path": str(calibration_json), "total": len(calibration_records)},
                },
                "original_before_prune": {
                    "model": original_model_summary,
                    "evaluations": original_results,
                },
                "pruning": pruning_summary,
                "pruned_after_50_percent": {
                    "model": pruned_model_summary,
                    "evaluations": pruned_results,
                },
            }
            write_json(output_json, report)
            rank0_print(state, f"Wrote combined prune/eval report: {output_json}")
    finally:
        cleanup_distributed(state)


if __name__ == "__main__":
    main()
