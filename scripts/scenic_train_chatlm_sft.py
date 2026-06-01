#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults directly if you prefer running the file without CLI flags.
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"
REGULAR_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
CONTRASTIVE_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_anchor_positive_negative.json"
REGULAR_OUTPUT_DIR = PROJECT_ROOT / "models" / "chatlm_scenic_regular_sft"
CONTRASTIVE_OUTPUT_DIR = PROJECT_ROOT / "models" / "chatlm_scenic_triplet_sft"

PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "anchor", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")
POSITIVE_FIELDS = ("positive", "pos", "x_positive", "x_plus", "chosen")
NEGATIVE_FIELDS = ("negative", "neg", "x_negative", "x_minus", "rejected")


@dataclass
class DistributedState:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class RegularSFTConfig:
    model_name_or_path: str = MODEL_NAME_OR_PATH
    train_json: Path = REGULAR_TRAIN_JSON
    output_dir: Path = REGULAR_OUTPUT_DIR
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_source_length: int = 128
    max_target_length: int = 96
    max_grad_norm: float = 1.0
    seed: int = 42
    max_examples: int | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False
    fp16: bool = False
    bf16: bool = False
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 20
    save_every_steps: int = 0


@dataclass
class ContrastiveSFTConfig(RegularSFTConfig):
    train_json: Path = CONTRASTIVE_TRAIN_JSON
    output_dir: Path = CONTRASTIVE_OUTPUT_DIR
    alignment_weight: float = 0.1
    margin: float = 0.5
    negative_field: str = "negative"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def read_records(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser()
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain one JSON object per line.")
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
        raise ValueError(f"{path} must contain a JSON list, JSON object, or JSONL records.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains at least one non-object record.")
    return list(records)


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str | None]:
    for field in fields:
        value = clean_text(record.get(field))
        if value:
            return value, field
    return "", None


def extract_prompt_response(record: dict[str, Any]) -> tuple[str, str]:
    prompt, prompt_field = first_text(record, PROMPT_FIELDS)
    response, _ = first_text(record, RESPONSE_FIELDS)

    instruction = clean_text(record.get("instruction"))
    extra_input = clean_text(record.get("input"))
    if instruction and extra_input and prompt_field == "instruction":
        prompt = f"{instruction}\n{extra_input}"

    return prompt, response


def load_regular_examples(path: Path) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for index, record in enumerate(read_records(path)):
        prompt, response = extract_prompt_response(record)
        if not prompt or not response:
            raise ValueError(f"{path}:{index} is missing a prompt/instruction or response/output field.")
        examples.append({"prompt": prompt, "response": response})
    return examples


def load_contrastive_examples(path: Path, negative_field: str = "negative") -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []
    for index, record in enumerate(read_records(path)):
        anchor = clean_text(record.get("anchor")) or first_text(record, PROMPT_FIELDS)[0]
        positive = clean_text(record.get("positive")) or first_text(record, POSITIVE_FIELDS)[0]
        negative = clean_text(record.get(negative_field))
        if not negative:
            negative = first_text(record, NEGATIVE_FIELDS)[0]
        response, _ = first_text(record, RESPONSE_FIELDS)

        if not anchor or not positive or not negative or not response:
            raise ValueError(
                f"{path}:{index} must contain anchor, positive, {negative_field or 'negative'}, and response."
            )
        examples.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "response": response,
            }
        )
    return examples


def setup_distributed() -> DistributedState:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedState()
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA. Use a single-process CPU/MPS run instead.")

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedState(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.destroy_process_group()


def sync_distributed(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    dist.barrier()


def rank0_print(state: DistributedState, message: str) -> None:
    if state.is_main:
        print(message, flush=True)


def force_huggingface_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def resolve_device(device_name: str, state: DistributedState) -> Any:
    import torch

    if state.enabled:
        return torch.device("cuda", state.local_rank)
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_model(model: Any) -> Any:
    while hasattr(model, "module"):
        model = model.module
    return model


def model_for_save(model: Any) -> Any:
    model = unwrap_model(model)
    if getattr(model, "_is_scenic_triplet_sft_wrapper", False):
        return model.base_model
    return model


def load_chatlm_stack(config: RegularSFTConfig, state: DistributedState) -> tuple[Any, Any, Any]:
    if config.local_files_only:
        force_huggingface_offline()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = resolve_device(config.device, state)
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
    }
    if config.cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.cache_dir.expanduser())
    dtype = model_load_dtype(config, device)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    # In DDP, let rank 0 touch the network/cache first. Otherwise eight ranks can
    # simultaneously initialize Hugging Face HTTP clients and mutate the same cache.
    if state.enabled and not state.is_main and not config.local_files_only:
        sync_distributed(state)
        load_kwargs["local_files_only"] = True

    rank0_print(state, f"Loading tokenizer from {config.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **load_kwargs)
    except Exception as exc:
        raise_model_load_error(config, exc)

    rank0_print(state, f"Loading model from {config.model_name_or_path}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    except Exception as exc:
        raise_model_load_error(config, exc)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    rank0_print(state, f"Model loaded on {device}.")
    if state.enabled and state.is_main and not config.local_files_only:
        sync_distributed(state)
    return tokenizer, model, device


def model_load_dtype(config: RegularSFTConfig, device: Any) -> Any | None:
    if device.type != "cuda":
        return None
    import torch

    if config.bf16:
        return torch.bfloat16
    if config.fp16:
        return torch.float16
    return None


def pad_multiple(config: RegularSFTConfig) -> int | None:
    return 8 if config.fp16 or config.bf16 else None


def raise_model_load_error(config: RegularSFTConfig, exc: Exception) -> None:
    message = str(exc).lower()
    if "ssl" in message or "certificate" in message or "max retries" in message:
        hint = (
            "Failed to load ChatLM model files because the Hugging Face request hit an SSL/certificate error. "
            "This is usually a VPN/proxy/captive-network/CA-bundle issue, not a training issue. "
            "Try upgrading certifi/huggingface_hub/transformers, disabling the proxy or VPN, or download the model once "
            "with `huggingface-cli download charent/ChatLM-mini-Chinese --local-dir models/ChatLM-mini-Chinese` "
            "and rerun with `--model models/ChatLM-mini-Chinese --local-files-only`."
        )
        raise RuntimeError(hint) from exc
    raise exc


def tokenize_targets(tokenizer: Any, targets: list[str], max_length: int) -> Any:
    try:
        labels = tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
    except TypeError:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=max_length,
                pad_to_multiple_of=8,
                return_tensors="pt",
            )
    return labels["input_ids"]


def mask_pad_tokens(labels: Any, pad_token_id: int | None) -> Any:
    if pad_token_id is None:
        return labels
    labels = labels.clone()
    labels[labels == pad_token_id] = -100
    return labels


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def remove_token_type_ids(encoded: Any) -> Any:
    encoded.pop("token_type_ids", None)
    return encoded


def pad_source_batch(encoded: dict[str, Any], target_length: int) -> dict[str, Any]:
    import torch.nn.functional as functional

    padded: dict[str, Any] = {}
    for key, value in encoded.items():
        if hasattr(value, "dim") and value.dim() == 2 and value.size(1) < target_length:
            value = functional.pad(value, (0, target_length - value.size(1)), value=0)
        padded[key] = value
    return padded


def combine_source_batches(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    import torch

    target_length = max(first["input_ids"].size(1), second["input_ids"].size(1))
    first = pad_source_batch(first, target_length)
    second = pad_source_batch(second, target_length)
    return {key: torch.cat([first[key], second[key]], dim=0) for key in first.keys() & second.keys()}


def make_regular_collate(tokenizer: Any, config: RegularSFTConfig) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    def collate(batch: list[dict[str, str]]) -> dict[str, Any]:
        sources = [item["prompt"] for item in batch]
        targets = [item["response"] for item in batch]
        encoded = remove_token_type_ids(
            tokenizer(
                sources,
                padding=True,
                truncation=True,
                max_length=config.max_source_length,
                pad_to_multiple_of=pad_multiple(config),
                return_tensors="pt",
            )
        )
        labels = tokenize_targets(tokenizer, targets, config.max_target_length)
        encoded["labels"] = mask_pad_tokens(labels, tokenizer.pad_token_id)
        return encoded

    return collate


def make_contrastive_collate(
    tokenizer: Any, config: ContrastiveSFTConfig
) -> Callable[[list[dict[str, str]]], dict[str, Any]]:
    def encode_sources(texts: list[str]) -> dict[str, Any]:
        return remove_token_type_ids(
            tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_source_length,
                pad_to_multiple_of=pad_multiple(config),
                return_tensors="pt",
            )
        )

    def collate(batch: list[dict[str, str]]) -> dict[str, Any]:
        labels = tokenize_targets(tokenizer, [item["response"] for item in batch], config.max_target_length)
        labels = mask_pad_tokens(labels, tokenizer.pad_token_id)
        return {
            "anchor": encode_sources([item["anchor"] for item in batch]),
            "positive": encode_sources([item["positive"] for item in batch]),
            "negative": encode_sources([item["negative"] for item in batch]),
            "labels": labels,
        }

    return collate


def make_dataloader(
    examples: list[dict[str, str]],
    batch_size: int,
    shuffle: bool,
    collate_fn: Any,
    num_workers: int,
    state: DistributedState,
) -> tuple[Any, Any]:
    from torch.utils.data import DataLoader, DistributedSampler

    sampler = DistributedSampler(
        examples,
        num_replicas=state.world_size,
        rank=state.rank,
        shuffle=shuffle,
    ) if state.enabled else None

    dataloader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return dataloader, sampler


def make_optimizer_and_scheduler(model: Any, config: RegularSFTConfig, batches_per_epoch: int) -> tuple[Any, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    update_steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_update_steps = max(1, update_steps_per_epoch * config.epochs)
    warmup_steps = int(total_update_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    return optimizer, scheduler


def autocast_context(config: RegularSFTConfig, device: Any) -> Any:
    import torch

    if device.type != "cuda":
        return nullcontext()
    if config.bf16:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if config.fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def save_model(tokenizer: Any, model: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model_for_save(model).save_pretrained(output_dir)


class TripletSFTModule:
    def __init__(self, base_model: Any) -> None:
        import torch

        class _TripletSFTModule(torch.nn.Module):
            def __init__(self, wrapped_model: Any) -> None:
                super().__init__()
                self._is_scenic_triplet_sft_wrapper = True
                self.base_model = wrapped_model

            def forward(
                self,
                anchor: dict[str, Any],
                positive: dict[str, Any],
                negative: dict[str, Any],
                labels: Any,
                margin: float,
                alignment_weight: float,
            ) -> tuple[Any, Any, Any]:
                import torch
                import torch.nn.functional as functional

                generation_inputs = combine_source_batches(anchor, positive)
                generation_labels = torch.cat([labels, labels], dim=0)
                generation_outputs = self.base_model(**generation_inputs, labels=generation_labels, return_dict=True)
                gen_loss = generation_outputs.loss

                encoder = self.base_model.get_encoder()
                negative_outputs = encoder(
                    input_ids=negative["input_ids"],
                    attention_mask=negative.get("attention_mask"),
                    return_dict=True,
                )
                batch_size = labels.size(0)
                anchor_hidden = generation_outputs.encoder_last_hidden_state[:batch_size]
                positive_hidden = generation_outputs.encoder_last_hidden_state[batch_size:]
                anchor_attention_mask = generation_inputs["attention_mask"][:batch_size]
                positive_attention_mask = generation_inputs["attention_mask"][batch_size:]

                anchor_rep = mean_pool_encoder(anchor_hidden, anchor_attention_mask)
                positive_rep = mean_pool_encoder(positive_hidden, positive_attention_mask)
                negative_rep = mean_pool_encoder(negative_outputs.last_hidden_state, negative["attention_mask"])

                positive_distance = 1.0 - (anchor_rep * positive_rep).sum(dim=-1)
                negative_distance = 1.0 - (anchor_rep * negative_rep).sum(dim=-1)
                align_loss = functional.relu(margin + positive_distance - negative_distance).mean()
                loss = gen_loss + alignment_weight * align_loss
                return loss, gen_loss, align_loss

        self.module = _TripletSFTModule(base_model)


def wrap_for_distributed(model: Any, state: DistributedState, mode: str) -> Any:
    if mode == "contrastive":
        model = TripletSFTModule(model).module
    if not state.enabled:
        return model

    import torch
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        model,
        device_ids=[state.local_rank],
        output_device=state.local_rank,
        find_unused_parameters=False,
    )


def train_regular_sft(config: RegularSFTConfig | None = None) -> Path:
    import torch
    from tqdm.auto import tqdm

    config = config or RegularSFTConfig()
    state = setup_distributed()
    seed_everything(config.seed)
    examples = load_regular_examples(config.train_json)
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    rank0_print(state, f"Loaded {len(examples)} regular SFT examples from {config.train_json}.")
    tokenizer, model, device = load_chatlm_stack(config, state)
    model = wrap_for_distributed(model, state, mode="regular")
    dataloader, sampler = make_dataloader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_regular_collate(tokenizer, config),
        num_workers=config.num_workers,
        state=state,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(model, config, len(dataloader))
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16 and device.type == "cuda" and not config.bf16)
    rank0_print(
        state,
        "Starting regular SFT: "
        f"{config.epochs} epoch(s), {len(dataloader)} batch(es)/epoch/process, "
        f"world_size={state.world_size}, per_gpu_batch={config.batch_size}.",
    )

    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = tqdm(dataloader, desc=f"regular sft epoch {epoch}/{config.epochs}", disable=not state.is_main)
        running_loss = 0.0
        for batch_index, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            with autocast_context(config, device):
                loss = model(**batch).loss
                scaled_loss = loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.detach().cpu())

            if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if config.log_every and global_step % config.log_every == 0:
                    progress.set_postfix(loss=f"{running_loss / batch_index:.4f}", step=global_step)
                if state.is_main and config.save_every_steps and global_step % config.save_every_steps == 0:
                    save_model(tokenizer, model, config.output_dir / f"checkpoint-step-{global_step}")

        if state.is_main:
            save_model(tokenizer, model, config.output_dir / f"checkpoint-epoch-{epoch}")

    if state.is_main:
        save_model(tokenizer, model, config.output_dir)
    sync_distributed(state)
    cleanup_distributed(state)
    return config.output_dir


def mean_pool_encoder(hidden_states: Any, attention_mask: Any) -> Any:
    import torch.nn.functional as functional

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return functional.normalize(pooled, p=2, dim=-1)


def encoder_representation(model: Any, encoded_inputs: dict[str, Any]) -> Any:
    encoder = model_for_save(model).get_encoder()
    outputs = encoder(
        input_ids=encoded_inputs["input_ids"],
        attention_mask=encoded_inputs.get("attention_mask"),
        return_dict=True,
    )
    return mean_pool_encoder(outputs.last_hidden_state, encoded_inputs["attention_mask"])


def train_contrastive_triplet_sft(config: ContrastiveSFTConfig | None = None) -> Path:
    import torch
    from tqdm.auto import tqdm

    config = config or ContrastiveSFTConfig()
    if not 0.0 < config.margin < 2.0:
        raise ValueError("Triplet margin should be in (0, 2) for cosine distance.")

    state = setup_distributed()
    seed_everything(config.seed)
    examples = load_contrastive_examples(config.train_json, negative_field=config.negative_field)
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    rank0_print(state, f"Loaded {len(examples)} contrastive tuples from {config.train_json}.")
    tokenizer, model, device = load_chatlm_stack(config, state)
    model = wrap_for_distributed(model, state, mode="contrastive")
    dataloader, sampler = make_dataloader(
        examples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=make_contrastive_collate(tokenizer, config),
        num_workers=config.num_workers,
        state=state,
    )
    optimizer, scheduler = make_optimizer_and_scheduler(model, config, len(dataloader))
    scaler = torch.cuda.amp.GradScaler(enabled=config.fp16 and device.type == "cuda" and not config.bf16)
    rank0_print(
        state,
        "Starting triplet SFT: "
        f"{config.epochs} epoch(s), {len(dataloader)} batch(es)/epoch/process, "
        f"world_size={state.world_size}, per_gpu_batch={config.batch_size}.",
    )

    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, config.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        progress = tqdm(dataloader, desc=f"triplet sft epoch {epoch}/{config.epochs}", disable=not state.is_main)
        running_total = 0.0
        running_gen = 0.0
        running_align = 0.0
        for batch_index, batch in enumerate(progress, start=1):
            anchor = move_batch_to_device(batch["anchor"], device)
            positive = move_batch_to_device(batch["positive"], device)
            negative = move_batch_to_device(batch["negative"], device)
            labels = batch["labels"].to(device)

            with autocast_context(config, device):
                loss, gen_loss, align_loss = model(
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    labels=labels,
                    margin=config.margin,
                    alignment_weight=config.alignment_weight,
                )
                scaled_loss = loss / config.gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            running_total += float(loss.detach().cpu())
            running_gen += float(gen_loss.detach().cpu())
            running_align += float(align_loss.detach().cpu())

            if batch_index % config.gradient_accumulation_steps == 0 or batch_index == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if config.log_every and global_step % config.log_every == 0:
                    progress.set_postfix(
                        total=f"{running_total / batch_index:.4f}",
                        gen=f"{running_gen / batch_index:.4f}",
                        align=f"{running_align / batch_index:.4f}",
                        step=global_step,
                    )
                if state.is_main and config.save_every_steps and global_step % config.save_every_steps == 0:
                    save_model(tokenizer, model, config.output_dir / f"checkpoint-step-{global_step}")

        if state.is_main:
            save_model(tokenizer, model, config.output_dir / f"checkpoint-epoch-{epoch}")

    if state.is_main:
        save_model(tokenizer, model, config.output_dir)
    sync_distributed(state)
    cleanup_distributed(state)
    return config.output_dir


def print_dataset_summary(mode: str, path: Path, negative_field: str) -> None:
    if mode == "regular":
        examples = load_regular_examples(path)
        unique_responses = len({item["response"] for item in examples})
        print(f"regular examples: {len(examples)}")
        print(f"unique responses: {unique_responses}")
        print(json.dumps(examples[0], ensure_ascii=False, indent=2))
        return

    examples = load_contrastive_examples(path, negative_field=negative_field)
    print(f"contrastive tuples: {len(examples)}")
    print(json.dumps(examples[0], ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regular and compatibility-aware triplet SFT for ChatLM-mini-Chinese.")
    parser.add_argument("--mode", choices=("regular", "contrastive"), default="regular")
    parser.add_argument("--model", default=MODEL_NAME_OR_PATH, help="Hugging Face model id or local model directory.")
    parser.add_argument("--train-json", default=None, help="Training .json or .jsonl path. Defaults depend on --mode.")
    parser.add_argument("--output-dir", default=None, help="Directory for final model and checkpoints.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None, help="Use only the first N examples for a smoke test.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer only from local files.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--fp16", action="store_true", help="Use CUDA fp16 autocast.")
    parser.add_argument("--bf16", action="store_true", help="Use CUDA bf16 autocast.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--negative-field",
        default="negative",
        help="Contrastive negative field to use. Set to invalid_negative if you want invalid hard negatives.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize data without loading the model.")
    return parser.parse_args()


def regular_config_from_args(args: argparse.Namespace) -> RegularSFTConfig:
    train_json = Path(args.train_json).expanduser() if args.train_json else REGULAR_TRAIN_JSON
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else REGULAR_OUTPUT_DIR
    return RegularSFTConfig(
        model_name_or_path=args.model,
        train_json=train_json,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_examples=args.max_examples,
        cache_dir=Path(args.cache_dir).expanduser() if args.cache_dir else None,
        local_files_only=args.local_files_only,
        fp16=args.fp16,
        bf16=args.bf16,
        device=args.device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_every_steps=args.save_every_steps,
    )


def contrastive_config_from_args(args: argparse.Namespace) -> ContrastiveSFTConfig:
    train_json = Path(args.train_json).expanduser() if args.train_json else CONTRASTIVE_TRAIN_JSON
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else CONTRASTIVE_OUTPUT_DIR
    return ContrastiveSFTConfig(
        model_name_or_path=args.model,
        train_json=train_json,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_examples=args.max_examples,
        cache_dir=Path(args.cache_dir).expanduser() if args.cache_dir else None,
        local_files_only=args.local_files_only,
        fp16=args.fp16,
        bf16=args.bf16,
        device=args.device,
        num_workers=args.num_workers,
        log_every=args.log_every,
        save_every_steps=args.save_every_steps,
        alignment_weight=args.alignment_weight,
        margin=args.margin,
        negative_field=args.negative_field,
    )


def main() -> None:
    args = parse_args()
    if args.dry_run:
        default_path = REGULAR_TRAIN_JSON if args.mode == "regular" else CONTRASTIVE_TRAIN_JSON
        train_path = Path(args.train_json).expanduser() if args.train_json else default_path
        print_dataset_summary(args.mode, train_path, args.negative_field)
        return

    if args.mode == "regular":
        output_dir = train_regular_sft(regular_config_from_args(args))
    else:
        output_dir = train_contrastive_triplet_sft(contrastive_config_from_args(args))
    print(f"saved model to {output_dir}")


if __name__ == "__main__":
    main()
