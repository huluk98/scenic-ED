#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import random
import shutil
import signal
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit these defaults directly if you prefer running the file without CLI flags.
MODEL_NAME_OR_PATH = "charent/ChatLM-mini-Chinese"
REGULAR_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
CONTRASTIVE_TRAIN_JSON = PROJECT_ROOT / "data" / "SCENIC_full_anchor_positive_negative.json"
REGULAR_OUTPUT_DIR = PROJECT_ROOT / "sft5"
CONTRASTIVE_OUTPUT_DIR = PROJECT_ROOT / "models" / "chatlm_scenic_triplet_sft"

REGULAR_EPOCHS = 5
REGULAR_MAX_SOURCE_LENGTH = 256
REGULAR_MAX_TARGET_LENGTH = 128
REGULAR_LOG_EVERY = 50
REGULAR_SAVE_EVERY_STEPS = 500
REGULAR_SAVE_TOTAL_LIMIT = 2
REGULAR_FP16 = True
REGULAR_NUM_WORKERS = 4

CONTRASTIVE_EPOCHS = 5
CONTRASTIVE_MAX_SOURCE_LENGTH = 128
CONTRASTIVE_MAX_TARGET_LENGTH = 96
CONTRASTIVE_LOG_EVERY = 20
CONTRASTIVE_SAVE_EVERY_STEPS = 0
CONTRASTIVE_FP16 = True
CONTRASTIVE_NUM_WORKERS = 0

PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "anchor", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")
POSITIVE_FIELDS = ("positive", "pos", "x_positive", "x_plus", "chosen")
NEGATIVE_FIELDS = ("negative", "neg", "x_negative", "x_minus", "rejected")
DEFAULT_DDP_TIMEOUT_MINUTES = 10
BAD_TOKENIZER_CLASSES = {"tokenizersbackend"}
TOKENIZER_ASSET_FILENAMES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
)
FAST_TOKENIZER_CONFIG_KEYS = ("tokenizer_file", "fast_tokenizer_files")
CUSTOM_CODE_GLOBS = (
    "configuration*.py",
    "modeling*.py",
    "tokenization*.py",
    "generation*.py",
    "processing*.py",
)


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
    epochs: int = REGULAR_EPOCHS
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_source_length: int = REGULAR_MAX_SOURCE_LENGTH
    max_target_length: int = REGULAR_MAX_TARGET_LENGTH
    max_grad_norm: float = 1.0
    seed: int = 42
    max_examples: int | None = None
    cache_dir: Path | None = None
    local_files_only: bool = False
    fp16: bool = REGULAR_FP16
    bf16: bool = False
    device: str = "auto"
    num_workers: int = REGULAR_NUM_WORKERS
    log_every: int = REGULAR_LOG_EVERY
    save_every_steps: int = REGULAR_SAVE_EVERY_STEPS
    save_total_limit: int = REGULAR_SAVE_TOTAL_LIMIT
    save_epoch_checkpoints: bool = False
    final_save_on_cpu: bool = True
    safe_serialization: bool = True


@dataclass
class ContrastiveSFTConfig(RegularSFTConfig):
    train_json: Path = CONTRASTIVE_TRAIN_JSON
    output_dir: Path = CONTRASTIVE_OUTPUT_DIR
    epochs: int = CONTRASTIVE_EPOCHS
    max_source_length: int = CONTRASTIVE_MAX_SOURCE_LENGTH
    max_target_length: int = CONTRASTIVE_MAX_TARGET_LENGTH
    fp16: bool = CONTRASTIVE_FP16
    bf16: bool = False
    num_workers: int = CONTRASTIVE_NUM_WORKERS
    log_every: int = CONTRASTIVE_LOG_EVERY
    save_every_steps: int = CONTRASTIVE_SAVE_EVERY_STEPS
    save_epoch_checkpoints: bool = True
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

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    timeout_minutes = int(os.environ.get("SCENIC_DDP_TIMEOUT_MINUTES", "10"))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    try:
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=timeout_minutes),
            device_id=torch.device("cuda", local_rank),
        )
    except TypeError:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    return DistributedState(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def release_cuda_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def cleanup_active_distributed() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    release_cuda_memory()


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: object) -> None:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Received signal {signum}; cleaning up distributed/CUDA state.", flush=True)
        cleanup_active_distributed()
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, handle_signal)


def sync_distributed(state: DistributedState) -> None:
    if not state.enabled:
        return
    import torch.distributed as dist

    try:
        dist.barrier(device_ids=[state.local_rank])
    except TypeError:
        dist.barrier()


def rank0_print(state: DistributedState, message: str) -> None:
    if state.is_main:
        print(message, flush=True)


def print_distributed_launch(state: DistributedState) -> None:
    import torch

    if not state.enabled:
        rank0_print(state, "Distributed training disabled: using a single process.")
        return

    device_name = torch.cuda.get_device_name(state.local_rank)
    print(
        "DDP launch: "
        f"rank={state.rank}/{state.world_size} "
        f"local_rank={state.local_rank} "
        f"cuda_device={torch.cuda.current_device()} "
        f"gpu={device_name}",
        flush=True,
    )
    sync_distributed(state)


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


def json_safe_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {key: json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(json_safe_value(item) for item in value)
    return value


def sanitize_config_for_json(config: Any) -> None:
    if config is None:
        return
    for key, value in list(vars(config).items()):
        safe_value = json_safe_value(value)
        if safe_value is not value:
            setattr(config, key, safe_value)


def sanitize_model_for_save(model: Any) -> Any:
    model = model_for_save(model)
    sanitize_config_for_json(getattr(model, "config", None))
    sanitize_config_for_json(getattr(model, "generation_config", None))
    return model


def sanitize_tokenizer_for_save(tokenizer: Any) -> Any:
    for attr in ("init_kwargs", "special_tokens_map"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, dict):
            safe_value = json_safe_value(value)
            try:
                setattr(tokenizer, attr, safe_value)
            except AttributeError:
                value.clear()
                value.update(safe_value)
    return tokenizer


def is_bad_tokenizer_class(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.replace("_", "").replace("-", "").lower()
    return normalized in BAD_TOKENIZER_CLASSES or normalized.endswith(".tokenizersbackend")


def read_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def path_is_same(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def tokenizer_source_dir(tokenizer: Any) -> Path | None:
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attr, None)
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if path.is_dir():
            return path
    return None


def checkpoint_tokenizer_source_dir(checkpoint_dir: Path) -> Path | None:
    for filename in ("tokenizer_config.json", "config.json"):
        config = read_json_dict(checkpoint_dir / filename)
        if not config:
            continue
        for key in ("_name_or_path", "name_or_path"):
            value = config.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.is_dir() and not path_is_same(path, checkpoint_dir):
                return path
    return None


def flatten_auto_map(auto_map: Any) -> list[str]:
    if not isinstance(auto_map, dict):
        return []
    values: list[str] = []
    for value in auto_map.values():
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, list):
            values.extend(item for item in value if isinstance(item, str))
    return values


def custom_code_filenames_from_config(config: dict[str, Any] | None) -> list[str]:
    if not config:
        return []
    filenames: list[str] = []
    for value in flatten_auto_map(config.get("auto_map")):
        module_reference = value.split("--", 1)[-1]
        module_name = module_reference.split(".", 1)[0]
        if module_name and not module_name.startswith("transformers"):
            filename = f"{module_name}.py"
            if filename not in filenames:
                filenames.append(filename)
    return filenames


def copy_file_if_missing(source: Path, destination: Path) -> None:
    if source.exists() and source.is_file() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)


def copy_missing_tokenizer_assets(source_dir: Path | None, output_dir: Path) -> None:
    if source_dir is None or not source_dir.is_dir() or path_is_same(source_dir, output_dir):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_ASSET_FILENAMES:
        copy_file_if_missing(source_dir / filename, output_dir / filename)
    for source in source_dir.glob("*.model"):
        copy_file_if_missing(source, output_dir / source.name)


def copy_missing_custom_code_assets(source_dir: Path | None, output_dir: Path) -> None:
    if source_dir is None or not source_dir.is_dir() or path_is_same(source_dir, output_dir):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(custom_code_filenames_from_config(read_json_dict(output_dir / "config.json")))
    wanted.update(custom_code_filenames_from_config(read_json_dict(output_dir / "tokenizer_config.json")))
    wanted.update(custom_code_filenames_from_config(read_json_dict(source_dir / "config.json")))
    wanted.update(custom_code_filenames_from_config(read_json_dict(source_dir / "tokenizer_config.json")))
    for filename in sorted(wanted):
        copy_file_if_missing(source_dir / filename, output_dir / filename)
    for pattern in CUSTOM_CODE_GLOBS:
        for source in source_dir.glob(pattern):
            copy_file_if_missing(source, output_dir / source.name)


def strip_fast_tokenizer_config(config: dict[str, Any]) -> None:
    removed = [key for key in FAST_TOKENIZER_CONFIG_KEYS if key in config]
    for key in removed:
        config.pop(key, None)
    if removed:
        config["_scenic_removed_fast_tokenizer_keys"] = removed


def normalize_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    safe_config = json_safe_value(config)
    tokenizer_class = safe_config.get("tokenizer_class")
    if is_bad_tokenizer_class(tokenizer_class):
        safe_config.pop("tokenizer_class", None)
        safe_config["_scenic_removed_tokenizer_class"] = tokenizer_class
        strip_fast_tokenizer_config(safe_config)
    return safe_config


def valid_source_tokenizer_config(source_dir: Path | None) -> dict[str, Any] | None:
    if source_dir is None:
        return None
    config = read_json_dict(source_dir / "tokenizer_config.json")
    if not config:
        return None
    return normalize_tokenizer_config(config)


def repair_tokenizer_files_for_auto_load(output_dir: Path, source_dir: Path | None = None) -> None:
    output_dir = Path(output_dir).expanduser()
    source_dir = source_dir or checkpoint_tokenizer_source_dir(output_dir)
    copy_missing_tokenizer_assets(source_dir, output_dir)
    copy_missing_custom_code_assets(source_dir, output_dir)

    config_path = output_dir / "tokenizer_config.json"
    config = read_json_dict(config_path)
    if config is None:
        source_config = valid_source_tokenizer_config(source_dir)
        if source_config is not None:
            write_json_dict(config_path, source_config)
        return

    raw_tokenizer_class = config.get("tokenizer_class")
    if is_bad_tokenizer_class(raw_tokenizer_class):
        source_config = valid_source_tokenizer_config(source_dir)
        if source_config is not None:
            safe_config = source_config
        else:
            safe_config = json_safe_value(config)
            safe_config.pop("tokenizer_class", None)
            safe_config["_scenic_removed_tokenizer_class"] = raw_tokenizer_class
            strip_fast_tokenizer_config(safe_config)
    else:
        safe_config = normalize_tokenizer_config(config)
        tokenizer_class = safe_config.get("tokenizer_class")
        if not tokenizer_class and not safe_config.get("auto_map"):
            strip_fast_tokenizer_config(safe_config)

    if safe_config != config:
        write_json_dict(config_path, safe_config)


def save_tokenizer_for_auto_load(tokenizer: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = tokenizer_source_dir(tokenizer)
    sanitize_tokenizer_for_save(tokenizer).save_pretrained(output_dir)
    repair_tokenizer_files_for_auto_load(output_dir, source_dir=source_dir)


def repair_checkpoint_for_auto_load(output_dir: Path, tokenizer: Any | None = None) -> None:
    source_dir = tokenizer_source_dir(tokenizer) if tokenizer is not None else None
    repair_tokenizer_files_for_auto_load(output_dir, source_dir=source_dir)


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
    tokenizer_kwargs = {**load_kwargs, "use_fast": False}
    model_kwargs = dict(load_kwargs)
    dtype = model_load_dtype(config, device)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    # In DDP, let rank 0 touch the network/cache first. Otherwise eight ranks can
    # simultaneously initialize Hugging Face HTTP clients and mutate the same cache.
    if state.enabled and not state.is_main and not config.local_files_only:
        sync_distributed(state)
        tokenizer_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True

    rank0_print(state, f"Loading tokenizer from {config.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **tokenizer_kwargs)
    except Exception as exc:
        raise_model_load_error(config, exc)

    rank0_print(state, f"Loading model from {config.model_name_or_path}...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, **model_kwargs)
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
            "with `hf download charent/ChatLM-mini-Chinese --local-dir models/ChatLM-mini-Chinese` "
            "and rerun with `--model models/ChatLM-mini-Chinese --local-files-only`."
        )
        raise RuntimeError(hint) from exc
    raise exc


def append_eos_to_targets(
    input_ids: Any,
    attention_mask: Any | None,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_length: int,
) -> tuple[Any, Any | None]:
    if eos_token_id is None:
        return input_ids, attention_mask

    import torch

    if attention_mask is None:
        if pad_token_id is None:
            attention_mask = input_ids.new_ones(input_ids.shape)
        else:
            attention_mask = input_ids.ne(pad_token_id).long()

    input_ids = input_ids.clone()
    attention_mask = attention_mask.clone()
    needs_extra_column = False
    for row_index in range(input_ids.size(0)):
        length = int(attention_mask[row_index].sum().item())
        has_eos = length > 0 and int(input_ids[row_index, length - 1].item()) == eos_token_id
        if not has_eos and length >= input_ids.size(1):
            needs_extra_column = True
            break

    if needs_extra_column and input_ids.size(1) < max_length:
        pad_value = eos_token_id if pad_token_id is None else pad_token_id
        pad_column = input_ids.new_full((input_ids.size(0), 1), pad_value)
        mask_column = attention_mask.new_zeros((attention_mask.size(0), 1))
        input_ids = torch.cat((input_ids, pad_column), dim=1)
        attention_mask = torch.cat((attention_mask, mask_column), dim=1)

    for row_index in range(input_ids.size(0)):
        length = int(attention_mask[row_index].sum().item())
        if length > 0 and int(input_ids[row_index, length - 1].item()) == eos_token_id:
            continue
        eos_position = min(length, input_ids.size(1) - 1)
        input_ids[row_index, eos_position] = eos_token_id
        attention_mask[row_index, eos_position] = 1
    return input_ids, attention_mask


def tokenize_targets(tokenizer: Any, targets: list[str], max_length: int) -> tuple[Any, Any | None]:
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


def move_batch_to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def remove_token_type_ids(encoded: Any) -> Any:
    encoded.pop("token_type_ids", None)
    return encoded


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
        labels, label_attention_mask = tokenize_targets(tokenizer, targets, config.max_target_length)
        encoded["labels"] = mask_pad_tokens(labels, tokenizer.pad_token_id, label_attention_mask)
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
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        labels, label_attention_mask = tokenize_targets(
            tokenizer,
            [item["response"] for item in batch],
            config.max_target_length,
        )
        labels = mask_pad_tokens(labels, tokenizer.pad_token_id, label_attention_mask)
        return {
            "generation": encode_sources(anchors + positives),
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
        persistent_workers=num_workers > 0,
        pin_memory=state.enabled,
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


def save_model(tokenizer: Any, model: Any, output_dir: Path, safe_serialization: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_for_auto_load(tokenizer, output_dir)
    sanitize_model_for_save(model).save_pretrained(output_dir, safe_serialization=safe_serialization)
    repair_checkpoint_for_auto_load(output_dir, tokenizer=tokenizer)


def save_rank0_then_sync(
    state: DistributedState,
    tokenizer: Any,
    model: Any,
    output_dir: Path,
    label: str,
) -> None:
    if state.is_main:
        rank0_print(state, f"Saving {label} to {output_dir}...")
        save_model(tokenizer, model, output_dir)
        rank0_print(state, f"Finished saving {label}.")
    sync_distributed(state)


def save_final_rank0_after_training(
    state: DistributedState,
    tokenizer: Any,
    model: Any,
    output_dir: Path,
    label: str,
    save_on_cpu: bool,
    safe_serialization: bool,
) -> None:
    rank0_print(state, f"Training finished; synchronizing ranks before saving {label}.")
    sync_distributed(state)
    cleanup_distributed(state)

    if not state.is_main:
        release_cuda_memory()
        return

    model_to_save = sanitize_model_for_save(model)
    if save_on_cpu:
        rank0_print(state, f"Moving {label} to CPU before saving...")
        model_to_save.to("cpu")
        release_cuda_memory()

    rank0_print(state, f"Saving {label} to {output_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer_for_auto_load(tokenizer, output_dir)
    rank0_print(state, f"Tokenizer saved for {label}; saving model weights...")
    model_to_save.save_pretrained(output_dir, safe_serialization=safe_serialization)
    repair_checkpoint_for_auto_load(output_dir, tokenizer=tokenizer)
    rank0_print(state, f"Finished saving {label}.")
    release_cuda_memory()


def pair_balanced_generation_loss(outputs: Any, labels: Any, batch_size: int) -> Any:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("Model output must include either logits or loss for triplet SFT generation loss.")
        return loss

    import torch.nn.functional as functional

    if logits.size(1) != labels.size(1):
        common_length = min(logits.size(1), labels.size(1))
        logits = logits[:, :common_length, :]
        labels = labels[:, :common_length]

    token_losses = functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    token_counts = labels.ne(-100).sum(dim=1).clamp(min=1).to(token_losses.dtype)
    per_example_losses = token_losses.sum(dim=1) / token_counts

    if per_example_losses.size(0) != batch_size * 2:
        return per_example_losses.mean()
    paired_losses = 0.5 * (per_example_losses[:batch_size] + per_example_losses[batch_size:])
    return paired_losses.mean()


class TripletSFTModule:
    def __init__(self, base_model: Any) -> None:
        import torch

        class _TripletSFTModule(torch.nn.Module):
            def __init__(self, wrapped_model: Any) -> None:
                super().__init__()
                self._is_scenic_triplet_sft_wrapper = True
                self.base_model = wrapped_model

            def _encoder(self) -> Any:
                if not hasattr(self.base_model, "get_encoder"):
                    raise RuntimeError("Triplet SFT currently requires an encoder-decoder model with get_encoder().")
                encoder = self.base_model.get_encoder()
                if encoder is None:
                    raise RuntimeError("Triplet SFT could not obtain an encoder from the base model.")
                return encoder

            def _representations(self, encoded_inputs: dict[str, Any]) -> Any:
                encoder_outputs = self._encoder()(
                    input_ids=encoded_inputs["input_ids"],
                    attention_mask=encoded_inputs.get("attention_mask"),
                    return_dict=True,
                )
                return mean_pool_encoder(encoder_outputs.last_hidden_state, encoded_inputs["attention_mask"])

            def forward(
                self,
                generation: dict[str, Any],
                negative: dict[str, Any],
                labels: Any,
                margin: float,
                alignment_weight: float,
            ) -> tuple[Any, Any, Any]:
                import torch
                import torch.nn.functional as functional

                batch_size = labels.size(0)
                generation_labels = torch.cat([labels, labels], dim=0)
                generation_outputs = self.base_model(**generation, labels=generation_labels, return_dict=True)
                gen_loss = pair_balanced_generation_loss(generation_outputs, generation_labels, batch_size)

                encoder_hidden = getattr(generation_outputs, "encoder_last_hidden_state", None)
                if encoder_hidden is None:
                    generation_reps = self._representations(generation)
                    anchor_rep = generation_reps[:batch_size]
                    positive_rep = generation_reps[batch_size:]
                else:
                    anchor_hidden = encoder_hidden[:batch_size]
                    positive_hidden = encoder_hidden[batch_size:]
                    anchor_attention_mask = generation["attention_mask"][:batch_size]
                    positive_attention_mask = generation["attention_mask"][batch_size:]
                    anchor_rep = mean_pool_encoder(anchor_hidden, anchor_attention_mask)
                    positive_rep = mean_pool_encoder(positive_hidden, positive_attention_mask)
                negative_rep = self._representations(negative)

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


def tokenize_regular_examples_like_original(
    examples: list[dict[str, str]],
    tokenizer: Any,
    config: RegularSFTConfig,
) -> list[dict[str, Any]]:
    encoded_examples: list[dict[str, Any]] = []
    for example in examples:
        inputs = tokenizer(
            example["prompt"],
            max_length=config.max_source_length,
            truncation=True,
            padding="max_length",
        )

        try:
            labels = tokenizer(
                text_target=example["response"],
                max_length=config.max_target_length,
                truncation=True,
                padding="max_length",
            )
        except TypeError:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    example["response"],
                    max_length=config.max_target_length,
                    truncation=True,
                    padding="max_length",
                )

        inputs["labels"] = labels["input_ids"]
        inputs.pop("token_type_ids", None)
        encoded_examples.append(inputs)
    return encoded_examples


def make_seq2seq_training_args(config: RegularSFTConfig) -> Any:
    import inspect

    from transformers import Seq2SeqTrainingArguments

    fp16 = config.fp16 and not config.bf16
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "logging_steps": config.log_every,
        "save_total_limit": config.save_total_limit,
        "fp16": fp16,
        "bf16": config.bf16,
        "dataloader_num_workers": config.num_workers,
        "ddp_find_unused_parameters": False,
        "report_to": "none",
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "seed": config.seed,
        "save_safetensors": config.safe_serialization,
    }
    if config.save_every_steps <= 0:
        kwargs["save_strategy"] = "no"
    else:
        kwargs["save_steps"] = config.save_every_steps

    supported_kwargs = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
    return Seq2SeqTrainingArguments(**kwargs)


def make_seq2seq_trainer(model: Any, training_args: Any, dataset: Any, tokenizer: Any) -> Any:
    from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer

    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer, model=model),
    }
    try:
        return Seq2SeqTrainer(**kwargs, tokenizer=tokenizer)
    except TypeError as exc:
        if "tokenizer" not in str(exc):
            raise
        return Seq2SeqTrainer(**kwargs, processing_class=tokenizer)


def train_regular_sft(config: RegularSFTConfig | None = None) -> Path:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    config = config or RegularSFTConfig()
    if config.local_files_only:
        force_huggingface_offline()
    seed_everything(config.seed)

    examples = load_regular_examples(config.train_json)
    if config.max_examples is not None:
        examples = examples[: config.max_examples]
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"Loaded {len(examples)} regular SFT examples from {config.train_json}.", flush=True)

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": config.local_files_only,
    }
    if config.cache_dir is not None:
        load_kwargs["cache_dir"] = str(config.cache_dir.expanduser())

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False, **load_kwargs)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, **load_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    dataset = tokenize_regular_examples_like_original(examples, tokenizer, config)
    trainer = make_seq2seq_trainer(model, make_seq2seq_training_args(config), dataset, tokenizer)
    trainer.train()
    trainer.save_model(str(config.output_dir))

    if trainer.is_world_process_zero():
        save_tokenizer_for_auto_load(tokenizer, config.output_dir)
        repair_checkpoint_for_auto_load(config.output_dir, tokenizer=tokenizer)
        print(f"SFT complete. Model saved to {config.output_dir}.", flush=True)

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
    print_distributed_launch(state)
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
            generation = move_batch_to_device(batch["generation"], device)
            negative = move_batch_to_device(batch["negative"], device)
            labels = batch["labels"].to(device)

            with autocast_context(config, device):
                loss, gen_loss, align_loss = model(
                    generation=generation,
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
                if config.save_every_steps and global_step % config.save_every_steps == 0:
                    save_rank0_then_sync(
                        state,
                        tokenizer,
                        model,
                        config.output_dir / f"checkpoint-step-{global_step}",
                        f"triplet step {global_step} checkpoint",
                    )

        if config.save_epoch_checkpoints:
            save_rank0_then_sync(
                state,
                tokenizer,
                model,
                config.output_dir / f"checkpoint-epoch-{epoch}",
                f"triplet epoch {epoch} checkpoint",
            )
        else:
            rank0_print(state, f"Finished triplet epoch {epoch}; synchronizing ranks.")
            sync_distributed(state)

    save_final_rank0_after_training(
        state,
        tokenizer,
        model,
        config.output_dir,
        "triplet final model",
        save_on_cpu=config.final_save_on_cpu,
        safe_serialization=config.safe_serialization,
    )
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
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count. Defaults to 5 for both modes.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-source-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=None, help="Use only the first N examples for a smoke test.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer only from local files.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use CUDA fp16. Regular and contrastive SFT default on.",
    )
    parser.add_argument("--bf16", action="store_true", help="Use CUDA bf16 autocast.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every-steps", type=int, default=None)
    parser.add_argument(
        "--epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save checkpoint-epoch-N directories. Use --no-epoch-checkpoints to save only the final model.",
    )
    parser.add_argument(
        "--final-save-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move the final rank-0 model to CPU before save_pretrained to release GPU memory while saving.",
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use safetensors for the final model save. Use --no-safe-serialization to write pytorch_model.bin.",
    )
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=DEFAULT_DDP_TIMEOUT_MINUTES,
        help="NCCL/DDP timeout. A failed rank should abort instead of hanging forever.",
    )
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--negative-field",
        default="negative",
        help="Contrastive negative field to use. Set to invalid_negative if you want invalid hard negatives.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize data without loading the model.")
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ["SCENIC_DDP_TIMEOUT_MINUTES"] = str(args.ddp_timeout_minutes)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def defaulted(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def fp16_default(args: argparse.Namespace, fallback: bool) -> bool:
    if args.bf16:
        return False
    if args.fp16 is not None:
        return args.fp16
    return fallback


def regular_config_from_args(args: argparse.Namespace) -> RegularSFTConfig:
    train_json = Path(args.train_json).expanduser() if args.train_json else REGULAR_TRAIN_JSON
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else REGULAR_OUTPUT_DIR
    return RegularSFTConfig(
        model_name_or_path=args.model,
        train_json=train_json,
        output_dir=output_dir,
        epochs=defaulted(args.epochs, REGULAR_EPOCHS),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_source_length=defaulted(args.max_source_length, REGULAR_MAX_SOURCE_LENGTH),
        max_target_length=defaulted(args.max_target_length, REGULAR_MAX_TARGET_LENGTH),
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_examples=args.max_examples,
        cache_dir=Path(args.cache_dir).expanduser() if args.cache_dir else None,
        local_files_only=args.local_files_only,
        fp16=fp16_default(args, REGULAR_FP16),
        bf16=args.bf16,
        device=args.device,
        num_workers=defaulted(args.num_workers, REGULAR_NUM_WORKERS),
        log_every=defaulted(args.log_every, REGULAR_LOG_EVERY),
        save_every_steps=defaulted(args.save_every_steps, REGULAR_SAVE_EVERY_STEPS),
        save_epoch_checkpoints=defaulted(args.epoch_checkpoints, False),
        final_save_on_cpu=args.final_save_on_cpu,
        safe_serialization=args.safe_serialization,
    )


def contrastive_config_from_args(args: argparse.Namespace) -> ContrastiveSFTConfig:
    train_json = Path(args.train_json).expanduser() if args.train_json else CONTRASTIVE_TRAIN_JSON
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else CONTRASTIVE_OUTPUT_DIR
    return ContrastiveSFTConfig(
        model_name_or_path=args.model,
        train_json=train_json,
        output_dir=output_dir,
        epochs=defaulted(args.epochs, CONTRASTIVE_EPOCHS),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_source_length=defaulted(args.max_source_length, CONTRASTIVE_MAX_SOURCE_LENGTH),
        max_target_length=defaulted(args.max_target_length, CONTRASTIVE_MAX_TARGET_LENGTH),
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        max_examples=args.max_examples,
        cache_dir=Path(args.cache_dir).expanduser() if args.cache_dir else None,
        local_files_only=args.local_files_only,
        fp16=fp16_default(args, CONTRASTIVE_FP16),
        bf16=args.bf16,
        device=args.device,
        num_workers=defaulted(args.num_workers, CONTRASTIVE_NUM_WORKERS),
        log_every=defaulted(args.log_every, CONTRASTIVE_LOG_EVERY),
        save_every_steps=defaulted(args.save_every_steps, CONTRASTIVE_SAVE_EVERY_STEPS),
        save_epoch_checkpoints=defaulted(args.epoch_checkpoints, True),
        final_save_on_cpu=args.final_save_on_cpu,
        safe_serialization=args.safe_serialization,
        alignment_weight=args.alignment_weight,
        margin=args.margin,
        negative_field=args.negative_field,
    )


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    install_signal_handlers()
    atexit.register(cleanup_active_distributed)
    try:
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
    finally:
        cleanup_active_distributed()


if __name__ == "__main__":
    main()
