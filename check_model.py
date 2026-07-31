#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM


# =====================================================
# EDIT THESE DEFAULTS IF YOU WANT TO RUN WITHOUT FLAGS
# =====================================================
MODEL_PATH = "./sft"
# Examples:
# MODEL_PATH = "./sft"
# MODEL_PATH = "charent/ChatLM-mini-Chinese"

TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = True
CACHE_DIR = None
TORCH_DTYPE = "auto"  # auto, float32, float16, bfloat16
COUNT_SPARSITY = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a Hugging Face model and print parameter statistics.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Local model path or Hugging Face model id.")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network requests. By default this script uses local/offline files only.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading custom model code.",
    )
    parser.add_argument(
        "--torch-dtype",
        default=TORCH_DTYPE,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="dtype to request while loading weights.",
    )
    parser.add_argument(
        "--skip-sparsity",
        action="store_true",
        help="Skip nonzero counting. This is faster for very large dense models.",
    )
    return parser.parse_args()


def force_huggingface_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def dtype_from_name(name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def build_load_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    local_files_only = LOCAL_FILES_ONLY and not args.allow_network
    if local_files_only:
        force_huggingface_offline()

    kwargs: dict[str, Any] = {
        "trust_remote_code": not args.no_trust_remote_code and TRUST_REMOTE_CODE,
        "local_files_only": local_files_only,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = args.cache_dir

    torch_dtype = dtype_from_name(args.torch_dtype)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    return kwargs


def detect_model_family(config: Any) -> tuple[str, list[Any]]:
    architectures = getattr(config, "architectures", None) or []

    if getattr(config, "is_encoder_decoder", False):
        return "Encoder-Decoder", [AutoModelForSeq2SeqLM, AutoModel]

    if any("CausalLM" in architecture for architecture in architectures):
        return "Decoder-Only", [AutoModelForCausalLM, AutoModel]

    if any("ConditionalGeneration" in architecture for architecture in architectures):
        return "Conditional Generation", [AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel]

    return "Encoder-Only or Unknown", [AutoModel]


def model_load_hint(exc: Exception) -> str:
    message = str(exc).lower()
    if "ssl" in message or "certificate" in message or "max retries" in message:
        return (
            "Hugging Face hit an SSL/certificate error. If the model is already downloaded, keep "
            "`LOCAL_FILES_ONLY = True` or run with a local path such as `--model-path ./sft`. "
            "If you need to download from the Hub, fix the network/CA issue or download once with "
            "`huggingface-cli download ... --local-dir ...`."
        )
    if "local_files_only" in message or "offline" in message:
        return (
            "The script is running in local/offline mode and could not find every required model file. "
            "Point `MODEL_PATH` at a complete local checkpoint, or rerun with `--allow-network`."
        )
    return ""


def load_model(model_path: str, load_kwargs: dict[str, Any]) -> tuple[Any, Any, str]:
    config_kwargs = dict(load_kwargs)
    config_kwargs.pop("torch_dtype", None)

    try:
        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
    except Exception as exc:
        hint = model_load_hint(exc)
        if hint:
            raise RuntimeError(hint) from exc
        raise

    family, model_classes = detect_model_family(config)
    print(f"Detected {family} model")
    print(f"Config model_type: {getattr(config, 'model_type', 'unknown')}")
    print(f"Config architectures: {getattr(config, 'architectures', None)}")

    last_error: Exception | None = None
    for model_class in model_classes:
        try:
            print(f"Trying {model_class.__name__}...")
            model = model_class.from_pretrained(model_path, **load_kwargs)
            print(f"Loaded with {model_class.__name__}")
            return model, config, family
        except Exception as exc:
            last_error = exc
            print(f"{model_class.__name__} failed: {exc}")

    assert last_error is not None
    hint = model_load_hint(last_error)
    if hint:
        raise RuntimeError(hint) from last_error
    raise RuntimeError("Could not load the model with any compatible AutoModel class.") from last_error


def count_nonzero_parameters(param: torch.Tensor) -> int:
    tensor = param.detach()
    if tensor.is_meta:
        return 0
    if tensor.is_sparse:
        return tensor._nnz()
    return int(torch.count_nonzero(tensor).item())


def get_model_statistics(model: Any, count_sparsity: bool = True) -> dict[str, Any]:
    total_params = 0
    trainable_params = 0
    nonzero_params = 0
    skipped_nonzero_params = 0
    parameter_bytes = 0
    dtype_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()

    with torch.no_grad():
        for param in model.parameters():
            n = param.numel()
            total_params += n
            parameter_bytes += n * param.element_size()
            dtype_counts[str(param.dtype)] += n
            device_counts[str(param.device)] += n

            if param.requires_grad:
                trainable_params += n

            if count_sparsity:
                if param.is_meta:
                    skipped_nonzero_params += n
                else:
                    nonzero_params += count_nonzero_parameters(param)

    frozen_params = total_params - trainable_params

    if count_sparsity and total_params > skipped_nonzero_params:
        counted_params = total_params - skipped_nonzero_params
        sparsity = 100.0 * (1 - nonzero_params / counted_params)
        active_ratio = 100.0 * nonzero_params / counted_params
    else:
        sparsity = None
        active_ratio = None

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "nonzero_params": nonzero_params if count_sparsity else None,
        "skipped_nonzero_params": skipped_nonzero_params,
        "sparsity_percent": sparsity,
        "active_ratio_percent": active_ratio,
        "parameter_bytes": parameter_bytes,
        "dtype_counts": dtype_counts,
        "device_counts": device_counts,
    }


def format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return f"{number:.2f} TB"


def print_counter(title: str, values: Counter[str]) -> None:
    print(f"\n{title}")
    for name, count in values.most_common():
        print(f"  {name:<16}: {count:,}")


def print_statistics(model: Any, config: Any, family: str, stats: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)
    print(f"Model Class           : {model.__class__.__name__}")
    print(f"Model Family          : {family}")
    print(f"Model Type            : {getattr(config, 'model_type', 'unknown')}")
    print(f"Encoder-Decoder       : {getattr(config, 'is_encoder_decoder', False)}")
    print(f"Total Parameters      : {stats['total_params']:,}")
    print(f"Trainable Parameters  : {stats['trainable_params']:,}")
    print(f"Frozen Parameters     : {stats['frozen_params']:,}")

    if stats["nonzero_params"] is not None:
        print(f"Non-Zero Parameters   : {stats['nonzero_params']:,}")
        if stats["skipped_nonzero_params"]:
            print(f"Skipped Meta Params   : {stats['skipped_nonzero_params']:,}")
        print(f"Active Parameters     : {stats['nonzero_params'] / 1e6:.2f} M")
        print(f"Active Ratio          : {stats['active_ratio_percent']:.2f}%")
        print(f"Sparsity              : {stats['sparsity_percent']:.2f}%")
    else:
        print("Non-Zero Parameters   : skipped")
        print("Active Ratio          : skipped")
        print("Sparsity              : skipped")

    print(f"Model Size            : {stats['total_params'] / 1e6:.2f} M")
    print(f"Parameter Memory      : {format_bytes(stats['parameter_bytes'])}")
    print_counter("Parameter dtypes", stats["dtype_counts"])
    print_counter("Parameter devices", stats["device_counts"])
    print("=" * 60)


def main() -> None:
    args = parse_args()
    print(f"Loading model from: {args.model_path}")
    print(f"Local files only: {LOCAL_FILES_ONLY and not args.allow_network}")

    load_kwargs = build_load_kwargs(args)
    model, config, family = load_model(args.model_path, load_kwargs)

    stats = get_model_statistics(
        model=model,
        count_sparsity=COUNT_SPARSITY and not args.skip_sparsity,
    )
    print_statistics(model, config, family, stats)


if __name__ == "__main__":
    main()
