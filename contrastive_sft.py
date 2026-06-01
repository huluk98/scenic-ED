#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import os
import signal
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import (  # noqa: E402
    ContrastiveSFTConfig,
    print_dataset_summary,
    train_contrastive_triplet_sft,
)


# =====================================================
# CHANGE THESE PATHS DIRECTLY
# =====================================================
MODEL_PATH = str(PROJECT_ROOT / "models" / "ChatLM-mini-Chinese-local")
# Examples:
# MODEL_PATH = str(PROJECT_ROOT / "models" / "ChatLM-mini-Chinese-local")
# MODEL_PATH = "/nvme1/home/luke/models/chatlm"
# MODEL_PATH = "charent/ChatLM-mini-Chinese"

TRAIN_JSON = str(PROJECT_ROOT / "data" / "SCENIC_full_anchor_positive_negative.json")
# Example:
# TRAIN_JSON = "/nvme1/home/luke/Encoder-Chinese-SLM/data/scenic/SCENIC_full_anchor_positive_negative.json"

OUTPUT_DIR = str(PROJECT_ROOT / "models" / "chatlm_scenic_triplet_sft")
# Example:
# OUTPUT_DIR = "/nvme1/home/luke/Encoder-Chinese-SLM/sft_contrastive"

EPOCHS = 3
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 96
MAX_GRAD_NORM = 1.0
SEED = 42
MAX_EXAMPLES = None
CACHE_DIR = None
LOCAL_FILES_ONLY = True
DEVICE = "auto"
FP16 = False
BF16 = True
NUM_WORKERS = 4
LOG_EVERY = 20
SAVE_EVERY_STEPS = 0
SAVE_EPOCH_CHECKPOINTS = False
EXPECTED_GPUS = 8
DDP_TIMEOUT_MINUTES = 10

# Triplet SFT objective controls.
ALIGNMENT_WEIGHT = 0.1
MARGIN = 0.5
NEGATIVE_FIELD = "negative"
# Use NEGATIVE_FIELD = "invalid_negative" if you want invalid hard negatives.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone contrastive triplet SFT runner for ChatLM-mini-Chinese."
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Hugging Face model id or local model directory.")
    parser.add_argument("--train-json", default=str(TRAIN_JSON), help="Contrastive .json or .jsonl path.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory for final model and checkpoints.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    parser.add_argument("--max-source-length", type=int, default=MAX_SOURCE_LENGTH)
    parser.add_argument("--max-target-length", type=int, default=MAX_TARGET_LENGTH)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-examples", type=int, default=MAX_EXAMPLES, help="Use only the first N tuples.")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=LOCAL_FILES_ONLY,
        help="Load model/tokenizer only from local files.",
    )
    parser.add_argument("--device", default=DEVICE, help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--fp16", action="store_true", default=FP16, help="Use CUDA fp16 autocast.")
    parser.add_argument("--bf16", action="store_true", default=BF16, help="Use CUDA bf16 autocast.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--save-every-steps", type=int, default=SAVE_EVERY_STEPS)
    parser.add_argument(
        "--epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=SAVE_EPOCH_CHECKPOINTS,
        help="Save checkpoint-epoch-N directories. Defaults off here to avoid long epoch-boundary saves.",
    )
    parser.add_argument("--alignment-weight", type=float, default=ALIGNMENT_WEIGHT)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--negative-field", default=NEGATIVE_FIELD)
    parser.add_argument("--expected-gpus", type=int, default=EXPECTED_GPUS)
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=int,
        default=DDP_TIMEOUT_MINUTES,
        help="NCCL/DDP timeout. A failed rank should abort instead of hanging forever.",
    )
    parser.add_argument(
        "--allow-single-gpu",
        action="store_true",
        help="Allow running without torchrun. By default this file expects 8 GPU torchrun training.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and summarize data without loading the model.")
    return parser.parse_args()


def contrastive_config_from_args(args: argparse.Namespace) -> ContrastiveSFTConfig:
    return ContrastiveSFTConfig(
        model_name_or_path=args.model,
        train_json=Path(args.train_json).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
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
        save_epoch_checkpoints=args.epoch_checkpoints,
        alignment_weight=args.alignment_weight,
        margin=args.margin,
        negative_field=args.negative_field,
    )


def print_run_config(args: argparse.Namespace) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print("Contrastive SFT run config:")
    print(f"  model: {args.model}")
    print(f"  train_json: {args.train_json}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size_per_gpu: {args.batch_size}")
    print(f"  gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"  bf16: {args.bf16}")
    print(f"  local_files_only: {args.local_files_only}")
    print(f"  expected_gpus: {args.expected_gpus}")
    print(f"  ddp_timeout_minutes: {args.ddp_timeout_minutes}")
    print(f"  epoch_checkpoints: {args.epoch_checkpoints}")
    print(f"  alignment_weight: {args.alignment_weight}")
    print(f"  margin: {args.margin}")
    print(f"  negative_field: {args.negative_field}")


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ["SCENIC_DDP_TIMEOUT_MINUTES"] = str(args.ddp_timeout_minutes)
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")


def cleanup_runtime() -> None:
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def install_signal_handlers() -> None:
    def handle_signal(signum: int, _frame: object) -> None:
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Received signal {signum}; cleaning up distributed/CUDA state.", flush=True)
        cleanup_runtime()
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, handle_signal)


def validate_gpu_launch(args: argparse.Namespace) -> None:
    if args.dry_run or args.allow_single_gpu:
        return

    import torch

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0

    if rank == 0:
        print("Torchrun/GPU launch check:")
        print(f"  WORLD_SIZE: {world_size}")
        print(f"  LOCAL_WORLD_SIZE: {local_world_size}")
        print(f"  torch.cuda.is_available: {cuda_available}")
        print(f"  torch.cuda.device_count: {cuda_count}")
        print(f"  CUDA_VISIBLE_DEVICES: {visible_devices}")

    if world_size <= 1:
        raise RuntimeError(
            "contrastive_sft.py is configured for multi-GPU training. "
            "Launch with `torchrun --nproc_per_node=8 contrastive_sft.py`, "
            "or pass `--allow-single-gpu` for a one-GPU/debug run."
        )
    if args.expected_gpus and world_size != args.expected_gpus:
        raise RuntimeError(
            f"Expected WORLD_SIZE={args.expected_gpus}, but torchrun started WORLD_SIZE={world_size}. "
            f"Use `torchrun --nproc_per_node={args.expected_gpus} contrastive_sft.py`."
        )
    if not cuda_available:
        raise RuntimeError("CUDA is not available, so NCCL/DDP cannot use the requested GPUs.")
    if cuda_count < local_world_size:
        raise RuntimeError(
            f"Only {cuda_count} CUDA device(s) are visible, but LOCAL_WORLD_SIZE={local_world_size}. "
            "Check CUDA_VISIBLE_DEVICES and nvidia-smi."
        )
    if local_rank >= cuda_count:
        raise RuntimeError(f"LOCAL_RANK={local_rank} is out of range for {cuda_count} visible CUDA device(s).")


def main() -> None:
    args = parse_args()
    configure_runtime(args)
    install_signal_handlers()
    atexit.register(cleanup_runtime)
    print_run_config(args)
    validate_gpu_launch(args)
    train_json = Path(args.train_json).expanduser()

    if args.dry_run:
        print_dataset_summary("contrastive", train_json, args.negative_field)
        return

    try:
        output_dir = train_contrastive_triplet_sft(contrastive_config_from_args(args))
        print(f"saved contrastive model to {output_dir}")
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
