#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import (  # noqa: E402
    CONTRASTIVE_OUTPUT_DIR,
    CONTRASTIVE_TRAIN_JSON,
    MODEL_NAME_OR_PATH,
    ContrastiveSFTConfig,
    print_dataset_summary,
    train_contrastive_triplet_sft,
)


# =====================================================
# EDIT THESE DEFAULTS IF YOU WANT TO RUN WITHOUT FLAGS
# =====================================================
MODEL_PATH = MODEL_NAME_OR_PATH
# Examples:
# MODEL_PATH = "charent/ChatLM-mini-Chinese"
# MODEL_PATH = "models/ChatLM-mini-Chinese"
# MODEL_PATH = "models/ChatLM-mini-Chinese-local"
# MODEL_PATH = "/nvme1/home/luke/models/chatlm"

TRAIN_JSON = CONTRASTIVE_TRAIN_JSON
OUTPUT_DIR = CONTRASTIVE_OUTPUT_DIR

EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 96
MAX_GRAD_NORM = 1.0
SEED = 42
MAX_EXAMPLES = None
CACHE_DIR = None
LOCAL_FILES_ONLY = False
DEVICE = "auto"
FP16 = False
BF16 = False
NUM_WORKERS = 0
LOG_EVERY = 20
SAVE_EVERY_STEPS = 0

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
    parser.add_argument("--alignment-weight", type=float, default=ALIGNMENT_WEIGHT)
    parser.add_argument("--margin", type=float, default=MARGIN)
    parser.add_argument("--negative-field", default=NEGATIVE_FIELD)
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
        alignment_weight=args.alignment_weight,
        margin=args.margin,
        negative_field=args.negative_field,
    )


def main() -> None:
    args = parse_args()
    train_json = Path(args.train_json).expanduser()

    if args.dry_run:
        print_dataset_summary("contrastive", train_json, args.negative_field)
        return

    output_dir = train_contrastive_triplet_sft(contrastive_config_from_args(args))
    print(f"saved contrastive model to {output_dir}")


if __name__ == "__main__":
    main()
