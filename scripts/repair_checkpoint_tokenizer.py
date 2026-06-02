#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from scenic_train_chatlm_sft import repair_tokenizer_files_for_auto_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair tokenizer files in a SCENIC checkpoint directory.")
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint directory to repair.")
    parser.add_argument("--source-tokenizer", default=None, help="Optional base model/tokenizer directory to copy assets from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    source = Path(args.source_tokenizer).expanduser() if args.source_tokenizer else None
    repair_tokenizer_files_for_auto_load(checkpoint, source_dir=source)
    print(f"Repaired tokenizer files for: {checkpoint}")


if __name__ == "__main__":
    main()
