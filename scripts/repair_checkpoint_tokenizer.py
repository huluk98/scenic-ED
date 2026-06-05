#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scenic_train_chatlm_sft import (
    CUSTOM_CODE_GLOBS,
    TOKENIZER_ASSET_FILENAMES,
    custom_code_filenames_from_config,
    read_json_dict,
    repair_tokenizer_files_for_auto_load,
)


ASSET_ALLOW_PATTERNS = (
    "config.json",
    "generation_config.json",
    *TOKENIZER_ASSET_FILENAMES,
    "*.model",
    *CUSTOM_CODE_GLOBS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair tokenizer and custom modeling files in a SCENIC checkpoint directory."
    )
    parser.add_argument("--checkpoint", required=True, help="Fine-tuned checkpoint directory to repair.")
    parser.add_argument(
        "--source-tokenizer",
        default=None,
        help="Optional base model/tokenizer directory or Hugging Face model ID to copy assets from.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve Hugging Face source assets from the local cache only.",
    )
    return parser.parse_args()


def resolve_source_assets(source: str | None, local_files_only: bool) -> Path | None:
    if not source:
        return None

    source_path = Path(source).expanduser()
    if source_path.is_dir():
        return source_path

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            f"Source assets '{source}' are not a local directory, and huggingface_hub is not available. "
            "Install huggingface_hub or pass SOURCE_ASSET_DIR=/path/to/local/base_model."
        ) from exc

    try:
        snapshot_dir = snapshot_download(
            repo_id=source,
            allow_patterns=list(dict.fromkeys(ASSET_ALLOW_PATTERNS)),
            local_files_only=local_files_only,
        )
    except Exception as exc:  # pragma: no cover - depends on HF cache/network
        mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub"
        raise RuntimeError(
            f"Could not resolve source assets for '{source}' from {mode}. "
            "Pass a local SOURCE_ASSET_DIR or run once with LOCAL_FILES_ONLY=0 so the source assets can be cached."
        ) from exc

    return Path(snapshot_dir).expanduser()


def missing_required_custom_code(checkpoint: Path, source: Path | None) -> list[str]:
    wanted = set(custom_code_filenames_from_config(read_json_dict(checkpoint / "config.json")))
    wanted.update(custom_code_filenames_from_config(read_json_dict(checkpoint / "tokenizer_config.json")))
    if source is not None:
        wanted.update(custom_code_filenames_from_config(read_json_dict(source / "config.json")))
        wanted.update(custom_code_filenames_from_config(read_json_dict(source / "tokenizer_config.json")))
    return sorted(filename for filename in wanted if not (checkpoint / filename).is_file())


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser()
    try:
        source = resolve_source_assets(args.source_tokenizer, local_files_only=args.local_files_only)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    repair_tokenizer_files_for_auto_load(checkpoint, source_dir=source)
    missing = missing_required_custom_code(checkpoint, source)
    if missing:
        missing_text = ", ".join(missing)
        print(
            f"ERROR: checkpoint is still missing required custom modeling/tokenization code: {missing_text}",
            file=sys.stderr,
        )
        if source is not None:
            print(f"Resolved source assets from: {source}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Repaired tokenizer/custom-code files for: {checkpoint}")
    if source is not None:
        print(f"Source assets: {source}")


if __name__ == "__main__":
    main()
