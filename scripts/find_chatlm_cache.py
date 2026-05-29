#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


REQUIRED_FILES = ("config.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find a local Hugging Face cache snapshot for ChatLM-mini-Chinese.")
    parser.add_argument("--model-id", default="charent/ChatLM-mini-Chinese")
    parser.add_argument("--cache-dir", default=None, help="Defaults to HF_HOME/hub or ~/.cache/huggingface/hub.")
    return parser.parse_args()


def default_hub_cache() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def cache_repo_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def score_snapshot(path: Path) -> tuple[int, float]:
    score = 0
    for name in REQUIRED_FILES:
        if (path / name).exists():
            score += 10
    for name in ("tokenizer.json", "tokenizer_config.json", "spiece.model", "pytorch_model.bin", "model.safetensors"):
        if (path / name).exists():
            score += 1
    newest = max((item.stat().st_mtime for item in path.rglob("*") if item.exists()), default=path.stat().st_mtime)
    return score, newest


def main() -> None:
    args = parse_args()
    hub_cache = Path(args.cache_dir).expanduser() if args.cache_dir else default_hub_cache()
    snapshots_dir = hub_cache / cache_repo_name(args.model_id) / "snapshots"
    if not snapshots_dir.exists():
        raise SystemExit(f"No snapshots found at {snapshots_dir}")

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        raise SystemExit(f"No snapshot directories found at {snapshots_dir}")

    snapshots.sort(key=score_snapshot, reverse=True)
    best = snapshots[0].resolve()
    print(best)
    print()
    print("Use:")
    print("export HF_HUB_OFFLINE=1")
    print("export TRANSFORMERS_OFFLINE=1")
    print(
        "torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py "
        f"--mode regular --model '{best}' --local-files-only --epochs 3 --bf16 --batch-size 16"
    )


if __name__ == "__main__":
    main()
