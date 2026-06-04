#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "charent/ChatLM-mini-Chinese"
DEFAULT_OUTPUT = Path("models") / "ChatLM-mini-Chinese-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained local ChatLM-mini-Chinese directory from the Hugging Face cache."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=None, help="Defaults to HF_HOME/hub or ~/.cache/huggingface/hub.")
    parser.add_argument("--modules-dir", default=None, help="Defaults to HF_MODULES_CACHE or ~/.cache/huggingface/modules.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when required custom remote-code files are missing after materialization.",
    )
    return parser.parse_args()


def default_hub_cache() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path("~/.cache/huggingface/hub").expanduser()


def default_modules_cache() -> Path:
    value = os.environ.get("HF_MODULES_CACHE")
    if value:
        return Path(value).expanduser()
    return Path("~/.cache/huggingface/modules").expanduser()


def cache_repo_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def find_best_snapshot(model_id: str, hub_cache: Path) -> Path:
    snapshots_dir = hub_cache / cache_repo_name(model_id) / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(f"No snapshots found at {snapshots_dir}")
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"No snapshot directories found at {snapshots_dir}")
    snapshots.sort(key=snapshot_score, reverse=True)
    return snapshots[0].resolve()


def snapshot_score(path: Path) -> tuple[int, float]:
    score = 0
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "spiece.model"):
        if (path / name).exists():
            score += 10
    for pattern in ("*.bin", "*.safetensors", "*.py"):
        score += len(list(path.glob(pattern)))
    newest = max((item.stat().st_mtime for item in path.rglob("*") if item.exists()), default=path.stat().st_mtime)
    return score, newest


def copy_snapshot(snapshot: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in snapshot.iterdir():
        destination = output_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination, symlinks=False)
        else:
            shutil.copy2(item, destination, follow_symlinks=True)


def required_custom_code_files(model_dir: Path) -> list[str]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    filenames: list[str] = []
    for value in flatten_auto_map(config.get("auto_map")):
        module_reference = value.split("--", 1)[-1]
        module_name = module_reference.split(".", 1)[0]
        if module_name and not module_name.startswith("transformers"):
            filename = f"{module_name}.py"
            if filename not in filenames:
                filenames.append(filename)
    return filenames


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


def candidate_module_dirs(model_id: str, modules_cache: Path) -> list[Path]:
    if not modules_cache.exists():
        return []
    org, name = model_id.split("/", 1)
    candidates: list[Path] = []
    for path in modules_cache.rglob("*"):
        if not path.is_dir():
            continue
        parts = set(path.parts)
        text = "/".join(path.parts)
        if (org in parts and name in parts) or model_id in text or cache_repo_name(model_id) in text:
            if list(path.glob("*.py")):
                candidates.append(path)
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates


def copy_custom_code(model_id: str, output_dir: Path, modules_cache: Path) -> list[str]:
    copied: set[str] = set()
    for directory in candidate_module_dirs(model_id, modules_cache):
        for py_file in directory.glob("*.py"):
            shutil.copy2(py_file, output_dir / py_file.name)
            copied.add(py_file.name)
    return sorted(copied)


def main() -> None:
    args = parse_args()
    hub_cache = Path(args.cache_dir).expanduser() if args.cache_dir else default_hub_cache()
    modules_cache = Path(args.modules_dir).expanduser() if args.modules_dir else default_modules_cache()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        snapshot = find_best_snapshot(args.model_id, hub_cache)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    copy_snapshot(snapshot, output_dir)
    copied = copy_custom_code(args.model_id, output_dir, modules_cache)
    required = required_custom_code_files(output_dir)
    missing = [name for name in required if not (output_dir / name).exists()]

    print(f"Snapshot: {snapshot}")
    print(f"Output: {output_dir}")
    print(f"Copied custom code: {', '.join(copied) if copied else 'none found'}")
    if missing:
        print(f"Missing custom code files: {', '.join(missing)}")
        print("If this list is non-empty, copy those .py files from another machine that can download the model.")
        if args.strict:
            raise SystemExit(3)
    print()
    print("Use:")
    print("export HF_HUB_OFFLINE=1")
    print("export TRANSFORMERS_OFFLINE=1")
    print(
        "torchrun --standalone --nproc_per_node=8 scripts/scenic_train_chatlm_sft.py "
        f"--mode regular --model '{output_dir}' --local-files-only --epochs 5 --fp16 --batch-size 16"
    )


if __name__ == "__main__":
    main()
