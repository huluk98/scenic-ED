#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROMPT_FIELDS = ("prompt", "anchor", "instruction", "question", "input", "x")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "target", "y")
ID_FIELDS = ("id", "sample_id", "uid", "index")


README_TEXT = """# Benchmark Difficulty Labeling Guidance

Fill the `difficulty` column in `benchmark_difficulty_template.csv` with one of:

- easy: Direct, single-intent, single-device command with explicit action and target.
  Example: "Turn on the bedroom light."
- medium: Paraphrased, indirect, multi-device, or slightly contextual command, but still unambiguous.
  Example: "It is too dark in the bedroom."
- hard: Indirect, compositional, conditional, rare-device, multi-step, negated, or potentially ambiguous command.
  Example: "If the room gets too warm, lower the AC and turn off the heater."

The sparsity experiment runner joins labels by `id`/`sample_id` first, then by exact `input`.
Do not leave difficulty blank for final benchmark runs.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a blank easy/medium/hard difficulty labeling CSV.")
    parser.add_argument("--benchmark_path", required=True)
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        rows.append(value)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "examples", "items"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        return [value]
    raise ValueError(f"{path} must be JSON, JSONL, or contain object records.")


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def row_id(record: dict[str, Any], index: int) -> str:
    for field in ID_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(index)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records(Path(args.benchmark_path))

    csv_path = output_dir / "benchmark_difficulty_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "input", "target", "difficulty"])
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow(
                {
                    "id": row_id(record, index),
                    "input": first_text(record, PROMPT_FIELDS),
                    "target": first_text(record, RESPONSE_FIELDS),
                    "difficulty": "",
                }
            )

    readme_path = output_dir / "benchmark_difficulty_template_README.md"
    readme_path.write_text(README_TEXT, encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
