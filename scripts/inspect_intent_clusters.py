#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_smart_home_dataset import infer_intents_from_pair, intent_key  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect repaired smart-home rows grouped by inferred canonical intent.")
    parser.add_argument("--input", default=str(DEFAULT_DATASET))
    parser.add_argument("--limit", type=int, default=0, help="Maximum inconsistent clusters to print; 0 means all.")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    rows = load_rows(input_path)
    clusters: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    unresolved: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        intents = infer_intents_from_pair(row.get("prompt", ""), row.get("response", ""))
        if not intents:
            unresolved.append({"index": index, "prompt": row.get("prompt", ""), "response": row.get("response", "")})
            continue
        clusters[intent_key(intents)].append((index, row))

    inconsistent = []
    for key, items in sorted(clusters.items()):
        responses = sorted({row["response"] for _, row in items})
        if len(responses) > 1:
            inconsistent.append(
                {
                    "intent_key": key,
                    "count": len(items),
                    "responses": responses,
                    "examples": [
                        {"index": index, "prompt": row["prompt"], "response": row["response"]}
                        for index, row in items[:10]
                    ],
                }
            )

    print(f"dataset: {input_path}")
    print(f"rows: {len(rows)}")
    print(f"intent_clusters: {len(clusters)}")
    print(f"unresolved_rows: {len(unresolved)}")
    print(f"inconsistent_clusters: {len(inconsistent)}")

    if unresolved:
        print("\nUNRESOLVED_ROWS")
        print(json.dumps(unresolved[:20], ensure_ascii=False, indent=2))

    if inconsistent:
        print("\nINCONSISTENT_CLUSTERS")
        shown = inconsistent if args.limit <= 0 else inconsistent[: args.limit]
        print(json.dumps(shown, ensure_ascii=False, indent=2))
    else:
        print("No inconsistent clusters found.")


if __name__ == "__main__":
    main()
