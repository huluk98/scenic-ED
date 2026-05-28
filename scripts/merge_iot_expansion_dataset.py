#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_EXPANSION = PROJECT_ROOT / "generated" / "iot_single_device_expansion.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "619_Luke_REPAIRED_plus_iot_expansion.json"
DEFAULT_REPORT = PROJECT_ROOT / "generated" / "iot_merge_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge repaired SCENIC data with the single-device IoT expansion.")
    parser.add_argument("--original", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--expansion", default=str(DEFAULT_EXPANSION))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"[\s。！？!?,，、；;：:]+", "", clean_text(prompt))


def load_dataset(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item.keys()) != {"prompt", "response"}:
            raise ValueError(f"{path}:{index} must contain exactly prompt and response")
        prompt = clean_text(item["prompt"])
        response = clean_text(item["response"])
        if not prompt or not response:
            raise ValueError(f"{path}:{index} has an empty prompt or response")
        rows.append({"prompt": prompt, "response": response})
    return rows


def merge_rows(original: list[dict[str, str]], expansion: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    merged: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    prompt_to_response: dict[str, str] = {}
    conflicts: list[dict[str, str]] = []
    skipped_duplicates: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for source_name, rows in (("original", original), ("expansion", expansion)):
        for row in rows:
            pair_key = (normalize_prompt(row["prompt"]), row["response"])
            prompt_key = pair_key[0]

            if pair_key in seen_pairs:
                skipped_duplicates[source_name] += 1
                continue
            existing_response = prompt_to_response.get(prompt_key)
            if existing_response is not None and existing_response != row["response"]:
                conflicts.append(
                    {
                        "source": source_name,
                        "prompt": row["prompt"],
                        "existing_response": existing_response,
                        "new_response": row["response"],
                    }
                )
                continue

            merged.append(row)
            seen_pairs.add(pair_key)
            prompt_to_response[prompt_key] = row["response"]
            source_counts[source_name] += 1

    response_counts = Counter(row["response"] for row in merged)
    report = {
        "original_input_count": len(original),
        "expansion_input_count": len(expansion),
        "merged_count": len(merged),
        "unique_prompts": len({normalize_prompt(row["prompt"]) for row in merged}),
        "unique_responses": len(response_counts),
        "source_counts": dict(source_counts),
        "skipped_exact_duplicate_pairs": dict(skipped_duplicates),
        "prompt_response_conflict_count": len(conflicts),
        "prompt_response_conflicts": conflicts[:100],
        "top_50_responses": [
            {"response": response, "count": count}
            for response, count in response_counts.most_common(50)
        ],
    }
    return merged, report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    original_path = Path(args.original).expanduser()
    expansion_path = Path(args.expansion).expanduser()
    output_path = Path(args.output).expanduser()
    report_path = Path(args.report).expanduser()

    original = load_dataset(original_path)
    expansion = load_dataset(expansion_path)
    merged, report = merge_rows(original, expansion)
    write_json(output_path, merged)
    write_json(report_path, report)

    print(f"original_input_count: {report['original_input_count']}")
    print(f"expansion_input_count: {report['expansion_input_count']}")
    print(f"merged_count: {report['merged_count']}")
    print(f"unique_prompts: {report['unique_prompts']}")
    print(f"unique_responses: {report['unique_responses']}")
    print(f"prompt_response_conflict_count: {report['prompt_response_conflict_count']}")
    print(f"output: {output_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
