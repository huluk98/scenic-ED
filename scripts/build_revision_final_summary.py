#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DIFFICULTY_FIELDS = ("difficulty", "complexity", "level")
EXPECTED_LEGACY_METHODS = (
    {"method": "magnitude", "target_sparsity": 0.3},
    {"method": "wanda", "target_sparsity": 0.3},
    {"method": "gradient", "target_sparsity": 0.3},
    {"method": "magnitude", "target_sparsity": 0.5},
    {"method": "wanda", "target_sparsity": 0.5},
    {"method": "gradient", "target_sparsity": 0.5},
    {"method": "nvidia24", "target_sparsity": 0.5},
)
EXPECTED_PROGRESSIVE_TARGETS = (0.3, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one final SCENIC revision summary JSON.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--regular-checkpoint", required=True)
    parser.add_argument("--contrastive-checkpoint", required=True)
    parser.add_argument("--regular-train-json", required=True)
    parser.add_argument("--contrastive-train-json", required=True)
    parser.add_argument("--legacy-50-json", required=True)
    parser.add_argument("--legacy-30-json", required=True)
    parser.add_argument("--regular-sparsity-summary-csv", required=True)
    parser.add_argument("--contrastive-sparsity-summary-csv", required=True)
    parser.add_argument("--onnx-regular-table", default=None)
    parser.add_argument("--onnx-contrastive-table", default=None)
    return parser.parse_args()


def read_json_if_exists(path: str | Path | None) -> dict[str, Any] | None:
    if path in (None, ""):
        return None
    json_path = Path(path).expanduser()
    if not json_path.exists() or json_path.is_dir():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{json_path} must contain a JSON object.")
    return value


def read_records(path: str | Path) -> list[dict[str, Any]]:
    data_path = Path(path).expanduser()
    if not data_path.exists():
        return []
    if data_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    records.append(row)
        return records

    with data_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "examples", "items"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [value]
    return []


def normalize_difficulty(value: Any) -> str | None:
    if value in (None, ""):
        return None
    label = str(value).strip().lower()
    aliases = {"simple": "easy", "moderate": "medium", "med": "medium", "complex": "hard"}
    label = aliases.get(label, label)
    return label if label in {"easy", "medium", "hard"} else None


def difficulty_from_record(record: dict[str, Any]) -> str | None:
    for field in DIFFICULTY_FIELDS:
        difficulty = normalize_difficulty(record.get(field))
        if difficulty is not None:
            return difficulty
    return None


def benchmark_difficulty_by_index(records: list[dict[str, Any]]) -> dict[int, str]:
    difficulty: dict[int, str] = {}
    for index, record in enumerate(records):
        label = difficulty_from_record(record)
        if label is not None:
            difficulty[index] = label
    return difficulty


def metric_pair(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {"available": False}
    return {
        "available": True,
        "total": metrics.get("total", 0),
        "em1": metrics.get("em1", 0.0),
        "em5": metrics.get("em5", 0.0),
        "em1_percent": metrics.get("em1_percent", 0.0),
        "em5_percent": metrics.get("em5_percent", 0.0),
    }


def phase_metrics(method_info: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_info = method_info.get(phase)
    if not isinstance(phase_info, dict):
        return {
            "training": {"available": False},
            "benchmark": {"available": False},
            "benchmark_by_difficulty": {"available": False},
        }
    return {
        "training": metric_pair(phase_info.get("training")),
        "benchmark": metric_pair(phase_info.get("benchmark")),
        "benchmark_by_difficulty": {"available": False},
    }


def outputs_for_phase(report: dict[str, Any] | None, phase: str, dataset: str) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    phase_info = report.get(phase)
    if not isinstance(phase_info, dict):
        return []
    evaluations = phase_info.get("evaluations")
    if not isinstance(evaluations, dict):
        return []
    dataset_info = evaluations.get(dataset)
    if not isinstance(dataset_info, dict):
        return []
    outputs = dataset_info.get("outputs")
    return [row for row in outputs if isinstance(row, dict)] if isinstance(outputs, list) else []


def difficulty_breakdown(outputs: list[dict[str, Any]], difficulty_by_index: dict[int, str]) -> dict[str, Any]:
    groups = {label: [] for label in ("easy", "medium", "hard")}
    for output in outputs:
        try:
            index = int(output.get("index"))
        except (TypeError, ValueError):
            continue
        difficulty = difficulty_by_index.get(index)
        if difficulty in groups:
            groups[difficulty].append(output)

    if not any(groups.values()):
        return {"available": False}

    breakdown: dict[str, Any] = {"available": True}
    for label, rows in groups.items():
        total = len(rows)
        em1_correct = sum(1 for row in rows if bool(row.get("em1_correct")))
        em5_correct = sum(1 for row in rows if bool(row.get("em5_correct")))
        em1 = em1_correct / total if total else 0.0
        em5 = em5_correct / total if total else 0.0
        breakdown[label] = {
            "total": total,
            "em1": em1,
            "em5": em5,
            "em1_percent": em1 * 100.0,
            "em5_percent": em5 * 100.0,
        }
    return breakdown


def summarize_legacy_json(
    aggregate_json: str | Path,
    target_sparsity: float,
    difficulty_by_index: dict[int, str],
) -> dict[str, list[dict[str, Any]]]:
    aggregate = read_json_if_exists(aggregate_json)
    if aggregate is None:
        return {}

    output: dict[str, list[dict[str, Any]]] = {}
    models = aggregate.get("models")
    if not isinstance(models, dict):
        return output

    for model_label, model_info in models.items():
        if not isinstance(model_info, dict):
            continue
        methods = model_info.get("methods")
        if not isinstance(methods, dict):
            continue
        entries = output.setdefault(model_label, [])
        for method, method_info in sorted(methods.items()):
            if not isinstance(method_info, dict):
                continue
            individual_report = read_json_if_exists(method_info.get("report_json", ""))
            original = phase_metrics(method_info, "original_before_prune")
            pruned = phase_metrics(method_info, "pruned_after_50_percent")
            original["benchmark_by_difficulty"] = difficulty_breakdown(
                outputs_for_phase(individual_report, "original_before_prune", "benchmark"),
                difficulty_by_index,
            )
            pruned["benchmark_by_difficulty"] = difficulty_breakdown(
                outputs_for_phase(individual_report, "pruned_after_50_percent", "benchmark"),
                difficulty_by_index,
            )
            entries.append(
                {
                    "method": method,
                    "target_sparsity": target_sparsity,
                    "report_json": method_info.get("report_json"),
                    "pruned_model_path": method_info.get("pruned_model_path"),
                    "pruning": method_info.get("pruning", {}),
                    "original_before_prune": original,
                    "pruned_after": pruned,
                }
            )
    return output


def parse_number(value: Any) -> float | int | str | None:
    if value in (None, ""):
        return None
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def csv_metric_block(row: dict[str, Any]) -> dict[str, Any]:
    def block(group: str) -> dict[str, Any]:
        suffix = "overall" if group == "overall" else group
        return {
            "total": parse_number(row.get(f"count_{suffix}")),
            "em1": parse_number(row.get(f"em1_{suffix}")),
            "em5": parse_number(row.get(f"em5_{suffix}")),
            "em1_retention": parse_number(row.get(f"em1_retention_{suffix}")),
            "em5_retention": parse_number(row.get(f"em5_retention_{suffix}")),
        }

    return {
        "overall": block("overall"),
        "easy": block("easy"),
        "medium": block("medium"),
        "hard": block("hard"),
    }


def summarize_sparsity_csv(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summaries: list[dict[str, Any]] = []
    for row in rows:
        summaries.append(
            {
                "pruning_mode": row.get("pruning_mode"),
                "pruning_method": row.get("pruning_method"),
                "target_sparsity": parse_number(row.get("target_sparsity")),
                "targeted_linear_sparsity_actual": parse_number(row.get("targeted_linear_sparsity_actual")),
                "whole_model_sparsity_actual": parse_number(row.get("whole_model_sparsity_actual")),
                "seed": parse_number(row.get("seed")),
                "training": {
                    "available": False,
                    "reason": "The linear sparsity matrix uses training data for recovery fine-tuning but does not separately evaluate training EM.",
                },
                "benchmark": csv_metric_block(row),
                "checkpoint_path": row.get("checkpoint_path"),
                "mask_path": row.get("mask_path"),
            }
        )
    return summaries


def merge_legacy_by_model(
    legacy_30: dict[str, list[dict[str, Any]]],
    legacy_50: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    labels = sorted(set(legacy_30) | set(legacy_50))
    merged: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        entries = [*legacy_30.get(label, []), *legacy_50.get(label, [])]
        merged[label] = sorted(entries, key=lambda item: (float(item["target_sparsity"]), item["method"]))
    return merged


def build_model_summary(
    *,
    label: str,
    checkpoint: str,
    train_json: str,
    legacy_entries: list[dict[str, Any]],
    sparsity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    progressive_rows = [
        row for row in sparsity_rows
        if row.get("pruning_mode") == "progressive"
        and row.get("target_sparsity") in EXPECTED_PROGRESSIVE_TARGETS
    ]
    return {
        "checkpoint": checkpoint,
        "train_json": train_json,
        "original_pruning_methods": legacy_entries,
        "added_progressive_pruning": progressive_rows,
        "linear_sparsity_matrix": sparsity_rows,
        "counts": {
            "original_pruning_method_outputs": len(legacy_entries),
            "added_progressive_outputs": len(progressive_rows),
        },
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    benchmark_records = read_records(args.benchmark_json)
    difficulty_by_index = benchmark_difficulty_by_index(benchmark_records)
    legacy_30 = summarize_legacy_json(args.legacy_30_json, 0.3, difficulty_by_index)
    legacy_50 = summarize_legacy_json(args.legacy_50_json, 0.5, difficulty_by_index)
    legacy_by_model = merge_legacy_by_model(legacy_30, legacy_50)
    regular_sparsity = summarize_sparsity_csv(args.regular_sparsity_summary_csv)
    contrastive_sparsity = summarize_sparsity_csv(args.contrastive_sparsity_summary_csv)

    models = {
        "regular_sft": build_model_summary(
            label="regular_sft",
            checkpoint=args.regular_checkpoint,
            train_json=args.regular_train_json,
            legacy_entries=legacy_by_model.get("regular_sft", []),
            sparsity_rows=regular_sparsity,
        ),
        "contrastive_sft": build_model_summary(
            label="contrastive_sft",
            checkpoint=args.contrastive_checkpoint,
            train_json=args.contrastive_train_json,
            legacy_entries=legacy_by_model.get("contrastive_sft", []),
            sparsity_rows=contrastive_sparsity,
        ),
    }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "sft_epochs": args.epochs,
        "accuracy_definition": "accuracy is exact-match@1 / EM@1",
        "expected_outputs_per_model": {
            "original_pruning_methods": len(EXPECTED_LEGACY_METHODS),
            "original_pruning_method_plan": list(EXPECTED_LEGACY_METHODS),
            "added_progressive_pruning": len(EXPECTED_PROGRESSIVE_TARGETS),
            "added_progressive_targets": list(EXPECTED_PROGRESSIVE_TARGETS),
            "nvidia24_note": "nvidia24 is listed only at 50% because 2:4 pruning is effectively 50% selected-weight sparsity.",
        },
        "inputs": {
            "benchmark_json": args.benchmark_json,
            "benchmark_difficulty_labels_found": bool(difficulty_by_index),
            "regular_train_json": args.regular_train_json,
            "contrastive_train_json": args.contrastive_train_json,
        },
        "source_reports": {
            "legacy_50_json": args.legacy_50_json,
            "legacy_30_json": args.legacy_30_json,
            "regular_sparsity_summary_csv": args.regular_sparsity_summary_csv,
            "contrastive_sparsity_summary_csv": args.contrastive_sparsity_summary_csv,
            "onnx_regular_table": args.onnx_regular_table,
            "onnx_contrastive_table": args.onnx_contrastive_table,
        },
        "models": models,
        "notes": [
            "Legacy original pruning method entries include training and benchmark EM@1/EM@5 before and after pruning.",
            "Benchmark difficulty breakdowns are included when the benchmark records contain difficulty, complexity, or level labels.",
            "Added progressive pruning entries are benchmark summaries from the linear sparsity matrix; training data is used for recovery fine-tuning and is not separately reported by that matrix.",
        ],
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    write_json(args.output_json, build_summary(args))
    print(f"Wrote final revision summary JSON: {args.output_json}")


if __name__ == "__main__":
    main()
