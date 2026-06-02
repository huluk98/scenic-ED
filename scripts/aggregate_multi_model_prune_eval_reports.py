#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aggregate_prune_eval_reports import (
    PHASES,
    canonical_method_name,
    compact_dataset_metrics,
    prediction_diagnostics,
    read_json,
    report_phase_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SCENIC prune/eval reports across multiple SFT models.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--model", action="append", default=[], help="Model metadata as label=/path/to/model.")
    parser.add_argument("--train-json", action="append", default=[], help="Training data metadata as label=/path/train.json.")
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Per-method report as label:method=/path/prune_eval_report.json.",
    )
    parser.add_argument("--diagnostic-examples", type=int, default=5)
    return parser.parse_args()


def parse_label_value(value: str, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"{option} must be label=value, got {value!r}")
    label, raw = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"{option} has an empty label: {value!r}")
    if not raw:
        raise ValueError(f"{option} has an empty value for {label!r}.")
    return label, raw


def parse_report(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise ValueError(f"--report must be label:method=/path/report.json, got {value!r}")
    label_method, raw_path = value.split("=", 1)
    label, method = label_method.split(":", 1)
    label = label.strip()
    method = canonical_method_name(method)
    if not label or not method:
        raise ValueError(f"--report has an empty label or method: {value!r}")
    report_path = Path(raw_path).expanduser()
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report for {label}:{method}: {report_path}")
    return label, method, report_path


def collect_mapping(values: list[str], option: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        label, raw = parse_label_value(value, option)
        mapping[label] = raw
    return mapping


def build_model_summary(
    *,
    label: str,
    model_path: str,
    train_json: str | None,
    method_reports: dict[str, Path],
    diagnostic_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not method_reports:
        raise ValueError(f"No method reports were provided for model label {label!r}.")

    methods: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    first_report: dict[str, Any] | None = None

    for method, report_path in sorted(method_reports.items()):
        report = read_json(report_path)
        if first_report is None:
            first_report = report

        method_summary: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for phase in PHASES:
            phase_metrics = compact_dataset_metrics(report_phase_metrics(report, phase))
            method_summary[phase] = phase_metrics
            full_phase = report.get(phase, {})
            full_evaluations = full_phase.get("evaluations", {}) if isinstance(full_phase, dict) else {}
            phase_diagnostics: dict[str, Any] = {}

            for dataset, metrics in phase_metrics.items():
                full_dataset_metrics = (
                    full_evaluations.get(dataset, {}) if isinstance(full_evaluations, dict) else {}
                )
                if isinstance(full_dataset_metrics, dict):
                    phase_diagnostics[dataset] = prediction_diagnostics(
                        full_dataset_metrics,
                        limit=diagnostic_examples,
                    )
                table.append(
                    {
                        "model_label": label,
                        "method": method,
                        "phase": phase,
                        "dataset": dataset,
                        "total": metrics["total"],
                        "em1": metrics["em1"],
                        "em5": metrics["em5"],
                        "em1_percent": metrics["em1_percent"],
                        "em5_percent": metrics["em5_percent"],
                        "accuracy": metrics["accuracy"],
                        "accuracy_percent": metrics["accuracy_percent"],
                    }
                )
            diagnostics[phase] = phase_diagnostics

        methods[method] = {
            "report_json": str(report_path),
            "pruned_model_path": report.get("pruned_model_path"),
            "generation": report.get("generation", {}),
            "pruning": report.get("pruning", {}),
            "diagnostics": diagnostics,
            **method_summary,
        }

    assert first_report is not None
    return (
        {
            "model_path": model_path,
            "train_json": train_json,
            "generation": first_report.get("generation", {}),
            "datasets": first_report.get("datasets", {}),
            "original_before_prune": compact_dataset_metrics(
                report_phase_metrics(first_report, "original_before_prune")
            ),
            "methods": methods,
        },
        table,
    )


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    model_paths = collect_mapping(args.model, "--model")
    train_jsons = collect_mapping(args.train_json, "--train-json")
    grouped_reports: dict[str, dict[str, Path]] = {}
    for label, method, report_path in [parse_report(value) for value in args.report]:
        grouped_reports.setdefault(label, {})[method] = report_path

    labels = sorted(set(model_paths) | set(train_jsons) | set(grouped_reports))
    if not labels:
        raise ValueError("At least one model/report group is required.")

    models: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for label in labels:
        model_summary, model_table = build_model_summary(
            label=label,
            model_path=model_paths.get(label, "unknown"),
            train_json=train_jsons.get(label),
            method_reports=grouped_reports.get(label, {}),
            diagnostic_examples=args.diagnostic_examples,
        )
        models[label] = model_summary
        table.extend(model_table)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": args.base_model,
        "epochs": args.epochs,
        "target_sparsity": args.sparsity,
        "accuracy_definition": "accuracy is exact-match@1 / EM@1",
        "models": models,
        "table": table,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).expanduser()
    write_json(output_json, build_report(args))
    print(f"Wrote multi-model prune/eval JSON: {output_json}")


if __name__ == "__main__":
    main()
