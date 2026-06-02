#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASETS = ("benchmark", "training")
PHASES = ("original_before_prune", "pruned_after_50_percent")
METHOD_ALIASES = {
    "2:4": "nvidia24",
    "2of4": "nvidia24",
    "nvidia": "nvidia24",
    "nvidia24": "nvidia24",
}
KNOWN_METHOD_DIRS = {
    "magnitude_50": "magnitude",
    "wanda_50": "wanda",
    "gradient_50": "gradient",
    "nvidia_50": "nvidia24",
    "nvidia24_50": "nvidia24",
    "2of4_50": "nvidia24",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-method SCENIC prune/eval JSON reports.")
    parser.add_argument("--run-dir", default=None, help="Directory containing per-method *_50/prune_eval_report.json files.")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--contrastive-model", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--contrastive-train-json", default=None)
    parser.add_argument("--report", action="append", default=[], help="Method report in method=/path/report.json form.")
    return parser.parse_args()


def canonical_method_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace("_", "")
    return METHOD_ALIASES.get(normalized, normalized)


def parse_method_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--report must be method=/path/report.json, got {value!r}")
    method, path = value.split("=", 1)
    method = canonical_method_name(method)
    if not method:
        raise ValueError(f"--report has an empty method: {value!r}")
    report_path = Path(path).expanduser()
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report for {method}: {report_path}")
    return method, report_path


def infer_method_from_report_path(report_path: Path) -> str | None:
    parent_name = report_path.parent.name
    if parent_name in KNOWN_METHOD_DIRS:
        return KNOWN_METHOD_DIRS[parent_name]
    for suffix in ("_50", "-50"):
        if parent_name.endswith(suffix):
            return canonical_method_name(parent_name[: -len(suffix)])
    return None


def discover_method_reports(run_dir: Path) -> list[tuple[str, Path]]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    reports: list[tuple[str, Path]] = []
    for report_path in sorted(run_dir.glob("*/prune_eval_report.json")):
        method = infer_method_from_report_path(report_path)
        if method is not None:
            reports.append((method, report_path))
    if not reports:
        raise FileNotFoundError(f"No per-method prune_eval_report.json files found under {run_dir}")
    deduped: dict[str, Path] = {}
    for method, report_path in reports:
        deduped[method] = report_path
    return sorted(deduped.items())


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def infer_run_metadata(
    method_reports: list[tuple[str, Path]],
    run_dir: Path | None,
) -> dict[str, Any]:
    first_report = read_json(method_reports[0][1])
    inferred: dict[str, Any] = {
        "base_model": None,
        "contrastive_model": first_report.get("input_model_path"),
        "contrastive_train_json": None,
        "sparsity": None,
    }
    datasets = first_report.get("datasets", {})
    if isinstance(datasets, dict):
        calibration = datasets.get("calibration", {})
        if isinstance(calibration, dict):
            inferred["contrastive_train_json"] = calibration.get("path")
    pruning = first_report.get("pruning", {})
    if isinstance(pruning, dict):
        inferred["sparsity"] = pruning.get("sparsity")
    if run_dir is not None:
        base_model_dir = run_dir / "base_model"
        if base_model_dir.exists():
            inferred["base_model"] = str(base_model_dir)
    return inferred


def compact_dataset_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for dataset in DATASETS:
        dataset_metrics = metrics.get(dataset)
        if not isinstance(dataset_metrics, dict):
            continue
        compact[dataset] = {
            "total": dataset_metrics.get("total", 0),
            "em1": dataset_metrics.get("em1", 0.0),
            "em5": dataset_metrics.get("em5", 0.0),
            "em1_percent": dataset_metrics.get("em1_percent", 0.0),
            "em5_percent": dataset_metrics.get("em5_percent", 0.0),
            "accuracy": dataset_metrics.get("accuracy", dataset_metrics.get("em1", 0.0)),
            "accuracy_percent": dataset_metrics.get("accuracy_percent", dataset_metrics.get("em1_percent", 0.0)),
        }
    return compact


def report_phase_metrics(report: dict[str, Any], phase: str) -> dict[str, Any]:
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    phase_metrics = summary.get(phase, {})
    return phase_metrics if isinstance(phase_metrics, dict) else {}


def build_aggregate_report(
    *,
    base_model: str,
    contrastive_model: str,
    epochs: int,
    sparsity: float,
    method_reports: list[tuple[str, Path]],
    contrastive_train_json: str | None = None,
) -> dict[str, Any]:
    if not method_reports:
        raise ValueError("At least one method report is required.")

    methods: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    first_report: dict[str, Any] | None = None

    for method, report_path in method_reports:
        report = read_json(report_path)
        if first_report is None:
            first_report = report

        method_summary: dict[str, Any] = {}
        for phase in PHASES:
            phase_metrics = compact_dataset_metrics(report_phase_metrics(report, phase))
            method_summary[phase] = phase_metrics
            for dataset, metrics in phase_metrics.items():
                table.append(
                    {
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

        methods[method] = {
            "report_json": str(report_path),
            "pruned_model_path": report.get("pruned_model_path"),
            "pruning": report.get("pruning", {}),
            **method_summary,
        }

    assert first_report is not None
    top_level_original = compact_dataset_metrics(report_phase_metrics(first_report, "original_before_prune"))

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_model_path": base_model,
        "contrastive_model_path": contrastive_model,
        "contrastive_train_json": contrastive_train_json,
        "contrastive_epochs": epochs,
        "target_sparsity": sparsity,
        "accuracy_definition": "accuracy is exact-match@1 / EM@1",
        "datasets": first_report.get("datasets", {}),
        "original_before_prune": top_level_original,
        "methods": methods,
        "table": table,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else None
    method_reports = [parse_method_report(value) for value in args.report]
    if run_dir is not None:
        method_reports.extend(discover_method_reports(run_dir))
    if not method_reports:
        raise SystemExit("Provide --run-dir or at least one --report method=/path/prune_eval_report.json.")
    method_reports = sorted(dict(method_reports).items())
    inferred = infer_run_metadata(method_reports, run_dir)
    output_json = Path(args.output_json).expanduser() if args.output_json else (
        run_dir / "all_pruning_em_report.json" if run_dir is not None else Path("all_pruning_em_report.json")
    )
    aggregate = build_aggregate_report(
        base_model=args.base_model or inferred["base_model"] or "unknown",
        contrastive_model=args.contrastive_model or inferred["contrastive_model"] or "unknown",
        contrastive_train_json=args.contrastive_train_json or inferred["contrastive_train_json"],
        epochs=args.epochs,
        sparsity=args.sparsity if args.sparsity is not None else float(inferred["sparsity"] or 0.5),
        method_reports=method_reports,
    )
    write_json(output_json, aggregate)
    print(f"Wrote all-method prune/eval JSON: {output_json}")


if __name__ == "__main__":
    main()
