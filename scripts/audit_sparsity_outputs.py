#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPECTED_ROWS = (
    ("dense", 0.0),
    ("oneshot", 0.3),
    ("oneshot", 0.5),
    ("progressive", 0.3),
    ("progressive", 0.5),
)
GROUPS = ("overall", "easy", "medium", "hard")
METRICS = ("em1", "em5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SCENIC sparsity summary rows against run artifacts.")
    parser.add_argument("--summary-csv", required=True, help="Combined or per-job summary_metrics.csv.")
    parser.add_argument("--job-root", default=None, help="Optional jobs/ directory for parallel run artifacts.")
    parser.add_argument("--output-json", default=None, help="Optional path to write machine-readable audit results.")
    parser.add_argument(
        "--expected-row",
        action="append",
        default=None,
        metavar="MODE:SPARSITY",
        help="Expected row such as progressive:0.3. May be repeated. Defaults to full dense/oneshot/progressive matrix.",
    )
    parser.add_argument("--fail-on-warning", action="store_true", help="Exit non-zero when warnings are found.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def filename_sparsity(sparsity: float) -> str:
    return str(sparsity).replace(".", "p")


def row_label(row: dict[str, Any]) -> str:
    return (
        f"{row.get('model_family', '')} "
        f"{row.get('pruning_mode', '')} "
        f"{row.get('target_sparsity', '')} "
        f"seed={row.get('seed', '')}"
    ).strip()


def resolve_existing_path(raw: str | None, summary_dir: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, summary_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def find_artifact(
    row: dict[str, Any],
    summary_dir: Path,
    job_root: Path | None,
    field: str,
    filename: str,
) -> Path | None:
    explicit = resolve_existing_path(row.get(field), summary_dir)
    if explicit is not None and explicit.exists():
        return explicit
    candidates = [summary_dir / filename]
    if job_root is not None:
        candidates.extend(sorted(job_root.glob(f"*/{filename}")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return explicit or (candidates[0] if candidates else None)


def expected_prediction_filename(row: dict[str, Any]) -> str | None:
    model = row.get("model_family")
    mode = row.get("pruning_mode")
    target = as_float(row.get("target_sparsity"))
    seed = row.get("seed")
    if model in (None, "") or mode in (None, "") or target is None or seed in (None, ""):
        return None
    return f"predictions_{model}_{mode}_{filename_sparsity(target)}_{seed}.csv"


def expected_progressive_log_filename(row: dict[str, Any]) -> str | None:
    model = row.get("model_family")
    target = as_float(row.get("target_sparsity"))
    seed = row.get("seed")
    if model in (None, "") or target is None or seed in (None, ""):
        return None
    return f"progressive_logs_{model}_{filename_sparsity(target)}_{seed}.csv"


def parse_expected_rows(values: list[str] | None) -> tuple[tuple[str, float], ...]:
    if not values:
        return EXPECTED_ROWS
    rows: list[tuple[str, float]] = []
    for value in values:
        if ":" not in value:
            raise SystemExit(f"Expected row must use MODE:SPARSITY format, got: {value}")
        mode, sparsity = value.split(":", 1)
        mode = mode.strip()
        if mode not in {"dense", "oneshot", "progressive"}:
            raise SystemExit(f"Unknown expected row mode: {mode}")
        try:
            target = float(sparsity)
        except ValueError as exc:
            raise SystemExit(f"Invalid expected row sparsity: {sparsity}") from exc
        rows.append((mode, target))
    return tuple(rows)


def summarize_predictions(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    summary: dict[str, Any] = {"count_total": len(rows)}
    for group in GROUPS:
        group_rows = rows if group == "overall" else [row for row in rows if row.get("difficulty") == group]
        suffix = "overall" if group == "overall" else group
        summary[f"count_{suffix}"] = len(group_rows)
        for metric in METRICS:
            values = [int(row.get(metric, 0)) for row in group_rows]
            summary[f"{metric}_{suffix}"] = sum(values) / len(values) if values else 0.0
    return summary


def add_metric_warnings(row: dict[str, Any], prediction_summary: dict[str, Any], warnings: list[str]) -> None:
    label = row_label(row)
    summary_count = as_int(row.get("count_total"))
    if summary_count is not None and summary_count != prediction_summary["count_total"]:
        warnings.append(f"{label}: count_total is {summary_count}, predictions contain {prediction_summary['count_total']}")
    for group in GROUPS:
        suffix = "overall" if group == "overall" else group
        for metric in METRICS:
            key = f"{metric}_{suffix}"
            reported = as_float(row.get(key))
            recomputed = as_float(prediction_summary.get(key))
            if reported is not None and recomputed is not None and abs(reported - recomputed) > 1e-9:
                warnings.append(f"{label}: {key} reported {reported}, predictions recompute {recomputed}")


def add_required_row_warnings(
    rows: list[dict[str, Any]],
    warnings: list[str],
    expected_rows: tuple[tuple[str, float], ...],
) -> None:
    models = sorted({row.get("model_family", "") for row in rows if row.get("model_family")})
    existing = {
        (row.get("model_family"), row.get("pruning_mode"), as_float(row.get("target_sparsity")))
        for row in rows
    }
    for model in models:
        for mode, target in expected_rows:
            if (model, mode, target) not in existing:
                warnings.append(f"{model}: missing expected {mode} target_sparsity={target}")


def audit_row(row: dict[str, Any], summary_dir: Path, job_root: Path | None, dense_by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    warnings: list[str] = []
    label = row_label(row)
    mode = row.get("pruning_mode")
    target = as_float(row.get("target_sparsity"))

    prediction_filename = expected_prediction_filename(row)
    prediction_path = (
        find_artifact(row, summary_dir, job_root, "prediction_path", prediction_filename)
        if prediction_filename is not None else None
    )
    prediction_exists = bool(prediction_path and prediction_path.exists())
    if not prediction_exists:
        warnings.append(f"{label}: missing predictions CSV")
    else:
        add_metric_warnings(row, summarize_predictions(prediction_path), warnings)

    checkpoint_path = resolve_existing_path(row.get("checkpoint_path"), summary_dir)
    if checkpoint_path is not None and not checkpoint_path.exists():
        warnings.append(f"{label}: checkpoint_path does not exist: {checkpoint_path}")

    mask_path = resolve_existing_path(row.get("mask_path"), summary_dir)
    if mode != "dense" and (mask_path is None or not mask_path.exists()):
        warnings.append(f"{label}: mask_path does not exist")

    progressive_log_path = None
    if mode == "progressive":
        log_filename = expected_progressive_log_filename(row)
        progressive_log_path = (
            find_artifact(row, summary_dir, job_root, "progressive_log_path", log_filename)
            if log_filename is not None else None
        )
        if progressive_log_path is None or not progressive_log_path.exists():
            warnings.append(f"{label}: missing progressive log CSV")
        else:
            log_rows = read_csv(progressive_log_path)
            if not log_rows:
                warnings.append(f"{label}: progressive log is empty")
            else:
                final_row = log_rows[-1]
                final_targeted = as_float(final_row.get("targeted_linear_sparsity_actual"))
                reported_targeted = as_float(row.get("targeted_linear_sparsity_actual"))
                if (
                    final_targeted is not None
                    and reported_targeted is not None
                    and abs(final_targeted - reported_targeted) > 1e-9
                ):
                    warnings.append(
                        f"{label}: final progressive targeted sparsity {final_targeted} "
                        f"does not match summary {reported_targeted}"
                    )

        dense = dense_by_model.get(str(row.get("model_family")))
        dense_em1 = as_float(dense.get("em1_overall")) if dense else None
        row_em1 = as_float(row.get("em1_overall"))
        if dense_em1 is not None and dense_em1 > 0.1 and row_em1 == 0.0:
            warnings.append(f"{label}: progressive EM@1 is zero while dense EM@1 is {dense_em1}")

    return {
        "label": label,
        "mode": mode,
        "target_sparsity": target,
        "prediction_path": str(prediction_path) if prediction_path else None,
        "prediction_exists": prediction_exists,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_exists": bool(checkpoint_path and checkpoint_path.exists()),
        "mask_path": str(mask_path) if mask_path else None,
        "mask_exists": bool(mask_path and mask_path.exists()),
        "progressive_log_path": str(progressive_log_path) if progressive_log_path else None,
        "progressive_log_exists": bool(progressive_log_path and progressive_log_path.exists()),
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser()
    summary_dir = summary_csv.parent
    job_root = Path(args.job_root).expanduser() if args.job_root else None
    rows = read_csv(summary_csv)
    warnings: list[str] = []
    expected_rows = parse_expected_rows(args.expected_row)
    add_required_row_warnings(rows, warnings, expected_rows)

    dense_by_model = {
        str(row.get("model_family")): row
        for row in rows
        if row.get("pruning_mode") == "dense" and abs((as_float(row.get("target_sparsity")) or 0.0)) < 1e-9
    }
    row_audits = [audit_row(row, summary_dir, job_root, dense_by_model) for row in rows]
    for audit in row_audits:
        warnings.extend(audit["warnings"])

    payload = {
        "summary_csv": str(summary_csv),
        "job_root": str(job_root) if job_root else None,
        "row_count": len(rows),
        "expected_rows": [{"pruning_mode": mode, "target_sparsity": target} for mode, target in expected_rows],
        "warning_count": len(warnings),
        "warnings": warnings,
        "rows": row_audits,
    }
    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Audited {len(rows)} sparsity rows from {summary_csv}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("No warnings.")
    if args.fail_on_warning and warnings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
