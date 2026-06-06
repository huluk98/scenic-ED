#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


MODE_ORDER = {"dense": 0, "oneshot": 1, "progressive": 2}
RETENTION_GROUPS = ("overall", "easy", "medium", "hard")
RETENTION_METRICS = ("em1", "em5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate split SCENIC sparsity job CSVs.")
    parser.add_argument("--job-root", required=True, help="Directory containing per-condition job directories.")
    parser.add_argument("--output-dir", required=True, help="Directory for combined summary CSVs.")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def as_float(value: Any) -> float | None:
    if value in (None, "", "None", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    mode = str(row.get("pruning_mode", ""))
    sparsity = as_float(row.get("target_sparsity"))
    return (MODE_ORDER.get(mode, 99), sparsity if sparsity is not None else 99.0, mode)


def recompute_retention(rows: list[dict[str, Any]]) -> None:
    dense = next(
        (
            row for row in rows
            if str(row.get("pruning_mode")) == "dense"
            and abs((as_float(row.get("target_sparsity")) or 0.0) - 0.0) < 1e-9
        ),
        None,
    )
    if dense is None:
        return

    for row in rows:
        for group in RETENTION_GROUPS:
            for metric in RETENTION_METRICS:
                metric_key = f"{metric}_{group}"
                retention_key = f"{metric}_retention_{group}"
                numerator = as_float(row.get(metric_key))
                denominator = as_float(dense.get(metric_key))
                if numerator is None or denominator in (None, 0.0):
                    row[retention_key] = ""
                else:
                    row[retention_key] = numerator / denominator


def collect_csv_rows(job_root: Path, filename: str) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in sorted(job_root.glob(f"*/{filename}")):
        current_fields, current_rows = read_csv(path)
        if not fieldnames:
            fieldnames = current_fields
        for field in current_fields:
            if field not in fieldnames:
                fieldnames.append(field)
        rows.extend(current_rows)
    return fieldnames, rows


def main() -> None:
    args = parse_args()
    job_root = Path(args.job_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not job_root.exists():
        raise SystemExit(f"Job root does not exist: {job_root}")

    summary_fields, summary_rows = collect_csv_rows(job_root, "summary_metrics.csv")
    if not summary_rows:
        raise SystemExit(f"No summary_metrics.csv files found under {job_root}")
    summary_rows.sort(key=sort_key)
    recompute_retention(summary_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_fields, summary_rows)

    paper_fields, paper_rows = collect_csv_rows(job_root, "paper_table_sparsity_difficulty.csv")
    if paper_rows:
        paper_rows.sort(key=sort_key)
        write_csv(output_dir / "paper_table_sparsity_difficulty.csv", paper_fields, paper_rows)

    print(f"Wrote combined sparsity summaries to: {output_dir}")


if __name__ == "__main__":
    main()
