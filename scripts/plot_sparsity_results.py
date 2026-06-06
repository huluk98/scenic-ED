#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SCENIC sparsity experiment results.")
    parser.add_argument("--summary_csv", required=True, help="Path to summary_metrics.csv.")
    parser.add_argument("--output_dir", default=None, help="Defaults to <summary_csv parent>/figures.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def plot_metric_vs_sparsity(rows: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        label = f"{row.get('model_family', '')} / {row.get('pruning_mode', '')}"
        grouped.setdefault(label, []).append((as_float(row.get("target_sparsity")), as_float(row.get(metric))))

    plt.figure(figsize=(8, 5))
    for label, points in sorted(grouped.items()):
        points = sorted(points)
        plt.plot([item[0] for item in points], [item[1] for item in points], marker="o", label=label)
    plt.xlabel("Target sparsity")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_difficulty_bars(rows: list[dict[str, Any]], metric_prefix: str, ylabel: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{row.get('model_family', '')}\n{row.get('pruning_mode', '')} {row.get('target_sparsity', '')}" for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    difficulties = ["easy", "medium", "hard"]

    plt.figure(figsize=(max(9, len(rows) * 0.8), 5))
    for offset, difficulty in enumerate(difficulties):
        values = [as_float(row.get(f"{metric_prefix}_{difficulty}")) for row in rows]
        positions = [item + (offset - 1) * width for item in x]
        plt.bar(positions, values, width=width, label=difficulty)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    summary_csv = Path(args.summary_csv)
    output_dir = Path(args.output_dir) if args.output_dir else summary_csv.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(summary_csv)
    if not rows:
        raise SystemExit(f"No rows found in {summary_csv}")
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting. Install it with: python -m pip install matplotlib") from exc

    plot_metric_vs_sparsity(rows, "em1_overall", "Overall EM@1", output_dir / "em1_vs_sparsity.png")
    plot_metric_vs_sparsity(rows, "em5_overall", "Overall EM@5", output_dir / "em5_vs_sparsity.png")
    plot_difficulty_bars(rows, "em1", "EM@1 by difficulty", output_dir / "em1_difficulty_breakdown.png")
    plot_difficulty_bars(rows, "em5", "EM@5 by difficulty", output_dir / "em5_difficulty_breakdown.png")
    print(f"Wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
