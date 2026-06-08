#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize latency test-drive runtime benchmark JSON files."
    )
    parser.add_argument("--input-glob", required=True, help="Glob for gpu_*_runtime_benchmark.json files.")
    parser.add_argument("--output-json", help="Optional summary JSON output path.")
    parser.add_argument("--output-csv", help="Optional summary CSV output path.")
    parser.add_argument("--output-md", help="Optional Markdown table output path.")
    return parser.parse_args()


def gpu_label(path: Path) -> str:
    match = re.search(r"gpu_([^/_]+)_runtime_benchmark\.json$", path.name)
    return match.group(1) if match else "unknown"


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for row in payload.get("rows", []):
            rows.append({**row, "gpu": gpu_label(path), "source_json": str(path)})
    return rows


def row_name(row: dict[str, Any]) -> str:
    runtime = str(row.get("runtime_label") or row.get("runtime") or "")
    variant = str(row.get("model_variant") or "")
    return f"{runtime} {variant}".strip()


def baseline_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("gpu", "")), int(row.get("input_length", 0)))


def add_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("model_variant") == "dense" and row.get("runtime_label") == "ONNX FP16":
            baselines[baseline_key(row)] = row

    out: list[dict[str, Any]] = []
    for row in rows:
        baseline = baselines.get(baseline_key(row))
        mean_ms = float(row.get("mean_latency_ms_per_query") or 0.0)
        p95_ms = float(row.get("p95_latency_ms_per_query") or 0.0)
        qps = float(row.get("throughput_queries_per_second") or 0.0)
        item = {
            "gpu": row.get("gpu"),
            "variant": row_name(row),
            "model_variant": row.get("model_variant"),
            "runtime_label": row.get("runtime_label"),
            "precision": row.get("precision"),
            "input_length": row.get("input_length"),
            "queries": row.get("queries"),
            "warmup_queries": row.get("warmup_queries"),
            "max_new_tokens": row.get("max_new_tokens"),
            "mean_latency_ms": mean_ms,
            "p95_latency_ms": p95_ms,
            "throughput_qps": qps,
            "model_size_mb": row.get("model_size_mb"),
            "peak_gpu_memory_mb_nvidia_smi": row.get("peak_gpu_memory_mb_nvidia_smi"),
            "source_json": row.get("source_json"),
        }
        if baseline is not None:
            base_mean = float(baseline.get("mean_latency_ms_per_query") or 0.0)
            base_p95 = float(baseline.get("p95_latency_ms_per_query") or 0.0)
            base_qps = float(baseline.get("throughput_queries_per_second") or 0.0)
            item["mean_delta_ms_vs_onnx_fp16_dense"] = mean_ms - base_mean
            item["p95_delta_ms_vs_onnx_fp16_dense"] = p95_ms - base_p95
            item["speedup_vs_onnx_fp16_dense"] = base_mean / mean_ms if mean_ms > 0 else None
            item["throughput_ratio_vs_onnx_fp16_dense"] = qps / base_qps if base_qps > 0 else None
        else:
            item["mean_delta_ms_vs_onnx_fp16_dense"] = None
            item["p95_delta_ms_vs_onnx_fp16_dense"] = None
            item["speedup_vs_onnx_fp16_dense"] = None
            item["throughput_ratio_vs_onnx_fp16_dense"] = None
        out.append(item)
    return sorted(out, key=lambda item: (str(item["gpu"]), int(item["input_length"]), str(item["variant"])))


def fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "GPU",
        "Variant",
        "Seq",
        "Queries",
        "Mean ms",
        "p95 ms",
        "QPS",
        "Mean Δ vs FP16 dense",
        "p95 Δ vs FP16 dense",
        "Speedup",
        "Size MB",
        "Peak GPU MB",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["gpu"]),
                    str(row["variant"]),
                    str(row["input_length"]),
                    str(row["queries"]),
                    fmt(row["mean_latency_ms"]),
                    fmt(row["p95_latency_ms"]),
                    fmt(row["throughput_qps"]),
                    fmt(row["mean_delta_ms_vs_onnx_fp16_dense"]),
                    fmt(row["p95_delta_ms_vs_onnx_fp16_dense"]),
                    fmt(row["speedup_vs_onnx_fp16_dense"], suffix="x"),
                    fmt(row["model_size_mb"]),
                    fmt(row["peak_gpu_memory_mb_nvidia_smi"], digits=0),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"rows": rows}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    if args.output_csv:
        path = Path(args.output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    table = markdown_table(rows)
    if args.output_md:
        path = Path(args.output_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table, encoding="utf-8")
    print(table, end="")


def main() -> None:
    args = parse_args()
    paths = sorted(Path(item) for item in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No runtime benchmark JSON files matched: {args.input_glob}")
    rows = add_comparisons(load_rows(paths))
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
