#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from check_pruned_model_sparsity import inspect_active_parameters


# Edit this path, then run:
#   python scripts/count_active_params_simple.py
MODEL_PATH = "prune_eval_outputs/<run>/contrastive_sft/gradient_50/pruned_model"

# For the current reference-style pruning runs, keep this as encoder-linear.
EXPECTED_SCOPE = "encoder-linear"

# Leave empty to only print results. Set a path to also write JSON.
OUTPUT_JSON = ""


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def main() -> None:
    result = inspect_active_parameters(
        MODEL_PATH,
        expected_scope=EXPECTED_SCOPE,
        label=Path(MODEL_PATH).name or "model",
    )
    if result.get("status") != "ok":
        print(f"Status: {result.get('status')}")
        print(f"Path: {result.get('model_path')}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        return

    overall = result["overall"]
    scope = result["expected_scope_result"]

    print(f"Model path: {result['model_path']}")
    print()
    print("Full checkpoint")
    print(f"  Total parameters:  {fmt_int(overall['parameter_count'])}")
    print(f"  Zero parameters:   {fmt_int(overall['zero_parameter_count'])}")
    print(f"  Active parameters: {fmt_int(overall['active_parameter_count'])}")
    print(f"  Sparsity:          {fmt_pct(overall['sparsity'])}")
    print()
    print(f"{EXPECTED_SCOPE}")
    print(f"  Linear layers:     {fmt_int(scope.get('layers', 0))}")
    print(f"  Total parameters:  {fmt_int(scope.get('parameter_count', 0))}")
    print(f"  Zero parameters:   {fmt_int(scope.get('zero_parameter_count', 0))}")
    print(f"  Active parameters: {fmt_int(scope.get('active_parameter_count', 0))}")
    print(f"  Sparsity:          {fmt_pct(scope.get('sparsity', 0.0))}")

    if OUTPUT_JSON:
        output_path = Path(OUTPUT_JSON).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print()
        print(f"Wrote JSON: {output_path}")


if __name__ == "__main__":
    main()
