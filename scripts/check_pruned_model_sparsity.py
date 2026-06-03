#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRUNE_SCOPES = ("encoder-linear", "decoder-linear", "all-linear")


@dataclass
class ModelEntry:
    label: str
    method: str
    model_dir: Path
    expected_scope: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auto-detect generated SCENIC model checkpoints and report full-model plus targeted "
            "linear sparsity without loading custom model code."
        )
    )
    parser.add_argument("--report-json", default=None, help="Combined prune/eval JSON report to inspect.")
    parser.add_argument("--run-dir", default=None, help="Run directory containing pruned_model folders.")
    parser.add_argument(
        "--model-path",
        action="append",
        default=[],
        help="Model directory or one checkpoint weight file to inspect directly. Can be used more than once.",
    )
    parser.add_argument("--base-dir", default=".", help="Base directory for relative paths in reports.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--include-originals", action="store_true", help="Also check dense SFT model directories.")
    parser.add_argument("--expected-sparsity", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--default-prune-scope",
        choices=PRUNE_SCOPES,
        default="encoder-linear",
        help="Scope to use when a report does not record prune_scope.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(raw_path: str | Path, *, base_dir: Path, report_dir: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = [base_dir / path, Path.cwd() / path]
    if report_dir is not None:
        candidates.extend([report_dir / path, report_dir.parent / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def method_from_dir(path: Path) -> str:
    parent = path.parent.name
    for suffix in ("_50", "-50"):
        if parent.endswith(suffix):
            return parent[: -len(suffix)]
    return parent or "unknown"


def entries_from_report(path: Path, base_dir: Path, default_scope: str, include_originals: bool) -> list[ModelEntry]:
    report = read_json(path)
    report_dir = path.parent
    entries: list[ModelEntry] = []

    if isinstance(report, dict) and isinstance(report.get("models"), dict):
        for label, model_info in report["models"].items():
            if include_originals and model_info.get("model_path"):
                entries.append(
                    ModelEntry(
                        label=label,
                        method="dense",
                        model_dir=resolve_path(model_info["model_path"], base_dir=base_dir, report_dir=report_dir),
                        expected_scope=default_scope,
                    )
                )
            for method, method_info in (model_info.get("methods") or {}).items():
                pruned_path = method_info.get("pruned_model_path")
                if not pruned_path:
                    continue
                pruning = method_info.get("pruning") if isinstance(method_info.get("pruning"), dict) else {}
                entries.append(
                    ModelEntry(
                        label=label,
                        method=method,
                        model_dir=resolve_path(pruned_path, base_dir=base_dir, report_dir=report_dir),
                        expected_scope=pruning.get("prune_scope") or default_scope,
                    )
                )
        return entries

    if isinstance(report, dict) and isinstance(report.get("methods"), dict):
        label = "model"
        if include_originals and report.get("contrastive_model_path"):
            entries.append(
                ModelEntry(
                    label="contrastive_sft",
                    method="dense",
                    model_dir=resolve_path(report["contrastive_model_path"], base_dir=base_dir, report_dir=report_dir),
                    expected_scope=default_scope,
                )
            )
        for method, method_info in report["methods"].items():
            pruned_path = method_info.get("pruned_model_path")
            if not pruned_path:
                continue
            pruning = method_info.get("pruning") if isinstance(method_info.get("pruning"), dict) else {}
            entries.append(
                ModelEntry(
                    label=label,
                    method=method,
                    model_dir=resolve_path(pruned_path, base_dir=base_dir, report_dir=report_dir),
                    expected_scope=pruning.get("prune_scope") or default_scope,
                )
            )
    return entries


def entries_from_run_dir(path: Path, default_scope: str, include_originals: bool) -> list[ModelEntry]:
    entries: list[ModelEntry] = []
    for model_dir in sorted(path.glob("**/pruned_model")):
        relative = model_dir.relative_to(path)
        parts = relative.parts
        label = parts[0] if len(parts) >= 3 else "model"
        entries.append(
            ModelEntry(
                label=label,
                method=method_from_dir(model_dir),
                model_dir=model_dir,
                expected_scope=default_scope,
            )
        )

    if include_originals:
        for model_dir in sorted(path.glob("*_sft_5epoch")):
            entries.append(
                ModelEntry(
                    label=model_dir.name,
                    method="dense",
                    model_dir=model_dir,
                    expected_scope=default_scope,
                )
            )
    return entries


def checkpoint_weight_files(model_path: Path) -> list[Path]:
    if model_path.is_file():
        if model_path.name.endswith((".safetensors", ".bin")):
            return [model_path]
        return []

    model_dir = model_path
    index_files = sorted(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        weight_files: set[Path] = set()
        for index_file in index_files:
            index = read_json(index_file)
            for shard in (index.get("weight_map") or {}).values():
                weight_files.add(model_dir / shard)
        return sorted(weight_files)

    safetensors = sorted(model_dir.glob("*.safetensors"))
    if safetensors:
        return safetensors

    index_files = sorted(model_dir.glob("pytorch_model*.bin.index.json"))
    if index_files:
        weight_files: set[Path] = set()
        for index_file in index_files:
            index = read_json(index_file)
            for shard in (index.get("weight_map") or {}).values():
                weight_files.add(model_dir / shard)
        return sorted(weight_files)

    return sorted(model_dir.glob("pytorch_model*.bin"))


def load_state_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install safetensors to inspect .safetensors checkpoints.") from exc
        return dict(load_file(str(path), device="cpu"))

    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        return payload["state_dict"]
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a state dict.")
    return payload


def linear_module_in_scope(name: str, scope: str) -> bool:
    if scope == "all-linear":
        return True
    normalized = name.lower().replace("_", ".")
    parts = normalized.split(".")
    is_encoder = "encoder" in parts
    is_decoder = "decoder" in parts
    if scope == "encoder-linear":
        return is_encoder and not is_decoder
    if scope == "decoder-linear":
        return is_decoder
    raise ValueError(f"Unknown scope: {scope}")


def looks_like_linear_weight(name: str, ndim: int) -> bool:
    if not name.endswith(".weight") or ndim != 2:
        return False

    module_name = name[: -len(".weight")]
    normalized = module_name.lower().replace("_", ".")
    parts = set(normalized.split("."))
    non_linear_tokens = {
        "embed",
        "embeds",
        "embedding",
        "embeddings",
        "position",
        "positions",
        "token",
        "tokens",
        "shared",
        "relative",
        "bias",
    }
    return not bool(parts & non_linear_tokens)


def empty_scope_counts() -> dict[str, int]:
    return {"layers": 0, "params": 0, "zeros": 0}


def add_counts(counts: dict[str, int], total: int, zeros: int, layer: bool = False) -> None:
    counts["params"] += total
    counts["zeros"] += zeros
    if layer:
        counts["layers"] += 1


def with_active_count(counts: dict[str, Any]) -> dict[str, Any]:
    params = int(counts.get("params", 0))
    zeros = int(counts.get("zeros", 0))
    return {
        **counts,
        "active": params - zeros,
        "active_parameter_count": params - zeros,
        "zero_parameter_count": zeros,
        "parameter_count": params,
    }


def inspect_model_dir(entry: ModelEntry, expected_sparsity: float, tolerance: float) -> dict[str, Any]:
    model_dir = entry.model_dir
    result: dict[str, Any] = {
        "label": entry.label,
        "method": entry.method,
        "model_dir": str(model_dir),
        "model_path": str(model_dir),
        "exists": model_dir.exists(),
        "expected_sparsity": expected_sparsity,
        "expected_scope": entry.expected_scope,
    }
    if not model_dir.exists():
        result["status"] = "missing_model_dir"
        return result

    weight_files = checkpoint_weight_files(model_dir)
    result["weight_files"] = [str(path) for path in weight_files]
    if not weight_files:
        result["status"] = "missing_weight_files"
        return result

    overall = {"params": 0, "zeros": 0}
    scopes = {scope: empty_scope_counts() for scope in PRUNE_SCOPES}

    try:
        for weight_file in weight_files:
            state = load_state_file(weight_file)
            for name, tensor in state.items():
                if not hasattr(tensor, "numel"):
                    continue
                total = int(tensor.numel())
                if total == 0:
                    continue
                zeros = int(tensor.eq(0).sum().item())
                add_counts(overall, total, zeros)

                if not looks_like_linear_weight(name, getattr(tensor, "ndim", 0)):
                    continue
                module_name = name[: -len(".weight")]
                for scope in PRUNE_SCOPES:
                    if linear_module_in_scope(module_name, scope):
                        add_counts(scopes[scope], total, zeros, layer=True)
    except Exception as exc:
        result["status"] = "inspection_failed"
        result["error"] = str(exc)
        return result

    overall_sparsity = overall["zeros"] / overall["params"] if overall["params"] else 0.0
    scope_results: dict[str, Any] = {}
    for scope, counts in scopes.items():
        sparsity = counts["zeros"] / counts["params"] if counts["params"] else 0.0
        scope_results[scope] = {
            **with_active_count(counts),
            "sparsity": sparsity,
            "matches_expected": abs(sparsity - expected_sparsity) <= tolerance if counts["params"] else False,
        }

    expected_scope_result = scope_results.get(entry.expected_scope, {})
    result.update(
        {
            "status": "ok",
            "overall": {
                **with_active_count(overall),
                "sparsity": overall_sparsity,
                "matches_expected": abs(overall_sparsity - expected_sparsity) <= tolerance,
            },
            "linear_scopes": scope_results,
            "expected_scope_result": expected_scope_result,
            "full_model_is_50_percent_sparse": abs(overall_sparsity - expected_sparsity) <= tolerance,
            "expected_scope_is_50_percent_sparse": bool(expected_scope_result.get("matches_expected")),
        }
    )
    return result


def inspect_active_parameters(
    model_path: str | Path,
    *,
    expected_scope: str = "encoder-linear",
    expected_sparsity: float = 0.5,
    tolerance: float = 0.01,
    label: str = "model",
    method: str = "direct",
) -> dict[str, Any]:
    """Return active/nonzero parameter counts for a model directory or weight file.

    Active parameters are parameters whose stored value is non-zero. This is meant
    for pruned checkpoints, so it reads raw checkpoint tensors and does not need
    custom ChatLM model code.
    """
    entry = ModelEntry(
        label=label,
        method=method,
        model_dir=Path(model_path).expanduser(),
        expected_scope=expected_scope,
    )
    return inspect_model_dir(entry, expected_sparsity=expected_sparsity, tolerance=tolerance)


def print_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'label':18s} {'method':12s} {'status':18s} {'overall':>9s} "
        f"{'active':>14s} {'scope':15s} {'scope_sp':>9s} {'scope_act':>14s} "
        f"{'full50':>7s} {'scope50':>8s}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result.get("status") != "ok":
            print(
                f"{result.get('label','')[:18]:18s} {result.get('method','')[:12]:12s} "
                f"{result.get('status','')[:18]:18s}"
            )
            continue
        scope = result["expected_scope"]
        scope_sparsity = result["expected_scope_result"].get("sparsity", 0.0)
        print(
            f"{result['label'][:18]:18s} {result['method'][:12]:12s} {result['status']:18s} "
            f"{result['overall']['sparsity']:9.4f} {result['overall']['active']:14d} "
            f"{scope:15s} {scope_sparsity:9.4f} "
            f"{result['expected_scope_result'].get('active', 0):14d} "
            f"{str(result['full_model_is_50_percent_sparse']):>7s} "
            f"{str(result['expected_scope_is_50_percent_sparse']):>8s}"
        )


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    ok_results = [result for result in results if result.get("status") == "ok"]
    full_50 = [result for result in ok_results if result.get("full_model_is_50_percent_sparse")]
    scope_50 = [result for result in ok_results if result.get("expected_scope_is_50_percent_sparse")]
    return {
        "total_entries": len(results),
        "ok_entries": len(ok_results),
        "status_counts": status_counts,
        "full_model_50_percent_sparse_count": len(full_50),
        "expected_scope_50_percent_sparse_count": len(scope_50),
        "all_existing_models_full_50_percent_sparse": bool(ok_results) and len(full_50) == len(ok_results),
        "all_existing_models_expected_scope_50_percent_sparse": bool(ok_results) and len(scope_50) == len(ok_results),
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser()
    entries: list[ModelEntry] = []

    for raw_model_path in args.model_path:
        model_path = Path(raw_model_path).expanduser()
        entries.append(
            ModelEntry(
                label=model_path.name or "model",
                method="direct",
                model_dir=model_path,
                expected_scope=args.default_prune_scope,
            )
        )
    if args.report_json:
        entries.extend(
            entries_from_report(
                Path(args.report_json).expanduser(),
                base_dir=base_dir,
                default_scope=args.default_prune_scope,
                include_originals=args.include_originals,
            )
        )
    if args.run_dir:
        entries.extend(
            entries_from_run_dir(
                Path(args.run_dir).expanduser(),
                default_scope=args.default_prune_scope,
                include_originals=args.include_originals,
            )
        )
    if not entries:
        default_run_dir = Path("prune_eval_outputs")
        if default_run_dir.exists():
            entries.extend(
                entries_from_run_dir(
                    default_run_dir,
                    default_scope=args.default_prune_scope,
                    include_originals=args.include_originals,
                )
            )
    if not entries:
        raise SystemExit("No model directories found. Provide --report-json or --run-dir.")

    deduped = {(entry.label, entry.method, str(entry.model_dir)): entry for entry in entries}
    results = [
        inspect_model_dir(entry, expected_sparsity=args.expected_sparsity, tolerance=args.tolerance)
        for entry in sorted(deduped.values(), key=lambda item: (item.label, item.method, str(item.model_dir)))
    ]
    print_table(results)
    summary = build_summary(results)
    print(
        "Summary: "
        f"ok={summary['ok_entries']}/{summary['total_entries']} "
        f"scope50={summary['expected_scope_50_percent_sparse_count']} "
        f"full50={summary['full_model_50_percent_sparse_count']}"
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "expected_sparsity": args.expected_sparsity,
                    "tolerance": args.tolerance,
                    "summary": summary,
                    "models": results,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        print(f"Wrote sparsity report: {output_path}")


if __name__ == "__main__":
    main()
