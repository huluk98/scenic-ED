from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_sparsity_experiments import (  # noqa: E402
    apply_magnitude_masks,
    apply_masks,
    build_examples,
    collect_prunable_linear_modules,
    compute_em_for_predictions,
    make_initial_masks,
    summarize_prediction_rows,
    sparsity_summary,
    write_summary_csv,
)


try:
    import torch
except Exception:
    torch = None


class TinyModel(torch.nn.Module if torch is not None else object):
    def __init__(self) -> None:
        if torch is None:
            return
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.encoder = torch.nn.Module()
        self.encoder.linear = torch.nn.Linear(4, 4, bias=True)
        self.lm_head = torch.nn.Linear(4, 4, bias=False)
        self.classifier = torch.nn.Linear(4, 2, bias=False)


def test_linear_collection_excludes_embeddings_norms_biases_and_heads() -> None:
    if torch is None:
        pytest.skip("torch is required for pruning tests")
    model = TinyModel()
    modules = collect_prunable_linear_modules(model, prune_scope="linear_weights", prune_output_heads=False)
    names = [name for name, _ in modules]

    assert names == ["encoder.linear"]


def test_magnitude_pruning_reaches_30_percent_targeted_linear_sparsity() -> None:
    if torch is None:
        pytest.skip("torch is required for pruning tests")
    model = TinyModel()
    modules = collect_prunable_linear_modules(model)
    with torch.no_grad():
        modules[0][1].weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4))

    masks, _ = apply_magnitude_masks(modules, 0.30)
    targeted, _ = sparsity_summary(model, modules)

    assert masks
    assert targeted == pytest.approx(0.30, abs=0.04)


def test_magnitude_pruning_reaches_50_percent_targeted_linear_sparsity() -> None:
    if torch is None:
        pytest.skip("torch is required for pruning tests")
    model = TinyModel()
    modules = collect_prunable_linear_modules(model)
    with torch.no_grad():
        modules[0][1].weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4))

    apply_magnitude_masks(modules, 0.50)
    targeted, _ = sparsity_summary(model, modules)

    assert targeted == pytest.approx(0.50, abs=0.01)


def test_mask_enforcement_keeps_pruned_weights_zero_after_optimizer_step() -> None:
    if torch is None:
        pytest.skip("torch is required for pruning tests")
    model = TinyModel()
    modules = collect_prunable_linear_modules(model)
    masks, _ = apply_magnitude_masks(modules, 0.50)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = sum(parameter.sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()
    apply_masks(modules, masks)

    key = "encoder.linear.weight"
    assert torch.all(modules[0][1].weight.detach()[~masks[key].to(modules[0][1].weight.device)] == 0)


def test_em1_and_em5_on_synthetic_predictions() -> None:
    assert compute_em_for_predictions(["打开灯", "关闭灯"], "打开灯", "ignore_spaces") == (True, True)
    assert compute_em_for_predictions(["关闭灯", "打开灯"], "打开灯", "ignore_spaces") == (False, True)
    assert compute_em_for_predictions(["关闭灯"], "打开灯", "ignore_spaces") == (False, False)


def test_difficulty_join_by_id_and_input(tmp_path: Path) -> None:
    difficulty_csv = tmp_path / "difficulty.csv"
    with difficulty_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "input", "difficulty"])
        writer.writeheader()
        writer.writerow({"id": "a", "input": "", "difficulty": "easy"})
        writer.writerow({"id": "", "input": "调暗卧室灯", "difficulty": "hard"})

    examples = build_examples(
        [
            {"id": "a", "prompt": "打开灯", "response": "好的"},
            {"id": "b", "prompt": "调暗卧室灯", "response": "好的"},
        ],
        difficulty_csv,
    )

    assert [example.difficulty for example in examples] == ["easy", "hard"]


def test_summary_csv_includes_difficulty_counts(tmp_path: Path) -> None:
    rows = [
        {"difficulty": "easy", "em1": 1, "em5": 1},
        {"difficulty": "medium", "em1": 0, "em5": 1},
        {"difficulty": "hard", "em1": 0, "em5": 0},
    ]
    metrics = summarize_prediction_rows(rows, bootstrap_resamples=10, seed=1)
    output = tmp_path / "summary_metrics.csv"
    write_summary_csv(output, [{"experiment_name": "x", **metrics}])

    text = output.read_text(encoding="utf-8")
    assert "count_easy" in text
    assert "count_medium" in text
    assert "count_hard" in text
