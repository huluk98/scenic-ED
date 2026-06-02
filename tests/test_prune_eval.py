from __future__ import annotations

import sys
from pathlib import Path
import json
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from aggregate_prune_eval_reports import build_aggregate_report
from scenic_prune_eval import (
    CalibrationDataset,
    DistributedState,
    compact_metrics,
    finalize_eval_result,
    magnitude_prune,
    normalize_text,
    summarize_model,
    wanda_prune,
)


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts=None,
        text_target=None,
        padding=True,
        truncation=True,
        max_length=4,
        return_tensors=None,
    ):
        values = text_target if text_target is not None else texts
        if isinstance(values, str):
            values = [values]
        batch_size = len(values)
        return {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }


class TinySeq2Seq(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        features = torch.nn.functional.one_hot(input_ids.clamp(max=3), num_classes=4).float()
        logits = self.linear(features)
        return SimpleNamespace(logits=logits, loss=logits.mean())


class VariableLengthTargetTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts=None,
        text_target=None,
        padding=True,
        truncation=True,
        max_length=8,
        return_tensors=None,
    ):
        values = text_target if text_target is not None else texts
        if isinstance(values, str):
            values = [values]
        batch_size = len(values)
        if text_target is not None and padding != "max_length":
            length = min(max(len(str(value)) for value in values), max_length)
        else:
            length = max_length
        return {
            "input_ids": torch.ones((batch_size, length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, length), dtype=torch.long),
        }


def test_normalize_text_can_ignore_spaces() -> None:
    assert normalize_text(" 好 的 ", ignore_spaces=False) == "好 的"
    assert normalize_text(" 好 的 ", ignore_spaces=True) == "好的"


def test_finalize_eval_result_reports_em_and_accuracy() -> None:
    result = finalize_eval_result(
        {
            "total": 4,
            "em1_correct": 2,
            "em5_correct": 3,
            "outputs": [],
        }
    )

    assert result["em1"] == 0.5
    assert result["em5"] == 0.75
    assert result["accuracy"] == result["em1"]
    assert result["accuracy_definition"] == "accuracy is exact-match@1 / EM@1"


def test_compact_metrics_keeps_top_level_accuracy_easy_to_find() -> None:
    compact = compact_metrics(
        {
            "benchmark": {
                "total": 2,
                "em1": 0.5,
                "em5": 1.0,
                "em1_percent": 50.0,
                "em5_percent": 100.0,
                "accuracy": 0.5,
                "accuracy_percent": 50.0,
                "outputs": [{"large": "payload"}],
            }
        }
    )

    assert compact == {
        "benchmark": {
            "total": 2,
            "em1": 0.5,
            "em5": 1.0,
            "em1_percent": 50.0,
            "em5_percent": 100.0,
            "accuracy": 0.5,
            "accuracy_percent": 50.0,
        }
    }


def test_magnitude_prune_sets_half_of_linear_weights_to_zero() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(8, 1, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(torch.arange(1, 9, dtype=torch.float32).reshape(1, 8))

    before = summarize_model(model, "tiny")
    summary = magnitude_prune(model, sparsity=0.5)
    after = summarize_model(model, "tiny")

    assert before["linear_sparsity"] == 0.0
    assert summary["pruned_linear_layers"] == 1
    assert after["linear_zero_weight_count"] == 4
    assert after["linear_sparsity"] == 0.5


def test_wanda_prune_sets_half_of_linear_weights_to_zero() -> None:
    model = TinySeq2Seq()
    with torch.no_grad():
        model.linear.weight.copy_(torch.arange(1, 5, dtype=torch.float32).reshape(1, 4))
    args = SimpleNamespace(
        max_input_len=4,
        max_target_len=4,
        calibration_batch_size=2,
        calibration_batches=1,
        sparsity=0.5,
    )
    state = DistributedState(enabled=False, rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))

    summary = wanda_prune(
        model,
        FakeTokenizer(),
        [{"prompt": "turn on light", "response": "ok"}, {"prompt": "turn off light", "response": "ok"}],
        args,
        state,
    )

    assert summary["pruned_linear_layers"] == 1
    assert summary["skipped_linear_layers"] == 0
    assert int(model.linear.weight.eq(0).sum().item()) == 2


def test_calibration_dataset_pads_variable_length_labels_for_collate() -> None:
    dataset = CalibrationDataset(
        records=[
            {"prompt": "a", "response": "short"},
            {"prompt": "b", "response": "much longer response"},
        ],
        tokenizer=VariableLengthTargetTokenizer(),
        max_input_len=8,
        max_target_len=16,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    batch = next(iter(loader))

    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 16)


def test_aggregate_prune_eval_reports_keeps_all_method_em_metrics(tmp_path: Path) -> None:
    report_path = tmp_path / "magnitude.json"
    report_path.write_text(
        json.dumps(
            {
                "pruned_model_path": "runs/magnitude/pruned_model",
                "datasets": {
                    "benchmark": {"path": "benchmark.json", "total": 2},
                    "training": {"path": "train.json", "total": 4},
                },
                "summary": {
                    "original_before_prune": {
                        "benchmark": {
                            "total": 2,
                            "em1": 0.5,
                            "em5": 1.0,
                            "em1_percent": 50.0,
                            "em5_percent": 100.0,
                            "accuracy": 0.5,
                            "accuracy_percent": 50.0,
                        },
                        "training": {
                            "total": 4,
                            "em1": 0.75,
                            "em5": 1.0,
                            "em1_percent": 75.0,
                            "em5_percent": 100.0,
                            "accuracy": 0.75,
                            "accuracy_percent": 75.0,
                        },
                    },
                    "pruned_after_50_percent": {
                        "benchmark": {
                            "total": 2,
                            "em1": 0.25,
                            "em5": 0.5,
                            "em1_percent": 25.0,
                            "em5_percent": 50.0,
                            "accuracy": 0.25,
                            "accuracy_percent": 25.0,
                        }
                    },
                },
                "pruning": {"method": "magnitude", "sparsity": 0.5},
            }
        ),
        encoding="utf-8",
    )

    aggregate = build_aggregate_report(
        base_model="charent/ChatLM-mini-Chinese",
        contrastive_model="runs/contrastive",
        epochs=5,
        sparsity=0.5,
        method_reports=[("magnitude", report_path)],
        contrastive_train_json="data/SCENIC_full_anchor_positive_negative.json",
    )

    assert aggregate["contrastive_epochs"] == 5
    assert aggregate["methods"]["magnitude"]["original_before_prune"]["training"]["em1"] == 0.75
    assert aggregate["methods"]["magnitude"]["pruned_after_50_percent"]["benchmark"]["em5"] == 0.5
    assert {
        "method": "magnitude",
        "phase": "pruned_after_50_percent",
        "dataset": "benchmark",
        "total": 2,
        "em1": 0.25,
        "em5": 0.5,
        "em1_percent": 25.0,
        "em5_percent": 50.0,
        "accuracy": 0.25,
        "accuracy_percent": 25.0,
    } in aggregate["table"]
