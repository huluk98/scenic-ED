from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import TripletSFTModule, pair_balanced_generation_loss


class FakeEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.calls = 0

    def forward(self, input_ids, attention_mask=None, return_dict=True):
        self.calls += 1
        hidden = torch.ones(input_ids.size(0), input_ids.size(1), self.hidden_size)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeSeq2Seq(torch.nn.Module):
    def __init__(self, hidden_size: int = 3, include_encoder_hidden: bool = True) -> None:
        super().__init__()
        self.encoder = FakeEncoder(hidden_size)
        self.include_encoder_hidden = include_encoder_hidden
        self.calls = 0
        self.last_input_shape = None
        self.last_labels_shape = None

    def get_encoder(self):
        return self.encoder

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        self.calls += 1
        self.last_input_shape = tuple(input_ids.shape)
        self.last_labels_shape = tuple(labels.shape)
        hidden = torch.ones(input_ids.size(0), input_ids.size(1), self.encoder.hidden_size)
        if self.include_encoder_hidden:
            return SimpleNamespace(loss=torch.tensor(2.0), encoder_last_hidden_state=hidden)
        return SimpleNamespace(loss=torch.tensor(2.0))


def test_triplet_forward_combines_anchor_and_positive_generation_pass() -> None:
    base_model = FakeSeq2Seq()
    model = TripletSFTModule(base_model).module
    labels = torch.ones(2, 4, dtype=torch.long)

    loss, gen_loss, align_loss = model(
        generation={
            "input_ids": torch.ones(4, 5, dtype=torch.long),
            "attention_mask": torch.ones(4, 5, dtype=torch.long),
        },
        negative={
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        },
        labels=labels,
        margin=0.5,
        alignment_weight=0.1,
    )

    assert base_model.calls == 1
    assert base_model.encoder.calls == 1
    assert base_model.last_input_shape == (4, 5)
    assert base_model.last_labels_shape == (4, 4)
    assert torch.isclose(gen_loss, torch.tensor(2.0))
    assert loss >= gen_loss
    assert align_loss >= 0


def test_triplet_forward_recomputes_generation_reps_when_outputs_omit_encoder_hidden() -> None:
    base_model = FakeSeq2Seq(include_encoder_hidden=False)
    model = TripletSFTModule(base_model).module
    labels = torch.ones(2, 4, dtype=torch.long)

    loss, gen_loss, align_loss = model(
        generation={
            "input_ids": torch.ones(4, 5, dtype=torch.long),
            "attention_mask": torch.ones(4, 5, dtype=torch.long),
        },
        negative={
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        },
        labels=labels,
        margin=0.5,
        alignment_weight=0.1,
    )

    assert base_model.calls == 1
    assert base_model.encoder.calls == 2
    assert torch.isclose(gen_loss, torch.tensor(2.0))
    assert loss >= gen_loss
    assert align_loss >= 0


def test_pair_balanced_generation_loss_averages_anchor_positive_by_tuple() -> None:
    labels = torch.tensor(
        [
            [0, -100, -100],
            [1, 1, 1],
            [0, 0, 0],
            [1, -100, -100],
        ]
    )
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 2.0], [0.0, -2.0]],
            [[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0]],
            [[0.0, 3.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    outputs = SimpleNamespace(loss=torch.tensor(999.0), logits=logits)

    loss = pair_balanced_generation_loss(outputs, labels, batch_size=2)

    token_losses = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    token_counts = labels.ne(-100).sum(dim=1).clamp(min=1)
    per_example_losses = token_losses.sum(dim=1) / token_counts
    expected = (0.5 * (per_example_losses[:2] + per_example_losses[2:])).mean()
    token_weighted = token_losses.sum() / token_counts.sum()

    assert torch.isclose(loss, expected)
    assert not torch.isclose(loss, token_weighted)
