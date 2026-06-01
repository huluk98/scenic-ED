from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import TripletSFTModule


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
    def __init__(self, hidden_size: int = 3) -> None:
        super().__init__()
        self.encoder = FakeEncoder(hidden_size)
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
        return SimpleNamespace(loss=torch.tensor(2.0), encoder_last_hidden_state=hidden)


def test_triplet_forward_combines_anchor_and_positive_generation_pass() -> None:
    base_model = FakeSeq2Seq()
    model = TripletSFTModule(base_model).module
    labels = torch.ones(2, 4, dtype=torch.long)

    loss, gen_loss, align_loss = model(
        anchor={
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        },
        positive={
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
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
