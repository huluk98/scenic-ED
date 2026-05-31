from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import ContrastiveSFTConfig, RegularSFTConfig, make_contrastive_collate, make_regular_collate


class FakeTokenizer:
    pad_token_id = 0

    def __call__(
        self,
        texts=None,
        text_target=None,
        padding=True,
        truncation=True,
        max_length=16,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        values = text_target if text_target is not None else texts
        batch_size = len(values)
        return {
            "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
            "token_type_ids": torch.zeros((batch_size, 4), dtype=torch.long),
        }


def test_regular_collate_removes_token_type_ids() -> None:
    collate = make_regular_collate(FakeTokenizer(), RegularSFTConfig())
    batch = collate([{"prompt": "10点关空调。", "response": "好的，已为空调设置10点关闭。"}])

    assert "token_type_ids" not in batch
    assert set(batch) == {"input_ids", "attention_mask", "labels"}


def test_contrastive_collate_removes_token_type_ids() -> None:
    collate = make_contrastive_collate(FakeTokenizer(), ContrastiveSFTConfig())
    batch = collate(
        [
            {
                "anchor": "10点关空调。",
                "positive": "请在10点关闭空调。",
                "negative": "空调1点半关。",
                "response": "好的，已为空调设置10点关闭。",
            }
        ]
    )

    for key in ("anchor", "positive", "negative"):
        assert "token_type_ids" not in batch[key]
        assert set(batch[key]) == {"input_ids", "attention_mask"}
    assert "labels" in batch
