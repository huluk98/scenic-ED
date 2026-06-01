from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import TripletSFTModule, model_for_save


class FakeSeq2SeqModel:
    def __init__(self) -> None:
        self.base_model = object()


def test_model_for_save_keeps_regular_seq2seq_lm_head_model() -> None:
    model = FakeSeq2SeqModel()

    assert model_for_save(model) is model


def test_model_for_save_unwraps_only_scenic_triplet_wrapper() -> None:
    model = FakeSeq2SeqModel()
    wrapped = TripletSFTModule(model).module

    assert model_for_save(wrapped) is model
