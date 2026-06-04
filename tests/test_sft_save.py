from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scenic_train_chatlm_sft import (
    ContrastiveSFTConfig,
    TripletSFTModule,
    model_load_dtype,
    model_for_save,
    repair_tokenizer_files_for_auto_load,
    sanitize_config_for_json,
    sanitize_tokenizer_for_save,
)


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


def test_sanitize_config_for_json_converts_torch_dtypes() -> None:
    config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        nested={"dtype": torch.float16},
        dtypes=[torch.float32],
    )

    sanitize_config_for_json(config)

    assert config.torch_dtype == "bfloat16"
    assert config.nested == {"dtype": "float16"}
    assert config.dtypes == ["float32"]
    json.dumps(vars(config))


def test_sanitize_tokenizer_for_save_converts_init_kwargs_dtypes() -> None:
    tokenizer = SimpleNamespace(init_kwargs={"torch_dtype": torch.bfloat16})

    sanitize_tokenizer_for_save(tokenizer)

    assert tokenizer.init_kwargs == {"torch_dtype": "bfloat16"}
    json.dumps(tokenizer.init_kwargs)


def test_fp16_contrastive_keeps_default_model_load_dtype_for_grad_scaler() -> None:
    config = ContrastiveSFTConfig(fp16=True, bf16=False)

    assert model_load_dtype(config, torch.device("cuda")) is None


def test_bf16_contrastive_uses_bfloat16_model_load_dtype() -> None:
    config = ContrastiveSFTConfig(fp16=False, bf16=True)

    assert model_load_dtype(config, torch.device("cuda")) == torch.bfloat16


def test_repair_tokenizer_config_replaces_tokenizersbackend_from_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "base"
    output_dir = tmp_path / "fine_tuned"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "ChatLMTokenizer", "auto_map": {"AutoTokenizer": "tokenization_chatlm.ChatLMTokenizer"}}),
        encoding="utf-8",
    )
    (source_dir / "tokenization_chatlm.py").write_text("# custom tokenizer\n", encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding="utf-8",
    )

    repair_tokenizer_files_for_auto_load(output_dir, source_dir=source_dir)

    repaired = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert repaired["tokenizer_class"] == "ChatLMTokenizer"
    assert (output_dir / "tokenization_chatlm.py").exists()


def test_repair_tokenizer_config_removes_tokenizersbackend_without_source(tmp_path: Path) -> None:
    output_dir = tmp_path / "fine_tuned"
    output_dir.mkdir()
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "tokenizer_file": "tokenizer.json"}),
        encoding="utf-8",
    )

    repair_tokenizer_files_for_auto_load(output_dir)

    repaired = json.loads((output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert "tokenizer_class" not in repaired
    assert "tokenizer_file" not in repaired
    assert repaired["_scenic_removed_tokenizer_class"] == "TokenizersBackend"
    assert repaired["_scenic_removed_fast_tokenizer_keys"] == ["tokenizer_file"]


def test_repair_checkpoint_copies_custom_modeling_code_from_source(tmp_path: Path) -> None:
    source_dir = tmp_path / "base"
    output_dir = tmp_path / "fine_tuned"
    source_dir.mkdir()
    output_dir.mkdir()
    (source_dir / "modeling_chat.py").write_text("# custom model code\n", encoding="utf-8")
    (source_dir / "configuration_chat.py").write_text("# custom config code\n", encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": "configuration_chat.ChatConfig",
                    "AutoModelForSeq2SeqLM": "modeling_chat.ChatForConditionalGeneration",
                }
            }
        ),
        encoding="utf-8",
    )

    repair_tokenizer_files_for_auto_load(output_dir, source_dir=source_dir)

    assert (output_dir / "modeling_chat.py").exists()
    assert (output_dir / "configuration_chat.py").exists()
