from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_original_chatlm import (  # noqa: E402
    DEFAULT_MODEL,
    eos_contains,
    eos_diagnostics,
    generated_sequence_ended_with_eos,
    parse_args,
)


def test_parse_args_accepts_huggingface_model_id_as_only_argument() -> None:
    args = parse_args(["charent/ChatLM-mini-Chinese"])

    assert args.model == "charent/ChatLM-mini-Chinese"


def test_parse_args_defaults_to_chatlm_mini_chinese() -> None:
    args = parse_args([])

    assert args.model == DEFAULT_MODEL


def test_parse_args_keeps_model_flag_for_compatibility() -> None:
    args = parse_args(["--model", "local/chatlm"])

    assert args.model == "local/chatlm"


def test_eos_diagnostics_fills_missing_model_and_generation_eos() -> None:
    tokenizer = SimpleNamespace(eos_token="</s>", eos_token_id=2)
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=None),
        generation_config=SimpleNamespace(eos_token_id=None),
    )

    diagnostics = eos_diagnostics(tokenizer, model, ensure_eos=True)

    assert diagnostics["model_config_eos_token_id_after"] == 2
    assert diagnostics["generation_config_eos_token_id_after"] == 2
    assert diagnostics["model_config_contains_tokenizer_eos"] is True
    assert diagnostics["generation_config_contains_tokenizer_eos"] is True
    assert diagnostics["changed_model_config"] is True
    assert diagnostics["changed_generation_config"] is True


def test_eos_diagnostics_preserves_existing_list_and_appends_tokenizer_eos() -> None:
    tokenizer = SimpleNamespace(eos_token="</s>", eos_token_id=2)
    model = SimpleNamespace(
        config=SimpleNamespace(eos_token_id=[1]),
        generation_config=SimpleNamespace(eos_token_id=[1, 2]),
    )

    diagnostics = eos_diagnostics(tokenizer, model, ensure_eos=True)

    assert diagnostics["model_config_eos_token_id_after"] == [1, 2]
    assert diagnostics["generation_config_eos_token_id_after"] == [1, 2]
    assert diagnostics["changed_model_config"] is True
    assert diagnostics["changed_generation_config"] is False


def test_eos_contains_accepts_int_and_list_values() -> None:
    assert eos_contains(2, 2)
    assert eos_contains([1, 2], 2)
    assert not eos_contains([1, 3], 2)
    assert not eos_contains(None, 2)


def test_generated_sequence_ended_with_eos_ignores_pad_tokens() -> None:
    sequence = torch.tensor([0, 5, 6, 2, 0, 0])

    assert generated_sequence_ended_with_eos(sequence, eos_token_id=2, pad_token_id=0)
    assert not generated_sequence_ended_with_eos(sequence, eos_token_id=7, pad_token_id=0)
