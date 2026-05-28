from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGULAR_DATASET = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
CONTRASTIVE_DATASET = PROJECT_ROOT / "data" / "SCENIC_full_anchor_positive_negative.json"
IOT_BENCHMARK = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200.json"


def load_json(path: Path):
    assert path.exists(), f"missing expected dataset: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_only_expected_dataset_artifacts_are_tracked() -> None:
    expected = {
        REGULAR_DATASET.relative_to(PROJECT_ROOT).as_posix(),
        CONTRASTIVE_DATASET.relative_to(PROJECT_ROOT).as_posix(),
        IOT_BENCHMARK.relative_to(PROJECT_ROOT).as_posix(),
    }
    actual = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for folder in ("data", "generated")
        for path in (PROJECT_ROOT / folder).glob("*")
        if path.is_file()
    }
    assert actual == expected


def test_regular_sft_dataset_shape() -> None:
    rows = load_json(REGULAR_DATASET)
    assert isinstance(rows, list)
    assert len(rows) == 9772
    assert all(set(row) == {"prompt", "response"} for row in rows)
    assert all(isinstance(row["prompt"], str) and row["prompt"].strip() for row in rows)
    assert all(isinstance(row["response"], str) and row["response"].strip() for row in rows)


def test_contrastive_dataset_shape() -> None:
    rows = load_json(CONTRASTIVE_DATASET)
    required = {"anchor", "positive", "negative", "response"}
    assert isinstance(rows, list)
    assert len(rows) == 9772
    assert all(required <= set(row) for row in rows)
    for row in rows:
        assert all(isinstance(row[key], str) and row[key].strip() for key in required)


def test_iot_benchmark_shape() -> None:
    rows = load_json(IOT_BENCHMARK)
    assert isinstance(rows, list)
    assert len(rows) == 200
    assert all("prompt" in row and "response" in row for row in rows)
    assert all(isinstance(row["prompt"], str) and row["prompt"].strip() for row in rows)
    assert all(isinstance(row["response"], str) and row["response"].strip() for row in rows)
