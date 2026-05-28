#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "x", "INSTRUCTION")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "y", "RESPONSE")
PREDICTION_FIELDS = ("prediction", "predicted_response", "generated_response", "model_output", "model_response")
POSITIVE_FIELDS = ("positive", "pos", "x_positive", "chosen", "x_plus")
NEGATIVE_FIELDS = ("negative", "neg", "x_negative", "rejected", "x_minus")

DEFAULT_SCENIC_RUBRIC = """SCENIC framework rubric from the paper:
SCENIC means Semantic Conditioned Edge-aware Neural framework for structured IoT
Command generation. It models smart-home understanding as many-to-one normalized
IoT response generation: multiple natural-language smart-home commands may map
to the same deterministic device-control response.

Evaluate each row as supervised fine-tuning data for this task:
1. smart_home_command: the prompt is a valid natural-language smart-home command,
   including direct, indirect, single-device, or multi-device control.
2. normalized_iot_response: the response is a compact deterministic IoT control
   target, not a chatty explanation, essay, or open-ended assistant response.
3. prompt_response_alignment: the response executes the command implied by the
   prompt and does not control unrelated devices or attributes.
4. many_to_one_consistency: semantically equivalent prompt variants should map
   to the same normalized response; tolerate paraphrases with identical intent.
5. semantic_alignment_fields: if positive/negative fields are present, positive
   should preserve the same normalized response class and negative should express
   a different response class.
6. edge_ready_structure: the row is concise, deterministic, and suitable for
   compact edge deployment where partial matches can be invalid actions.

Score each dimension from 0.0 to 1.0. The SCENIC score is the mean of all six
dimension scores. Rows pass when the score is at least the configured pass
threshold and both prompt and response are valid.
"""

SCENIC_KEYS = (
    "smart_home_command",
    "normalized_iot_response",
    "prompt_response_alignment",
    "many_to_one_consistency",
    "semantic_alignment_fields",
    "edge_ready_structure",
)

EVAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "input_valid": {"type": "boolean"},
        "response_valid": {"type": "boolean"},
        "scores": {
            "type": "object",
            "properties": {key: {"type": "number"} for key in SCENIC_KEYS},
            "required": list(SCENIC_KEYS),
            "additionalProperties": False,
        },
        "accuracy": {"type": "number"},
        "scenic_score": {"type": "number"},
        "passes_scenic": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "suggested_prompt": {"type": "string"},
        "suggested_response": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "input_valid",
        "response_valid",
        "scores",
        "accuracy",
        "scenic_score",
        "passes_scenic",
        "issues",
        "suggested_prompt",
        "suggested_response",
        "rationale",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt/response JSON data against a SCENIC rubric with gpt-5-mini."
    )
    parser.add_argument("--input", required=True, help="Path to a .json or .jsonl prompt/response dataset.")
    parser.add_argument("--output", default=None, help="Optional path for a JSON report. Defaults to stdout.")
    parser.add_argument("--model", default="gpt-5-mini", help="OpenAI model to use.")
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=("minimal", "low", "medium", "high"),
        help="Reasoning effort for GPT-5 style models.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many rows.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset to start from.")
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.8,
        help="Minimum SCENIC score required for passes_scenic.",
    )
    parser.add_argument(
        "--rubric-file",
        default=None,
        help="Optional text file containing your exact SCENIC framework. Overrides the default rubric.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate file parsing and prompt/response extraction; do not call the OpenAI API.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str | None]:
    for field in fields:
        value = clean_text(record.get(field))
        if value:
            return value, field
    return "", None


def extract_prompt_response(record: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    prompt, prompt_field = first_text(record, PROMPT_FIELDS)
    response, response_field = first_text(record, RESPONSE_FIELDS)

    extra_input = clean_text(record.get("input")) if "instruction" in record else ""
    if extra_input and extra_input != prompt:
        prompt = f"{prompt}\n{extra_input}" if prompt else extra_input

    return prompt, response, prompt_field, response_field


def normalize_for_exact_match(text: str) -> str:
    return "".join(text.split())


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object.")
                records.append(value)
        return records

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)

    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        for key in ("records", "data", "examples", "items"):
            if isinstance(value.get(key), list):
                records = value[key]
                break
        else:
            records = [value]
    else:
        raise ValueError(f"{path} must contain a JSON object, array, or JSONL objects.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains at least one non-object row.")
    return list(records)


def load_rubric(path: str | None) -> str:
    if not path:
        return DEFAULT_SCENIC_RUBRIC
    return Path(path).expanduser().read_text(encoding="utf-8").strip()


def build_system_instructions(rubric: str, pass_threshold: float) -> str:
    return (
        "You are a strict local dataset quality agent for supervised fine-tuning data. "
        "Evaluate whether each prompt/response pair is usable training data. "
        "Use the supplied SCENIC paper rubric exactly, not a generic QA rubric. "
        "Return only JSON that matches the schema. "
        "Keep suggested_prompt and suggested_response identical to the original values unless a small, "
        "obvious repair would make the row better training data. "
        f"Set passes_scenic true only when scenic_score >= {pass_threshold:.3f} and both fields are valid. "
        "Do not compute exact-match prediction accuracy; local code handles that when a prediction field exists.\n\n"
        f"{rubric}"
    )


def normalize_row(record: dict[str, Any], index: int) -> dict[str, Any]:
    prompt, response, prompt_field, response_field = extract_prompt_response(record)
    prediction, prediction_field = first_text(record, PREDICTION_FIELDS)
    positive, positive_field = first_text(record, POSITIVE_FIELDS)
    negative, negative_field = first_text(record, NEGATIVE_FIELDS)
    issues: list[str] = []
    if not prompt:
        issues.append("missing prompt-like field")
    if not response:
        issues.append("missing response-like field")
    exact_match = None
    if prediction:
        exact_match = normalize_for_exact_match(prediction) == normalize_for_exact_match(response)
    return {
        "index": index,
        "prompt": prompt,
        "response": response,
        "prediction": prediction,
        "positive": positive,
        "negative": negative,
        "prompt_field": prompt_field,
        "response_field": response_field,
        "prediction_field": prediction_field,
        "positive_field": positive_field,
        "negative_field": negative_field,
        "exact_match": exact_match,
        "local_issues": issues,
    }


def get_openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI Python SDK is not installed. Install dependencies with `pip install -e .` "
            "or `pip install openai`."
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it in your shell before running this script; "
            "do not paste API keys into source files."
        )
    return OpenAI()


def extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)

    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        raise ValueError("Could not read text from OpenAI response object.")

    chunks: list[str] = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    if chunks:
        return "\n".join(chunks)
    raise ValueError("OpenAI response did not include output text.")


def clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    return min(1.0, max(0.0, number))


def normalize_eval_result(result: dict[str, Any], row: dict[str, Any], pass_threshold: float) -> dict[str, Any]:
    scores = {key: clamp_score(result.get("scores", {}).get(key)) for key in SCENIC_KEYS}
    scenic_score = sum(scores.values()) / len(scores)
    accuracy = float(row["exact_match"]) if row["exact_match"] is not None else scenic_score

    input_valid = bool(result.get("input_valid")) and bool(row["prompt"])
    response_valid = bool(result.get("response_valid")) and bool(row["response"])
    issues = [str(issue).strip() for issue in result.get("issues", []) if str(issue).strip()]
    issues.extend(row["local_issues"])

    return {
        "index": row["index"],
        "prompt": row["prompt"],
        "response": row["response"],
        "prediction": row["prediction"],
        "positive": row["positive"],
        "negative": row["negative"],
        "prompt_field": row["prompt_field"],
        "response_field": row["response_field"],
        "prediction_field": row["prediction_field"],
        "positive_field": row["positive_field"],
        "negative_field": row["negative_field"],
        "exact_match": row["exact_match"],
        "input_valid": input_valid,
        "response_valid": response_valid,
        "scores": scores,
        "scenic_score": scenic_score,
        "accuracy": accuracy,
        "passes_scenic": input_valid and response_valid and scenic_score >= pass_threshold,
        "issues": sorted(set(issues)),
        "suggested_prompt": clean_text(result.get("suggested_prompt")) or row["prompt"],
        "suggested_response": clean_text(result.get("suggested_response")) or row["response"],
        "rationale": clean_text(result.get("rationale")),
    }


def evaluate_row(
    client: Any,
    row: dict[str, Any],
    model: str,
    instructions: str,
    reasoning_effort: str,
    pass_threshold: float,
) -> dict[str, Any]:
    if row["local_issues"]:
        result = {
            "input_valid": bool(row["prompt"]),
            "response_valid": bool(row["response"]),
            "scores": {key: 0.0 for key in SCENIC_KEYS},
            "accuracy": 0.0,
            "scenic_score": 0.0,
            "passes_scenic": False,
            "issues": row["local_issues"],
            "suggested_prompt": row["prompt"],
            "suggested_response": row["response"],
            "rationale": "Local validation found missing prompt or response data.",
        }
        return normalize_eval_result(result, row, pass_threshold)

    payload = {
        "row_index": row["index"],
        "prompt": row["prompt"],
        "response": row["response"],
        "prediction": row["prediction"],
        "positive": row["positive"],
        "negative": row["negative"],
    }
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            }
        ],
        reasoning={"effort": reasoning_effort},
        store=False,
        text={
            "format": {
                "type": "json_schema",
                "name": "scenic_eval_result",
                "schema": EVAL_SCHEMA,
                "strict": True,
            }
        },
    )
    result = json.loads(extract_output_text(response))
    return normalize_eval_result(result, row, pass_threshold)


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in results if row["input_valid"] and row["response_valid"]]
    scenic_scores: list[float] = []
    for row in results:
        scenic_score = row.get("scenic_score")
        if scenic_score is not None:
            scenic_scores.append(float(scenic_score))
    predicted = [row for row in valid if row.get("exact_match") is not None]
    exact_match_accuracy = None
    if predicted:
        exact_match_accuracy = sum(1 for row in predicted if row["exact_match"]) / len(predicted)
    return {
        "total_records": len(results),
        "valid_records": len(valid),
        "invalid_records": len(results) - len(valid),
        "scored_records": len(scenic_scores),
        "predicted_records": len(predicted),
        "passing_records": sum(1 for row in valid if row["passes_scenic"]),
        "overall_scenic_score": (sum(scenic_scores) / len(scenic_scores)) if scenic_scores else None,
        "exact_match_accuracy": exact_match_accuracy,
        "overall_accuracy": exact_match_accuracy
        if exact_match_accuracy is not None
        else ((sum(scenic_scores) / len(scenic_scores)) if scenic_scores else None),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    records = read_records(input_path)
    selected = records[max(0, args.start) :]
    if args.limit is not None:
        selected = selected[: max(0, args.limit)]

    rows = [normalize_row(record, index=args.start + offset) for offset, record in enumerate(selected)]
    rubric = load_rubric(args.rubric_file)
    instructions = build_system_instructions(rubric, args.pass_threshold)

    results: list[dict[str, Any]] = []
    client = None if args.dry_run else get_openai_client()
    for row in rows:
        if args.dry_run:
            results.append(
                {
                    "index": row["index"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "prediction": row["prediction"],
                    "positive": row["positive"],
                    "negative": row["negative"],
                    "prompt_field": row["prompt_field"],
                    "response_field": row["response_field"],
                    "prediction_field": row["prediction_field"],
                    "positive_field": row["positive_field"],
                    "negative_field": row["negative_field"],
                    "exact_match": row["exact_match"],
                    "input_valid": bool(row["prompt"]),
                    "response_valid": bool(row["response"]),
                    "scores": {key: None for key in SCENIC_KEYS},
                    "scenic_score": None,
                    "accuracy": float(row["exact_match"]) if row["exact_match"] is not None else None,
                    "passes_scenic": False,
                    "issues": row["local_issues"],
                    "suggested_prompt": row["prompt"],
                    "suggested_response": row["response"],
                    "rationale": "Dry run: OpenAI API evaluation was skipped.",
                }
            )
        else:
            results.append(
                evaluate_row(
                    client=client,
                    row=row,
                    model=args.model,
                    instructions=instructions,
                    reasoning_effort=args.reasoning_effort,
                    pass_threshold=args.pass_threshold,
                )
            )
            print(f"evaluated row {row['index']}", file=sys.stderr)

    report = {
        "input_path": str(input_path),
        "model": None if args.dry_run else args.model,
        "rubric": "custom" if args.rubric_file else "default_scenic",
        "pass_threshold": args.pass_threshold,
        "summary": summarize(results),
        "records": results,
    }

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output_path}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
