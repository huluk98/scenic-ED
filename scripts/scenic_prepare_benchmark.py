#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "x", "INSTRUCTION")
RESPONSE_FIELDS = ("response", "output", "answer", "completion", "y", "RESPONSE")

DEVICE_TERMS: dict[str, tuple[str, ...]] = {
    "light": ("灯", "灯光", "台灯", "照明"),
    "air_conditioner": ("空调", "温度", "风速", "风向", "制冷", "制热", "送风"),
    "tv": ("电视",),
    "speaker": ("音箱", "音乐", "播放", "暂停", "歌曲", "音量"),
    "dehumidifier": ("抽湿机", "除湿", "湿度"),
}

ACTION_TERMS = (
    "打开",
    "开启",
    "关闭",
    "调",
    "设",
    "定",
    "提高",
    "降低",
    "升",
    "降",
    "播放",
    "暂停",
    "切换",
    "换",
)

INDIRECT_TERMS = (
    "太热",
    "好热",
    "太冷",
    "好冷",
    "太暗",
    "太亮",
    "有点暗",
    "有点亮",
    "不够亮",
    "不够大",
    "不舒服",
    "看不清",
    "睡觉",
    "学习",
    "冷",
    "热",
)

TIMER_RE = re.compile(r"(\d+|一|二|三|四|五|六|七|八|九|十|半).{0,4}(分钟|小时|点|时)")
NUMBER_RE = re.compile(r"\d+")
TRAILING_PUNCTUATION_RE = re.compile(r"[。！？!?,，、\s]+$")
FALSE_NEGATIVE_RESPONSE_SIMILARITY = 0.82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a SCENIC smart-home benchmark with positive/negative samples and difficulty labels."
    )
    parser.add_argument("--input", required=True, help="Path to the source .json or .jsonl benchmark.")
    parser.add_argument("--output", required=True, help="Path for the augmented dataset.")
    parser.add_argument("--report", default=None, help="Optional JSON quality report path.")
    parser.add_argument(
        "--format",
        choices=("jsonl", "json"),
        default=None,
        help="Output format. Defaults to the output extension, or jsonl.",
    )
    parser.add_argument("--seed", type=int, default=619, help="Sampling seed for reproducible negatives.")
    parser.add_argument(
        "--singleton-positive",
        choices=("synthetic", "self", "blank"),
        default="synthetic",
        help="How to fill positive when a response class has only one prompt.",
    )
    parser.add_argument(
        "--keep-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep scenic_metadata with sampling and quality details.",
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
        text = clean_text(record.get(field))
        if text:
            return text, field
    return "", None


def normalize_for_match(text: str) -> str:
    return "".join(clean_text(text).split())


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
        raise ValueError(f"{path} must contain JSON objects.")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{path} contains at least one non-object row.")
    return list(records)


def extract_pair(record: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    prompt, prompt_field = first_text(record, PROMPT_FIELDS)
    response, response_field = first_text(record, RESPONSE_FIELDS)
    extra_input = clean_text(record.get("input")) if "instruction" in record else ""
    if extra_input and extra_input != prompt:
        prompt = f"{prompt}\n{extra_input}" if prompt else extra_input
    return prompt, response, prompt_field, response_field


def infer_devices(*texts: str) -> set[str]:
    joined = " ".join(texts)
    devices: set[str] = set()
    for device, terms in DEVICE_TERMS.items():
        if any(term in joined for term in terms):
            devices.add(device)
    return devices


def action_count(response: str) -> int:
    normalized = clean_text(response)
    count = normalized.count("已")
    if count:
        return count
    return max(1, sum(1 for term in ACTION_TERMS if term in normalized))


def has_timer(text: str) -> bool:
    return bool(TIMER_RE.search(text))


def has_number(text: str) -> bool:
    return bool(NUMBER_RE.search(text)) or has_timer(text)


def has_indirect_signal(prompt: str) -> bool:
    return any(term in prompt for term in INDIRECT_TERMS)


def classify_difficulty(prompt: str, response: str) -> tuple[str, dict[str, Any]]:
    devices = infer_devices(prompt, response)
    actions = action_count(response)
    combined_text = f"{prompt} {response}"
    timer = has_timer(combined_text)
    numeric = has_number(combined_text)
    indirect = has_indirect_signal(prompt)
    conjunctions = sum(prompt.count(term) + response.count(term) for term in ("并", "同时", "和", "以及"))
    prompt_len = len(prompt)
    response_len = len(response)

    if len(devices) >= 2 or actions >= 3 or (timer and actions >= 2) or prompt_len >= 42 or conjunctions >= 2:
        difficulty = "hard"
    elif actions == 2 or timer or numeric or indirect or prompt_len >= 20 or response_len >= 28 or conjunctions == 1:
        difficulty = "medium"
    else:
        difficulty = "easy"

    features = {
        "devices": sorted(devices),
        "device_count": len(devices),
        "action_count": actions,
        "has_timer": timer,
        "has_number": numeric,
        "has_indirect_signal": indirect,
        "conjunction_count": conjunctions,
        "prompt_length": prompt_len,
        "response_length": response_len,
    }
    return difficulty, features


def make_synthetic_positive(prompt: str) -> str:
    base = TRAILING_PUNCTUATION_RE.sub("", clean_text(prompt))
    if not base:
        return ""
    if base.startswith(("请", "麻烦", "帮我", "把")):
        candidate = f"{base}吧"
    else:
        candidate = f"请{base}"
    if normalize_for_match(candidate) == normalize_for_match(prompt):
        candidate = f"麻烦{base}"
    return f"{candidate}。"


def response_similarity(left: str, right: str) -> float:
    left_set = set(normalize_for_match(left))
    right_set = set(normalize_for_match(right))
    return jaccard_similarity(left_set, right_set)


def jaccard_similarity(left_set: set[str], right_set: set[str]) -> float:
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def choose_positive(
    row_index: int,
    rows: list[dict[str, Any]],
    response_groups: dict[str, list[int]],
    singleton_positive: str,
    rng: random.Random,
) -> tuple[str, str, str]:
    response_key = rows[row_index]["_response_key"]
    candidates = [idx for idx in response_groups[response_key] if idx != row_index]
    if candidates:
        selected = rng.choice(candidates)
        return rows[selected]["prompt"], rows[selected]["response"], "same_response"

    if singleton_positive == "synthetic":
        return make_synthetic_positive(rows[row_index]["prompt"]), rows[row_index]["response"], "synthetic_singleton"
    if singleton_positive == "self":
        return rows[row_index]["prompt"], rows[row_index]["response"], "self_singleton"
    return "", rows[row_index]["response"], "blank_singleton"


def choose_negative(row_index: int, rows: list[dict[str, Any]], rng: random.Random) -> tuple[str, str, str]:
    row = rows[row_index]
    row_devices = set(row["features"]["devices"])
    candidates = [
        idx
        for idx, candidate in enumerate(rows)
        if idx != row_index and candidate["_response_key"] != row["_response_key"]
    ]
    safe_candidates = [
        idx
        for idx in candidates
        if rows[idx]["prompt_norm"] != row["prompt_norm"]
        and jaccard_similarity(row["response_chars"], rows[idx]["response_chars"]) < FALSE_NEGATIVE_RESPONSE_SIMILARITY
    ]
    if safe_candidates:
        candidates = safe_candidates
    if not candidates:
        return "", "", "none"

    same_device = [
        idx
        for idx in candidates
        if row_devices and row_devices.intersection(rows[idx]["features"]["devices"])
    ]
    pool = same_device or candidates
    pool = sorted(
        pool,
        key=lambda idx: (
            jaccard_similarity(row["response_chars"], rows[idx]["response_chars"]),
            -jaccard_similarity(row["prompt_chars"], rows[idx]["prompt_chars"]),
            -abs(row["features"]["action_count"] - rows[idx]["features"]["action_count"]),
        ),
        reverse=True,
    )
    hard_pool = pool[: min(25, len(pool))]
    selected = rng.choice(hard_pool)
    source = "same_device_hard_negative" if selected in same_device else "different_response"
    return rows[selected]["prompt"], rows[selected]["response"], source


def quality_flags(record: dict[str, Any], prompt_to_responses: dict[str, set[str]]) -> list[str]:
    flags: list[str] = []
    prompt = record["prompt"]
    response = record["response"]
    if not prompt:
        flags.append("missing_prompt")
    if not response:
        flags.append("missing_response")
    if prompt and len(prompt) <= 2:
        flags.append("very_short_prompt")
    if response and not response.endswith("。"):
        flags.append("response_missing_final_period")
    if response and not response.startswith("好的"):
        flags.append("response_not_standard_acknowledgement")
    if not record["features"]["devices"]:
        flags.append("unknown_device")
    if len(prompt_to_responses[prompt]) > 1:
        flags.append("same_prompt_multiple_responses")
    return flags


def summarize_lengths(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.mean(values), 2),
        "max": max(values),
    }


def build_report(
    rows: list[dict[str, Any]],
    augmented: list[dict[str, Any]],
    response_groups: dict[str, list[int]],
) -> dict[str, Any]:
    prompt_counter = Counter(row["prompt"] for row in rows)
    response_counter = Counter(row["response"] for row in rows)
    pair_counter = Counter((row["prompt"], row["response"]) for row in rows)
    flag_counter = Counter(flag for row in augmented for flag in row.get("scenic_metadata", {}).get("quality_flags", []))
    class_sizes = [len(indices) for indices in response_groups.values()]
    difficulty_counter = Counter(row["difficulty"] for row in augmented)
    positive_source_counter = Counter(row["scenic_metadata"]["positive_source"] for row in augmented)
    negative_source_counter = Counter(row["scenic_metadata"]["negative_source"] for row in augmented)
    quality_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in augmented:
        flags = row.get("scenic_metadata", {}).get("quality_flags", [])
        for flag in flags:
            if len(quality_examples[flag]) >= 5:
                continue
            quality_examples[flag].append(
                {
                    "index": row.get("scenic_metadata", {}).get("index"),
                    "prompt": row.get("prompt"),
                    "response": row.get("response"),
                    "difficulty": row.get("difficulty"),
                }
            )

    return {
        "total_records": len(rows),
        "valid_records": sum(1 for row in rows if row["prompt"] and row["response"]),
        "missing_prompt": sum(1 for row in rows if not row["prompt"]),
        "missing_response": sum(1 for row in rows if not row["response"]),
        "unique_prompts": len(prompt_counter),
        "duplicate_prompt_extra_rows": sum(count - 1 for count in prompt_counter.values() if count > 1),
        "conflicting_prompt_count": sum(1 for prompt in prompt_counter if len({row["response"] for row in rows if row["prompt"] == prompt}) > 1),
        "unique_responses": len(response_counter),
        "singleton_response_classes": sum(1 for count in response_counter.values() if count == 1),
        "response_classes_with_positive_candidates": sum(1 for count in response_counter.values() if count > 1),
        "exact_duplicate_pair_extra_rows": sum(count - 1 for count in pair_counter.values() if count > 1),
        "response_class_size": summarize_lengths(class_sizes),
        "prompt_length": summarize_lengths([len(row["prompt"]) for row in rows]),
        "response_length": summarize_lengths([len(row["response"]) for row in rows]),
        "difficulty": dict(sorted(difficulty_counter.items())),
        "positive_source": dict(sorted(positive_source_counter.items())),
        "negative_source": dict(sorted(negative_source_counter.items())),
        "quality_flags": dict(sorted(flag_counter.items())),
        "quality_examples": {key: value for key, value in sorted(quality_examples.items())},
    }


def output_format(path: Path, requested: str | None) -> str:
    if requested:
        return requested
    return "json" if path.suffix.lower() == ".json" else "jsonl"


def write_records(path: Path, records: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    records = read_records(input_path)

    rows: list[dict[str, Any]] = []
    prompt_to_responses: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        prompt, response, prompt_field, response_field = extract_pair(record)
        difficulty, features = classify_difficulty(prompt, response)
        row = {
            "index": index,
            "source": dict(record),
            "prompt": prompt,
            "response": response,
            "prompt_norm": normalize_for_match(prompt),
            "response_norm": normalize_for_match(response),
            "prompt_chars": set(normalize_for_match(prompt)),
            "response_chars": set(normalize_for_match(response)),
            "prompt_field": prompt_field,
            "response_field": response_field,
            "difficulty": difficulty,
            "features": features,
            "_response_key": normalize_for_match(response),
        }
        rows.append(row)
        prompt_to_responses[prompt].add(response)

    response_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        response_groups[row["_response_key"]].append(index)

    augmented: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        positive, positive_response, positive_source = choose_positive(
            index, rows, response_groups, args.singleton_positive, rng
        )
        negative, negative_response, negative_source = choose_negative(index, rows, rng)
        flags = quality_flags(row, prompt_to_responses)
        output = dict(row["source"])
        output.update(
            {
                "prompt": row["prompt"],
                "response": row["response"],
                "positive": positive,
                "negative": negative,
                "difficulty": row["difficulty"],
            }
        )
        metadata = {
            "index": row["index"],
            "response_class_size": len(response_groups[row["_response_key"]]),
            "positive_response": positive_response,
            "negative_response": negative_response,
            "positive_source": positive_source,
            "negative_source": negative_source,
            "features": row["features"],
            "quality_flags": flags,
        }
        if args.keep_metadata:
            output["scenic_metadata"] = metadata
        augmented.append(output)

    fmt = output_format(output_path, args.format)
    write_records(output_path, augmented, fmt)
    report = build_report(rows, augmented, response_groups)

    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote augmented dataset: {output_path}")
    if args.report:
        print(f"wrote quality report: {Path(args.report).expanduser()}")


if __name__ == "__main__":
    main()
