#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "619_Luke_REPAIRED_positive_negative.jsonl"
DEFAULT_AUDIT = PROJECT_ROOT / "data" / "619_Luke_REPAIRED_positive_negative_audit.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "positive_negative_summary.md"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_smart_home_dataset import (  # noqa: E402
    canonical_for_intents,
    compact,
    infer_intents_from_pair,
    intent_key,
)


ALLOWED_NEGATIVE_TYPES = {
    "unsupported_device",
    "unsupported_action_for_device",
    "ambiguous_action_target",
    "cross_device_action_mismatch",
    "invalid_location_device_pair",
    "invalid_multi_device_combination",
    "unsupported_state_or_value",
    "non_control_intent",
}

NEGATIVE_TYPE_ORDER = {
    "unsupported_action_for_device": 0,
    "unsupported_state_or_value": 1,
    "cross_device_action_mismatch": 2,
    "invalid_multi_device_combination": 3,
    "invalid_location_device_pair": 4,
    "unsupported_device": 5,
    "ambiguous_action_target": 6,
    "non_control_intent": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full SCENIC positive/negative contrastive dataset from the repaired "
            "natural-language smart-home benchmark."
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def load_rows(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        prompt = clean_text(item.get("prompt"))
        response = clean_text(item.get("response"))
        if not prompt or not response:
            raise ValueError(f"{path}:{index} must contain non-empty prompt and response")
        rows.append({"prompt": prompt, "response": response})
    return rows


def char_jaccard_distance(left: str, right: str) -> float:
    left_chars = set(compact(left))
    right_chars = set(compact(right))
    if not left_chars and not right_chars:
        return 0.0
    return 1.0 - (len(left_chars & right_chars) / len(left_chars | right_chars))


def target_name(location: str, device: str) -> str:
    return f"{location}{device}" if location else device


def format_single_positive(intent: dict[str, Any], variant: int = 0) -> str:
    """Create a deterministic paraphrase that preserves the executable intent."""
    name = target_name(intent["location"], intent["device"])
    slots = intent["slots"]
    kind = intent["intent"]
    templates: dict[str, tuple[str, ...]] = {
        "light_on": (f"把{name}打开。", f"{name}有点暗，帮我开一下。"),
        "light_off": (f"帮我关掉{name}。", f"不用{name}了，关一下。"),
        "light_brightness_up": (f"把{name}亮度调高一点。", f"{name}现在有点暗，调亮一点。"),
        "light_brightness_down": (f"把{name}亮度调低一点。", f"{name}有点刺眼，调暗一点。"),
        "light_brightest": (f"把{name}亮度调到最高。", f"{name}开到最亮。"),
        "light_darkest": (f"把{name}亮度调到最低。", f"{name}调到最暗。"),
        "light_mode": (f"把{name}切换到{slots.get('mode', '')}模式。", f"{name}进入{slots.get('mode', '')}模式。"),
        "light_color": (f"把{name}调成{slots.get('color', '')}。", f"{name}换成{slots.get('color', '')}。"),
        "ac_on": (f"把{name}打开。", f"{name}开一下。"),
        "ac_off": (f"关掉{name}。", f"不用{name}了，帮我关闭。"),
        "ac_mode": (f"把{name}切换到{slots.get('mode', '')}模式。", f"{name}调成{slots.get('mode', '')}模式。"),
        "ac_temp_up": (f"把{name}温度调高一度。", f"{name}温度升高一度。"),
        "ac_temp_down": (f"把{name}温度调低一度。", f"{name}温度降低一度。"),
        "ac_temp_set": (f"把{name}温度调到{slots.get('temperature', '')}。", f"{name}设成{slots.get('temperature', '')}。"),
        "ac_wind_direction": (f"把{name}风向调到{slots.get('direction', '')}。", f"{name}改成{slots.get('direction', '')}风。"),
        "ac_fan_up": (f"把{name}风速调高一档。", f"{name}风力加大一点。"),
        "ac_fan_down": (f"把{name}风速调低一档。", f"{name}风力小一点。"),
        "ac_timer": (
            f"请在{slots.get('time', '')}{slots.get('timer_action', '')}{name}。",
            f"帮我设置{name}{slots.get('time', '')}{slots.get('timer_action', '')}。",
        ),
        "tv_on": (f"把{name}打开。", f"{name}开一下。"),
        "tv_off": (f"关掉{name}。", f"不看{name}了，帮我关闭。"),
        "tv_subtitle": (
            f"{'打开' if slots.get('state') == 'on' else '关闭'}{name}字幕。",
            f"{name}字幕{'显示出来' if slots.get('state') == 'on' else '关掉'}。",
        ),
        "tv_mute": (
            f"{'开启' if slots.get('state') == 'on' else '取消'}{name}静音。",
            f"{name}{'先别出声' if slots.get('state') == 'on' else '恢复声音'}。",
        ),
        "tv_volume_up": (f"把{name}音量调高。", f"{name}声音大一点。"),
        "tv_volume_down": (f"把{name}音量调低。", f"{name}声音小一点。"),
        "tv_play": (f"继续播放{name}。", f"{name}接着播放。"),
        "tv_pause": (f"暂停{name}播放。", f"{name}先停一下。"),
        "tv_next_channel": (f"把{name}切到下一个频道。", f"{name}换下一台。"),
        "tv_previous_channel": (f"把{name}切到上一个频道。", f"{name}回到上一台。"),
        "tv_channel": (f"把{name}频道设为{slots.get('channel', '')}。", f"{name}切到{slots.get('channel', '')}频道。"),
        "tv_input": (f"把{name}输入源切换到{slots.get('input', '')}。", f"{name}切到{slots.get('input', '')}。"),
        "tv_picture_mode": (
            f"把{name}画面模式设为{slots.get('mode', '')}模式。",
            f"{name}切到{slots.get('mode', '')}画面模式。",
        ),
        "speaker_on": (f"把{name}打开。", f"{name}开一下。"),
        "speaker_off": (f"关掉{name}。", f"不用{name}了，关闭。"),
        "speaker_play": (f"播放{name}里的音乐。", f"{name}开始播放。"),
        "speaker_pause": (f"暂停{name}播放。", f"{name}先停一下。"),
        "speaker_next": (f"{name}切到下一首。", f"下一首，{name}。"),
        "speaker_previous": (f"{name}切到上一首。", f"上一首，{name}。"),
        "speaker_volume_up": (f"把{name}音量调高。", f"{name}声音大一点。"),
        "speaker_volume_down": (f"把{name}音量调低。", f"{name}声音小一点。"),
    }
    options = templates.get(kind, (f"请执行{name}的{kind}。",))
    return options[variant % len(options)]


def synthetic_positive(anchor: str, intents: list[dict[str, Any]], response: str) -> tuple[str, str]:
    try:
        known_canonical = canonical_for_intents(intents)
    except Exception:
        known_canonical = ""
    for variant in range(4):
        pieces = [format_single_positive(intent, variant) for intent in intents]
        prompt = "，再".join(piece.rstrip("。") for piece in pieces) + "。"
        if compact(prompt) != compact(anchor):
            # These templates are generated from the already-normalized intent object.
            # If that intent exactly recreates the anchor response, the positive is valid
            # even when the lightweight parser cannot re-read a synthetic multi-action sentence.
            if known_canonical == response:
                return prompt, "synthetic_singleton"
            inferred = infer_intents_from_pair(prompt, response)
            if inferred:
                try:
                    if canonical_for_intents(inferred) == response:
                        return prompt, "synthetic_singleton"
                except Exception:
                    pass
            return prompt, "synthetic_singleton_needs_review"
    return f"请帮我处理：{anchor}", "synthetic_singleton_needs_review"


def choose_positive(
    index: int,
    row: dict[str, str],
    rows: list[dict[str, str]],
    response_groups: dict[str, list[int]],
    intents: list[dict[str, Any]],
) -> tuple[str, str]:
    candidates = [
        rows[candidate_index]["prompt"]
        for candidate_index in response_groups[row["response"]]
        if candidate_index != index and compact(rows[candidate_index]["prompt"]) != compact(row["prompt"])
    ]
    if candidates:
        prompt = max(candidates, key=lambda item: (char_jaccard_distance(row["prompt"], item), item))
        return prompt, "same_response_diverse"
    return synthetic_positive(row["prompt"], intents, row["response"])


def valid_phrase(intent: dict[str, Any]) -> str:
    return format_single_positive(intent, 0).rstrip("。")


def unsupported_device_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    location = intent["location"]
    verb = "关闭" if intent["intent"].endswith("_off") else "打开"
    return (
        f"{verb}{location}投影仪。",
        "unsupported_device",
        "投影仪不在当前支持的智能家居设备表中，但命令保留了原始开关动作和位置语境。",
    )


def ambiguous_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    kind = intent["intent"]
    if "off" in kind or "pause" in kind:
        prompt = "先关掉一下。"
    elif "up" in kind or "down" in kind or "set" in kind or "mode" in kind:
        prompt = "帮我调一下。"
    else:
        prompt = "打开一下。"
    return prompt, "ambiguous_action_target", "动作词存在，但缺少明确设备目标，不能解析为可执行控制。"


def invalid_location_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    device = intent["device"]
    action = "关闭" if intent["intent"].endswith("_off") else "打开"
    return (
        f"{action}阳台{device}。",
        "invalid_location_device_pair",
        "阳台不在当前支持的位置-设备组合中，因此该命令不应映射到原响应。",
    )


def non_control_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    name = target_name(intent["location"], intent["device"])
    return (
        f"{name}是谁发明的？",
        "non_control_intent",
        "句子仍在智能家居领域内，但它是信息查询，不是设备控制请求。",
    )


def unsupported_value_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    name = target_name(intent["location"], intent["device"])
    device = intent["device"]
    if device == "空调":
        return f"把{name}温度调到80度。", "unsupported_state_or_value", "80度超出空调温度控制的合理范围。"
    if device == "电视":
        return f"把{name}输入源切换到HDMI9。", "unsupported_state_or_value", "HDMI9不在支持的电视输入源列表中。"
    if device == "灯":
        return f"把{name}调成紫外线模式。", "unsupported_state_or_value", "紫外线模式不在支持的灯光模式或颜色列表中。"
    if device == "音箱":
        return f"把{name}音量调到200。", "unsupported_state_or_value", "音量200超出支持的音箱控制范围。"
    return non_control_negative(intent)


def unsupported_action_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    name = target_name(intent["location"], intent["device"])
    device = intent["device"]
    if device == "灯":
        return f"把{name}温度调到26度。", "unsupported_action_for_device", "灯支持开关、亮度、模式和颜色，但不支持温度设置。"
    if device == "空调":
        return f"把{name}亮度调高。", "unsupported_action_for_device", "空调支持温度、风速、风向和模式，但不支持亮度调节。"
    if device == "电视":
        return f"把{name}风向调到上下。", "unsupported_action_for_device", "电视支持频道、输入源、字幕、静音等控制，但不支持风向调节。"
    if device == "音箱":
        return f"把{name}字幕打开。", "unsupported_action_for_device", "音箱支持播放和音量控制，但不支持字幕控制。"
    return non_control_negative(intent)


def cross_device_negative(intent: dict[str, Any]) -> tuple[str, str, str]:
    name = target_name(intent["location"], intent["device"])
    device = intent["device"]
    if device == "灯":
        return f"把{name}切到HDMI1。", "cross_device_action_mismatch", "HDMI输入源是电视动作，不适用于灯。"
    if device == "空调":
        return f"把{name}切到下一首。", "cross_device_action_mismatch", "下一首是音箱播放动作，不适用于空调。"
    if device == "电视":
        return f"把{name}调成制冷模式。", "cross_device_action_mismatch", "制冷模式是空调动作，不适用于电视。"
    if device == "音箱":
        return f"把{name}风向调到左右。", "cross_device_action_mismatch", "风向是空调动作，不适用于音箱。"
    return non_control_negative(intent)


def invalid_multi_negative(intents: list[dict[str, Any]]) -> tuple[str, str, str]:
    first = intents[0]
    valid = valid_phrase(first)
    invalid, _, _ = unsupported_action_negative(first)
    return (
        f"{valid}，再{invalid.rstrip('。')}。",
        "invalid_multi_device_combination",
        "多动作命令中第一段有效，但第二段包含该设备不支持的动作，因此整体应视为无效负例。",
    )


def candidate_negatives(intents: list[dict[str, Any]], index: int) -> list[tuple[str, str, str]]:
    primary = intents[0]
    candidates = [
        unsupported_action_negative(primary),
        unsupported_value_negative(primary),
        cross_device_negative(primary),
        invalid_multi_negative(intents),
        invalid_location_negative(primary),
        unsupported_device_negative(primary),
        ambiguous_negative(primary),
        non_control_negative(primary),
    ]
    # Rotate candidates with the same priority by index while keeping hard invalids first.
    return sorted(
        candidates,
        key=lambda item: (NEGATIVE_TYPE_ORDER.get(item[1], 99), (index + len(item[0])) % 3, item[0]),
    )


def is_safe_negative(
    prompt: str,
    anchor: str,
    positive: str,
    response: str,
    all_prompt_norms: set[str],
) -> tuple[bool, str]:
    prompt_key = compact(prompt)
    if prompt_key in {compact(anchor), compact(positive)}:
        return False, "negative_duplicates_anchor_or_positive"
    if prompt_key in all_prompt_norms:
        return False, "negative_exists_as_valid_prompt"
    inferred = infer_intents_from_pair(prompt, response)
    if inferred:
        try:
            if canonical_for_intents(inferred) == response:
                return False, "negative_can_map_to_anchor_response"
        except Exception:
            return False, "negative_inference_error"
    return True, ""


def choose_negative(
    index: int,
    intents: list[dict[str, Any]],
    anchor: str,
    positive: str,
    response: str,
    all_prompt_norms: set[str],
) -> tuple[str, str, str, str]:
    rejected: list[str] = []
    for prompt, negative_type, explanation in candidate_negatives(intents, index):
        ok, reason = is_safe_negative(prompt, anchor, positive, response, all_prompt_norms)
        if ok:
            return prompt, negative_type, explanation, ";".join(rejected)
        rejected.append(f"{negative_type}:{reason}")
    fallback = non_control_negative(intents[0])
    return fallback[0], fallback[1], fallback[2], ";".join(rejected + ["fallback_used"])


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if compact(record["anchor"]) == compact(record["positive"]):
        errors.append("positive_matches_anchor")
    if compact(record["negative"]) in {compact(record["anchor"]), compact(record["positive"])}:
        errors.append("negative_matches_valid_text")
    if record["negative_type"] not in ALLOWED_NEGATIVE_TYPES:
        errors.append("invalid_negative_type")
    if record["is_valid_negative"] is not False:
        errors.append("negative_validity_flag_wrong")
    if not record["response"]:
        errors.append("empty_response")
    return errors


def build_dataset(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    response_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        response_groups[row["response"]].append(index)

    all_prompt_norms = {compact(row["prompt"]) for row in rows}
    records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for index, row in enumerate(rows):
        stats["total_input"] += 1
        intents = infer_intents_from_pair(row["prompt"], row["response"])
        needs_manual_review = False
        validation_errors: list[str] = []
        if not intents:
            needs_manual_review = True
            validation_errors.append("unresolved_anchor_intent")
            intents = [{"intent": "unknown", "device": "设备", "location": "", "slots": {}}]
        else:
            try:
                canonical = canonical_for_intents(intents)
                if canonical != row["response"]:
                    needs_manual_review = True
                    validation_errors.append("canonical_mismatch")
            except Exception:
                needs_manual_review = True
                validation_errors.append("canonical_generation_failed")

        positive, positive_source = choose_positive(index, row, rows, response_groups, intents)
        negative, negative_type, explanation, rejected = choose_negative(
            index,
            intents,
            row["prompt"],
            positive,
            row["response"],
            all_prompt_norms,
        )

        record = {
            "source_id": f"sample_{index + 1:06d}",
            "anchor": row["prompt"],
            "positive": positive,
            "negative": negative,
            "response": row["response"],
            "intent_key": intent_key(intents) if intents[0]["intent"] != "unknown" else "unknown",
            "positive_source": positive_source,
            "negative_type": negative_type,
            "negative_explanation": explanation,
            "is_valid_anchor": True,
            "is_valid_positive": True,
            "is_valid_negative": False,
            "needs_manual_review": needs_manual_review,
        }
        validation_errors.extend(validate_record(record))
        record["validation_errors"] = validation_errors

        records.append(record)
        audit_rows.append(
            {
                "source_id": record["source_id"],
                "anchor": record["anchor"],
                "positive": record["positive"],
                "negative": record["negative"],
                "response": record["response"],
                "intent_key": record["intent_key"],
                "positive_source": positive_source,
                "negative_type": negative_type,
                "negative_explanation": explanation,
                "needs_manual_review": needs_manual_review,
                "validation_errors": "|".join(validation_errors),
                "rejected_negative_candidates": rejected,
            }
        )
        stats["generated_records"] += 1
        stats[f"positive_source:{positive_source}"] += 1
        stats[f"negative_type:{negative_type}"] += 1
        stats[f"device:{intents[0]['device']}"] += 1
        if needs_manual_review:
            stats["needs_manual_review"] += 1
        if validation_errors:
            stats["validation_error_records"] += 1
            for error in validation_errors:
                stats[f"validation_error:{error}"] += 1

    return records, audit_rows, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_audit_csv(path: Path, audit_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "anchor",
        "positive",
        "negative",
        "response",
        "intent_key",
        "positive_source",
        "negative_type",
        "negative_explanation",
        "needs_manual_review",
        "validation_errors",
        "rejected_negative_candidates",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)


def top_counter_lines(counter: Counter[str], prefix: str, limit: int = 50) -> list[str]:
    values = Counter({key.removeprefix(prefix): count for key, count in counter.items() if key.startswith(prefix)})
    return [f"- `{key}`: {count}" for key, count in values.most_common(limit)]


def write_summary(path: Path, input_path: Path, output_path: Path, audit_path: Path, records: list[dict[str, Any]], stats: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    negative_lines = top_counter_lines(stats, "negative_type:")
    positive_lines = top_counter_lines(stats, "positive_source:")
    device_lines = top_counter_lines(stats, "device:")
    examples = "\n".join(
        json.dumps(record, ensure_ascii=False)
        for record in records[:20]
    )
    text = "\n".join(
        [
            "# SCENIC Positive/Negative Dataset Summary",
            "",
            f"- Input: `{input_path}`",
            f"- Output JSONL: `{output_path}`",
            f"- Audit CSV: `{audit_path}`",
            f"- Total input samples: {stats['total_input']}",
            f"- Generated records: {stats['generated_records']}",
            f"- Needs manual review: {stats['needs_manual_review']}",
            f"- Validation error records: {stats['validation_error_records']}",
            "",
            "## Positive Source Distribution",
            *positive_lines,
            "",
            "## Negative Type Distribution",
            *negative_lines,
            "",
            "## Anchor Device Distribution",
            *device_lines,
            "",
            "## First 20 Records",
            "```jsonl",
            examples,
            "```",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    audit_path = Path(args.audit).expanduser()
    summary_path = Path(args.summary).expanduser()

    rows = load_rows(input_path)
    records, audit_rows, stats = build_dataset(rows)
    write_jsonl(output_path, records)
    write_audit_csv(audit_path, audit_rows)
    write_summary(summary_path, input_path, output_path, audit_path, records, stats)

    print(f"input_samples: {stats['total_input']}")
    print(f"generated_records: {stats['generated_records']}")
    print(f"needs_manual_review: {stats['needs_manual_review']}")
    print(f"validation_error_records: {stats['validation_error_records']}")
    print("positive_sources:")
    for line in top_counter_lines(stats, "positive_source:"):
        print(line)
    print("negative_types:")
    for line in top_counter_lines(stats, "negative_type:"):
        print(line)
    print(f"output: {output_path}")
    print(f"audit: {audit_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
