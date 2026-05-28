#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Editable schema/config
# ---------------------------------------------------------------------------
# Negatives are invalid under this smart-home schema. This avoids the weak
# contrastive pattern of pairing a valid prompt with a different valid response.

SUPPORTED_DEVICES = {"灯", "空调", "电视", "音箱", "抽湿机"}
SUPPORTED_ROOMS = {"客厅", "卧室", "主卧", "主卧室", "次卧", "厨房", "书房", "阳台", "餐厅", "浴室", "洗手间", "房间", "未指定"}

DEVICE_ROOM_AVAILABILITY = {
    "灯": {"客厅", "卧室", "主卧", "主卧室", "次卧", "厨房", "书房", "阳台", "餐厅", "浴室", "洗手间", "房间", "未指定"},
    "空调": {"客厅", "卧室", "主卧", "主卧室", "次卧", "书房", "未指定"},
    "电视": {"客厅", "卧室", "主卧", "主卧室", "次卧", "未指定"},
    "音箱": {"客厅", "卧室", "主卧", "主卧室", "次卧", "厨房", "书房", "餐厅", "未指定"},
    "抽湿机": {"浴室", "洗手间", "卧室", "未指定"},
}

ALLOWED_ACTIONS = {
    "灯": {"打开", "关闭", "设置亮度", "调高", "调低", "设置模式"},
    "空调": {"打开", "关闭", "设置温度", "调高", "调低", "设置风速", "设置风向", "设置模式", "设置定时"},
    "电视": {"打开", "关闭", "设置电源", "调整音量", "切换频道", "设置频道", "设置输入源", "播放控制", "设置画面模式", "设置字幕", "设置静音"},
    "音箱": {"打开", "关闭", "播放", "暂停", "下一首", "调高", "调低"},
    "抽湿机": {"打开", "关闭"},
}

VALUE_RANGES = {
    ("空调", "设置温度"): (16, 30),
    ("灯", "设置亮度"): (0, 100),
    ("电视", "调整音量"): (0, 100),
    ("音箱", "调高"): (0, 100),
    ("音箱", "调低"): (0, 100),
}

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

PROMPT_FIELDS = ("prompt", "instruction", "input", "query", "user_command")
RESPONSE_FIELDS = ("response", "output", "answer", "target")
SOURCE_ID_FIELDS = ("source_id", "id", "uid", "sample_id")
FIELD_ORDER = ("device", "location", "action", "attribute", "value")
CHINESE_FIELDS = {
    "device": "设备",
    "location": "位置",
    "action": "动作",
    "attribute": "属性",
    "value": "值",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SCENIC invalid-hard-negative contrastive JSONL data.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL dataset.")
    parser.add_argument("--output", required=True, help="Output contrastive JSONL path.")
    parser.add_argument("--audit", required=True, help="CSV audit output path.")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    parser.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True, help="Dedupe by anchor and response.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def normalize_text(value: str) -> str:
    return "".join(clean_text(value).split())


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(item)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    raise ValueError(f"{path} must contain a JSON object, JSON list, or JSONL objects")


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = clean_text(record.get(field))
        if value:
            return value
    return ""


def source_id(record: dict[str, Any], index: int) -> str:
    explicit = first_text(record, SOURCE_ID_FIELDS)
    if explicit:
        return explicit
    metadata = record.get("scenic_metadata")
    if isinstance(metadata, dict) and metadata.get("source_index") is not None:
        return f"source_{metadata['source_index']}"
    return f"sample_{index + 1:06d}"


def parse_structured_nlp(response: str) -> list[dict[str, str]]:
    if "设备=" not in response or "动作=" not in response:
        return []
    ops: list[dict[str, str]] = []
    for segment in response.split("||"):
        values: dict[str, str] = {}
        for part in segment.split(";"):
            if "=" not in part:
                continue
            key, value = [piece.strip() for piece in part.split("=", 1)]
            for internal, chinese in CHINESE_FIELDS.items():
                if key == chinese:
                    values[internal] = value
        if values.get("device") and values.get("action"):
            ops.append({field: values.get(field, "未知") for field in FIELD_ORDER})
    return ops


def parse_structured_response(record: dict[str, Any], response: str) -> list[dict[str, str]]:
    raw = record.get("structured_response")
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    ops: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if all(field in item for field in FIELD_ORDER):
            ops.append({field: clean_text(item.get(field)) or "未知" for field in FIELD_ORDER})
    return ops or parse_structured_nlp(response)


def normalize_location(location: str) -> str:
    location = clean_text(location)
    if location == "主卧室":
        return "主卧"
    if not location or location in {"未知", "none", "unspecified"}:
        return "未指定"
    return location


def target_name(op: dict[str, str]) -> str:
    device = clean_text(op.get("device")) or "设备"
    location = normalize_location(op.get("location", "未指定"))
    return device if location == "未指定" else f"{location}{device}"


def first_op(ops: list[dict[str, str]]) -> dict[str, str]:
    return ops[0] if ops else {"device": "设备", "location": "未指定", "action": "控制", "attribute": "功能", "value": "未知"}


def value_number(value: str) -> int | None:
    match = re.search(r"\d+", clean_text(value))
    return int(match.group(0)) if match else None


def display_value(value: str) -> str:
    value = clean_text(value)
    value = value.replace("关闭闭", "关闭").replace("开启启", "开启")
    value = value.replace("舒睡模式一模式", "舒睡模式一").replace("舒睡模式二模式", "舒睡模式二")
    value = re.sub(r"模式模式$", "模式", value)
    return value


def is_supported_device(device: str) -> bool:
    return clean_text(device) in SUPPORTED_DEVICES


def is_supported_room(room: str) -> bool:
    return normalize_location(room) in SUPPORTED_ROOMS


def is_valid_device_room(device: str, room: str) -> bool:
    device = clean_text(device)
    room = normalize_location(room)
    return room in DEVICE_ROOM_AVAILABILITY.get(device, set())


def is_allowed_action(device: str, action: str) -> bool:
    return clean_text(action) in ALLOWED_ACTIONS.get(clean_text(device), set())


def is_value_in_range(op: dict[str, str]) -> bool:
    key = (clean_text(op.get("device")), clean_text(op.get("action")))
    if key not in VALUE_RANGES:
        return True
    number = value_number(clean_text(op.get("value")))
    if number is None:
        return True
    low, high = VALUE_RANGES[key]
    return low <= number <= high


def schema_valid_ops(ops: list[dict[str, str]]) -> bool:
    if not ops:
        return False
    for op in ops:
        device = clean_text(op.get("device"))
        location = clean_text(op.get("location"))
        action = clean_text(op.get("action"))
        if not is_supported_device(device):
            return False
        if not is_supported_room(location):
            return False
        if not is_valid_device_room(device, location):
            return False
        if not is_allowed_action(device, action):
            return False
        if not is_value_in_range(op):
            return False
    return True


def positive_for_op(op: dict[str, str]) -> str:
    device = clean_text(op.get("device"))
    action = clean_text(op.get("action"))
    attribute = clean_text(op.get("attribute"))
    value = display_value(op.get("value", ""))
    target = target_name(op)

    if action in {"打开", "设置电源"} and value in {"开启", "打开", "on"}:
        return f"把{target}打开"
    if action == "关闭" and attribute == "模式" and value not in {"", "未知", "关闭", "关", "off"}:
        return f"关闭{target}{value}"
    if action in {"关闭", "设置电源"} and value in {"关闭", "关", "off"}:
        return f"帮我关掉{target}"
    if action == "打开":
        return f"把{target}打开"
    if action == "关闭":
        return f"帮我关掉{target}"
    if action == "设置温度":
        return f"把{target}温度设为{value}"
    if action == "设置风速":
        return f"把{target}风速调到{value}"
    if action == "设置风向":
        return f"把{target}风向调到{value}"
    if action == "设置亮度":
        return f"把{target}亮度调到{value}"
    if action == "设置模式":
        return f"把{target}切换到{value}"
    if action == "设置定时":
        return f"给{target}设置{value}"
    if action == "调高":
        return f"把{target}{attribute}调高一点"
    if action == "调低":
        return f"把{target}{attribute}调低一点"
    if device == "电视":
        if action == "调整音量":
            return f"把{target}音量{value}"
        if action == "切换频道":
            return f"把{target}切到{value}频道"
        if action == "设置频道":
            return f"把{target}频道设为{value}"
        if action == "设置输入源":
            return f"把{target}输入源切到{value}"
        if action == "播放控制":
            return f"让{target}{value}"
        if action == "设置画面模式":
            return f"把{target}画面模式设为{value}"
        if action == "设置字幕":
            return f"{'打开' if value == '开启' else '关闭'}{target}字幕"
        if action == "设置静音":
            return f"{'开启' if value == '开启' else '关闭'}{target}静音"
    if device == "音箱":
        if action == "播放":
            return f"让{target}播放音乐"
        if action == "暂停":
            return f"暂停{target}播放"
        if action == "下一首":
            return f"{target}切到下一首"
    return f"请执行{target}{attribute}{action}{value}"


def make_positive(anchor: str, ops: list[dict[str, str]]) -> str:
    pieces = [positive_for_op(op) for op in ops]
    positive = "，然后".join(pieces)
    if normalize_text(positive) == normalize_text(anchor):
        positive = f"请{positive}"
    if normalize_text(positive) == normalize_text(anchor):
        positive = f"麻烦{positive}"
    return positive


def action_word(op: dict[str, str]) -> str:
    action = clean_text(op.get("action"))
    value = display_value(op.get("value", ""))
    if action in {"打开", "设置电源"} and value == "开启":
        return "打开"
    if action in {"关闭", "设置电源"} and value == "关闭":
        return "关闭"
    if action == "调高":
        return "调高"
    if action == "调低":
        return "调低"
    if action.startswith("设置"):
        return "设置"
    return action or "控制"


def unsupported_device_negative(op: dict[str, str]) -> tuple[str, str]:
    location = normalize_location(op.get("location", "未指定"))
    prefix = "" if location == "未指定" else location
    verb = action_word(op)
    if verb == "关闭":
        command = f"关闭{prefix}投影仪"
    elif verb in {"调高", "调低"}:
        command = f"把{prefix}投影仪{clean_text(op.get('attribute'))}{verb}"
    else:
        command = f"{verb}{prefix}投影仪"
    return command, "投影仪不在当前支持的设备列表中；负例保留了房间或动作词，属于设备不支持的 hard negative。"


def unsupported_action_negative(op: dict[str, str]) -> tuple[str, str] | None:
    device = clean_text(op.get("device"))
    target = target_name(op)
    templates = {
        "空调": (f"把{target}的亮度调高", "空调支持开关、温度、风速、风向、模式和定时等控制，但不支持亮度调节。"),
        "灯": (f"把{target}温度调到26度", "灯支持开关、亮度、模式和色温控制，但不支持温度设置。"),
        "电视": (f"把{target}风向调到上下", "电视支持电源、音量、频道、输入源、播放、画面、字幕和静音控制，但不支持风向控制。"),
        "音箱": (f"把{target}风向调到上下", "音箱支持播放、电源和音量控制，但不支持风向控制。"),
        "抽湿机": (f"把{target}频道调到CCTV-1", "抽湿机支持开关等除湿控制，但不支持电视频道设置。"),
    }
    return templates.get(device)


def ambiguous_target_negative(op: dict[str, str]) -> tuple[str, str]:
    verb = action_word(op)
    if verb in {"调高", "调低"}:
        return f"{verb}一点", f"“{verb}一点”缺少设备目标，多个设备可能涉及调节，无法生成确定控制指令。"
    if verb == "设置":
        return "帮我设置一下", "设置动作缺少设备、属性和值，无法确定应该控制哪个智能家居设备。"
    return f"{verb}一下", f"“{verb}一下”只有动作词，没有明确设备目标，因此属于歧义控制命令。"


def cross_device_negative(op: dict[str, str]) -> tuple[str, str] | None:
    action = clean_text(op.get("action"))
    attribute = clean_text(op.get("attribute"))
    value = display_value(op.get("value", ""))
    location = normalize_location(op.get("location", "未指定"))
    loc = "" if location == "未指定" else location

    if attribute in {"亮度", "色温"} or clean_text(op.get("device")) == "灯":
        return f"把{loc}空调{attribute or '亮度'}调高", "该命令把灯光类能力套到了空调上，设备存在但动作能力不匹配。"
    if attribute in {"温度", "风速", "风向"} or clean_text(op.get("device")) == "空调":
        return f"把{loc}电视{attribute or '温度'}调到{value if value and value != '未知' else '26度'}", "该命令把空调类能力套到了电视上，属于跨设备动作不匹配。"
    if clean_text(op.get("device")) == "电视":
        return f"把{loc}灯{attribute or '频道'}设为{value if value and value != '未知' else 'CCTV-1'}", "该命令把电视类能力套到了灯上，设备和动作范围不兼容。"
    if action in {"播放", "暂停", "下一首"} or clean_text(op.get("device")) == "音箱":
        return f"让{loc}空调{action or '播放'}音乐", "该命令把音箱播放类能力套到了空调上，属于跨设备动作不匹配。"
    return None


def invalid_location_negative(op: dict[str, str]) -> tuple[str, str] | None:
    device = clean_text(op.get("device"))
    bad_room = {
        "空调": "阳台",
        "电视": "浴室",
        "音箱": "浴室",
        "抽湿机": "客厅",
    }.get(device)
    if not bad_room or is_valid_device_room(device, bad_room):
        return None
    verb = action_word(op)
    command = f"{verb}{bad_room}{device}" if verb in {"打开", "关闭"} else f"把{bad_room}{device}{clean_text(op.get('attribute'))}{verb}"
    return command, f"{device}是支持设备，但配置中没有{bad_room}{device}，因此房间和设备组合无效。"


def invalid_multi_device_negative(op: dict[str, str]) -> tuple[str, str]:
    valid_part = positive_for_op(op)
    device = clean_text(op.get("device"))
    if device == "空调":
        invalid_part = f"把{target_name(op)}亮度调高"
        reason = "第一条子命令有效，但第二条要求空调调亮度，空调不支持亮度控制。"
    else:
        invalid_part = "把空调亮度调高"
        reason = "第一条子命令有效，但第二条要求空调调亮度，空调不支持亮度控制。"
    return f"{valid_part}，再{invalid_part}", reason


def unsupported_value_negative(op: dict[str, str]) -> tuple[str, str] | None:
    device = clean_text(op.get("device"))
    action = clean_text(op.get("action"))
    target = target_name(op)
    if device == "空调" and action in {"设置温度", "调高", "调低"}:
        return f"把{target}调到80度", "空调温度允许范围是16到30度，80度超出支持范围。"
    if device == "灯" and clean_text(op.get("attribute")) == "亮度":
        return f"把{target}亮度调到200%", "灯光亮度允许范围是0到100%，200%超出支持范围。"
    if device in {"电视", "音箱"} and clean_text(op.get("attribute")) == "音量":
        return f"把{target}音量调到200%", "音量允许范围是0到100%，200%超出支持范围。"
    return None


def non_control_negative(op: dict[str, str]) -> tuple[str, str]:
    target = target_name(op)
    return f"{target}是谁发明的", "该句属于智能家居相关的信息询问，不是设备控制意图。"


NEGATIVE_BUILDERS = {
    "unsupported_device": unsupported_device_negative,
    "unsupported_action_for_device": unsupported_action_negative,
    "ambiguous_action_target": ambiguous_target_negative,
    "cross_device_action_mismatch": cross_device_negative,
    "invalid_location_device_pair": invalid_location_negative,
    "invalid_multi_device_combination": invalid_multi_device_negative,
    "unsupported_state_or_value": unsupported_value_negative,
    "non_control_intent": non_control_negative,
}

NEGATIVE_ORDER = (
    "unsupported_action_for_device",
    "unsupported_state_or_value",
    "cross_device_action_mismatch",
    "invalid_multi_device_combination",
    "invalid_location_device_pair",
    "ambiguous_action_target",
    "unsupported_device",
    "non_control_intent",
)


def build_negative(anchor: str, positive: str, ops: list[dict[str, str]], index: int) -> tuple[str, str, str]:
    op = first_op(ops)
    rotation = index % len(NEGATIVE_ORDER)
    ordered_types = NEGATIVE_ORDER[rotation:] + NEGATIVE_ORDER[:rotation]
    for negative_type in ordered_types:
        candidate = NEGATIVE_BUILDERS[negative_type](op)
        if not candidate:
            continue
        negative, explanation = candidate
        if normalize_text(negative) in {normalize_text(anchor), normalize_text(positive)}:
            continue
        return negative, negative_type, explanation
    negative, explanation = non_control_negative(op)
    return negative, "non_control_intent", explanation


def negative_might_match_response(negative: str, ops: list[dict[str, str]]) -> bool:
    # Conservative guard against accidentally generating a paraphrase of y_i.
    key = normalize_text(negative)
    for op in ops:
        device = clean_text(op.get("device"))
        location = normalize_location(op.get("location", "未指定"))
        action = clean_text(op.get("action"))
        value = clean_text(op.get("value"))
        mentions_device = device and device in key
        mentions_location = location == "未指定" or location in key
        if mentions_device and mentions_location:
            if action in {"打开", "设置电源"} and value == "开启" and any(word in key for word in ("打开", "开启", "开一下", "开机")):
                return True
            if action in {"关闭", "设置电源"} and value == "关闭" and any(word in key for word in ("关闭", "关掉", "关一下", "关机")):
                return True
            if value and value != "未知" and value in key and clean_text(op.get("attribute")) in key:
                return True
    return False


def validate_record(record: dict[str, Any], original_response: str) -> list[str]:
    errors: list[str] = []
    if normalize_text(record["anchor"]) == normalize_text(record["positive"]):
        errors.append("anchor_positive_identical")
    if normalize_text(record["negative"]) == normalize_text(record["anchor"]):
        errors.append("negative_anchor_identical")
    if normalize_text(record["negative"]) == normalize_text(record["positive"]):
        errors.append("negative_positive_identical")
    if record["negative_type"] not in ALLOWED_NEGATIVE_TYPES:
        errors.append("invalid_negative_type")
    if record["response"] != original_response:
        errors.append("response_changed")
    if record["is_valid_negative"] is not False:
        errors.append("negative_not_marked_invalid")
    return errors


def build_records(rows: list[dict[str, Any]], dedupe: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    stats: Counter[str] = Counter()
    validation_errors: Counter[str] = Counter()

    for index, row in enumerate(rows):
        anchor = first_text(row, PROMPT_FIELDS)
        response = first_text(row, RESPONSE_FIELDS)
        sid = source_id(row, index)
        ops = parse_structured_response(row, response)
        needs_manual_review = not schema_valid_ops(ops)

        if not anchor or not response:
            stats["skipped_missing_prompt_or_response"] += 1
            continue

        dedupe_key = (normalize_text(anchor), normalize_text(response))
        if dedupe and dedupe_key in seen:
            stats["deduped_records"] += 1
            continue
        seen.add(dedupe_key)

        positive = make_positive(anchor, ops)
        negative, negative_type, explanation = build_negative(anchor, positive, ops, index)
        if negative_might_match_response(negative, ops):
            # If a rotated template is too close to y_i, use a domain-related
            # non-control invalid utterance rather than a wrong valid response.
            negative, explanation = non_control_negative(first_op(ops))
            negative_type = "non_control_intent"
        if not ops:
            needs_manual_review = True
            negative, negative_type, explanation = (
                f"{anchor}需要多少钱",
                "non_control_intent",
                "无法可靠解析源样本的设备或动作；输出保留但需要人工复核。",
            )

        record = {
            "source_id": sid,
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
            "response": response,
            "negative_type": negative_type,
            "negative_explanation": explanation,
            "is_valid_anchor": True,
            "is_valid_positive": True,
            "is_valid_negative": False,
            "needs_manual_review": needs_manual_review,
        }
        errors = validate_record(record, response)
        for error in errors:
            validation_errors[error] += 1
        if errors:
            record["validation_errors"] = errors
            record["needs_manual_review"] = True

        output.append(record)
        stats[f"negative_type:{negative_type}"] += 1
        stats["manual_review"] += int(bool(record["needs_manual_review"]))
        audit_rows.append(
            {
                "source_id": sid,
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "response": response,
                "negative_type": negative_type,
                "negative_explanation": explanation,
                "needs_manual_review": record["needs_manual_review"],
            }
        )

    report = {
        "total_input_samples": len(rows),
        "total_generated_records": len(output),
        "deduped_records": stats.get("deduped_records", 0),
        "skipped_missing_prompt_or_response": stats.get("skipped_missing_prompt_or_response", 0),
        "count_by_negative_type": {
            key.replace("negative_type:", ""): value
            for key, value in sorted(stats.items())
            if key.startswith("negative_type:")
        },
        "manual_review_required": stats.get("manual_review", 0),
        "validation_errors": dict(sorted(validation_errors.items())),
    }
    return output, audit_rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "source_id",
        "anchor",
        "positive",
        "negative",
        "response",
        "negative_type",
        "negative_explanation",
        "needs_manual_review",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    audit_path = Path(args.audit).expanduser()
    rows = read_records(input_path)
    output, audit_rows, report = build_records(rows, args.dedupe)
    write_jsonl(output_path, output)
    write_audit(audit_path, audit_rows)
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote contrastive dataset: {output_path}")
    print(f"wrote audit CSV: {audit_path}")
    if args.report:
        print(f"wrote report: {Path(args.report).expanduser()}")


if __name__ == "__main__":
    main()
