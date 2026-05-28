#!/usr/bin/env python3
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
FIELD_ORDER = ("device", "location", "action", "attribute", "value")
CHINESE_FIELDS = {
    "device": "设备",
    "location": "位置",
    "action": "动作",
    "attribute": "属性",
    "value": "值",
}

LOCATION_TERMS = (
    ("主卧室", ("主卧室", "主卧")),
    ("客厅", ("客厅",)),
    ("次卧", ("次卧",)),
    ("卧室", ("卧室", "房间")),
    ("书房", ("书房",)),
    ("厨房", ("厨房",)),
    ("阳台", ("阳台",)),
    ("餐厅", ("餐厅",)),
    ("浴室", ("浴室", "洗手间", "卫生间")),
)
LIGHT_MODES = ("学习模式", "夜灯模式", "睡眠模式", "舒睡模式", "普通模式", "自然光", "冷色光", "冷色灯", "暖色光", "暖色灯")
AC_MODES = ("舒睡模式二", "舒睡模式一", "舒睡二", "舒睡一", "制冷模式", "制热模式", "自动模式", "环保模式", "抽湿模式", "送风模式", "通风模式", "睡眠模式", "强劲模式", "静音模式", "制冷", "制热", "自动", "环保", "抽湿", "送风", "通风", "强劲", "静音")
NUMBER_RE = re.compile(r"(\d{1,3})\s*度")
TIMER_RE = re.compile(r"(\d+|半|一|二|三|四|五|六|七|八|九|十|三十|一点|二点).{0,4}(分钟|小时|点|时).{0,6}(开启|打开|关闭|关)")
PUNCT_RE = re.compile(r"[。！？!?,，、\s]+$")
FALSE_NEGATIVE_RESPONSE_SIMILARITY = 0.985
FALSE_NEGATIVE_PROMPT_SIMILARITY = 0.86
TOP_NEGATIVE_CANDIDATES = 25
HARD_NEGATIVE_POOL = 700
FALLBACK_NEGATIVE_SAMPLE = 200

DEVICE_CATEGORY_KEYWORDS = {
    "light": ("灯", "灯光", "台灯", "照明", "亮度", "光线"),
    "air_conditioner": ("空调", "温度", "风速", "风向", "制冷", "制热", "送风", "冷气"),
    "tv": ("电视",),
    "speaker": ("音箱", "音乐", "播放", "暂停", "歌曲", "音量", "声音"),
    "dehumidifier": ("抽湿机", "除湿机", "除湿", "湿度"),
}
ACTION_TYPE_KEYWORDS = {
    "turn_on": ("打开", "开启", "开一下", "启动", "开机", "开灯", "亮灯"),
    "turn_off": ("关闭", "关掉", "关一下", "停止", "关机", "关灯", "不用了", "别开"),
    "increase": ("调高", "提高", "增加", "升高", "亮一点", "大一点", "调大", "加大"),
    "decrease": ("调低", "降低", "减少", "降", "暗一点", "小一点", "调小", "减小"),
    "set_value": ("设置", "设为", "设成", "调到", "调成", "定到", "改成"),
    "play": ("播放", "放音乐", "放歌"),
    "pause": ("暂停", "停一下"),
    "switch_mode": ("切换", "换成", "模式", "制冷", "制热", "送风"),
}
ATTRIBUTE_TYPE_KEYWORDS = {
    "brightness": ("亮度", "亮", "暗", "光线"),
    "temperature": ("温度", "度", "冷", "热"),
    "fan_speed": ("风速", "风力", "风量"),
    "airflow_direction": ("风向", "风口"),
    "mode": ("模式", "制冷", "制热", "送风", "抽湿", "自动"),
    "volume": ("音量", "声音"),
    "humidity": ("湿度", "除湿", "抽湿"),
}
INDIRECT_COMMAND_CUES = (
    "有点",
    "太",
    "不够",
    "看不清",
    "不用",
    "不需要",
    "想看",
    "想听",
    "热",
    "冷",
    "闷",
    "安静",
    "吵",
    "刺眼",
    "舒服",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean SCENIC data into fixed-slot structured NLP responses.")
    parser.add_argument("--input", required=True, help="Raw or augmented JSON/JSONL input.")
    parser.add_argument("--output", required=True, help="Cleaned JSONL/JSON output.")
    parser.add_argument("--report", default=None, help="Quality report path.")
    parser.add_argument("--format", choices=("jsonl", "json"), default=None)
    parser.add_argument("--seed", type=int, default=619)
    parser.add_argument("--drop-uncertain", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--singleton-positive",
        choices=("synthetic", "self", "empty"),
        default="synthetic",
        help="How to fill positives for response classes with only one prompt.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").strip()


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, str | None]:
    for field in fields:
        value = clean_text(record.get(field))
        if value:
            return value, field
    return "", None


def normalize_key(text: str) -> str:
    return "".join(clean_text(text).split())


def char_jaccard_similarity(left: str, right: str) -> float:
    left_chars = set(normalize_key(left))
    right_chars = set(normalize_key(right))
    if not left_chars and not right_chars:
        return 1.0
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)


def char_jaccard_distance(left: str, right: str) -> float:
    return 1.0 - char_jaccard_similarity(left, right)


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                records.append(row)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    raise ValueError(f"{path} must contain JSON objects")


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


def location_from_text(text: str) -> str:
    for location, terms in LOCATION_TERMS:
        if any(term in text for term in terms):
            return location
    return "未指定"


def op(device: str, location: str, action: str, attribute: str, value: str) -> dict[str, str]:
    return {
        "device": device,
        "location": location or "未指定",
        "action": action,
        "attribute": attribute,
        "value": value,
    }


def format_ops(ops: list[dict[str, str]]) -> str:
    segments = []
    for item in ops:
        parts = [f"{CHINESE_FIELDS[key]}={item.get(key, '未知')}" for key in FIELD_ORDER]
        segments.append("; ".join(parts))
    return " || ".join(segments)


def normalize_mode(mode: str) -> str:
    return mode if mode.endswith("模式") or mode.endswith("光") or mode.endswith("灯") else f"{mode}模式"


def parse_structured_json_response(response: str) -> list[dict[str, str]]:
    try:
        value = json.loads(response)
    except Exception:
        return []
    items = value if isinstance(value, list) else [value]
    ops: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if all(field in item for field in FIELD_ORDER):
            normalized = {field: clean_text(item.get(field)) or "未知" for field in FIELD_ORDER}
            normalized["location"] = normalized["location"] or "未指定"
            ops.append(normalized)
            continue
        device = {"tv": "电视", "light": "灯", "air_conditioner": "空调"}.get(str(item.get("device", "")), str(item.get("device", "未知")))
        location = {
            "unspecified": "未指定",
            "living_room": "客厅",
            "bedroom": "卧室",
            "study": "书房",
        }.get(str(item.get("location", "unspecified")), str(item.get("location", "未指定")))
        action = str(item.get("action", "未知"))
        value_text = str(item.get("value", "未知"))
        attr = {
            "set_power": "电源",
            "adjust_volume": "音量",
            "set_mute": "静音",
            "change_channel": "频道",
            "set_channel": "频道",
            "set_source": "输入源",
            "playback": "播放",
            "set_picture_mode": "画面模式",
            "set_subtitles": "字幕",
        }.get(action, "控制")
        action_text = {
            "set_power": "设置电源",
            "adjust_volume": "调整音量",
            "set_mute": "设置静音",
            "change_channel": "切换频道",
            "set_channel": "设置频道",
            "set_source": "设置输入源",
            "playback": "播放控制",
            "set_picture_mode": "设置画面模式",
            "set_subtitles": "设置字幕",
        }.get(action, action)
        value_map = {"on": "开启", "off": "关闭", "up": "调高", "down": "调低", "next": "下一个", "previous": "上一个", "play": "播放", "pause": "暂停"}
        ops.append(op(device, location, action_text, attr, value_map.get(value_text, value_text)))
    return ops


def parse_structured_nlp_response(response: str) -> list[dict[str, str]]:
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
            ops.append({key: values.get(key, "未知") for key in FIELD_ORDER})
    return ops


def parse_light_ops(text: str) -> list[dict[str, str]]:
    location = location_from_text(text)
    ops: list[dict[str, str]] = []
    has_light_cue = any(term in text for term in ("灯", "灯光", "照明", "光线")) or any(mode in text for mode in LIGHT_MODES)
    if not has_light_cue:
        return ops

    for mode in LIGHT_MODES:
        if mode in text:
            attr = "色温" if mode in {"自然光", "冷色光", "冷色灯", "暖色光", "暖色灯"} else "模式"
            ops.append(op("灯", location, "设置模式", attr, mode.replace("冷色灯", "冷色光").replace("暖色灯", "暖色光")))
            break

    if any(term in text for term in ("最亮", "全开")):
        ops.append(op("灯", location, "设置亮度", "亮度", "最高"))
    elif any(term in text for term in ("最暗", "最低")):
        ops.append(op("灯", location, "设置亮度", "亮度", "最低"))
    elif any(term in text for term in ("调亮", "亮一点", "亮一度", "太暗", "看不清", "不够亮", "昏暗")) and not any(term in text for term in ("开灯", "打开灯", "亮灯")):
        ops.append(op("灯", location, "调高", "亮度", "一档"))
    elif any(term in text for term in ("调暗", "暗一点", "暗些", "太亮", "刺眼", "柔和")):
        ops.append(op("灯", location, "调低", "亮度", "一档"))

    if not ops:
        if any(term in text for term in ("关闭", "关灯", "关掉灯", "灯关", "关一下灯")):
            ops.append(op("灯", location, "关闭", "电源", "关闭"))
        elif any(term in text for term in ("打开", "开启", "开灯", "开一下", "开个灯", "亮灯", "灯开")):
            ops.append(op("灯", location, "打开", "电源", "开启"))
    return dedupe_ops(ops)


def parse_ac_ops(text: str) -> list[dict[str, str]]:
    location = location_from_text(text)
    ac_cue = any(term in text for term in ("空调", "温度", "风速", "风向", "风力", "风口", "风太", "风吹", "制冷", "制热", "送风", "抽湿", "环保"))
    if not ac_cue:
        return []
    ops: list[dict[str, str]] = []

    for mode in AC_MODES:
        if mode in text:
            mode_value = normalize_mode(mode.replace("舒睡一", "舒睡模式一").replace("舒睡二", "舒睡模式二"))
            action = "关闭" if any(close in text for close in ("关闭", "关掉")) and "关闭空调" not in text else "设置模式"
            ops.append(op("空调", location, action, "模式", mode_value))
            break

    match = NUMBER_RE.search(text)
    if match:
        ops.append(op("空调", location, "设置温度", "温度", f"{match.group(1)}度"))
    elif any(term in text for term in ("温度调高", "温度升高", "升高一度", "调高一度", "升一度")):
        ops.append(op("空调", location, "调高", "温度", "一度"))
    elif any(term in text for term in ("温度调低", "温度降低", "降低一度", "调低一度", "降一度")):
        ops.append(op("空调", location, "调低", "温度", "一度"))

    if any(term in text for term in ("风速调到最高", "风速最高", "风速最大", "最大风", "风力最大")):
        ops.append(op("空调", location, "设置风速", "风速", "最高"))
    elif any(term in text for term in ("风速调到最低", "风速最低", "风速最小", "最小风", "减至最小")):
        ops.append(op("空调", location, "设置风速", "风速", "最低"))
    elif "中速" in text:
        ops.append(op("空调", location, "设置风速", "风速", "中速"))
    elif any(term in text for term in ("风速调高", "风速提高", "已提高", "风不够大", "开大点", "风大点", "风再大", "风更大", "风力调大", "风力不够", "风力太小", "风速太小", "加大风", "加大点风", "风来得更猛烈", "更猛", "更给力", "强点", "风速快一点")):
        ops.append(op("空调", location, "调高", "风速", "一档"))
    elif any(term in text for term in ("风速调低", "风速降低", "已降低", "风小点", "风太大", "风速太大", "减小风", "减小点风", "调小", "调低", "降低")):
        ops.append(op("空调", location, "调低", "风速", "一档"))

    if any(term in text for term in ("上下左右", "上下及左右", "吹上下及左右")):
        ops.append(op("空调", location, "设置风向", "风向", "上下左右"))
    elif "向左" in text or "左边" in text or "往左" in text:
        ops.append(op("空调", location, "设置风向", "风向", "左"))
    elif "向下" in text or "下吹" in text or "往下" in text:
        ops.append(op("空调", location, "设置风向", "风向", "下"))
    elif "上下" in text or "向上" in text:
        ops.append(op("空调", location, "设置风向", "风向", "上下"))
    elif "左右" in text:
        ops.append(op("空调", location, "设置风向", "风向", "左右"))
    elif "不动" in text or "不变" in text or "不用动" in text:
        ops.append(op("空调", location, "设置风向", "风向", "不动"))

    timer = TIMER_RE.search(text)
    if timer:
        time_value = f"{timer.group(1)}{timer.group(2)}后{timer.group(3)}".replace("关", "关闭").replace("打开", "开启")
        ops.append(op("空调", location, "设置定时", "定时", time_value))
    elif "运行" in text and re.search(r"\d+分钟后", text):
        minutes = re.search(r"(\d+)分钟后", text).group(1)  # type: ignore[union-attr]
        ops.append(op("空调", location, "设置定时", "定时", f"{minutes}分钟后开启"))

    if not ops:
        close_cues = (
            "关闭",
            "关掉",
            "关了",
            "关下",
            "关一下",
            "关机",
            "停止",
            "停了",
            "不用了",
            "不需要开",
            "别开",
            "休息",
            "下班",
            "退下",
            "退休",
            "休眠",
            "已关闭",
        )
        open_cues = (
            "打开",
            "开启",
            "启动",
            "开机",
            "开一下",
            "开个",
            "开空调",
            "凉快",
            "冷气",
            "好热",
            "太热",
            "热死",
            "冒汗",
            "闷",
            "已打开",
            "已开启",
        )
        if any(term in text for term in close_cues):
            ops.append(op("空调", location, "关闭", "电源", "关闭"))
        elif any(term in text for term in open_cues):
            ops.append(op("空调", location, "打开", "电源", "开启"))
    return dedupe_ops(ops)


def parse_tv_ops(text: str) -> list[dict[str, str]]:
    if "电视" not in text:
        return []
    location = location_from_text(text)
    if any(term in text for term in ("关闭", "关掉", "关一下", "关下", "关机", "不看电视", "电视不用", "已关闭")):
        return [op("电视", location, "关闭", "电源", "关闭")]
    if any(term in text for term in ("打开", "开启", "开机", "启动", "看电视", "看会儿电视", "电视时间", "已开启", "已打开")):
        return [op("电视", location, "打开", "电源", "开启")]
    return []


def parse_speaker_ops(text: str) -> list[dict[str, str]]:
    speaker_cue = any(term in text for term in ("音箱", "音乐", "歌曲", "歌", "播放", "暂停")) and "电视" not in text
    if not speaker_cue:
        return []
    location = location_from_text(text)
    if any(term in text for term in ("音量调高", "音量加", "音量增加", "提高音箱音量", "音量太低", "音量不够", "音量调大", "音量稍微调高", "音量已调高", "声音加大", "声音调大", "声音调高", "声音太小", "更大的音箱声音")):
        return [op("音箱", location, "调高", "音量", "一档")]
    if any(term in text for term in ("音量调低", "音量减少", "降低音箱音量", "音量太高", "音量太大", "音量调小", "音量已调低", "声音调小", "声音调低", "声音太大", "更小的音箱声音")):
        return [op("音箱", location, "调低", "音量", "一档")]
    if any(term in text for term in ("暂停", "停一下")):
        return [op("音箱", location, "暂停", "播放", "暂停")]
    if any(term in text for term in ("下一首", "下首")):
        return [op("音箱", location, "下一首", "播放", "下一首")]
    if any(term in text for term in ("播放", "放音乐", "放歌")):
        return [op("音箱", location, "播放", "播放", "播放")]
    if any(term in text for term in ("关机", "关闭")):
        return [op("音箱", location, "关闭", "电源", "关闭")]
    if any(term in text for term in ("开机", "打开", "开启")):
        return [op("音箱", location, "打开", "电源", "开启")]
    return []


def parse_dehumidifier_ops(text: str) -> list[dict[str, str]]:
    if not any(term in text for term in ("抽湿机", "除湿机")):
        return []
    location = location_from_text(text)
    if any(term in text for term in ("关闭", "关掉", "关机")):
        return [op("抽湿机", location, "关闭", "电源", "关闭")]
    if any(term in text for term in ("打开", "开启", "开机")):
        return [op("抽湿机", location, "打开", "电源", "开启")]
    return []


def parse_text_ops(text: str) -> list[dict[str, str]]:
    text = clean_text(text)
    ops: list[dict[str, str]] = []
    # Specific operations first so contextual phrases such as “看电视，调整自然光” map to the light.
    ops.extend(parse_light_ops(text))
    ops.extend(parse_ac_ops(text))
    ops.extend(parse_speaker_ops(text))
    ops.extend(parse_dehumidifier_ops(text))
    if not ops:
        ops.extend(parse_tv_ops(text))
    return dedupe_ops(ops)


def dedupe_ops(ops: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, str]] = []
    for item in ops:
        key = tuple(item[field] for field in FIELD_ORDER)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def extract_pair(record: dict[str, Any]) -> tuple[str, str]:
    prompt, _ = first_text(record, PROMPT_FIELDS)
    response, _ = first_text(record, RESPONSE_FIELDS)
    extra_input = clean_text(record.get("input")) if "instruction" in record else ""
    if extra_input and extra_input != prompt:
        prompt = f"{prompt}\n{extra_input}" if prompt else extra_input
    return prompt, response


def canonicalize_record(record: dict[str, Any], index: int, drop_uncertain: bool) -> tuple[dict[str, Any] | None, list[str]]:
    prompt, response = extract_pair(record)
    flags: list[str] = []
    if not prompt or not response:
        flags.append("missing_prompt_or_response")
        return None, flags

    structured_source = record.get("structured_response")
    source_structured_ops: list[dict[str, str]] = []
    if isinstance(structured_source, dict) or isinstance(structured_source, list):
        source_structured_ops = parse_structured_json_response(json.dumps(structured_source, ensure_ascii=False))

    prompt_ops = parse_text_ops(prompt)
    response_ops = source_structured_ops or parse_structured_nlp_response(response) or parse_structured_json_response(response) or parse_text_ops(response)

    if prompt_ops:
        chosen_ops = prompt_ops
        source = "prompt_inferred"
        if response_ops and format_ops(response_ops) != format_ops(prompt_ops):
            flags.append("response_corrected_from_prompt")
    elif response_ops:
        chosen_ops = response_ops
        source = "response_inferred"
        flags.append("prompt_intent_uncertain")
    else:
        flags.append("unparsed_intent")
        if drop_uncertain:
            return None, flags
        chosen_ops = [op("未知", "未指定", "未知", "未知", "未知")]
        source = "unknown"

    canonical = format_ops(chosen_ops)
    difficulty = record.get("difficulty") or classify_difficulty(prompt, chosen_ops)
    output = dict(record)
    output.update(
        {
            "prompt": prompt,
            "response": canonical,
            "structured_response": chosen_ops,
            "difficulty": difficulty,
        }
    )
    metadata = dict(output.get("scenic_metadata") or {})
    metadata.update(
        {
            "source_index": index,
            "source_response": response,
            "canonical_source": source,
            "quality_flags": sorted(set(flags)),
        }
    )
    output["scenic_metadata"] = metadata
    output.pop("positive", None)
    output.pop("negative", None)
    return output, flags


def classify_difficulty(prompt: str, ops: list[dict[str, str]]) -> str:
    if len(ops) >= 3 or len(prompt) >= 42:
        return "hard"
    if len(ops) == 2 or len(prompt) >= 20 or any(item["attribute"] in {"定时", "温度", "风向", "输入源", "频道", "画面模式"} for item in ops):
        return "medium"
    return "easy"


def row_structured_ops(row: dict[str, Any]) -> list[dict[str, str]]:
    raw_ops = row.get("structured_response")
    items = raw_ops if isinstance(raw_ops, list) else [raw_ops] if isinstance(raw_ops, dict) else []
    ops: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        normalized = {field: clean_text(item.get(field)) or "未知" for field in FIELD_ORDER}
        normalized["location"] = normalized["location"] or "未指定"
        ops.append(normalized)
    return ops or parse_structured_nlp_response(clean_text(row.get("response"))) or parse_text_ops(clean_text(row.get("prompt")))


def infer_device_categories(text: str) -> set[str]:
    return {category for category, keywords in DEVICE_CATEGORY_KEYWORDS.items() if any(keyword in text for keyword in keywords)}


def infer_action_types(text: str) -> set[str]:
    return {action for action, keywords in ACTION_TYPE_KEYWORDS.items() if any(keyword in text for keyword in keywords)}


def infer_attribute_types(text: str) -> set[str]:
    return {attribute for attribute, keywords in ATTRIBUTE_TYPE_KEYWORDS.items() if any(keyword in text for keyword in keywords)}


def infer_numeric_values(text: str) -> set[str]:
    values = set(re.findall(r"\d+(?:\.\d+)?\s*(?:度|分钟|小时|档|%|台|频道)?", text))
    return {normalize_key(value) for value in values if value.strip()}


def infer_timer_values(text: str) -> set[str]:
    timers: set[str] = set()
    for amount, unit, action in TIMER_RE.findall(text):
        timers.add(f"{amount}{unit}后{action}".replace("打开", "开启").replace("关", "关闭"))
    return timers


def action_types_from_op(item: dict[str, str]) -> set[str]:
    text = f"{item.get('action', '')}{item.get('attribute', '')}{item.get('value', '')}"
    actions = infer_action_types(text)
    action = item.get("action", "")
    attribute = item.get("attribute", "")
    if "打开" in action or item.get("value") == "开启":
        actions.add("turn_on")
    if "关闭" in action or item.get("value") == "关闭":
        actions.add("turn_off")
    if "调高" in action:
        actions.add("increase")
    if "调低" in action:
        actions.add("decrease")
    if "播放" in action:
        actions.add("play")
    if "暂停" in action:
        actions.add("pause")
    if "模式" in action or attribute == "模式":
        actions.add("switch_mode")
    elif "设置" in action or attribute in {"温度", "亮度", "风速", "风向", "音量", "频道", "输入源"}:
        actions.add("set_value")
    return actions


def semantic_features(row: dict[str, Any]) -> dict[str, Any]:
    cached = row.get("_semantic_features")
    if isinstance(cached, dict):
        return cached

    prompt = clean_text(row.get("prompt"))
    response = clean_text(row.get("response"))
    ops = row_structured_ops(row)
    devices: set[str] = set()
    locations: set[str] = set()
    actions: set[str] = set()
    attributes: set[str] = set()
    values: set[str] = set()
    numeric_values: set[str] = set()
    timer_values: set[str] = set()

    for item in ops:
        op_text = "".join(item.get(field, "") for field in FIELD_ORDER)
        devices.update(infer_device_categories(item.get("device", "")) or infer_device_categories(op_text))
        location = item.get("location", "")
        if location and location not in {"未指定", "未知", "none", "unspecified"}:
            locations.add(location)
        actions.update(action_types_from_op(item))
        attributes.update(infer_attribute_types(f"{item.get('action', '')}{item.get('attribute', '')}{item.get('value', '')}"))
        value = clean_text(item.get("value"))
        if value and value != "未知":
            values.add(value)
        numeric_values.update(infer_numeric_values(op_text))
        if item.get("attribute") == "定时":
            timer_values.add(value)

    combined_text = f"{prompt}{response}"
    devices.update(infer_device_categories(combined_text))
    locations.update(location for location, terms in LOCATION_TERMS if any(term in prompt for term in terms))
    actions.update(infer_action_types(prompt))
    attributes.update(infer_attribute_types(prompt))
    numeric_values.update(infer_numeric_values(combined_text))
    timer_values.update(infer_timer_values(combined_text))

    features = {
        "devices": devices,
        "locations": locations,
        "actions": actions,
        "attributes": attributes,
        "values": values,
        "numeric_values": numeric_values,
        "timer_values": timer_values,
        "indirect": any(cue in prompt for cue in INDIRECT_COMMAND_CUES),
        "prompt_key": normalize_key(prompt),
        "response_key": normalize_key(response),
    }
    row["_semantic_features"] = features
    return features


def sorted_feature_values(features: dict[str, Any], key: str) -> list[str]:
    values = features.get(key, set())
    return sorted(values) if isinstance(values, set) else []


def first_feature_value(features: dict[str, Any], key: str) -> str:
    values = sorted_feature_values(features, key)
    return values[0] if values else ""


def same_device_instance(anchor_features: dict[str, Any], candidate_features: dict[str, Any]) -> bool:
    if not (anchor_features["devices"] & candidate_features["devices"]):
        return False
    anchor_locations = anchor_features["locations"]
    candidate_locations = candidate_features["locations"]
    if anchor_locations and candidate_locations:
        return bool(anchor_locations & candidate_locations)
    return True


def different_location(anchor_features: dict[str, Any], candidate_features: dict[str, Any]) -> bool:
    anchor_locations = anchor_features["locations"]
    candidate_locations = candidate_features["locations"]
    return bool(anchor_locations and candidate_locations and not (anchor_locations & candidate_locations))


def feature_sets_match(anchor_features: dict[str, Any], candidate_features: dict[str, Any]) -> bool:
    fields = ("devices", "locations", "actions", "attributes", "numeric_values", "timer_values", "values")
    return all(anchor_features[field] == candidate_features[field] for field in fields)


def likely_false_negative(anchor: dict[str, Any], candidate: dict[str, Any]) -> bool:
    anchor_features = semantic_features(anchor)
    candidate_features = semantic_features(candidate)
    if anchor_features["response_key"] == candidate_features["response_key"]:
        return True
    if anchor_features["prompt_key"] == candidate_features["prompt_key"]:
        return True

    response_similarity = char_jaccard_similarity(clean_text(anchor.get("response")), clean_text(candidate.get("response")))
    if response_similarity >= FALSE_NEGATIVE_RESPONSE_SIMILARITY and feature_sets_match(anchor_features, candidate_features):
        return True

    same_core = (
        anchor_features["devices"] == candidate_features["devices"]
        and anchor_features["locations"] == candidate_features["locations"]
        and anchor_features["actions"] == candidate_features["actions"]
        and anchor_features["attributes"] == candidate_features["attributes"]
        and anchor_features["numeric_values"] == candidate_features["numeric_values"]
        and anchor_features["timer_values"] == candidate_features["timer_values"]
    )
    if same_core and anchor_features["values"] == candidate_features["values"]:
        return True

    prompt_similarity = char_jaccard_similarity(clean_text(anchor.get("prompt")), clean_text(candidate.get("prompt")))
    if prompt_similarity >= FALSE_NEGATIVE_PROMPT_SIMILARITY and same_core:
        return True
    return False


def score_negative_candidate(anchor: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, str, str]:
    anchor_features = semantic_features(anchor)
    candidate_features = semantic_features(candidate)
    if likely_false_negative(anchor, candidate):
        return -10.0, "rejected_false_negative", "candidate likely represents the same executable action"

    shared_device_category = bool(anchor_features["devices"] & candidate_features["devices"])
    same_device = same_device_instance(anchor_features, candidate_features)
    shared_action = bool(anchor_features["actions"] & candidate_features["actions"])
    different_action = bool(anchor_features["actions"] and candidate_features["actions"] and anchor_features["actions"] != candidate_features["actions"])
    different_attribute = bool(anchor_features["attributes"] and candidate_features["attributes"] and anchor_features["attributes"] != candidate_features["attributes"])
    different_value = bool(
        (anchor_features["numeric_values"] or anchor_features["values"] or anchor_features["timer_values"])
        and (
            anchor_features["numeric_values"] != candidate_features["numeric_values"]
            or anchor_features["values"] != candidate_features["values"]
            or anchor_features["timer_values"] != candidate_features["timer_values"]
        )
    )
    prompt_similarity = char_jaccard_similarity(clean_text(anchor.get("prompt")), clean_text(candidate.get("prompt")))

    score = 0.0
    source = "random_different_response"
    reason = "different normalized response fallback"
    if same_device and different_action:
        score += 5.0
        source = "same_device_different_action"
        reason = "same device category and location but different action type"
    elif same_device and (different_attribute or different_value):
        score += 4.0
        source = "same_device_different_attribute_or_value"
        reason = "same device category but different attribute or value"
    elif shared_action and (not shared_device_category or different_location(anchor_features, candidate_features)):
        score += 3.0
        source = "same_action_different_device_or_location"
        reason = "same action type but different device or location"
    elif shared_device_category:
        score += 2.0
        source = "same_device_category_different_response"
        reason = "same device category with a different normalized response"
    else:
        score -= 2.0
        reason = "different device and action, easier contrast"

    if prompt_similarity >= 0.35:
        score += 1.0
    return score, source, reason


def negative_hardness(score: float, source: str) -> str:
    if source in {"same_device_different_action", "same_device_different_attribute_or_value"} or score >= 5:
        return "hard"
    if source in {"same_action_different_device_or_location", "same_device_category_different_response"} or score >= 3:
        return "medium"
    return "easy"


def make_synthetic_positive(prompt: str, features: dict[str, Any] | None = None) -> str:
    features = features or {"devices": set(), "locations": set(), "actions": set(), "attributes": set(), "numeric_values": set(), "values": set()}
    base = PUNCT_RE.sub("", prompt)
    location = first_feature_value(features, "locations")
    location_prefix = location if location else ""
    devices = features.get("devices", set())
    actions = features.get("actions", set())
    attributes = features.get("attributes", set())
    numeric = first_feature_value(features, "numeric_values")

    if "light" in devices:
        light_name = f"{location_prefix}灯" if location_prefix else "灯"
        if "turn_on" in actions:
            return f"{location_prefix}有点暗，帮我开一下灯。" if location_prefix else "有点暗，帮我开一下灯。"
        if "turn_off" in actions:
            return f"不用{light_name}了，帮我关掉。"
        if "increase" in actions or ("brightness" in attributes and "set_value" in actions):
            return f"{location_prefix}现在有点暗，把亮度调高一点。" if location_prefix else "现在有点暗，把亮度调高一点。"
        if "decrease" in actions:
            return f"{location_prefix}现在有点亮，把亮度调低一点。" if location_prefix else "现在有点亮，把亮度调低一点。"

    if "air_conditioner" in devices:
        ac_name = f"{location_prefix}空调" if location_prefix else "空调"
        if "temperature" in attributes and numeric:
            return f"把{ac_name}温度设成{numeric}。"
        if "increase" in actions:
            return f"把{ac_name}温度调高一点。"
        if "decrease" in actions:
            return f"把{ac_name}温度调低一点。"
        if "turn_on" in actions:
            return f"{location_prefix}有点闷，帮我开一下空调。" if location_prefix else "有点闷，帮我开一下空调。"
        if "turn_off" in actions:
            return f"不用{ac_name}了，帮我关掉。"

    if "tv" in devices:
        tv_name = f"{location_prefix}电视" if location_prefix else "电视"
        if "turn_on" in actions:
            return f"想看会儿{tv_name}，帮我打开。"
        if "turn_off" in actions:
            return f"不看{tv_name}了，帮我关掉。"

    if "speaker" in devices:
        if "play" in actions:
            return "想听点音乐，帮我播放。"
        if "pause" in actions:
            return "先暂停一下音乐。"
        if "increase" in actions:
            return "声音有点小，把音量调高一点。"
        if "decrease" in actions:
            return "声音有点大，把音量调低一点。"

    if "dehumidifier" in devices:
        if "turn_on" in actions:
            return f"{location_prefix}有点潮，帮我开一下除湿。" if location_prefix else "有点潮，帮我开一下除湿。"
        if "turn_off" in actions:
            return f"{location_prefix}不用除湿了，帮我关掉。" if location_prefix else "不用除湿了，帮我关掉。"

    if not base:
        return ""
    if base.startswith(("请", "麻烦", "帮我", "把")):
        return f"{base}吧。"
    return f"麻烦帮我{base}。"


def attach_contrastive(rows: list[dict[str, Any]], seed: int, singleton_positive: str = "synthetic") -> Counter[str]:
    rng = random.Random(seed)
    stats: Counter[str] = Counter()
    response_groups: dict[str, list[int]] = defaultdict(list)
    device_index: dict[str, set[int]] = defaultdict(set)
    action_index: dict[str, set[int]] = defaultdict(set)
    attribute_index: dict[str, set[int]] = defaultdict(set)
    all_indices = list(range(len(rows)))

    for index, row in enumerate(rows):
        row.setdefault("scenic_metadata", {})
        features = semantic_features(row)
        response_groups[features["response_key"]].append(index)
        for device in features["devices"]:
            device_index[device].add(index)
        for action in features["actions"]:
            action_index[action].add(index)
        for attribute in features["attributes"]:
            attribute_index[attribute].add(index)

    for index, row in enumerate(rows):
        metadata = row.setdefault("scenic_metadata", {})
        features = semantic_features(row)

        # Positives are same normalized response class C(y_i). Pick the most
        # surface-diverse paraphrase so alignment learns invariance to wording.
        same_response = response_groups[features["response_key"]]
        positive_candidates = [
            candidate
            for candidate in same_response
            if candidate != index and semantic_features(rows[candidate])["prompt_key"] != features["prompt_key"]
        ]
        if not positive_candidates:
            positive_candidates = [candidate for candidate in same_response if candidate != index]
        if positive_candidates:
            best_distance = max(char_jaccard_distance(row["prompt"], rows[candidate]["prompt"]) for candidate in positive_candidates)
            best_candidates = [
                candidate
                for candidate in positive_candidates
                if char_jaccard_distance(row["prompt"], rows[candidate]["prompt"]) == best_distance
            ]
            positive_index = rng.choice(best_candidates)
            row["positive"] = rows[positive_index]["prompt"]
            metadata["positive_source"] = "same_normalized_response_diverse"
            metadata["positive_distance"] = round(best_distance, 4)
        else:
            stats["singleton_positive"] += 1
            if singleton_positive == "self":
                row["positive"] = row["prompt"]
                metadata["positive_source"] = "singleton_self"
            elif singleton_positive == "empty":
                row["positive"] = ""
                metadata["positive_source"] = "singleton_empty"
            else:
                row["positive"] = make_synthetic_positive(row["prompt"], features)
                metadata["positive_source"] = "synthetic_singleton_template"

        # Hard negatives must be a different normalized response, but should
        # share smart-home structure when possible. False negatives are rejected
        # first because mislabeled paraphrases damage contrastive alignment.
        candidate_pool: set[int] = set()
        for device in features["devices"]:
            candidate_pool.update(device_index.get(device, set()))
        for action in features["actions"]:
            candidate_pool.update(action_index.get(action, set()))
        for attribute in features["attributes"]:
            candidate_pool.update(attribute_index.get(attribute, set()))
        if len(candidate_pool) > HARD_NEGATIVE_POOL:
            candidate_pool = set(rng.sample(sorted(candidate_pool), HARD_NEGATIVE_POOL))
        if len(rows) <= FALLBACK_NEGATIVE_SAMPLE:
            candidate_pool.update(all_indices)
        else:
            candidate_pool.update(rng.sample(all_indices, FALLBACK_NEGATIVE_SAMPLE))

        scored: list[tuple[float, int, str, str]] = []
        rejected_false_negatives = 0
        for candidate_index in candidate_pool:
            if candidate_index == index:
                continue
            candidate = rows[candidate_index]
            candidate_features = semantic_features(candidate)
            if candidate_features["response_key"] == features["response_key"]:
                continue
            if candidate_features["prompt_key"] == features["prompt_key"]:
                rejected_false_negatives += 1
                continue
            if char_jaccard_similarity(row["response"], candidate["response"]) >= FALSE_NEGATIVE_RESPONSE_SIMILARITY:
                rejected_false_negatives += 1
                continue
            score, source, reason = score_negative_candidate(row, candidate)
            if score <= -10:
                rejected_false_negatives += 1
                continue
            scored.append((score, candidate_index, source, reason))
        stats["false_negative_rejected"] += rejected_false_negatives

        if not scored:
            fallback = [
                candidate
                for candidate in all_indices
                if candidate != index and semantic_features(rows[candidate])["response_key"] != features["response_key"]
            ]
            negative_index = rng.choice(fallback) if fallback else index
            score, source, reason = (-2.0, "random_different_response", "different response fallback after all hard candidates were rejected")
        else:
            scored.sort(key=lambda item: (-item[0], item[1]))
            top = scored[:TOP_NEGATIVE_CANDIDATES]
            best_score = top[0][0]
            strong_top = [item for item in top if item[0] >= max(1.0, best_score - 1.0)]
            score, negative_index, source, reason = rng.choice(strong_top)

        negative = rows[negative_index]
        negative_features = semantic_features(negative)
        row["negative"] = negative["prompt"]
        metadata.update(
            {
                "negative_response": negative["response"],
                "negative_source": source,
                "negative_hardness": negative_hardness(score, source),
                "negative_reason": reason,
                "negative_score": round(score, 4),
                "anchor_devices": sorted_feature_values(features, "devices"),
                "negative_devices": sorted_feature_values(negative_features, "devices"),
                "anchor_actions": sorted_feature_values(features, "actions"),
                "negative_actions": sorted_feature_values(negative_features, "actions"),
                "anchor_locations": sorted_feature_values(features, "locations"),
                "negative_locations": sorted_feature_values(negative_features, "locations"),
                "anchor_attributes": sorted_feature_values(features, "attributes"),
                "negative_attributes": sorted_feature_values(negative_features, "attributes"),
            }
        )
        stats[f"negative_source:{source}"] += 1
        stats[f"negative_hardness:{metadata['negative_hardness']}"] += 1
        stats[f"positive_source:{metadata['positive_source']}"] += 1

    for row in rows:
        row.pop("_semantic_features", None)
    return stats


def resolve_prompt_conflicts(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_key(row["prompt"])].append(row)
    cleaned: list[dict[str, Any]] = []
    dropped = 0
    for items in grouped.values():
        responses = Counter(item["response"] for item in items)
        if len(responses) == 1:
            cleaned.extend(items)
            continue
        winner, count = responses.most_common(1)[0]
        if count == 1:
            dropped += len(items)
            continue
        for item in items:
            if item["response"] == winner:
                item["scenic_metadata"].setdefault("quality_flags", []).append("resolved_prompt_conflict")
                cleaned.append(item)
            else:
                dropped += 1
    return cleaned, dropped


def summarize_lengths(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "mean": round(statistics.mean(values), 2), "max": max(values)}


def metadata_distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(
        row.get("scenic_metadata", {}).get(key)
        for row in rows
        if row.get("scenic_metadata", {}).get(key)
    )
    return dict(sorted(counter.items()))


def build_report(
    raw_count: int,
    rows: list[dict[str, Any]],
    dropped_uncertain: int,
    dropped_conflicts: int,
    flags: Counter[str],
    contrastive_stats: Counter[str],
) -> dict[str, Any]:
    response_counter = Counter(row["response"] for row in rows)
    return {
        "input_records": raw_count,
        "clean_records": len(rows),
        "dropped_uncertain_records": dropped_uncertain,
        "dropped_conflict_records": dropped_conflicts,
        "unique_canonical_responses": len(response_counter),
        "singleton_canonical_responses": sum(1 for count in response_counter.values() if count == 1),
        "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "canonical_source": dict(sorted(Counter(row["scenic_metadata"].get("canonical_source") for row in rows).items())),
        "quality_flags": dict(sorted(flags.items())),
        "positive_source": metadata_distribution(rows, "positive_source"),
        "negative_source": metadata_distribution(rows, "negative_source"),
        "negative_hardness": metadata_distribution(rows, "negative_hardness"),
        "false_negative_rejected": int(contrastive_stats.get("false_negative_rejected", 0)),
        "singleton_positive": int(contrastive_stats.get("singleton_positive", 0)),
        "prompt_length": summarize_lengths([len(row["prompt"]) for row in rows]),
        "response_length": summarize_lengths([len(row["response"]) for row in rows]),
        "response_schema": "设备=<device>; 位置=<location>; 动作=<action>; 属性=<attribute>; 值=<value>",
        "examples": rows[:5],
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    raw_records = read_records(input_path)
    flags = Counter()
    rows: list[dict[str, Any]] = []
    dropped_uncertain = 0
    for index, record in enumerate(raw_records):
        clean, row_flags = canonicalize_record(record, index, args.drop_uncertain)
        flags.update(row_flags)
        if clean is None:
            dropped_uncertain += 1
            continue
        rows.append(clean)

    rows, dropped_conflicts = resolve_prompt_conflicts(rows)
    contrastive_stats = attach_contrastive(rows, args.seed, args.singleton_positive)
    report = build_report(len(raw_records), rows, dropped_uncertain, dropped_conflicts, flags, contrastive_stats)

    if not args.keep_metadata:
        for row in rows:
            row.pop("scenic_metadata", None)

    write_records(output_path, rows, output_format(output_path, args.format))
    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))
    print(f"wrote cleaned dataset: {output_path}")
    if args.report:
        print(f"wrote cleaning report: {Path(args.report).expanduser()}")


if __name__ == "__main__":
    main()
