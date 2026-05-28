#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_SINGLE = PROJECT_ROOT / "generated" / "iot_single_device_expansion.json"
DEFAULT_MULTI = PROJECT_ROOT / "generated" / "iot_multi_device_expansion.json"
DEFAULT_BASE_CONTRASTIVE = PROJECT_ROOT / "data" / "619_Luke_REPAIRED_positive_negative.jsonl"
DEFAULT_TRAINING_OUTPUT = PROJECT_ROOT / "data" / "SCENIC_full_training_dataset.json"
DEFAULT_CONTRASTIVE_OUTPUT = PROJECT_ROOT / "data" / "SCENIC_full_anchor_positive_negative.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "SCENIC_full_training_and_contrastive_report.json"


DEVICE_ORDER = (
    "扫地机器人",
    "空气净化器",
    "智能插座",
    "除湿机",
    "加湿器",
    "热水器",
    "摄像头",
    "窗帘",
    "空调",
    "电视",
    "音箱",
    "风扇",
    "门锁",
    "灯",
)
LOCATIONS = ("客厅", "卧室", "书房", "厨房", "浴室", "阳台", "玄关")

INVALID_ACTIONS = {
    "灯": ("温度调到26度", "灯支持开关、亮度、模式和颜色，但不支持温度设置。"),
    "空调": ("亮度调高", "空调支持温度、风速、风向和模式，但不支持亮度调节。"),
    "电视": ("风向调到上下", "电视支持频道、输入源、字幕、静音等控制，但不支持风向调节。"),
    "音箱": ("字幕打开", "音箱支持播放和音量控制，但不支持字幕控制。"),
    "窗帘": ("音量调高", "窗帘支持开合和半开合，但不支持音量控制。"),
    "扫地机器人": ("频道设为CCTV-1", "扫地机器人支持清扫、暂停和回充，但不支持电视频道控制。"),
    "空气净化器": ("字幕打开", "空气净化器支持开关、模式和风速，但不支持字幕控制。"),
    "加湿器": ("制冷模式", "加湿器支持开关和加湿量调节，但不支持制冷模式。"),
    "除湿机": ("播放音乐", "除湿机支持开关和除湿强度调节，但不支持音乐播放。"),
    "热水器": ("风向调到上下", "热水器支持开关和水温控制，但不支持风向控制。"),
    "风扇": ("温度调到26度", "风扇支持开关、风速和摇头，但不支持温度设置。"),
    "门锁": ("音量调高", "门锁支持上锁、解锁和状态检查，但不支持音量控制。"),
    "摄像头": ("温度调到26度", "摄像头支持开关、隐私模式和录像，但不支持温度设置。"),
    "智能插座": ("画面模式设为电影", "智能插座支持开关和定时供电，但不支持电视画面模式。"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the full SCENIC training JSON and full anchor-positive-negative contrastive JSON.")
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--single", default=str(DEFAULT_SINGLE))
    parser.add_argument("--multi", default=str(DEFAULT_MULTI))
    parser.add_argument("--base-contrastive", default=str(DEFAULT_BASE_CONTRASTIVE))
    parser.add_argument("--training-output", default=str(DEFAULT_TRAINING_OUTPUT))
    parser.add_argument("--contrastive-output", default=str(DEFAULT_CONTRASTIVE_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def compact(text: str) -> str:
    return re.sub(r"[\s。！？!?,，、；;：:]+", "", clean_text(text))


def load_rows(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item.keys()) != {"prompt", "response"}:
            raise ValueError(f"{path}:{index} must contain exactly prompt and response")
        prompt = clean_text(item["prompt"])
        response = clean_text(item["response"])
        if not prompt or not response:
            raise ValueError(f"{path}:{index} contains an empty prompt or response")
        rows.append({"prompt": prompt, "response": response})
    return rows


def load_base_positive_map(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    positives: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            anchor = clean_text(item.get("anchor"))
            response = clean_text(item.get("response"))
            positive = clean_text(item.get("positive"))
            if anchor and response and positive and compact(anchor) != compact(positive):
                positives[(compact(anchor), response)] = positive
    return positives


def merge_datasets(sources: list[tuple[str, list[dict[str, str]]]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    merged: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    prompt_to_response: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    exact_duplicates: Counter[str] = Counter()
    conflicts: list[dict[str, str]] = []

    for source_name, rows in sources:
        for row in rows:
            prompt_key = compact(row["prompt"])
            pair_key = (prompt_key, row["response"])
            if pair_key in seen_pairs:
                exact_duplicates[source_name] += 1
                continue
            existing = prompt_to_response.get(prompt_key)
            if existing is not None and existing != row["response"]:
                conflicts.append(
                    {
                        "source": source_name,
                        "prompt": row["prompt"],
                        "existing_response": existing,
                        "new_response": row["response"],
                    }
                )
                continue
            merged.append({"prompt": row["prompt"], "response": row["response"]})
            source_counts[source_name] += 1
            seen_pairs.add(pair_key)
            prompt_to_response[prompt_key] = row["response"]

    if conflicts:
        preview = json.dumps(conflicts[:20], ensure_ascii=False, indent=2)
        raise ValueError(f"Found {len(conflicts)} same-prompt/different-response conflicts:\n{preview}")

    return merged, {
        "source_counts": dict(source_counts),
        "skipped_exact_duplicate_pairs": dict(exact_duplicates),
        "prompt_response_conflict_count": len(conflicts),
        "prompt_response_conflicts": conflicts[:100],
    }


def char_jaccard_distance(left: str, right: str) -> float:
    left_chars = set(compact(left))
    right_chars = set(compact(right))
    if not left_chars and not right_chars:
        return 0.0
    return 1.0 - (len(left_chars & right_chars) / len(left_chars | right_chars))


def choose_positive(
    index: int,
    rows: list[dict[str, str]],
    response_groups: dict[str, list[int]],
    base_positive_map: dict[tuple[str, str], str],
) -> tuple[str, str]:
    anchor = rows[index]
    candidates = [
        rows[candidate_index]["prompt"]
        for candidate_index in response_groups[anchor["response"]]
        if candidate_index != index and compact(rows[candidate_index]["prompt"]) != compact(anchor["prompt"])
    ]
    if candidates:
        return max(candidates, key=lambda item: (char_jaccard_distance(anchor["prompt"], item), item)), "same_response_diverse"
    mapped_positive = base_positive_map.get((compact(anchor["prompt"]), anchor["response"]))
    if mapped_positive and compact(mapped_positive) != compact(anchor["prompt"]):
        return mapped_positive, "base_contrastive_positive"
    # This should not happen with the current full corpus, but keeps the artifact total.
    return f"请执行这个控制：{anchor['prompt']}", "synthetic_fallback"


def infer_location(text: str) -> str:
    for location in LOCATIONS:
        if location in text:
            return location
    return ""


def infer_device(text: str) -> str:
    for device in DEVICE_ORDER:
        if device in text:
            return device
    if "字幕" in text or "频道" in text or "HDMI" in text or "画面模式" in text:
        return "电视"
    if "音乐" in text or "音量" in text or "下一首" in text or "上一首" in text:
        return "音箱"
    if "亮度" in text or "冷色光" in text or "暖色光" in text or "自然光" in text:
        return "灯"
    if "温度" in text or "风向" in text or "制冷" in text or "抽湿" in text:
        return "空调"
    return "智能插座"


def infer_devices(text: str) -> set[str]:
    devices = {device for device in DEVICE_ORDER if device in text}
    if not devices:
        devices.add(infer_device(text))
    return devices


def infer_locations(text: str) -> set[str]:
    locations = {location for location in LOCATIONS if location in text}
    if not locations:
        locations.add("")
    return locations


def infer_action_family(text: str) -> str:
    if "；" in text:
        return "multi_action"
    if "设置" in text and ("开启" in text or "关闭" in text):
        return "timer"
    if "打开" in text or "开启" in text or "启动" in text or "开始" in text:
        return "open_or_start"
    if "关闭" in text or "关掉" in text or "暂停" in text or "停止" in text:
        return "close_or_stop"
    if "切换到" in text or "模式" in text or "输入源" in text or "频道" in text:
        return "mode_or_source"
    if "调高" in text or "调低" in text or "设置为" in text or "调到" in text:
        return "adjust_value"
    if "音量" in text:
        return "volume"
    if "亮度" in text:
        return "brightness"
    return "other"


def extract_values(text: str) -> set[str]:
    values = set(re.findall(r"(?:\d+|[一二三四五六七八九十两半]+)(?:度|点半|点|分钟后|小时后|分钟|小时)", text))
    for token in ("制冷", "制热", "送风", "抽湿", "自动", "环保", "舒睡", "学习", "夜灯", "电影", "游戏", "体育", "HDMI1", "HDMI2", "投屏", "CCTV-1", "冷色光", "暖色光", "自然光"):
        if token in text:
            values.add(token)
    return values


def row_features(row: dict[str, str]) -> dict[str, Any]:
    text = f"{row['prompt']} {row['response']}"
    return {
        "devices": infer_devices(text),
        "locations": infer_locations(text),
        "action_family": infer_action_family(row["response"]),
        "values": extract_values(text),
        "is_multi_action": "；" in row["response"],
    }


def format_target(device: str, location: str) -> str:
    if device in {"热水器", "门锁"}:
        return device
    return f"{location}{device}" if location else device


def make_negative(anchor: str, response: str, all_prompt_norms: set[str]) -> tuple[str, str, str]:
    text = f"{anchor} {response}"
    device = infer_device(text)
    location = infer_location(text)
    invalid_action, explanation = INVALID_ACTIONS[device]
    target = format_target(device, location)

    templates = [
        f"把{target}{invalid_action}。",
        f"顺便把{target}{invalid_action}。",
        f"这个场景下把{target}{invalid_action}。",
    ]
    for prompt in templates:
        if compact(prompt) not in all_prompt_norms and compact(prompt) != compact(anchor):
            return prompt, "unsupported_action_for_device", explanation
    fallback = f"请让{target}{invalid_action}。"
    return fallback, "unsupported_action_for_device", explanation


def score_valid_negative(anchor_features: dict[str, Any], candidate_features: dict[str, Any]) -> tuple[int, str]:
    shared_devices = anchor_features["devices"] & candidate_features["devices"]
    shared_locations = {item for item in anchor_features["locations"] & candidate_features["locations"] if item}
    same_action = anchor_features["action_family"] == candidate_features["action_family"]
    shared_values = anchor_features["values"] & candidate_features["values"]
    both_multi = anchor_features["is_multi_action"] and candidate_features["is_multi_action"]

    score = 0
    reason_parts: list[str] = []
    if shared_devices:
        score += 8
        reason_parts.append("shares device category")
    if shared_locations:
        score += 4
        reason_parts.append("shares location")
    if same_action:
        score += 5
        reason_parts.append("shares action family")
    if shared_values:
        score += 2
        reason_parts.append("shares slot/mode/value")
    if both_multi:
        score += 2
        reason_parts.append("both are multi-action commands")

    if shared_devices and shared_locations and not same_action:
        score += 6
        source = "same_device_location_different_action"
    elif shared_devices and same_action and not shared_values:
        score += 5
        source = "same_device_action_different_value_or_slot"
    elif shared_devices and not shared_locations:
        score += 3
        source = "same_device_different_location"
    elif same_action:
        score += 2
        source = "same_action_different_device"
    elif both_multi:
        score += 1
        source = "different_multi_action_response"
    else:
        score -= 2
        source = "valid_different_response_fallback"

    return score, source + ": " + ", ".join(reason_parts)


def build_feature_index(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, set[int]], dict[str, set[int]], dict[str, set[int]]]:
    features = [row_features(row) for row in rows]
    by_device: dict[str, set[int]] = defaultdict(set)
    by_location: dict[str, set[int]] = defaultdict(set)
    by_action: dict[str, set[int]] = defaultdict(set)
    for index, item in enumerate(features):
        for device in item["devices"]:
            by_device[device].add(index)
        for location in item["locations"]:
            by_location[location].add(index)
        by_action[item["action_family"]].add(index)
    return features, by_device, by_location, by_action


def take_deterministic_subset(indices: set[int], anchor_index: int, limit: int) -> set[int]:
    values = sorted(index for index in indices if index != anchor_index)
    if len(values) <= limit:
        return set(values)
    # Deterministic spread across the bucket, rotated by anchor index to avoid always
    # comparing against the same early rows in large device/action groups.
    step = max(1, len(values) // limit)
    start = anchor_index % step
    subset = values[start::step][:limit]
    if len(subset) < limit:
        subset.extend(values[: limit - len(subset)])
    return set(subset)


def choose_valid_hard_negative(
    index: int,
    rows: list[dict[str, str]],
    features: list[dict[str, Any]],
    by_device: dict[str, set[int]],
    by_location: dict[str, set[int]],
    by_action: dict[str, set[int]],
) -> tuple[str, str, str, str]:
    anchor = rows[index]
    anchor_features = features[index]
    candidate_indices: set[int] = set()
    same_device = set().union(*(by_device.get(device, set()) for device in anchor_features["devices"]))
    same_location = set().union(*(by_location.get(location, set()) for location in anchor_features["locations"]))
    same_action = by_action.get(anchor_features["action_family"], set())

    layers = [
        same_device & same_location,
        same_device & same_action,
        same_location & same_action,
        same_device,
        same_action,
        same_location,
        set(range(len(rows))),
    ]
    for layer in layers:
        candidate_indices.update(take_deterministic_subset(layer, index, 180))
        if len(candidate_indices) >= 720:
            break

    best: tuple[int, float, int, str] | None = None
    for candidate_index in candidate_indices:
        if candidate_index == index:
            continue
        candidate = rows[candidate_index]
        if candidate["response"] == anchor["response"]:
            continue
        if compact(candidate["prompt"]) == compact(anchor["prompt"]):
            continue
        score, reason = score_valid_negative(anchor_features, features[candidate_index])
        lexical = 1.0 - char_jaccard_distance(anchor["prompt"], candidate["prompt"])
        ranked = (score, lexical, -candidate_index, reason)
        if best is None or ranked > best:
            best = ranked

    if best is None:
        for candidate_index, candidate in enumerate(rows):
            if candidate_index != index and candidate["response"] != anchor["response"]:
                return candidate["prompt"], candidate["response"], "valid_different_response_fallback", "fallback different normalized response"
        raise ValueError("Unable to find a valid hard negative")

    _, _, neg_candidate_index, reason = best
    candidate_index = -neg_candidate_index
    source, _, detail = reason.partition(": ")
    return rows[candidate_index]["prompt"], rows[candidate_index]["response"], source, detail


def build_contrastive(
    rows: list[dict[str, str]],
    base_positive_map: dict[tuple[str, str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    response_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        response_groups[row["response"]].append(index)
    all_prompt_norms = {compact(row["prompt"]) for row in rows}
    features, by_device, by_location, by_action = build_feature_index(rows)

    records: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for index, row in enumerate(rows):
        positive, positive_source = choose_positive(index, rows, response_groups, base_positive_map)
        valid_negative, valid_negative_response, valid_negative_source, valid_negative_reason = choose_valid_hard_negative(
            index,
            rows,
            features,
            by_device,
            by_location,
            by_action,
        )
        invalid_negative, invalid_negative_type, invalid_negative_explanation = make_negative(row["prompt"], row["response"], all_prompt_norms)
        record = {
            "source_id": f"full_{index + 1:06d}",
            "anchor": row["prompt"],
            "positive": positive,
            "negative": valid_negative,
            "response": row["response"],
            "positive_source": positive_source,
            "negative_response": valid_negative_response,
            "negative_type": "valid_hard_different_response",
            "negative_source": valid_negative_source,
            "negative_explanation": f"Valid command with a different normalized response; {valid_negative_reason}.",
            "invalid_negative": invalid_negative,
            "invalid_negative_type": invalid_negative_type,
            "invalid_negative_explanation": invalid_negative_explanation,
            "is_valid_anchor": True,
            "is_valid_positive": True,
            "is_valid_negative": True,
            "is_valid_invalid_negative": False,
        }
        if compact(record["anchor"]) == compact(record["positive"]):
            raise ValueError(f"positive equals anchor at {index}: {record}")
        if compact(record["negative"]) in {compact(record["anchor"]), compact(record["positive"])}:
            raise ValueError(f"negative duplicates valid text at {index}: {record}")
        if record["negative_response"] == record["response"]:
            raise ValueError(f"valid negative has same response at {index}: {record}")
        if compact(record["negative"]) not in all_prompt_norms:
            raise ValueError(f"valid negative is not an existing valid prompt at {index}: {record}")
        if compact(record["invalid_negative"]) in all_prompt_norms:
            raise ValueError(f"invalid negative exists as valid prompt at {index}: {record}")
        records.append(record)
        stats[f"positive_source:{positive_source}"] += 1
        stats["negative_type:valid_hard_different_response"] += 1
        stats[f"negative_source:{valid_negative_source}"] += 1
        stats[f"invalid_negative_type:{invalid_negative_type}"] += 1
    report = {
        "contrastive_records": len(records),
        "positive_source_distribution": {
            key.removeprefix("positive_source:"): value
            for key, value in stats.items()
            if key.startswith("positive_source:")
        },
        "negative_type_distribution": {
            key.removeprefix("negative_type:"): value
            for key, value in stats.items()
            if key.startswith("negative_type:")
        },
        "negative_source_distribution": {
            key.removeprefix("negative_source:"): value
            for key, value in stats.items()
            if key.startswith("negative_source:")
        },
        "invalid_negative_type_distribution": {
            key.removeprefix("invalid_negative_type:"): value
            for key, value in stats.items()
            if key.startswith("invalid_negative_type:")
        },
    }
    return records, report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_report(training_rows: list[dict[str, str]], contrastive_rows: list[dict[str, Any]], merge_report: dict[str, Any], contrastive_report: dict[str, Any]) -> dict[str, Any]:
    response_counts = Counter(row["response"] for row in training_rows)
    return {
        "training_examples": len(training_rows),
        "training_unique_prompts": len({compact(row["prompt"]) for row in training_rows}),
        "training_unique_responses": len(response_counts),
        "training_multi_action_examples": sum(1 for row in training_rows if "；" in row["response"]),
        "training_top_50_responses": [
            {"response": response, "count": count}
            for response, count in response_counts.most_common(50)
        ],
        "contrastive_examples": len(contrastive_rows),
        **merge_report,
        **contrastive_report,
    }


def main() -> None:
    args = parse_args()
    base_rows = load_rows(Path(args.base).expanduser())
    single_rows = load_rows(Path(args.single).expanduser())
    multi_rows = load_rows(Path(args.multi).expanduser())
    base_positive_map = load_base_positive_map(Path(args.base_contrastive).expanduser())

    training_rows, merge_report = merge_datasets(
        [
            ("base_repaired", base_rows),
            ("single_device_expansion", single_rows),
            ("multi_device_expansion", multi_rows),
        ]
    )
    contrastive_rows, contrastive_report = build_contrastive(training_rows, base_positive_map)
    report = build_report(training_rows, contrastive_rows, merge_report, contrastive_report)

    training_output = Path(args.training_output).expanduser()
    contrastive_output = Path(args.contrastive_output).expanduser()
    report_output = Path(args.report).expanduser()
    write_json(training_output, training_rows)
    write_json(contrastive_output, contrastive_rows)
    write_json(report_output, report)

    print(f"training_examples: {report['training_examples']}")
    print(f"training_unique_prompts: {report['training_unique_prompts']}")
    print(f"training_unique_responses: {report['training_unique_responses']}")
    print(f"training_multi_action_examples: {report['training_multi_action_examples']}")
    print(f"contrastive_examples: {report['contrastive_examples']}")
    print(f"source_counts: {report['source_counts']}")
    print(f"positive_source_distribution: {report['positive_source_distribution']}")
    print(f"negative_type_distribution: {report['negative_type_distribution']}")
    print(f"training_output: {training_output}")
    print(f"contrastive_output: {contrastive_output}")
    print(f"report: {report_output}")


if __name__ == "__main__":
    main()
