#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clean_smart_home_dataset import infer_intents_from_pair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_BREAKDOWN = PROJECT_ROOT / "reports" / "iot_dataset_breakdown.json"
DEFAULT_RESPONSE_CSV = PROJECT_ROOT / "reports" / "iot_response_action_breakdown.csv"
DEFAULT_MANY_TO_ONE = PROJECT_ROOT / "reports" / "iot_many_to_one_examples.json"

CURRENT_DEVICES = ("空调", "灯", "电视", "音箱")
EXPANSION_DEVICES = ("窗帘", "扫地机器人", "空气净化器", "加湿器", "除湿机", "热水器", "风扇", "门锁", "摄像头", "智能插座")
ALL_DEVICE_CATEGORIES = CURRENT_DEVICES + EXPANSION_DEVICES

ACTION_CATEGORIES = (
    "open_device",
    "close_device",
    "mode_switch",
    "brightness_up",
    "brightness_down",
    "brightness_max",
    "brightness_min",
    "color_temperature",
    "temperature_up",
    "temperature_down",
    "temperature_set",
    "fan_speed_up",
    "fan_speed_down",
    "fan_direction",
    "timer_on",
    "timer_off",
    "tv_channel",
    "tv_input_source",
    "tv_picture_mode",
    "tv_subtitle",
    "tv_mute",
    "tv_volume",
    "tv_playback",
    "speaker_playback",
    "speaker_volume",
    "multi_device_action",
    "unknown",
)

INTENT_TO_ACTION = {
    "light_on": "open_device",
    "ac_on": "open_device",
    "tv_on": "open_device",
    "speaker_on": "open_device",
    "light_off": "close_device",
    "ac_off": "close_device",
    "tv_off": "close_device",
    "speaker_off": "close_device",
    "light_mode": "mode_switch",
    "ac_mode": "mode_switch",
    "light_brightness_up": "brightness_up",
    "light_brightness_down": "brightness_down",
    "light_brightest": "brightness_max",
    "light_darkest": "brightness_min",
    "light_color": "color_temperature",
    "ac_temp_up": "temperature_up",
    "ac_temp_down": "temperature_down",
    "ac_temp_set": "temperature_set",
    "ac_fan_up": "fan_speed_up",
    "ac_fan_down": "fan_speed_down",
    "ac_wind_direction": "fan_direction",
    "tv_channel": "tv_channel",
    "tv_input": "tv_input_source",
    "tv_picture_mode": "tv_picture_mode",
    "tv_subtitle": "tv_subtitle",
    "tv_mute": "tv_mute",
    "tv_volume_up": "tv_volume",
    "tv_volume_down": "tv_volume",
    "tv_play": "tv_playback",
    "tv_pause": "tv_playback",
    "tv_next_channel": "tv_channel",
    "tv_previous_channel": "tv_channel",
    "speaker_play": "speaker_playback",
    "speaker_pause": "speaker_playback",
    "speaker_next": "speaker_playback",
    "speaker_previous": "speaker_playback",
    "speaker_volume_up": "speaker_volume",
    "speaker_volume_down": "speaker_volume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the repaired SCENIC smart-home IoT dataset taxonomy.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--breakdown", default=str(DEFAULT_BREAKDOWN))
    parser.add_argument("--response-csv", default=str(DEFAULT_RESPONSE_CSV))
    parser.add_argument("--many-to-one", default=str(DEFAULT_MANY_TO_ONE))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_prompt(prompt: str) -> str:
    return "".join(clean_text(prompt).split())


def load_and_validate(path: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")

    rows: list[dict[str, str]] = []
    validation_errors: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            validation_errors.append({"index": index, "error": "item_is_not_object"})
            continue
        keys = set(item.keys())
        if keys != {"prompt", "response"}:
            validation_errors.append({"index": index, "error": "keys_not_exactly_prompt_response", "keys": sorted(keys)})
            continue
        prompt = clean_text(item["prompt"])
        response = clean_text(item["response"])
        if not isinstance(item["prompt"], str) or not isinstance(item["response"], str):
            validation_errors.append({"index": index, "error": "prompt_or_response_not_string"})
            continue
        if not prompt or not response:
            validation_errors.append({"index": index, "error": "empty_prompt_or_response"})
            continue
        rows.append({"prompt": prompt, "response": response})

    if validation_errors:
        preview = json.dumps(validation_errors[:20], ensure_ascii=False, indent=2)
        raise ValueError(f"Dataset validation failed with {len(validation_errors)} errors. First errors:\n{preview}")
    return rows, validation_errors


def action_for_intent(intent: dict[str, Any]) -> str:
    if intent["intent"] == "ac_timer":
        return "timer_on" if intent["slots"].get("timer_action") == "开启" else "timer_off"
    return INTENT_TO_ACTION.get(intent["intent"], "unknown")


def classify_row(row: dict[str, str]) -> dict[str, Any]:
    intents = infer_intents_from_pair(row["prompt"], row["response"])
    devices = sorted({intent["device"] for intent in intents}) if intents else ["unknown"]
    actions = sorted({action_for_intent(intent) for intent in intents}) if intents else ["unknown"]
    locations = sorted({intent["location"] or "未指定" for intent in intents}) if intents else ["unknown"]
    is_multi_action = len(intents) > 1 or "；" in row["response"]
    if is_multi_action and "multi_device_action" not in actions:
        actions.append("multi_device_action")
    return {
        "intents": intents,
        "devices": devices,
        "actions": sorted(actions),
        "locations": locations,
        "is_multi_action": is_multi_action,
    }


def seeded_counter(keys: tuple[str, ...]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for key in keys:
        counter[key] = 0
    return counter


def analyze(rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    response_counts = Counter(row["response"] for row in rows)
    prompt_counts = Counter(normalize_prompt(row["prompt"]) for row in rows)
    response_to_prompts: dict[str, list[str]] = defaultdict(list)
    response_to_indices: dict[str, list[int]] = defaultdict(list)
    response_classes: dict[str, dict[str, Any]] = {}

    device_distribution = seeded_counter(ALL_DEVICE_CATEGORIES)
    action_distribution = seeded_counter(ACTION_CATEGORIES)
    location_distribution: Counter[str] = Counter({"未指定": 0, "客厅": 0, "卧室": 0, "书房": 0, "unknown": 0})
    row_classifications: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    multi_action_count = 0

    for index, row in enumerate(rows):
        classification = classify_row(row)
        row_classifications.append(classification)
        response_to_prompts[row["response"]].append(row["prompt"])
        response_to_indices[row["response"]].append(index)
        if not classification["intents"]:
            unresolved.append({"index": index, "prompt": row["prompt"], "response": row["response"]})
        if classification["is_multi_action"]:
            multi_action_count += 1

        for device in classification["devices"]:
            device_distribution[device] += 1
        for action in classification["actions"]:
            action_distribution[action] += 1
        for location in classification["locations"]:
            location_distribution[location] += 1

    for response, prompts in response_to_prompts.items():
        indices = response_to_indices[response]
        devices: set[str] = set()
        actions: set[str] = set()
        locations: set[str] = set()
        is_multi_action = False
        for index in indices:
            classification = row_classifications[index]
            devices.update(classification["devices"])
            actions.update(classification["actions"])
            locations.update(classification["locations"])
            is_multi_action = is_multi_action or classification["is_multi_action"]
        response_classes[response] = {
            "response": response,
            "count": len(prompts),
            "devices": sorted(devices),
            "actions": sorted(actions),
            "locations": sorted(locations),
            "is_multi_action": is_multi_action,
            "sample_prompts": prompts[:10],
        }

    many_to_one_examples = [
        {
            "response": response,
            "count": response_counts[response],
            "devices": response_classes[response]["devices"],
            "actions": response_classes[response]["actions"],
            "locations": response_classes[response]["locations"],
            "sample_prompts": response_to_prompts[response][:20],
        }
        for response, count in response_counts.most_common()
        if count > 1
    ]
    unique_multi_action_responses = sum(1 for data in response_classes.values() if data["is_multi_action"])

    response_rows = [
        {
            "response": response,
            "count": data["count"],
            "devices": "|".join(data["devices"]),
            "actions": "|".join(data["actions"]),
            "locations": "|".join(data["locations"]),
            "is_multi_action": data["is_multi_action"],
            "sample_prompts": " || ".join(data["sample_prompts"][:5]),
        }
        for response, data in sorted(response_classes.items(), key=lambda item: (-item[1]["count"], item[0]))
    ]

    breakdown = {
        "input_file": str(DEFAULT_INPUT),
        "total_examples": len(rows),
        "unique_prompts": len(prompt_counts),
        "duplicate_prompt_count": sum(count - 1 for count in prompt_counts.values() if count > 1),
        "unique_responses": len(response_counts),
        "multi_action_response_count": multi_action_count,
        "unique_multi_action_response_count": unique_multi_action_responses,
        "unresolved_intent_count": len(unresolved),
        "top_50_responses": [{"response": response, "count": count} for response, count in response_counts.most_common(50)],
        "device_distribution": dict(device_distribution),
        "action_distribution": dict(action_distribution),
        "location_distribution": dict(location_distribution),
        "many_to_one_response_count": len(many_to_one_examples),
        "largest_many_to_one_examples": many_to_one_examples[:50],
        "current_device_categories": list(CURRENT_DEVICES),
        "expansion_device_categories": list(EXPANSION_DEVICES),
        "action_categories": list(ACTION_CATEGORIES),
        "unresolved_examples": unresolved[:50],
    }
    return breakdown, response_rows, many_to_one_examples


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_response_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["response", "count", "devices", "actions", "locations", "is_multi_action", "sample_prompts"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    breakdown_path = Path(args.breakdown).expanduser()
    response_csv_path = Path(args.response_csv).expanduser()
    many_to_one_path = Path(args.many_to_one).expanduser()

    rows, _ = load_and_validate(input_path)
    breakdown, response_rows, many_to_one_examples = analyze(rows)
    breakdown["input_file"] = str(input_path)

    write_json(breakdown_path, breakdown)
    write_response_csv(response_csv_path, response_rows)
    write_json(many_to_one_path, many_to_one_examples)

    print(f"input_file: {input_path}")
    print(f"total_examples: {breakdown['total_examples']}")
    print(f"unique_prompts: {breakdown['unique_prompts']}")
    print(f"unique_responses: {breakdown['unique_responses']}")
    print(f"multi_action_response_count: {breakdown['multi_action_response_count']}")
    print(f"unique_multi_action_response_count: {breakdown['unique_multi_action_response_count']}")
    print(f"many_to_one_response_count: {breakdown['many_to_one_response_count']}")
    print(f"unresolved_intent_count: {breakdown['unresolved_intent_count']}")
    print("device_distribution:")
    for device, count in breakdown["device_distribution"].items():
        print(f"- {device}: {count}")
    print("top_actions:")
    action_counts = Counter(breakdown["action_distribution"])
    for action, count in action_counts.most_common(10):
        print(f"- {action}: {count}")
    print(f"breakdown_report: {breakdown_path}")
    print(f"response_action_csv: {response_csv_path}")
    print(f"many_to_one_examples: {many_to_one_path}")


if __name__ == "__main__":
    main()
