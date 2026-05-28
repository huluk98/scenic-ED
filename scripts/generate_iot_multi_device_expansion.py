#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "iot_multi_device_expansion.json"
DEFAULT_BREAKDOWN = PROJECT_ROOT / "generated" / "iot_multi_device_expansion_breakdown.csv"
TARGET_TOTAL = 2000
SCENARIO_COUNT = 30


@dataclass(frozen=True)
class Action:
    device: str
    location: str
    action: str
    fragment: str
    command: str
    target: str
    feature: str
    value: str
    order: int


@dataclass(frozen=True)
class ResponseSpec:
    scenario: str
    actions: tuple[Action, ...]
    response: str


SCENARIO_CUES = {
    "sleep": ("我要睡觉了", "准备睡觉", "晚安", "该休息了", "睡前准备一下", "卧室准备入睡", "我要躺下了", "夜里模式来一下"),
    "leaving_home": ("我要出门了", "家里没人了", "准备离开家", "出门前帮我处理一下", "锁好家里", "离家模式", "没人看家了", "出门了"),
    "movie": ("今晚看电影", "电影夜准备一下", "我想看电影", "客厅调成观影状态", "准备放电影", "把客厅弄适合看电影", "影院感来一下", "电影时间到了"),
    "gaming": ("我要打游戏了", "主机游戏准备一下", "游戏模式", "手柄准备好了", "低延迟模式来一下", "客厅准备开玩", "游戏时间到了", "帮我切到游戏状态"),
    "study": ("我要学习了", "书房准备专注一下", "需要安静学习", "学习模式来一下", "我要看书了", "书房进入专注状态", "别打扰我学习", "准备复习了"),
    "cooking": ("我要做饭了", "厨房开始忙了", "厨房有点呛", "厨房味道有点重", "准备炒菜", "做饭模式来一下", "厨房处理一下", "饭点到了"),
    "guest_arrival": ("客人来了", "有人要来家里", "准备接待客人", "把客厅收拾成会客状态", "朋友快到了", "来客模式", "客厅准备一下", "有人过来了"),
    "humid_weather": ("今天太潮了", "房间有点湿", "梅雨天到了", "空气湿乎乎的", "屋里返潮了", "湿气有点重", "下雨天除除湿", "房间潮得不舒服"),
    "hot_weather": ("太热了", "我都出汗了", "屋里需要降温", "天气热起来了", "帮我凉快一下", "房间闷热", "夏天模式来一下", "快把屋里降降温"),
    "call_meeting": ("我要接电话了", "会议开始了", "保持安静", "我有个线上会议", "电话来了", "先安静一点", "开会模式", "别出声了"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic SCENIC multi-device IoT expansion examples.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--breakdown", default=str(DEFAULT_BREAKDOWN))
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def validate_input(path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item.keys()) != {"prompt", "response"}:
            raise ValueError(f"{path}:{index} must contain exactly prompt and response")
        if not clean_text(item["prompt"]) or not clean_text(item["response"]):
            raise ValueError(f"{path}:{index} has an empty prompt or response")


def name(location: str, device: str) -> str:
    return f"{location}{device}" if location else device


def act(device: str, location: str, action: str, fragment: str, command: str, feature: str, value: str, order: int) -> Action:
    return Action(device, location, action, fragment, command, name(location, device), feature, value, order)


def light_off(loc: str) -> Action:
    return act("灯", loc, "close_light", f"已关闭{name(loc, '灯')}", f"关闭{name(loc, '灯')}", "power", "off", 1)


def light_on(loc: str) -> Action:
    return act("灯", loc, "open_light", f"已打开{name(loc, '灯')}", f"打开{name(loc, '灯')}", "power", "on", 1)


def light_dim(loc: str) -> Action:
    return act("灯", loc, "brightness_down", f"已将{name(loc, '灯')}亮度调低一档", f"把{name(loc, '灯')}调暗一点", "brightness", "down", 3)


def light_mode(loc: str, mode: str) -> Action:
    return act("灯", loc, f"light_{mode}_mode", f"已将{name(loc, '灯')}切换到{mode}模式", f"把{name(loc, '灯')}切换到{mode}模式", "mode", mode, 2)


def ac_on(loc: str) -> Action:
    return act("空调", loc, "open_ac", f"已打开{name(loc, '空调')}", f"打开{name(loc, '空调')}", "power", "on", 1)


def ac_off(loc: str) -> Action:
    return act("空调", loc, "close_ac", f"已关闭{name(loc, '空调')}", f"关闭{name(loc, '空调')}", "power", "off", 1)


def ac_mode(loc: str, mode: str) -> Action:
    return act("空调", loc, f"ac_{mode}_mode", f"已将{name(loc, '空调')}切换到{mode}模式", f"把{name(loc, '空调')}切换到{mode}模式", "mode", mode, 2)


def ac_temp_down(loc: str) -> Action:
    return act("空调", loc, "temperature_down", f"已将{name(loc, '空调')}温度调低一度", f"把{name(loc, '空调')}温度调低一度", "temperature", "down", 3)


def tv_on(loc: str) -> Action:
    return act("电视", loc, "open_tv", f"已打开{name(loc, '电视')}", f"打开{name(loc, '电视')}", "power", "on", 1)


def tv_off(loc: str) -> Action:
    return act("电视", loc, "close_tv", f"已关闭{name(loc, '电视')}", f"关闭{name(loc, '电视')}", "power", "off", 1)


def tv_picture(loc: str, mode: str) -> Action:
    return act("电视", loc, f"tv_{mode}_picture", f"已将{name(loc, '电视')}画面模式设为{mode}", f"把{name(loc, '电视')}画面模式设为{mode}", "picture_mode", mode, 2)


def tv_input(loc: str, source: str) -> Action:
    return act("电视", loc, f"tv_input_{source}", f"已将{name(loc, '电视')}输入源切换到{source}", f"把{name(loc, '电视')}输入源切换到{source}", "input", source, 2)


def tv_mute(loc: str) -> Action:
    return act("电视", loc, "tv_mute", f"已开启{name(loc, '电视')}静音", f"开启{name(loc, '电视')}静音", "mute", "on", 3)


def tv_pause(loc: str) -> Action:
    return act("电视", loc, "tv_pause", f"已暂停{name(loc, '电视')}播放", f"暂停{name(loc, '电视')}播放", "playback", "pause", 3)


def speaker_on(loc: str) -> Action:
    return act("音箱", loc, "open_speaker", f"已打开{name(loc, '音箱')}", f"打开{name(loc, '音箱')}", "power", "on", 1)


def speaker_off(loc: str) -> Action:
    return act("音箱", loc, "close_speaker", f"已关闭{name(loc, '音箱')}", f"关闭{name(loc, '音箱')}", "power", "off", 1)


def speaker_volume_down(loc: str) -> Action:
    return act("音箱", loc, "speaker_volume_down", f"已将{name(loc, '音箱')}音量调低", f"把{name(loc, '音箱')}音量调低", "volume", "down", 3)


def speaker_volume_up(loc: str) -> Action:
    return act("音箱", loc, "speaker_volume_up", f"已将{name(loc, '音箱')}音量调高", f"把{name(loc, '音箱')}音量调高", "volume", "up", 3)


def curtain_open(loc: str) -> Action:
    return act("窗帘", loc, "open_curtain", f"已打开{name(loc, '窗帘')}", f"打开{name(loc, '窗帘')}", "power", "on", 1)


def curtain_close(loc: str) -> Action:
    return act("窗帘", loc, "close_curtain", f"已关闭{name(loc, '窗帘')}", f"关闭{name(loc, '窗帘')}", "power", "off", 1)


def purifier_open(loc: str) -> Action:
    return act("空气净化器", loc, "open_air_purifier", f"已打开{name(loc, '空气净化器')}", f"打开{name(loc, '空气净化器')}", "power", "on", 1)


def purifier_mode(loc: str, mode: str) -> Action:
    return act("空气净化器", loc, f"purifier_{mode}_mode", f"已将{name(loc, '空气净化器')}切换到{mode}模式", f"把{name(loc, '空气净化器')}切换到{mode}模式", "mode", mode, 2)


def dehumidifier_open(loc: str) -> Action:
    return act("除湿机", loc, "open_dehumidifier", f"已打开{name(loc, '除湿机')}", f"打开{name(loc, '除湿机')}", "power", "on", 1)


def fan_open(loc: str) -> Action:
    return act("风扇", loc, "open_fan", f"已打开{name(loc, '风扇')}", f"打开{name(loc, '风扇')}", "power", "on", 1)


def fan_speed_up(loc: str) -> Action:
    return act("风扇", loc, "fan_speed_up", f"已将{name(loc, '风扇')}风速调高", f"把{name(loc, '风扇')}风速调高", "fan_speed", "up", 3)


def smart_plug_close(loc: str) -> Action:
    return act("智能插座", loc, "close_smart_plug", f"已关闭{name(loc, '智能插座')}", f"关闭{name(loc, '智能插座')}", "power", "off", 1)


def door_lock() -> Action:
    return act("门锁", "", "lock_door", "已锁好门", "锁好门", "security", "locked", 4)


def camera_open(loc: str) -> Action:
    return act("摄像头", loc, "open_camera", f"已打开{name(loc, '摄像头')}", f"打开{name(loc, '摄像头')}", "security", "camera_on", 4)


def camera_privacy_on(loc: str) -> Action:
    return act("摄像头", loc, "camera_privacy_on", f"已开启{name(loc, '摄像头')}隐私模式", f"开启{name(loc, '摄像头')}隐私模式", "privacy", "on", 4)


def has_contradiction(actions: tuple[Action, ...]) -> bool:
    by_target_feature: dict[tuple[str, str], str] = {}
    power_state: dict[str, str] = {}
    for action in actions:
        key = (action.target, action.feature)
        previous = by_target_feature.get(key)
        if previous is not None and previous != action.value:
            return True
        by_target_feature[key] = action.value
        if action.feature == "power":
            power_state[action.target] = action.value

    for action in actions:
        if power_state.get(action.target) == "off" and action.feature != "power":
            return True
    return False


def response_from_actions(actions: tuple[Action, ...]) -> str:
    ordered = sorted(enumerate(actions), key=lambda item: (item[1].order, item[0]))
    return "好的，" + "；".join(action.fragment for _, action in ordered) + "。"


def make_spec(scenario: str, actions: tuple[Action, ...]) -> ResponseSpec:
    if len(actions) < 2:
        raise ValueError("multi-action spec must have at least two actions")
    if has_contradiction(actions):
        raise ValueError(f"contradictory action set: {actions}")
    return ResponseSpec(scenario, actions, response_from_actions(actions))


def choose_combos(
    scenario: str,
    pools: list[list[Action]],
    count: int = SCENARIO_COUNT,
    global_seen_responses: set[str] | None = None,
) -> list[ResponseSpec]:
    specs: list[ResponseSpec] = []
    seen_responses: set[str] = set()
    if global_seen_responses is None:
        global_seen_responses = set()
    for pool in pools:
        for size in (2, 3, 4):
            for combo in itertools.combinations(pool, size):
                if has_contradiction(combo):
                    continue
                spec = make_spec(scenario, combo)
                if spec.response in seen_responses or spec.response in global_seen_responses:
                    continue
                specs.append(spec)
                seen_responses.add(spec.response)
                global_seen_responses.add(spec.response)
                if len(specs) == count:
                    return specs
    raise ValueError(f"Only generated {len(specs)} specs for {scenario}; expected {count}")


def build_specs() -> list[ResponseSpec]:
    scenarios: dict[str, list[ResponseSpec]] = {}
    global_seen_responses: set[str] = set()

    scenarios["sleep"] = choose_combos(
        "sleep",
        [
            [
                light_off("卧室"),
                ac_mode("卧室", "舒睡"),
                tv_off("卧室"),
                curtain_close("卧室"),
                light_mode("卧室", "夜灯"),
                speaker_volume_down("卧室"),
                speaker_off("卧室"),
                door_lock(),
                camera_privacy_on("卧室"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["leaving_home"] = choose_combos(
        "leaving_home",
        [
            [
                light_off("客厅"),
                light_off("卧室"),
                ac_off("客厅"),
                ac_off("卧室"),
                tv_off("客厅"),
                speaker_off("客厅"),
                smart_plug_close("客厅"),
                smart_plug_close("厨房"),
                door_lock(),
                camera_open("玄关"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["movie"] = choose_combos(
        "movie",
        [
            [
                tv_on("客厅"),
                tv_picture("客厅", "电影"),
                light_dim("客厅"),
                curtain_close("客厅"),
                speaker_on("客厅"),
                speaker_volume_up("客厅"),
                purifier_open("客厅"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["gaming"] = choose_combos(
        "gaming",
        [
            [tv_on(loc), tv_input(loc, "HDMI2"), tv_picture(loc, "游戏"), light_dim(loc), speaker_on(loc)]
            for loc in ("客厅", "书房")
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["study"] = choose_combos(
        "study",
        [
            [
                light_mode("书房", "学习"),
                tv_off("书房"),
                tv_mute("书房"),
                speaker_volume_down("书房"),
                speaker_off("书房"),
                purifier_open("书房"),
                curtain_open("书房"),
                purifier_mode("书房", "自动"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["cooking"] = choose_combos(
        "cooking",
        [
            [
                light_on("厨房"),
                purifier_open("厨房"),
                fan_open("厨房"),
                fan_speed_up("厨房"),
                speaker_volume_down("厨房"),
                speaker_off("厨房"),
                smart_plug_close("厨房"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["guest_arrival"] = choose_combos(
        "guest_arrival",
        [
            [
                light_on("客厅"),
                ac_on("客厅"),
                purifier_open("客厅"),
                tv_on("客厅"),
                curtain_open("客厅"),
                fan_open("客厅"),
                speaker_on("客厅"),
            ]
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["humid_weather"] = choose_combos(
        "humid_weather",
        [
            [ac_mode(loc, "抽湿"), dehumidifier_open(loc), curtain_close(loc), fan_open(loc), purifier_mode(loc, "自动")]
            for loc in ("客厅", "卧室", "书房")
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["hot_weather"] = choose_combos(
        "hot_weather",
        [
            [ac_on(loc), ac_mode(loc, "制冷"), fan_open(loc), curtain_close(loc), ac_temp_down(loc), fan_speed_up(loc)]
            for loc in ("客厅", "卧室", "书房")
        ],
        global_seen_responses=global_seen_responses,
    )
    scenarios["call_meeting"] = choose_combos(
        "call_meeting",
        [
            [tv_mute(loc), tv_pause(loc), speaker_volume_down(loc), speaker_off(loc), curtain_close(loc), camera_privacy_on(loc)]
            for loc in ("客厅", "书房", "卧室")
        ],
        global_seen_responses=global_seen_responses,
    )

    specs: list[ResponseSpec] = []
    for scenario in SCENARIO_CUES:
        specs.extend(scenarios[scenario])
    if len(specs) != SCENARIO_COUNT * len(SCENARIO_CUES):
        raise ValueError(f"Expected 300 response specs, got {len(specs)}")
    return specs


def prompt_count_plan(specs: list[ResponseSpec], target_total: int) -> dict[ResponseSpec, int]:
    if not 3 * len(specs) <= target_total <= 8 * len(specs):
        raise ValueError(f"Cannot allocate {target_total} examples across {len(specs)} response groups with 3-8 prompts each")
    base = target_total // len(specs)
    remainder = target_total - base * len(specs)
    plan: dict[ResponseSpec, int] = {}
    for index, spec in enumerate(specs):
        plan[spec] = base + (1 if index < remainder else 0)
    return plan


def command_sequence(actions: tuple[Action, ...], joiner: str = "，再") -> str:
    ordered = sorted(enumerate(actions), key=lambda item: (item[1].order, item[0]))
    return joiner.join(action.command for _, action in ordered)


def prompts_for_spec(spec: ResponseSpec, count: int) -> list[str]:
    cues = SCENARIO_CUES[spec.scenario]
    sequence = command_sequence(spec.actions)
    compact_sequence = command_sequence(spec.actions, "并")
    first_location = next((action.location for action in spec.actions if action.location), "")
    devices = "、".join(dict.fromkeys(action.target for action in spec.actions))
    templates = [
        f"{cues[0]}，{sequence}。",
        f"{cues[1]}，帮我{sequence}。",
        f"{sequence}。",
        f"{cues[2]}，请{sequence}。",
        f"{cues[3]}，顺手{sequence}。",
        f"{cues[4]}，{compact_sequence}。",
        f"{cues[5]}，安排一下：{sequence}。",
        f"{cues[6]}，{first_location}这边{sequence}。",
        f"麻烦{sequence}。",
        f"{cues[7]}，需要{compact_sequence}。",
    ]
    selected: list[str] = []
    seen: set[str] = set()
    for prompt in templates:
        prompt = prompt.replace("，。", "。").replace("，这边", "这边")
        if prompt not in seen:
            selected.append(prompt)
            seen.add(prompt)
        if len(selected) == count:
            return selected
    raise ValueError(f"Only created {len(selected)} prompts for {spec.response}")


def validate_spec(spec: ResponseSpec) -> None:
    if not spec.response.startswith("好的，"):
        raise ValueError(f"response does not start with 好的，: {spec.response}")
    if "；" not in spec.response:
        raise ValueError(f"multi-action response missing semicolon: {spec.response}")
    if any(token in spec.response for token in ("- ", "{", "}", "[", "]")):
        raise ValueError(f"response has unsupported formatting: {spec.response}")
    if has_contradiction(spec.actions):
        raise ValueError(f"contradictory response spec: {spec.response}")


def generate_dataset(specs: list[ResponseSpec], target_total: int) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    plan = prompt_count_plan(specs, target_total)
    examples: list[dict[str, str]] = []
    breakdown: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for spec in specs:
        validate_spec(spec)
        prompts = prompts_for_spec(spec, plan[spec])
        for prompt in prompts:
            pair = (prompt, spec.response)
            if pair in seen_pairs:
                raise ValueError(f"duplicate prompt-response pair: {pair}")
            seen_pairs.add(pair)
            examples.append({"prompt": prompt, "response": spec.response})
        breakdown.append(
            {
                "scenario": spec.scenario,
                "devices": "|".join(dict.fromkeys(action.target for action in spec.actions)),
                "actions": "|".join(action.action for action in spec.actions),
                "response": spec.response,
                "prompt_count": len(prompts),
            }
        )

    response_counts = Counter(item["response"] for item in examples)
    bad_counts = [response for response, count in response_counts.items() if count < 3 or count > 8]
    if bad_counts:
        raise ValueError(f"response groups outside 3-8 prompt range: {bad_counts[:5]}")
    if len(examples) != target_total:
        raise ValueError(f"Generated {len(examples)} examples, expected {target_total}")
    return examples, breakdown


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_breakdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "devices", "actions", "response", "prompt_count"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(examples: list[dict[str, str]], breakdown: list[dict[str, Any]], output: Path, breakdown_path: Path) -> None:
    response_counts = Counter(item["response"] for item in examples)
    response_to_breakdown = {row["response"]: row for row in breakdown}
    per_scenario: Counter[str] = Counter()
    per_device_combo: Counter[str] = Counter()
    for row in breakdown:
        count = int(row["prompt_count"])
        per_scenario[str(row["scenario"])] += count
        per_device_combo[str(row["devices"])] += count

    print(f"total_examples: {len(examples)}")
    print(f"total_unique_responses: {len(response_counts)}")
    print(f"output: {output}")
    print(f"breakdown_csv: {breakdown_path}")
    print("examples_per_scenario:")
    for scenario, count in per_scenario.items():
        print(f"- {scenario}: {count}")
    print("examples_per_device_combination:")
    for combo, count in per_device_combo.most_common(30):
        print(f"- {combo}: {count}")
    print("top_20_many_to_one_response_groups:")
    for response, count in response_counts.most_common(20):
        row = response_to_breakdown[response]
        print(f"- {count} | {row['scenario']} | {row['devices']} | {response}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    breakdown_path = Path(args.breakdown).expanduser()

    validate_input(input_path)
    specs = build_specs()
    examples, breakdown = generate_dataset(specs, args.target_total)
    write_json(output_path, examples)
    write_breakdown(breakdown_path, breakdown)
    print_summary(examples, breakdown, output_path, breakdown_path)


if __name__ == "__main__":
    main()
