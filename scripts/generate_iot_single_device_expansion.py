#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "iot_single_device_expansion.json"
DEFAULT_BREAKDOWN = PROJECT_ROOT / "generated" / "iot_single_device_expansion_breakdown.csv"
TARGET_TOTAL = 3000

LOCATIONS = ("客厅", "卧室", "书房", "厨房", "浴室", "阳台", "玄关")
SMART_PLUG_TIMER_VALUES = ("15分钟后", "晚上10点")


@dataclass(frozen=True)
class Spec:
    device: str
    location: str
    action: str
    response: str
    time: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic SCENIC IoT single-device expansion examples.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--breakdown", default=str(DEFAULT_BREAKDOWN))
    parser.add_argument("--target-total", type=int, default=TARGET_TOTAL)
    return parser.parse_args()


def load_and_validate_input(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"prompt", "response"}:
            raise ValueError(f"{path}:{index} must contain exactly prompt and response")
        if not isinstance(item["prompt"], str) or not isinstance(item["response"], str):
            raise ValueError(f"{path}:{index} prompt and response must be strings")
        rows.append({"prompt": item["prompt"].strip(), "response": item["response"].strip()})
    return rows


def target(location: str, device: str) -> str:
    return f"{location}{device}" if location else device


def response_for(device: str, location: str, action: str, time: str = "") -> str:
    name = target(location, device)
    if device == "窗帘":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "open_half": f"好的，已将{name}开到一半。",
            "close_half": f"好的，已将{name}关闭一半。",
        }[action]
    if device == "扫地机器人":
        return {
            "start_clean": f"好的，已启动{name}清扫。",
            "pause_clean": f"好的，已暂停{name}清扫。",
            "return_charge": f"好的，已让{name}返回充电。",
            "clean_room": f"好的，已让扫地机器人清扫{location}。",
        }[action]
    if device == "空气净化器":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "auto_mode": f"好的，已将{name}切换到自动模式。",
            "sleep_mode": f"好的，已将{name}切换到睡眠模式。",
            "fan_up": f"好的，已将{name}风速调高。",
            "fan_down": f"好的，已将{name}风速调低。",
        }[action]
    if device == "加湿器":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "humidity_up": f"好的，已将{name}加湿量调高。",
            "humidity_down": f"好的，已将{name}加湿量调低。",
        }[action]
    if device == "除湿机":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "dehumidify_up": f"好的，已将{name}除湿强度调高。",
            "dehumidify_down": f"好的，已将{name}除湿强度调低。",
        }[action]
    if device == "热水器":
        return {
            "open": "好的，已打开热水器。",
            "close": "好的，已关闭热水器。",
            "temp_up": "好的，已将热水器温度调高。",
            "temp_down": "好的，已将热水器温度调低。",
            "set_45": "好的，已将热水器温度设置为45度。",
            "set_50": "好的，已将热水器温度设置为50度。",
        }[action]
    if device == "风扇":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "fan_up": f"好的，已将{name}风速调高。",
            "fan_down": f"好的，已将{name}风速调低。",
            "oscillate_on": f"好的，已开启{name}摇头。",
            "oscillate_off": f"好的，已关闭{name}摇头。",
        }[action]
    if device == "门锁":
        return {
            "lock": "好的，已锁好门。",
            "unlock": "好的，已解锁门。",
            "check": "好的，已检查门锁状态。",
        }[action]
    if device == "摄像头":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "privacy_on": f"好的，已开启{name}隐私模式。",
            "privacy_off": f"好的，已关闭{name}隐私模式。",
            "record_start": f"好的，已开始{name}录像。",
            "record_stop": f"好的，已停止{name}录像。",
        }[action]
    if device == "智能插座":
        return {
            "open": f"好的，已打开{name}。",
            "close": f"好的，已关闭{name}。",
            "timer_open": f"好的，已为{name}设置{time}开启。",
            "timer_close": f"好的，已为{name}设置{time}关闭。",
        }[action]
    raise ValueError(f"Unsupported spec: {device} {action}")


def build_specs() -> list[Spec]:
    specs: list[Spec] = []
    device_actions = {
        "窗帘": ("open", "close", "open_half", "close_half"),
        "扫地机器人": ("start_clean", "pause_clean", "return_charge", "clean_room"),
        "空气净化器": ("open", "close", "auto_mode", "sleep_mode", "fan_up", "fan_down"),
        "加湿器": ("open", "close", "humidity_up", "humidity_down"),
        "除湿机": ("open", "close", "dehumidify_up", "dehumidify_down"),
        "风扇": ("open", "close", "fan_up", "fan_down", "oscillate_on", "oscillate_off"),
        "摄像头": ("open", "close", "privacy_on", "privacy_off", "record_start", "record_stop"),
    }
    for device, actions in device_actions.items():
        for location in LOCATIONS:
            for action in actions:
                specs.append(Spec(device, location, action, response_for(device, location, action)))

    for action in ("open", "close", "temp_up", "temp_down", "set_45", "set_50"):
        specs.append(Spec("热水器", "浴室", action, response_for("热水器", "", action)))
    for action in ("lock", "unlock", "check"):
        specs.append(Spec("门锁", "玄关", action, response_for("门锁", "", action)))
    for location in LOCATIONS:
        for action in ("open", "close"):
            specs.append(Spec("智能插座", location, action, response_for("智能插座", location, action)))
        for time in SMART_PLUG_TIMER_VALUES:
            for action in ("timer_open", "timer_close"):
                specs.append(Spec("智能插座", location, action, response_for("智能插座", location, action, time), time))
    return specs


def curtain_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}拉开。",
            f"帮我把{name}打开吧。",
            f"{loc}有点暗，窗帘拉开。",
            f"想透透光，把{name}打开。",
            f"白天了，{name}开一下。",
            f"客人来了，把{name}拉开。",
            f"{loc}窗帘打开一下。",
            f"让{loc}亮一点，窗帘开开。",
            f"把{name}开起来。",
            f"晒点太阳，{name}拉开。",
            f"{loc}光线不够，把窗帘打开。",
            f"帮忙拉开{name}。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}拉上。",
            f"帮我把{name}关上吧。",
            f"外面太亮了，{name}拉上。",
            f"我要休息了，{name}关上。",
            f"{loc}窗帘关闭一下。",
            f"睡觉前把{name}拉好。",
            f"太阳太晒了，把{name}关上。",
            f"{loc}需要暗一点，窗帘拉上。",
            f"把{name}合上。",
            f"别让外面看进来，{name}关上。",
            f"帮忙关闭{name}。",
            f"{loc}窗帘收一下。",
        ],
        "open_half": [
            f"把{name}开到一半。",
            f"{name}拉开一半。",
            f"窗帘别全开，{loc}开一半。",
            f"{loc}透点光就行，窗帘开一半。",
            f"帮我把{name}半开。",
            f"{name}打开一半吧。",
            f"让{loc}稍微亮点，窗帘开一半。",
            f"{loc}窗帘开半边。",
            f"{name}拉到半开。",
            f"不要全开，{name}开到一半。",
            f"把{name}留一半光。",
            f"{loc}窗帘半开一下。",
            f"帮忙把{name}调到半开。",
        ],
        "close_half": [
            f"把{name}关闭一半。",
            f"{name}拉上一半。",
            f"窗帘别全关，{loc}关一半。",
            f"{loc}光有点强，窗帘关一半。",
            f"帮我把{name}半关。",
            f"{name}收上一半吧。",
            f"留点光，把{name}关一半。",
            f"{loc}窗帘拉回一半。",
            f"{name}合上一半。",
            f"不要全拉上，{name}关到一半。",
            f"把{name}遮一半。",
            f"{loc}窗帘半关一下。",
            f"帮忙把{name}调到半关。",
        ],
    }[spec.action]


def vacuum_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "start_clean": [
            f"启动{name}清扫。",
            f"让{name}开始扫地。",
            f"{name}开始工作。",
            f"帮我启动{name}。",
            f"{loc}扫地机器人开一下。",
            f"{loc}地面该清理了，机器人启动。",
            f"开始{name}清扫吧。",
            f"{name}去扫一下。",
            f"把{name}开起来清扫。",
            f"{loc}有点脏，扫地机器人启动。",
            f"让{name}处理地面。",
            f"清扫开始，{name}。",
            f"帮忙让{name}扫地。",
        ],
        "pause_clean": [
            f"暂停{name}清扫。",
            f"让{name}先停一下。",
            f"{name}暂停工作。",
            f"先别扫了，{name}暂停。",
            f"帮我暂停{name}。",
            f"{loc}扫地机器人停一下。",
            f"{name}清扫先暂停。",
            f"有人走动，{name}先停。",
            f"把{name}暂停一下。",
            f"别继续扫了，{name}停住。",
            f"{loc}清扫暂停。",
            f"先暂停{name}的清扫。",
            f"让{name}等一下。",
        ],
        "return_charge": [
            f"让{name}返回充电。",
            f"{name}回去充电。",
            f"让{name}回充。",
            f"清扫结束，{name}回充吧。",
            f"帮我让{name}回充电座。",
            f"{loc}扫地机器人回去充电。",
            f"{name}别扫了，回充。",
            f"电量留着，{name}返回充电。",
            f"把{name}送回充电座。",
            f"让{name}回到充电位。",
            f"{loc}机器人回充一下。",
            f"请{name}返回充电。",
            f"让{name}结束并回充。",
        ],
        "clean_room": [
            f"让扫地机器人清扫{loc}。",
            f"清扫{loc}。",
            f"{loc}地上有灰，处理一下。",
            f"让扫地机器人去{loc}扫一下。",
            f"帮我把{loc}地面清理一下。",
            f"{loc}有点脏，安排扫地机器人。",
            f"扫一下{loc}吧。",
            f"{loc}地面需要清扫。",
            f"让机器人重点清扫{loc}。",
            f"把{loc}扫干净。",
            f"扫地机器人去{loc}工作。",
            f"请清理{loc}地面。",
            f"{loc}来一次清扫。",
        ],
    }[spec.action]


def air_purifier_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"{loc}空气有点闷，净化一下。",
            f"客人来了，{name}开一下。",
            f"{loc}有点味道，打开空气净化器。",
            f"帮我开启{name}。",
            f"{loc}空气不太好，净化器开开。",
            f"让{name}运行。",
            f"把{name}启动起来。",
            f"{loc}需要换换空气。",
            f"空气有点浑，{name}打开。",
            f"请打开{name}。",
            f"{loc}空气净化一下。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"{loc}空气净化够了，关一下。",
            f"不用{name}了。",
            f"帮我关闭{name}。",
            f"{name}先停一下。",
            f"{loc}净化器关了吧。",
            f"把{name}停掉。",
            f"空气可以了，{name}关闭。",
            f"请关掉{name}。",
            f"{loc}空气净化器不用开了。",
            f"让{name}停止运行。",
            f"{name}关机。",
        ],
        "auto_mode": [
            f"把{name}切换到自动模式。",
            f"{name}设成自动模式。",
            f"让{name}自动调节。",
            f"{loc}空气情况自己判断，净化器自动。",
            f"帮我把{name}调到自动。",
            f"{name}进入自动模式。",
            f"自动模式，{name}。",
            f"{loc}净化器用自动档。",
            f"把{name}改成自动运行。",
            f"请将{name}设为自动。",
            f"让{name}自己控制风量。",
            f"{loc}空气净化器自动运行。",
            f"{name}自动一下。",
        ],
        "sleep_mode": [
            f"把{name}切换到睡眠模式。",
            f"{name}设成睡眠模式。",
            f"晚上了，{name}调睡眠。",
            f"睡觉前把{name}调安静。",
            f"帮我把{name}切到睡眠。",
            f"{name}进入睡眠模式。",
            f"睡眠模式，{name}。",
            f"{loc}净化器声音小一点，用睡眠模式。",
            f"把{name}改成睡眠运行。",
            f"请将{name}设为睡眠。",
            f"休息了，{name}睡眠模式。",
            f"{loc}空气净化器安静运行。",
            f"{name}夜里用睡眠模式。",
        ],
        "fan_up": [
            f"把{name}风速调高。",
            f"{name}风量大一点。",
            f"{loc}味道有点重，净化器风速调高。",
            f"帮我提高{name}风速。",
            f"{name}开大一点。",
            f"{loc}空气差，净化器加大风量。",
            f"{name}风速升一档。",
            f"让{name}净化快一点。",
            f"把{name}风调大。",
            f"请调高{name}风速。",
            f"{loc}净化器强一点。",
            f"{name}多吹一点。",
            f"{loc}空气净化器风速加大。",
        ],
        "fan_down": [
            f"把{name}风速调低。",
            f"{name}风量小一点。",
            f"{loc}净化器有点吵，风速调低。",
            f"帮我降低{name}风速。",
            f"{name}开小一点。",
            f"{loc}空气可以了，净化器减小风量。",
            f"{name}风速降一档。",
            f"让{name}安静一点。",
            f"把{name}风调小。",
            f"请调低{name}风速。",
            f"{loc}净化器弱一点。",
            f"{name}少吹一点。",
            f"{loc}空气净化器风速降低。",
        ],
    }[spec.action]


def humidifier_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"{loc}太干了，开加湿器。",
            f"房间有点干，{name}打开。",
            f"帮我开启{name}。",
            f"{loc}空气干，增加点湿度。",
            f"让{name}运行。",
            f"嘴唇有点干，{loc}加湿器开开。",
            f"把{name}启动起来。",
            f"{loc}需要加湿。",
            f"请打开{name}。",
            f"加点湿气，{name}打开。",
            f"{loc}干得不舒服，开加湿器。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"{loc}湿度够了，关加湿器。",
            f"不用{name}了。",
            f"帮我关闭{name}。",
            f"{name}先停一下。",
            f"{loc}加湿器关了吧。",
            f"把{name}停掉。",
            f"空气不干了，{name}关闭。",
            f"请关掉{name}。",
            f"{loc}加湿器不用开了。",
            f"让{name}停止运行。",
            f"{name}关机。",
        ],
        "humidity_up": [
            f"把{name}加湿量调高。",
            f"{name}加湿开大一点。",
            f"{loc}还是有点干，加湿量调高。",
            f"帮我提高{name}加湿量。",
            f"{name}雾量大一点。",
            f"{loc}湿度再高一点。",
            f"{name}加湿强一点。",
            f"让{name}多加湿。",
            f"把{name}档位调高。",
            f"请调高{name}加湿量。",
            f"{loc}加湿器开强一点。",
            f"加湿多一点，{name}。",
            f"{loc}空气再润一点。",
        ],
        "humidity_down": [
            f"把{name}加湿量调低。",
            f"{name}加湿开小一点。",
            f"{loc}有点潮，加湿量调低。",
            f"帮我降低{name}加湿量。",
            f"{name}雾量小一点。",
            f"{loc}湿度别太高。",
            f"{name}加湿弱一点。",
            f"让{name}少加湿。",
            f"把{name}档位调低。",
            f"请调低{name}加湿量。",
            f"{loc}加湿器开轻一点。",
            f"加湿少一点，{name}。",
            f"{loc}空气别太湿。",
        ],
    }[spec.action]


def dehumidifier_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"{loc}太潮了，开除湿机。",
            f"墙边有点潮，{name}打开。",
            f"帮我开启{name}。",
            f"{loc}湿气重，除湿一下。",
            f"让{name}运行。",
            f"空气黏糊糊的，{loc}除湿机开开。",
            f"把{name}启动起来。",
            f"{loc}需要除湿。",
            f"请打开{name}。",
            f"去点湿气，{name}打开。",
            f"{loc}潮得不舒服，开除湿机。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"{loc}不潮了，关除湿机。",
            f"不用{name}了。",
            f"帮我关闭{name}。",
            f"{name}先停一下。",
            f"{loc}除湿机关了吧。",
            f"把{name}停掉。",
            f"湿气差不多了，{name}关闭。",
            f"请关掉{name}。",
            f"{loc}除湿机不用开了。",
            f"让{name}停止运行。",
            f"{name}关机。",
        ],
        "dehumidify_up": [
            f"把{name}除湿强度调高。",
            f"{name}除湿开强一点。",
            f"{loc}还是潮，除湿强度调高。",
            f"帮我提高{name}除湿强度。",
            f"{name}除湿力度大一点。",
            f"{loc}湿气再降一点。",
            f"{name}除湿强一点。",
            f"让{name}多除湿。",
            f"把{name}档位调高。",
            f"请调高{name}除湿强度。",
            f"{loc}除湿机开强一点。",
            f"除湿快一点，{name}。",
            f"{loc}水汽重，除湿加大。",
        ],
        "dehumidify_down": [
            f"把{name}除湿强度调低。",
            f"{name}除湿开小一点。",
            f"{loc}已经不潮了，除湿强度调低。",
            f"帮我降低{name}除湿强度。",
            f"{name}除湿力度小一点。",
            f"{loc}别太干。",
            f"{name}除湿弱一点。",
            f"让{name}少除湿。",
            f"把{name}档位调低。",
            f"请调低{name}除湿强度。",
            f"{loc}除湿机开轻一点。",
            f"除湿慢一点，{name}。",
            f"{loc}湿度够了，除湿减弱。",
        ],
    }[spec.action]


def water_heater_prompts(spec: Spec) -> list[str]:
    return {
        "open": [
            "打开热水器。",
            "把热水器开一下。",
            "洗澡前把热水器打开。",
            "准备洗澡了，热水器开一下。",
            "帮我开启热水器。",
            "热水器启动。",
            "一会儿要用热水，先开热水器。",
            "浴室要用热水了。",
            "把热水器打开吧。",
            "请打开热水器。",
            "待会儿洗澡，热水器先开。",
            "让热水器开始工作。",
            "热水准备一下。",
        ],
        "close": [
            "关闭热水器。",
            "把热水器关掉。",
            "洗完澡了，关热水器。",
            "不用热水了，热水器关闭。",
            "帮我关闭热水器。",
            "热水器停一下。",
            "把热水器关了吧。",
            "请关掉热水器。",
            "热水器不用开了。",
            "让热水器停止运行。",
            "洗澡结束，热水器关掉。",
            "热水器关机。",
            "不需要热水了。",
        ],
        "temp_up": [
            "把热水器温度调高。",
            "热水器温度高一点。",
            "水有点凉，热水器调高。",
            "帮我提高热水器温度。",
            "热水再热一点。",
            "热水器升温。",
            "把热水器调热一点。",
            "请调高热水器温度。",
            "洗澡水不够热。",
            "热水器温度往上调。",
            "水温再高一点。",
            "热水器加热一点。",
            "把热水调热些。",
        ],
        "temp_down": [
            "把热水器温度调低。",
            "热水器温度低一点。",
            "水有点烫，热水器调低。",
            "帮我降低热水器温度。",
            "热水别太烫。",
            "热水器降温。",
            "把热水器调凉一点。",
            "请调低热水器温度。",
            "洗澡水太热了。",
            "热水器温度往下调。",
            "水温低一点。",
            "热水器少加热一点。",
            "把热水调温和些。",
        ],
        "set_45": [
            "把热水器设置到45度。",
            "热水器温度设为45度。",
            "洗澡水调到45度。",
            "帮我把热水器调成45度。",
            "热水器45度就行。",
            "请设置热水器45度。",
            "把水温定到45度。",
            "热水器调到45度吧。",
            "浴室水温用45度。",
            "热水器设45度。",
            "水温45度。",
            "一会儿洗澡，热水器45度。",
            "把热水器温度固定45度。",
        ],
        "set_50": [
            "把热水器设置到50度。",
            "热水器温度设为50度。",
            "洗澡水调到50度。",
            "帮我把热水器调成50度。",
            "热水器50度就行。",
            "请设置热水器50度。",
            "把水温定到50度。",
            "热水器调到50度吧。",
            "浴室水温用50度。",
            "热水器设50度。",
            "水温50度。",
            "需要热点水，热水器50度。",
            "把热水器温度固定50度。",
        ],
    }[spec.action]


def fan_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"{loc}有点热，开风扇。",
            f"帮我开启{name}。",
            f"{loc}闷得慌，风扇打开。",
            f"让{name}转起来。",
            f"把{name}启动。",
            f"需要点风，{name}开开。",
            f"{loc}风扇开一下吧。",
            f"请打开{name}。",
            f"{loc}通通风，风扇开。",
            f"把{name}开起来。",
            f"{loc}太热了，风扇来点风。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"{loc}不热了，关风扇。",
            f"不用{name}了。",
            f"帮我关闭{name}。",
            f"{name}先停一下。",
            f"{loc}风扇关了吧。",
            f"把{name}停掉。",
            f"风有点大，{name}关闭。",
            f"请关掉{name}。",
            f"{loc}风扇不用开了。",
            f"让{name}停止转动。",
            f"{name}关机。",
        ],
        "fan_up": [
            f"把{name}风速调高。",
            f"{name}风大一点。",
            f"{loc}还是热，风扇调高。",
            f"帮我提高{name}风速。",
            f"{name}开强一点。",
            f"风再大点，{name}。",
            f"{name}风速升一档。",
            f"让{name}吹强一点。",
            f"把{name}风调大。",
            f"请调高{name}风速。",
            f"{loc}风扇强一点。",
            f"{name}多吹一点。",
            f"{loc}风扇风速加大。",
        ],
        "fan_down": [
            f"把{name}风速调低。",
            f"{name}风小一点。",
            f"{loc}有点冷，风扇调低。",
            f"帮我降低{name}风速。",
            f"{name}开轻一点。",
            f"风别太大，{name}。",
            f"{name}风速降一档。",
            f"让{name}吹弱一点。",
            f"把{name}风调小。",
            f"请调低{name}风速。",
            f"{loc}风扇弱一点。",
            f"{name}少吹一点。",
            f"{loc}风扇风速降低。",
        ],
        "oscillate_on": [
            f"开启{name}摇头。",
            f"让{name}摇头。",
            f"{loc}风扇左右摆起来。",
            f"帮我打开{name}摇头。",
            f"{name}开始摇头。",
            f"让风吹均匀点，{name}摇头。",
            f"{loc}风扇摆风。",
            f"{name}打开摆头。",
            f"请开启{name}摇头。",
            f"{loc}风扇转头吹。",
            f"把{name}摇头功能打开。",
            f"{name}左右扫风。",
            f"让{name}摆起来。",
        ],
        "oscillate_off": [
            f"关闭{name}摇头。",
            f"别让{name}摇头。",
            f"{loc}风扇固定吹。",
            f"帮我关闭{name}摇头。",
            f"{name}停止摇头。",
            f"风别扫来扫去，{name}停摇头。",
            f"{loc}风扇不要摆风。",
            f"{name}关掉摆头。",
            f"请关闭{name}摇头。",
            f"{loc}风扇固定方向。",
            f"把{name}摇头功能关掉。",
            f"{name}别左右扫风。",
            f"让{name}定住。",
        ],
    }[spec.action]


def lock_prompts(spec: Spec) -> list[str]:
    return {
        "lock": [
            "把门锁上。",
            "睡觉前把门锁好。",
            "出门了，门锁好。",
            "帮我锁门。",
            "玄关门锁上。",
            "确认一下，把门锁好。",
            "我要休息了，锁门。",
            "离家前把门锁上。",
            "请锁好门。",
            "门口锁一下。",
            "把门锁住吧。",
            "晚上了，门锁起来。",
            "家里没人了，锁门。",
        ],
        "unlock": [
            "把门解锁。",
            "帮我开门锁。",
            "玄关门解锁。",
            "有人到了，把门锁打开。",
            "请解锁门。",
            "门口解一下锁。",
            "家人回来了，解锁。",
            "把门锁打开吧。",
            "门锁解除一下。",
            "帮我把门锁解开。",
            "门可以开了，解锁。",
            "把门锁放开。",
            "玄关解锁一下。",
        ],
        "check": [
            "检查门锁状态。",
            "看看门有没有锁好。",
            "出门了，门锁确认一下。",
            "帮我检查门锁。",
            "确认门锁状态。",
            "玄关门锁查一下。",
            "睡前看下门锁。",
            "门现在锁了吗？",
            "请检查一下门锁。",
            "帮我确认门口锁好了没。",
            "门锁状态看一下。",
            "查一下门有没有关好。",
            "离家前确认门锁。",
        ],
    }[spec.action]


def camera_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"帮我开启{name}。",
            f"{loc}需要看一下，摄像头打开。",
            f"{name}启动。",
            f"请打开{name}。",
            f"把{name}开起来。",
            f"{loc}摄像头开始工作。",
            f"看看{loc}情况，摄像头开。",
            f"{name}上线。",
            f"让{name}运行。",
            f"{loc}画面打开。",
            f"帮忙打开{name}。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"帮我关闭{name}。",
            f"{loc}不用看了，摄像头关掉。",
            f"{name}停止工作。",
            f"请关闭{name}。",
            f"把{name}关起来。",
            f"{loc}摄像头先停。",
            f"不用监看{loc}了。",
            f"{name}下线。",
            f"让{name}停止运行。",
            f"{loc}画面关掉。",
            f"帮忙关闭{name}。",
        ],
        "privacy_on": [
            f"开启{name}隐私模式。",
            f"把{name}调到隐私模式。",
            f"{loc}需要隐私，摄像头遮一下。",
            f"帮我打开{name}隐私模式。",
            f"{name}进入隐私模式。",
            f"请开启{name}隐私模式。",
            f"{loc}摄像头先保护隐私。",
            f"把{name}隐私保护打开。",
            f"不想被拍，{name}隐私模式。",
            f"{name}画面隐藏。",
            f"让{name}进入隐私保护。",
            f"{loc}暂时别拍，开隐私模式。",
            f"帮忙开启{name}隐私。",
        ],
        "privacy_off": [
            f"关闭{name}隐私模式。",
            f"取消{name}隐私模式。",
            f"{loc}可以恢复监看了。",
            f"帮我关闭{name}隐私模式。",
            f"{name}退出隐私模式。",
            f"请关闭{name}隐私模式。",
            f"{loc}摄像头恢复画面。",
            f"把{name}隐私保护关掉。",
            f"现在可以拍了，{name}退出隐私。",
            f"{name}画面恢复。",
            f"让{name}恢复监控。",
            f"{loc}不用隐私模式了。",
            f"帮忙关闭{name}隐私。",
        ],
        "record_start": [
            f"开始{name}录像。",
            f"让{name}开始录像。",
            f"{loc}摄像头录一下。",
            f"帮我开启{name}录像。",
            f"{name}开始录制。",
            f"请开始{name}录像。",
            f"{loc}情况记录下来。",
            f"把{name}录像打开。",
            f"开始记录{loc}画面。",
            f"{name}录制启动。",
            f"让{name}保存视频。",
            f"{loc}摄像头开始录。",
            f"帮忙启动{name}录像。",
        ],
        "record_stop": [
            f"停止{name}录像。",
            f"让{name}停止录像。",
            f"{loc}摄像头别录了。",
            f"帮我关闭{name}录像。",
            f"{name}停止录制。",
            f"请停止{name}录像。",
            f"{loc}不用记录了。",
            f"把{name}录像关掉。",
            f"停止记录{loc}画面。",
            f"{name}录制结束。",
            f"让{name}别保存视频了。",
            f"{loc}摄像头停止录。",
            f"帮忙停止{name}录像。",
        ],
    }[spec.action]


def smart_plug_prompts(spec: Spec) -> list[str]:
    name = target(spec.location, spec.device)
    loc = spec.location
    t = spec.time
    return {
        "open": [
            f"打开{name}。",
            f"把{name}开一下。",
            f"帮我开启{name}。",
            f"{loc}插座通电。",
            f"让{name}供电。",
            f"请打开{name}。",
            f"把{name}接通。",
            f"{loc}设备要用电，插座打开。",
            f"{name}开起来。",
            f"{loc}智能插座通一下电。",
            f"打开{loc}的插座。",
            f"让{name}开始供电。",
            f"帮忙打开{name}。",
        ],
        "close": [
            f"关闭{name}。",
            f"把{name}关掉。",
            f"帮我关闭{name}。",
            f"{loc}插座断电。",
            f"让{name}停止供电。",
            f"请关闭{name}。",
            f"把{name}断开。",
            f"{loc}设备不用电了，插座关闭。",
            f"{name}关起来。",
            f"{loc}智能插座断一下电。",
            f"关闭{loc}的插座。",
            f"让{name}停止通电。",
            f"帮忙关闭{name}。",
        ],
        "timer_open": [
            f"{t}打开{name}。",
            f"给{name}设置{t}开启。",
            f"{name}{t}通电。",
            f"帮我定时{t}打开{name}。",
            f"{loc}插座{t}开启。",
            f"请设置{name}{t}打开。",
            f"{t}让{name}供电。",
            f"把{name}定到{t}开启。",
            f"{name}{t}接通。",
            f"定时{t}开{name}。",
            f"{loc}智能插座{t}通电。",
            f"帮忙设置{name}{t}开。",
            f"{t}启动{name}。",
        ],
        "timer_close": [
            f"{t}关闭{name}。",
            f"给{name}设置{t}关闭。",
            f"{name}{t}断电。",
            f"帮我定时{t}关闭{name}。",
            f"{loc}插座{t}关闭。",
            f"请设置{name}{t}关掉。",
            f"{t}让{name}停止供电。",
            f"把{name}定到{t}关闭。",
            f"{name}{t}断开。",
            f"定时{t}关{name}。",
            f"{loc}智能插座{t}断电。",
            f"帮忙设置{name}{t}关。",
            f"{t}停止{name}。",
        ],
    }[spec.action]


PROMPT_BUILDERS: dict[str, Callable[[Spec], list[str]]] = {
    "窗帘": curtain_prompts,
    "扫地机器人": vacuum_prompts,
    "空气净化器": air_purifier_prompts,
    "加湿器": humidifier_prompts,
    "除湿机": dehumidifier_prompts,
    "热水器": water_heater_prompts,
    "风扇": fan_prompts,
    "门锁": lock_prompts,
    "摄像头": camera_prompts,
    "智能插座": smart_plug_prompts,
}


def prompt_count_plan(specs: list[Spec], target_total: int) -> dict[Spec, int]:
    if not 5 * len(specs) <= target_total <= 12 * len(specs):
        raise ValueError(
            f"Cannot allocate {target_total} examples across {len(specs)} responses while keeping 5-12 prompts per response"
        )
    base = target_total // len(specs)
    remainder = target_total - base * len(specs)
    if base < 5 or base > 12 or (base == 12 and remainder):
        raise ValueError(f"Invalid prompt count plan: base={base}, remainder={remainder}")
    plan: dict[Spec, int] = {}
    for index, spec in enumerate(specs):
        plan[spec] = base + (1 if index < remainder else 0)
    return plan


def generate_examples(specs: list[Spec], target_total: int) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    plan = prompt_count_plan(specs, target_total)
    examples: list[dict[str, str]] = []
    breakdown: list[dict[str, object]] = []
    used_pairs: set[tuple[str, str]] = set()
    used_prompts: set[str] = set()

    for spec in specs:
        candidates = PROMPT_BUILDERS[spec.device](spec)
        required = plan[spec]
        selected: list[str] = []
        for prompt in candidates:
            prompt = prompt.strip()
            pair = (prompt, spec.response)
            if pair in used_pairs or prompt in used_prompts:
                continue
            selected.append(prompt)
            used_pairs.add(pair)
            used_prompts.add(prompt)
            if len(selected) == required:
                break
        if len(selected) != required:
            raise ValueError(f"Only generated {len(selected)} prompts for {spec}; need {required}")

        for prompt in selected:
            examples.append({"prompt": prompt, "response": spec.response})
        breakdown.append(
            {
                "device": spec.device,
                "location": spec.location,
                "action": spec.action,
                "response": spec.response,
                "prompt_count": len(selected),
            }
        )

    if len(examples) != target_total:
        raise ValueError(f"Generated {len(examples)} examples, expected {target_total}")
    if len({(item["prompt"], item["response"]) for item in examples}) != len(examples):
        raise ValueError("Duplicate prompt-response pair detected")
    if len({item["prompt"] for item in examples}) != len(examples):
        raise ValueError("Duplicate prompt detected")
    return examples, breakdown


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_breakdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["device", "location", "action", "response", "prompt_count"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(examples: list[dict[str, str]], breakdown: list[dict[str, object]], output: Path, csv_path: Path) -> None:
    by_device: Counter[str] = Counter()
    by_action: Counter[str] = Counter()
    by_device_action: Counter[tuple[str, str]] = Counter()
    response_to_meta = {str(row["response"]): row for row in breakdown}

    for item in examples:
        meta = response_to_meta[item["response"]]
        device = str(meta["device"])
        action = str(meta["action"])
        by_device[device] += 1
        by_action[action] += 1
        by_device_action[(device, action)] += 1

    print(f"generated_examples: {len(examples)}")
    print(f"unique_prompts: {len({item['prompt'] for item in examples})}")
    print(f"unique_responses: {len({item['response'] for item in examples})}")
    print(f"response_classes: {len(breakdown)}")
    print(f"output: {output}")
    print(f"breakdown_csv: {csv_path}")
    print("examples_by_device:")
    for device, count in by_device.most_common():
        print(f"- {device}: {count}")
    print("examples_by_action:")
    for action, count in by_action.most_common():
        print(f"- {action}: {count}")
    print("examples_by_device_action:")
    for (device, action), count in sorted(by_device_action.items()):
        print(f"- {device}/{action}: {count}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    breakdown_path = Path(args.breakdown).expanduser()

    load_and_validate_input(input_path)
    specs = build_specs()
    examples, breakdown = generate_examples(specs, args.target_total)
    write_json(output_path, examples)
    write_breakdown(breakdown_path, breakdown)
    print_summary(examples, breakdown, output_path, breakdown_path)


if __name__ == "__main__":
    main()
