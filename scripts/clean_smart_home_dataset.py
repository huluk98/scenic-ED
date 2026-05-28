#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_dedup.json"
DEFAULT_REFERENCE = Path("/Users/luke/Downloads/619_Luke_fixed_dedup.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_LOG = PROJECT_ROOT / "reports" / "repair_log.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "repair_summary.md"

ROOM_ORDER = ("客厅", "书房", "卧室", "主卧室", "主卧", "次卧", "房间")
CANONICAL_ROOMS = {"客厅": "客厅", "书房": "书房", "卧室": "卧室", "主卧室": "卧室", "主卧": "卧室", "次卧": "卧室", "房间": "卧室"}
MODE_VALUES = ("制冷", "制热", "送风", "抽湿", "自动", "环保", "舒睡")
LIGHT_MODE_VALUES = ("普通", "学习", "夜灯", "睡眠", "舒睡")
LIGHT_COLOR_VALUES = ("冷色光", "暖色光", "自然光")
TV_PICTURE_MODES = ("游戏", "电影", "体育")
TV_INPUT_VALUES = ("HDMI1", "HDMI2", "投屏")
TV_CHANNEL_VALUES = ("CCTV-1", "体育")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair SCENIC smart-home natural-language responses with canonical many-to-one outputs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def compact(text: str) -> str:
    return re.sub(r"[\s。！？!?,，、；;：:]+", "", clean_text(text))


def load_json_list(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        rows.append({"prompt": clean_text(item.get("prompt")), "response": clean_text(item.get("response"))})
    return rows


def normalize_prompt_for_dedupe(prompt: str) -> str:
    key = compact(prompt)
    key = key.replace("吧吧", "吧").replace("谢谢谢谢", "谢谢").replace("可以吗可以吗", "可以吗")
    key = re.sub(r"(吧|嘛|呀|可以吗|谢谢)$", "", key)
    return key


def normalize_response_artifacts(response: str) -> str:
    fixed = clean_text(response)
    fixed = re.sub(r"已播放(.+?)播放", r"已继续播放\1", fixed)
    replacements = {
        "半时": "半小时",
        "一时": "一小时",
        "二时": "二小时",
        "三时": "三小时",
        "关闭闭": "关闭",
        "开启启": "开启",
        "舒睡模式一模式": "舒睡模式",
        "舒睡模式二模式": "舒睡模式",
        "模式模式": "模式",
    }
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)
    return fixed


def infer_location(prompt: str) -> str:
    text = clean_text(prompt)
    for room in ROOM_ORDER:
        if room in text:
            return CANONICAL_ROOMS[room]
    return ""


def infer_device(prompt: str) -> str | None:
    text = clean_text(prompt)
    lower = text.lower()
    if any(term in text for term in ("电视", "字幕", "投屏", "频道", "画面", "机顶盒", "节目", "中央一套", "新闻联播")) or "hdmi" in lower:
        return "电视"
    if any(term in text for term in ("音箱", "音乐", "歌曲", "下一首歌", "上一首歌", "放歌")):
        return "音箱"
    if any(term in text for term in ("空调", "温度", "风速", "风向", "制冷", "制热", "送风", "抽湿", "除湿", "冷气")):
        return "空调"
    if any(term in text for term in ("灯", "灯光", "光线", "亮度", "冷色光", "暖色光", "自然光", "太暗", "刺眼")):
        return "灯"
    return None


def infer_device_from_pair(prompt: str, response: str) -> str | None:
    return infer_device(prompt) or infer_device(response)


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def infer_light_intents(prompt: str) -> list[dict[str, Any]]:
    text = clean_text(prompt)
    key = compact(text)
    location = infer_location(text)
    intents: list[dict[str, Any]] = []

    if has_any(key, ("不要开灯", "不用开灯", "不需要开灯", "别开灯", "不要打开灯", "别打开灯")):
        return [{"intent": "light_off", "device": "灯", "location": location, "slots": {}}]
    if has_any(key, ("不要关灯", "不用关灯", "不需要关灯", "别关灯", "不要关闭灯", "别关闭灯")):
        return []

    for color in LIGHT_COLOR_VALUES:
        if color in text or color.replace("光", "灯") in text:
            intents.append({"intent": "light_color", "device": "灯", "location": location, "slots": {"color": color}})
            return intents
    for mode in LIGHT_MODE_VALUES:
        if f"{mode}模式" in text or (mode in text and "模式" in text):
            intents.append({"intent": "light_mode", "device": "灯", "location": location, "slots": {"mode": mode}})
            return intents

    if has_any(text, ("开灯", "打开灯", "开启灯", "我要开灯", "把灯打开", "灯打开", "开一下灯")):
        return [{"intent": "light_on", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("最亮", "全开", "亮度最高", "调到最高")):
        return [{"intent": "light_brightest", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("最暗", "最低", "亮度最低", "调到最低")):
        return [{"intent": "light_darkest", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("调亮", "调高", "亮一点", "亮一些", "太暗", "看不清", "不够亮")):
        return [{"intent": "light_brightness_up", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("调暗", "调低", "暗一点", "暗一些", "太亮", "刺眼", "柔和")):
        return [{"intent": "light_brightness_down", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("关灯", "关闭", "关掉", "关一下", "不用灯", "不需要灯", "睡觉")):
        return [{"intent": "light_off", "device": "灯", "location": location, "slots": {}}]
    if has_any(text, ("开灯", "打开", "开启", "开一下", "亮灯", "有点暗")):
        return [{"intent": "light_on", "device": "灯", "location": location, "slots": {}}]
    return []


def infer_tv_intents(prompt: str) -> list[dict[str, Any]]:
    text = clean_text(prompt)
    lower = text.lower()
    location = infer_location(text)

    if "字幕" in text or has_any(text, ("外语片看不懂", "对白太快", "显示文字", "不用显示文字")):
        state = "off" if has_any(text, ("关闭", "关掉", "不用", "不需要", "取消", "别显示")) else "on"
        return [{"intent": "tv_subtitle", "device": "电视", "location": location, "slots": {"state": state}}]
    if "静音" in text or has_any(text, ("声音先停", "先安静", "别吵", "别出声", "恢复声音", "声音回来", "继续播出声音")):
        state = "off" if has_any(text, ("关闭静音", "取消静音", "恢复声音", "声音回来", "继续播出声音", "可以恢复")) else "on"
        return [{"intent": "tv_mute", "device": "电视", "location": location, "slots": {"state": state}}]
    if has_any(text, ("音量调高", "声音太小", "听不清", "声音大点", "调大音量")):
        return [{"intent": "tv_volume_up", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("音量调低", "声音太大", "电视太吵", "别那么响", "声音小点", "调小音量")):
        return [{"intent": "tv_volume_down", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("暂停", "停住", "先别往下播", "我离开一下", "先停一下")):
        return [{"intent": "tv_pause", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("继续播放", "开始播放", "继续看", "继续刚才", "可以动了", "接上", "能看了", "电话打完", "节目继续", "电视开始")):
        return [{"intent": "tv_play", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("下一个频道", "下一台", "下一个台", "往下换", "换下一")):
        return [{"intent": "tv_next_channel", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("上一个频道", "上一台", "前一个频道", "前一个台", "刚才那个台", "回到上一个", "上一台那个节目")):
        return [{"intent": "tv_previous_channel", "device": "电视", "location": location, "slots": {}}]
    if "cctv-1" in lower or "cctv1" in lower or "中央一套" in text or "央视一套" in text or "新闻联播" in text or "全国新闻" in text:
        return [{"intent": "tv_channel", "device": "电视", "location": location, "slots": {"channel": "CCTV-1"}}]
    if ("频道" in text or "台" in text or "比赛" in text or "运动员" in text) and "体育" in text:
        return [{"intent": "tv_channel", "device": "电视", "location": location, "slots": {"channel": "体育"}}]
    if "hdmi1" in lower or "hdmi一" in lower or "hdmi 1" in lower or "一号接口" in text or "机顶盒" in text:
        return [{"intent": "tv_input", "device": "电视", "location": location, "slots": {"input": "HDMI1"}}]
    if "hdmi2" in lower or "hdmi二" in lower or "hdmi 2" in lower or "二号接口" in text or "主机" in text:
        return [{"intent": "tv_input", "device": "电视", "location": location, "slots": {"input": "HDMI2"}}]
    if "投屏" in text or "小屏幕" in text or "手机" in text or "相册" in text:
        return [{"intent": "tv_input", "device": "电视", "location": location, "slots": {"input": "投屏"}}]
    if has_any(text, ("画面模式", "画质", "手柄操作", "看电影", "我要看电影", "体育模式", "游戏模式")):
        mode = "游戏" if "游戏" in text or "手柄" in text else "电影" if "电影" in text else "体育" if "体育" in text else ""
        if mode:
            return [{"intent": "tv_picture_mode", "device": "电视", "location": location, "slots": {"mode": mode}}]
    if has_any(text, ("关闭电视", "关电视", "关掉电视", "不看电视", "电视不用", "屏幕可以黑", "准备睡觉", "我要休息", "电视别开着")):
        return [{"intent": "tv_off", "device": "电视", "location": location, "slots": {}}]
    if has_any(compact(text), ("不要开电视", "不用开电视", "不需要开电视", "别开电视", "不要打开电视", "别打开电视")):
        return [{"intent": "tv_off", "device": "电视", "location": location, "slots": {}}]
    if has_any(text, ("打开电视", "开电视", "开启电视", "看电视", "想看电视", "电视时间", "电视开机", "开起来")) or re.search(r"(开|打开|开启).{0,6}电视", text) or re.search(r"电视.{0,4}(开|打开|开启|开起来)", text):
        return [{"intent": "tv_on", "device": "电视", "location": location, "slots": {}}]
    return []


def infer_speaker_intents(prompt: str) -> list[dict[str, Any]]:
    text = clean_text(prompt)
    location = infer_location(text)
    if has_any(text, ("下一首", "下首")):
        return [{"intent": "speaker_next", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("上一首", "上首")):
        return [{"intent": "speaker_previous", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("停止播放音乐", "停止音箱中的音乐", "暂停", "停一下")):
        return [{"intent": "speaker_pause", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("播放", "放音乐", "放歌", "听歌", "想听")):
        return [{"intent": "speaker_play", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("音量调高", "声音调高", "声音太小", "音量太低", "调大音量")):
        return [{"intent": "speaker_volume_up", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("音量调低", "声音调低", "声音太大", "音量太高", "调小音量")):
        return [{"intent": "speaker_volume_down", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("关闭", "关机", "关掉")):
        return [{"intent": "speaker_off", "device": "音箱", "location": location, "slots": {}}]
    if has_any(compact(text), ("不要开音箱", "不用开音箱", "不需要开音箱", "别开音箱", "不要打开音箱", "别打开音箱")):
        return [{"intent": "speaker_off", "device": "音箱", "location": location, "slots": {}}]
    if has_any(text, ("打开", "开机", "开启")):
        return [{"intent": "speaker_on", "device": "音箱", "location": location, "slots": {}}]
    return []


def extract_timer(prompt: str) -> tuple[str, str] | None:
    text = clean_text(prompt)
    action = "关闭" if has_any(text, ("关", "关闭", "关掉")) else "开启" if has_any(text, ("开", "打开", "开启", "运行")) else ""
    absolute = re.search(r"((?:明早|明天|今晚|晚上|早上|上午|下午)?[零一二三四五六七八九十两\d]{1,3}点半)", text)
    if absolute and action:
        return absolute.group(1), action
    absolute = re.search(r"((?:明早|明天|今晚|晚上|早上|上午|下午)?[零一二三四五六七八九十两\d]{1,3}点)", text)
    if absolute and action:
        return absolute.group(1), action
    duration = re.search(r"((?:半|[零一二三四五六七八九十两\d]+)\s*(?:分钟|小时)后)", text)
    if duration and action:
        return duration.group(1).replace(" ", ""), action
    duration_without_after = re.search(r"((?:半|[零一二三四五六七八九十两\d]+)\s*(?:分钟|小时)).{0,6}(关闭|关|开启|打开|开|运行)", text)
    if duration_without_after:
        timer_action = "关闭" if duration_without_after.group(2) in {"关闭", "关"} else "开启"
        return duration_without_after.group(1).replace(" ", "") + "后", timer_action
    run_for = re.search(r"((?:半|[零一二三四五六七八九十两\d]+)\s*(?:分钟|小时))", text)
    if run_for and has_any(text, ("就开", "运行", "开")):
        return run_for.group(1).replace(" ", "") + "后", "关闭"
    return None


def infer_ac_intents(prompt: str) -> list[dict[str, Any]]:
    text = clean_text(prompt)
    location = infer_location(text)
    intents: list[dict[str, Any]] = []
    key = compact(text)

    if has_any(
        key,
        (
            "不要开空调",
            "不用开空调",
            "不需要开空调",
            "别开空调",
            "不要打开空调",
            "别打开空调",
            "空调不要开",
            "空调不用开",
            "空调不需要开",
            "空调别开",
            "空调不用开着",
        ),
    ):
        return [{"intent": "ac_off", "device": "空调", "location": location, "slots": {}}]

    explicit_mode_shutdown = has_any(text, ("结束", "退出", "停止")) and has_any(text, ("舒睡", "睡眠", "制冷", "制热", "送风", "抽湿", "除湿", "自动", "环保"))
    if explicit_mode_shutdown and not has_any(text, ("打开", "开启", "调到", "调成", "设为", "切换")):
        return [{"intent": "ac_off", "device": "空调", "location": location, "slots": {}}]

    for mode in MODE_VALUES:
        if mode == "自动" and re.search(r"自动[关开]", text):
            continue
        if mode in text or (mode == "抽湿" and "除湿" in text) or (mode == "舒睡" and has_any(text, ("舒睡", "睡眠"))):
            intents.append({"intent": "ac_mode", "device": "空调", "location": location, "slots": {"mode": mode}})
            break

    timer = extract_timer(text)
    temp_set = re.search(r"(?<!升高)(?<!降低)(?<!调高)(?<!调低)(\d{1,2})\s*度", text)
    if has_any(text, ("温度升高", "温度调高", "升高1度", "调高1度", "调高一度", "热一点")):
        intents.append({"intent": "ac_temp_up", "device": "空调", "location": location, "slots": {}})
    elif has_any(text, ("温度降低", "温度调低", "降低1度", "调低1度", "调低一度", "冷一点")):
        intents.append({"intent": "ac_temp_down", "device": "空调", "location": location, "slots": {}})
    elif temp_set:
        intents.append({"intent": "ac_temp_set", "device": "空调", "location": location, "slots": {"temperature": f"{temp_set.group(1)}度"}})

    if "风向" in text or has_any(text, ("上下左右", "上下风", "左右风", "向左", "向下", "不动")):
        direction = ""
        for value in ("上下左右", "不动", "上下", "左右", "左", "下"):
            if value in text:
                direction = value
                break
        if direction:
            intents.append({"intent": "ac_wind_direction", "device": "空调", "location": location, "slots": {"direction": direction}})
    if "风速" in text or "风力" in text or has_any(text, ("风太大", "风太小", "风大点", "风小点")):
        if has_any(text, ("调高", "提高", "最大", "最高", "风大点", "风太小", "不够大")):
            intents.append({"intent": "ac_fan_up", "device": "空调", "location": location, "slots": {}})
        elif has_any(text, ("调低", "降低", "最小", "最低", "风小点", "风太大")):
            intents.append({"intent": "ac_fan_down", "device": "空调", "location": location, "slots": {}})

    if timer:
        time_text, timer_action = timer
        intents.append({"intent": "ac_timer", "device": "空调", "location": location, "slots": {"time": time_text, "timer_action": timer_action}})

    if not intents:
        if has_any(text, ("关闭空调", "关空调", "关掉空调", "空调关闭", "空调关机", "不用空调")):
            intents.append({"intent": "ac_off", "device": "空调", "location": location, "slots": {}})
        elif has_any(text, ("打开空调", "开空调", "开启空调", "空调打开", "空调开机", "空调开下", "空调开一下", "太热", "好热", "冷气")) or re.search(r"(开|打开|开启).{0,6}空调", text) or re.search(r"空调.{0,4}(开|打开|开启|开下|开一下|开起来)", text):
            intents.append({"intent": "ac_on", "device": "空调", "location": location, "slots": {}})
    return intents


def infer_intent(prompt: str) -> list[dict[str, Any]]:
    device = infer_device(prompt)
    if device == "灯":
        return infer_light_intents(prompt)
    if device == "空调":
        return infer_ac_intents(prompt)
    if device == "电视":
        return infer_tv_intents(prompt)
    if device == "音箱":
        return infer_speaker_intents(prompt)
    return []


def infer_intents_from_pair(prompt: str, response: str) -> list[dict[str, Any]]:
    intents = infer_intent(prompt)
    if intents:
        return intents
    device = infer_device_from_pair(prompt, response)
    combined = f"{prompt} {response}"
    if device == "灯":
        return infer_light_intents(combined)
    if device == "空调":
        return infer_ac_intents(combined)
    if device == "电视":
        return infer_tv_intents(combined)
    if device == "音箱":
        return infer_speaker_intents(combined)
    return []


def with_location(location: str, device: str) -> str:
    return f"{location}{device}" if location else device


def canonical_response(intent: str, device: str, location: str, slots: dict[str, Any]) -> str:
    target = with_location(location, device)
    if intent == "light_on":
        return f"好的，已打开{target}。"
    if intent == "light_off":
        return f"好的，已关闭{target}。"
    if intent == "light_brightness_up":
        return f"好的，已将{target}亮度调高一档。"
    if intent == "light_brightness_down":
        return f"好的，已将{target}亮度调低一档。"
    if intent == "light_brightest":
        return f"好的，已将{target}亮度调到最高。"
    if intent == "light_darkest":
        return f"好的，已将{target}亮度调到最低。"
    if intent == "light_mode":
        return f"好的，已将{target}切换到{slots['mode']}模式。"
    if intent == "light_color":
        return f"好的，已将{target}调为{slots['color']}。"
    if intent == "ac_on":
        return f"好的，已打开{target}。"
    if intent == "ac_off":
        return f"好的，已关闭{target}。"
    if intent == "ac_mode":
        return f"好的，已将{target}切换到{slots['mode']}模式。"
    if intent == "ac_temp_up":
        return f"好的，已将{target}温度调高一度。"
    if intent == "ac_temp_down":
        return f"好的，已将{target}温度调低一度。"
    if intent == "ac_temp_set":
        return f"好的，已将{target}温度调到{slots['temperature']}。"
    if intent == "ac_wind_direction":
        return f"好的，已将{target}风向调到{slots['direction']}。"
    if intent == "ac_fan_up":
        return f"好的，已将{target}风速调高一档。"
    if intent == "ac_fan_down":
        return f"好的，已将{target}风速调低一档。"
    if intent == "ac_timer":
        return f"好的，已为{target}设置{slots['time']}{slots['timer_action']}。"
    if intent == "tv_on":
        return f"好的，已打开{target}。"
    if intent == "tv_off":
        return f"好的，已关闭{target}。"
    if intent == "tv_subtitle":
        action = "打开" if slots["state"] == "on" else "关闭"
        return f"好的，已{action}{target}字幕。"
    if intent == "tv_mute":
        action = "开启" if slots["state"] == "on" else "关闭"
        return f"好的，已{action}{target}静音。"
    if intent == "tv_volume_up":
        return f"好的，已将{target}音量调高。"
    if intent == "tv_volume_down":
        return f"好的，已将{target}音量调低。"
    if intent == "tv_play":
        return f"好的，已继续播放{target}。"
    if intent == "tv_pause":
        return f"好的，已暂停{target}播放。"
    if intent == "tv_next_channel":
        return f"好的，已将{target}切到下一个频道。"
    if intent == "tv_previous_channel":
        return f"好的，已将{target}切到上一个频道。"
    if intent == "tv_channel":
        return f"好的，已将{target}频道设为{slots['channel']}。"
    if intent == "tv_input":
        return f"好的，已将{target}输入源切换到{slots['input']}。"
    if intent == "tv_picture_mode":
        return f"好的，已将{target}画面模式设为{slots['mode']}。"
    if intent == "speaker_on":
        return f"好的，已打开{target}。"
    if intent == "speaker_off":
        return f"好的，已关闭{target}。"
    if intent == "speaker_play":
        return f"好的，已开始播放{target}。"
    if intent == "speaker_pause":
        return f"好的，已暂停{target}播放。"
    if intent == "speaker_next":
        return f"好的，已将{target}切到下一首。"
    if intent == "speaker_previous":
        return f"好的，已将{target}切到上一首。"
    if intent == "speaker_volume_up":
        return f"好的，已将{target}音量调高。"
    if intent == "speaker_volume_down":
        return f"好的，已将{target}音量调低。"
    raise ValueError(f"unsupported intent: {intent}")


def canonical_for_intents(intents: list[dict[str, Any]]) -> str:
    pieces: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in intents:
        response = canonical_response(item["intent"], item["device"], item["location"], item["slots"])
        phrase = response.removeprefix("好的，").removesuffix("。")
        key = (item["intent"], item["device"], item["location"], json.dumps(item["slots"], ensure_ascii=False, sort_keys=True))
        if key not in seen:
            seen.add(key)
            pieces.append(phrase)
    return "好的，" + "；".join(pieces) + "。"


def is_ungrammatical_prompt(prompt: str) -> tuple[bool, str]:
    text = compact(prompt)
    raw = clean_text(prompt)
    if not raw:
        return True, "empty_prompt"
    if has_any(text, ("吧吧", "谢谢谢谢", "可以吗可以吗")):
        return True, "duplicated_polite_suffix"
    if raw.startswith("刚好"):
        return True, "mechanical_prefix_ganghao"
    if raw.startswith("这会儿") and has_any(raw, ("吧吧", "现在能看了", "刚才停住的画面可以动了")):
        return True, "mechanical_prefix_zhehuier"
    if re.search(r"(电视电视|空调空调|音箱音箱|灯灯|客厅客厅|卧室卧室|书房书房)", text):
        return True, "duplicated_device_or_room_phrase"
    if re.search(r"(阳台|厨房|餐厅|浴室|洗手间|卫生间|温室).{0,2}(灯|空调|电视|音箱)", raw):
        return True, "unsupported_location_device_pair"
    if infer_location(raw) == "" and infer_device(raw) in {"灯", "空调"} and has_any(raw, ("阳台", "厨房", "餐厅", "浴室", "洗手间", "卫生间", "温室")):
        return True, "unsupported_location_device_pair"
    return False, ""


def is_invalid_response(prompt: str, response: str) -> bool:
    p = compact(prompt)
    r = compact(response)
    if re.search(r"已播放.*播放", response):
        return True
    if has_any(response, ("半时", "一时", "二时", "三时", "关闭闭", "开启启", "模式模式")):
        return True
    if infer_device(prompt) == "电视" and has_any(p, ("字幕", "静音", "投屏", "HDMI", "频道", "画面", "游戏", "电影", "体育", "播放", "暂停")) and "音箱" in response:
        return True
    if "电视" in p and has_any(p, ("字幕", "静音")) and has_any(r, ("已打开电视", "已关闭电视")):
        return True
    if has_any(p, ("不要开灯", "不用开灯", "不需要开灯", "别开灯")) and "已打开灯" in response:
        return True
    if infer_device(prompt) == "空调" and infer_mode_prompt(prompt) and r in {"好的，已打开空调。", "好的，已打开客厅空调。", "好的，已打开卧室空调。", "好的，已打开书房空调。"}:
        return True
    return False


def infer_mode_prompt(prompt: str) -> str:
    for mode in MODE_VALUES:
        if mode in prompt or (mode == "抽湿" and "除湿" in prompt) or (mode == "舒睡" and "睡眠" in prompt):
            return mode
    return ""


def intent_key(intents: list[dict[str, Any]]) -> str:
    return " || ".join(
        f"{item['device']}:{item['location']}:{item['intent']}:{json.dumps(item['slots'], ensure_ascii=False, sort_keys=True)}"
        for item in intents
    )


def repair_dataset(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    repaired: list[dict[str, str]] = []
    logs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_prompts: set[str] = set()
    canonical_by_intent: dict[str, str] = {}
    before_after: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        prompt = clean_text(row.get("prompt"))
        old_response = clean_text(row.get("response"))
        stats["original_count"] += 1

        prompt_key = normalize_prompt_for_dedupe(prompt)
        ungrammatical, grammar_reason = is_ungrammatical_prompt(prompt)
        if ungrammatical:
            stats["dropped_count"] += 1
            stats[f"drop_reason:{grammar_reason}"] += 1
            logs.append({"index": index, "action": "dropped", "reason": grammar_reason, "prompt": prompt, "old_response": old_response})
            continue
        if prompt_key in seen_prompts:
            stats["dropped_count"] += 1
            stats["drop_reason:duplicate_prompt"] += 1
            logs.append({"index": index, "action": "dropped", "reason": "duplicate_prompt", "prompt": prompt, "old_response": old_response})
            continue

        intents = infer_intents_from_pair(prompt, old_response)
        if not intents:
            stats["dropped_count"] += 1
            stats["drop_reason:unresolved_intent"] += 1
            logs.append({"index": index, "action": "dropped", "reason": "unresolved_intent", "prompt": prompt, "old_response": old_response})
            continue

        try:
            new_response = canonical_for_intents(intents)
        except Exception as exc:
            stats["dropped_count"] += 1
            stats["drop_reason:canonicalization_error"] += 1
            logs.append({"index": index, "action": "dropped", "reason": f"canonicalization_error:{exc}", "prompt": prompt, "old_response": old_response})
            continue

        key = intent_key(intents)
        previous = canonical_by_intent.setdefault(key, new_response)
        if previous != new_response:
            stats["dropped_count"] += 1
            stats["drop_reason:conflicting_canonical_response"] += 1
            logs.append({"index": index, "action": "dropped", "reason": "conflicting_canonical_response", "prompt": prompt, "old_response": old_response, "suggested_response": new_response, "canonical_response": previous})
            continue

        seen_prompts.add(prompt_key)
        repaired.append({"prompt": prompt, "response": new_response})
        stats["repaired_count"] += 1
        changed = old_response != new_response
        invalid = is_invalid_response(prompt, old_response)
        if changed:
            stats["changed_response_count"] += 1
            reason = "invalid_response_repaired" if invalid else "canonicalized_response_style"
            logs.append({"index": index, "action": "changed", "reason": reason, "prompt": prompt, "old_response": old_response, "new_response": new_response, "intent_key": key})
            if len(before_after) < 30:
                before_after.append({"index": index, "prompt": prompt, "old_response": old_response, "new_response": new_response, "reason": reason})
    return repaired, logs, stats, before_after


def write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def build_summary(input_path: Path, reference_path: Path, output_path: Path, source_rows: list[dict[str, str]], reference_rows: list[dict[str, str]], repaired: list[dict[str, str]], logs: list[dict[str, Any]], stats: Counter[str], before_after: list[dict[str, Any]]) -> str:
    top_responses = Counter(row["response"] for row in repaired).most_common(30)
    drop_reasons = sorted((key.replace("drop_reason:", ""), value) for key, value in stats.items() if key.startswith("drop_reason:"))
    lines = [
        "# Smart-Home Dataset Repair Summary",
        "",
        "The repair pipeline is deterministic and does not use LLM calls.",
        "",
        "## Files",
        f"- input: `{input_path}`",
        f"- reference: `{reference_path}`",
        f"- output: `{output_path}`",
        "",
        "## Counts",
        markdown_table(
            [
                ["original_count", len(source_rows)],
                ["reference_count", len(reference_rows)],
                ["repaired_count", len(repaired)],
                ["dropped_count", stats.get("dropped_count", 0)],
                ["changed_response_count", stats.get("changed_response_count", 0)],
                ["unique_prompts_after", len({normalize_prompt_for_dedupe(row["prompt"]) for row in repaired})],
                ["unique_responses_after", len({row["response"] for row in repaired})],
            ],
            ["metric", "value"],
        ),
        "",
        "## Drop Reasons",
        markdown_table([[reason, count] for reason, count in drop_reasons], ["reason", "count"]) if drop_reasons else "No drops.",
        "",
        "## Top 30 Canonical Responses",
        markdown_table([[response, count] for response, count in top_responses], ["response", "count"]),
        "",
        "## First 30 Before/After Repairs",
    ]
    lines.append(markdown_table([[item["index"], item["prompt"], item["old_response"], item["new_response"], item["reason"]] for item in before_after], ["index", "prompt", "old_response", "new_response", "reason"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    reference_path = Path(args.reference).expanduser()
    output_path = Path(args.output).expanduser()
    log_path = Path(args.log).expanduser()
    summary_path = Path(args.summary).expanduser()

    source_rows = load_json_list(input_path)
    reference_rows = load_json_list(reference_path)
    repaired, logs, stats, before_after = repair_dataset(source_rows)

    write_json(output_path, repaired)
    write_jsonl(log_path, logs)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        build_summary(input_path, reference_path, output_path, source_rows, reference_rows, repaired, logs, stats, before_after),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "original_count": len(source_rows),
                "repaired_count": len(repaired),
                "dropped_count": stats.get("dropped_count", 0),
                "changed_response_count": stats.get("changed_response_count", 0),
                "unique_responses_after": len({row["response"] for row in repaired}),
                "output": str(output_path),
                "repair_log": str(log_path),
                "repair_summary": str(summary_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
