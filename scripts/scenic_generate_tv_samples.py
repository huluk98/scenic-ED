#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LOCATIONS: dict[str, dict[str, str]] = {
    "未指定": {"cn": "", "prefix": "", "name": "unspecified"},
    "客厅": {"cn": "客厅", "prefix": "客厅的", "name": "living_room"},
    "卧室": {"cn": "卧室", "prefix": "卧室的", "name": "bedroom"},
    "书房": {"cn": "书房", "prefix": "书房的", "name": "study"},
}

INTENTS: list[dict[str, Any]] = [
    {
        "intent": "power_on",
        "action": "set_power",
        "value": "on",
        "category": "power",
        "direct": ["打开{tv}", "{tv}开一下", "帮我把{tv}打开", "{tv}开机", "启动{tv}", "把{tv}电源打开"],
        "indirect": ["我想看电视了", "有点无聊，想看会儿电视", "准备看节目了", "电视时间到了", "想看看新闻"],
        "hard_indirect": ["晚饭好了，边吃边看会儿", "球赛快开始了", "把屏幕亮起来，我要看节目"],
    },
    {
        "intent": "power_off",
        "action": "set_power",
        "value": "off",
        "category": "power",
        "direct": ["关闭{tv}", "{tv}关一下", "帮我把{tv}关掉", "{tv}关机", "关掉{tv}", "把{tv}电源关掉"],
        "indirect": ["不看电视了", "我要休息了，电视别开着", "节目看完了", "房间安静点，电视不用了", "准备睡觉了"],
        "hard_indirect": ["出门前别让电视一直亮着", "屏幕可以黑了", "客人走了，电视收起来吧"],
    },
    {
        "intent": "volume_up",
        "action": "adjust_volume",
        "value": "up",
        "category": "volume",
        "direct": ["{tv}音量调高", "把{tv}声音加大", "{tv}大声一点", "调高{tv}音量", "{tv}声音提高一档"],
        "indirect": ["声音太小了", "听不清电视", "台词有点听不到", "音量不够", "这个声音太轻"],
        "hard_indirect": ["厨房这边听不到电视", "新闻声音被空调盖住了", "离远点听不清主持人"],
    },
    {
        "intent": "volume_down",
        "action": "adjust_volume",
        "value": "down",
        "category": "volume",
        "direct": ["{tv}音量调低", "把{tv}声音小一点", "{tv}小声一点", "调低{tv}音量", "{tv}声音降低一档"],
        "indirect": ["电视太吵了", "声音有点大", "别吵到别人", "音量太高了", "电视声轻一点"],
        "hard_indirect": ["孩子睡了，电视别那么响", "晚上了，声音收一点", "邻居可能听得到，低一点"],
    },
    {
        "intent": "mute_on",
        "action": "set_mute",
        "value": "on",
        "category": "volume",
        "direct": ["{tv}静音", "把{tv}设为静音", "{tv}先别出声", "关闭{tv}声音", "{tv}声音关掉"],
        "indirect": ["我接个电话", "先安静一下", "别让电视出声", "需要安静几分钟", "电视别说话了"],
        "hard_indirect": ["会议开始了，电视声音先停", "电话来了，别让电视打扰", "录音的时候电视保持安静"],
    },
    {
        "intent": "mute_off",
        "action": "set_mute",
        "value": "off",
        "category": "volume",
        "direct": ["取消{tv}静音", "{tv}恢复声音", "打开{tv}声音", "{tv}别静音了", "恢复{tv}音量"],
        "indirect": ["电话打完了", "可以继续出声了", "现在能听电视了", "安静结束了", "声音回来吧"],
        "hard_indirect": ["会议结束了，让电视继续播出声音", "刚才静音的电视可以恢复", "录音完了，电视声打开"],
    },
    {
        "intent": "channel_next",
        "action": "change_channel",
        "value": "next",
        "category": "channel",
        "direct": ["{tv}下一个频道", "{tv}换到下一个台", "切到{tv}下一台", "{tv}频道往后", "换个电视台"],
        "indirect": ["这个台没意思", "看看下一个节目", "换一个频道吧", "这个节目不想看", "有没有别的台"],
        "hard_indirect": ["广告太多了，跳到下一个台", "这个节目看腻了，找别的", "继续往后找频道"],
    },
    {
        "intent": "channel_previous",
        "action": "change_channel",
        "value": "previous",
        "category": "channel",
        "direct": ["{tv}上一个频道", "{tv}换回上一个台", "切到{tv}上一台", "{tv}频道往前", "回到前一个电视台"],
        "indirect": ["刚才那个台更好", "回到上一个节目", "刚刚的频道不错", "往回换一个台", "前一个频道"],
        "hard_indirect": ["刚才的新闻还没看完，切回去", "上一台那个节目更合适", "退回刚刚那个频道"],
    },
    {
        "intent": "channel_cctv1",
        "action": "set_channel",
        "value": "CCTV-1",
        "category": "channel",
        "direct": ["{tv}切到CCTV-1", "{tv}打开央视一套", "把{tv}调到综合频道", "{tv}频道设为CCTV1"],
        "indirect": ["我想看新闻联播", "看看央视综合频道", "打开中央一套", "我要看央视节目"],
        "hard_indirect": ["七点新闻快开始了", "想看全国新闻", "把电视调到播新闻联播的台"],
    },
    {
        "intent": "channel_sports",
        "action": "set_channel",
        "value": "sports",
        "category": "channel",
        "direct": ["{tv}切到体育频道", "把{tv}调到体育台", "{tv}打开体育频道", "{tv}频道设为体育"],
        "indirect": ["我想看比赛", "球赛快开始了", "看看体育节目", "找个有比赛的频道"],
        "hard_indirect": ["今晚的决赛要开始了", "想看运动员比赛", "给我找体育直播"],
    },
    {
        "intent": "source_hdmi1",
        "action": "set_source",
        "value": "HDMI1",
        "category": "source",
        "direct": ["{tv}切到HDMI1", "{tv}输入源设为HDMI1", "把{tv}调到一号HDMI", "{tv}打开HDMI一"],
        "indirect": ["我要用机顶盒", "切到盒子画面", "看外接播放器", "打开HDMI一的设备"],
        "hard_indirect": ["机顶盒已经开了，电视切过去", "外接盒子的画面准备好了", "用一号接口看节目"],
    },
    {
        "intent": "source_hdmi2",
        "action": "set_source",
        "value": "HDMI2",
        "category": "source",
        "direct": ["{tv}切到HDMI2", "{tv}输入源设为HDMI2", "把{tv}调到二号HDMI", "{tv}打开HDMI二"],
        "indirect": ["我要打游戏", "切到游戏机", "打开主机画面", "游戏画面准备好了"],
        "hard_indirect": ["手柄已经连上了，电视切到游戏机", "今晚想玩主机游戏", "二号接口的设备要显示出来"],
    },
    {
        "intent": "source_cast",
        "action": "set_source",
        "value": "cast",
        "category": "source",
        "direct": ["{tv}切到投屏", "{tv}打开投屏模式", "把{tv}输入源设为投屏", "{tv}进入无线投屏"],
        "indirect": ["我要把手机画面放到电视上", "准备投屏", "手机要连电视", "把手机内容显示到电视"],
        "hard_indirect": ["相册想给大家一起看", "手机视频放到大屏上", "把小屏幕内容搬到电视"],
    },
    {
        "intent": "play",
        "action": "playback",
        "value": "play",
        "category": "playback",
        "direct": ["{tv}开始播放", "{tv}播放", "让{tv}继续播", "{tv}开始", "播放电视内容"],
        "indirect": ["可以继续看了", "接着播放吧", "暂停结束了", "继续刚才的节目", "现在能看了"],
        "hard_indirect": ["电话打完了，节目继续", "刚才停住的画面可以动了", "把暂停的内容接上"],
    },
    {
        "intent": "pause",
        "action": "playback",
        "value": "pause",
        "category": "playback",
        "direct": ["{tv}暂停", "暂停{tv}播放", "{tv}先停一下", "把{tv}画面暂停", "{tv}暂停播放"],
        "indirect": ["我去开个门", "先别往下播", "等我一下再看", "我离开一下", "暂停一下剧情"],
        "hard_indirect": ["电话来了，剧情别错过", "有人敲门，电视等一下", "我倒杯水，画面先停住"],
    },
    {
        "intent": "picture_movie",
        "action": "set_picture_mode",
        "value": "movie",
        "category": "picture",
        "direct": ["{tv}切到电影模式", "{tv}画面设为电影模式", "打开{tv}电影画质", "{tv}启用影院模式"],
        "indirect": ["我要看电影", "画面像影院一点", "晚上适合电影画质", "看大片用舒服点的画面"],
        "hard_indirect": ["电影夜开始了，画面调成影院感", "让画面适合看暗场电影", "今晚看大片，电视调到观影效果"],
    },
    {
        "intent": "picture_sports",
        "action": "set_picture_mode",
        "value": "sports",
        "category": "picture",
        "direct": ["{tv}切到体育模式", "{tv}画面设为体育模式", "打开{tv}运动画质", "{tv}启用比赛模式"],
        "indirect": ["我要看球赛", "运动画面清楚一点", "比赛画面流畅点", "看体育节目"],
        "hard_indirect": ["球赛开场了，画面适合高速运动", "比赛要看得清楚流畅", "让电视更适合看运动直播"],
    },
    {
        "intent": "picture_game",
        "action": "set_picture_mode",
        "value": "game",
        "category": "picture",
        "direct": ["{tv}切到游戏模式", "{tv}画面设为游戏模式", "打开{tv}游戏画质", "{tv}启用低延迟模式"],
        "indirect": ["我要玩游戏", "手柄操作别延迟", "游戏画面响应快点", "准备玩主机"],
        "hard_indirect": ["需要低延迟，等会要打游戏", "主机游戏开始前把画面调顺", "让电视适合实时操作"],
    },
    {
        "intent": "subtitles_on",
        "action": "set_subtitles",
        "value": "on",
        "category": "accessibility",
        "direct": ["{tv}打开字幕", "{tv}字幕开启", "给{tv}加上字幕", "把{tv}字幕显示出来"],
        "indirect": ["台词听不清，显示文字吧", "需要看字幕", "说话声有点含糊", "外语片看不懂"],
        "hard_indirect": ["这段英文没听懂，显示字幕", "环境有点吵，台词用文字看", "对白太快了，帮我显示字幕"],
    },
    {
        "intent": "subtitles_off",
        "action": "set_subtitles",
        "value": "off",
        "category": "accessibility",
        "direct": ["{tv}关闭字幕", "{tv}字幕关掉", "取消{tv}字幕", "把{tv}字幕隐藏"],
        "indirect": ["字幕挡住画面了", "不用显示文字了", "字幕有点碍眼", "我不需要字幕"],
        "hard_indirect": ["画面底部被字遮住了", "现在能听清楚，文字可以关", "别让字幕挡住比赛比分"],
    },
]

STYLE_PREFIXES = ["", "请", "麻烦", "帮我", "现在"]
STYLE_SUFFIXES = ["。", "吧。", "一下。", "可以吗？", "谢谢。"]
INDIRECT_PREFIXES = ["", "现在", "这会儿", "刚好"]
INDIRECT_SUFFIXES = ["。", "吧。"]

FIELD_ORDER = ("device", "location", "action", "attribute", "value")
CHINESE_FIELDS = {
    "device": "设备",
    "location": "位置",
    "action": "动作",
    "attribute": "属性",
    "value": "值",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SCENIC-style TV single-device single-action benchmark samples."
    )
    parser.add_argument("--output", default="data/tv_single_action_structured.jsonl")
    parser.add_argument("--report", default="reports/tv_single_action_report.json")
    parser.add_argument("--format", choices=("jsonl", "json"), default=None)
    parser.add_argument("--seed", type=int, default=619)
    parser.add_argument("--max-per-intent", type=int, default=0, help="0 means keep every generated command.")
    parser.add_argument("--merge-input", default=None, help="Optional JSON/JSONL dataset to append after generation.")
    parser.add_argument("--merged-output", default=None, help="Optional output path for generated + merge-input rows.")
    return parser.parse_args()


def clean_prompt(text: str) -> str:
    text = text.strip()
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def with_style(command: str, rng: random.Random, command_type: str) -> str:
    if command_type == "indirect":
        prefix = rng.choice(INDIRECT_PREFIXES)
        suffix = rng.choice(INDIRECT_SUFFIXES)
    else:
        prefix = rng.choice(STYLE_PREFIXES)
        suffix = rng.choice(STYLE_SUFFIXES)
    command = command.strip("。！？?!，, ")
    if prefix and command.startswith(prefix):
        prefix = ""
    return clean_prompt(f"{prefix}{command}{suffix}")


def difficulty_for(command_type: str, intent: dict[str, Any], location: str, hard: bool) -> str:
    if hard:
        return "hard"
    if command_type == "indirect" or location != "未指定" or intent["category"] in {"source", "picture", "channel"}:
        return "medium"
    return "easy"


def make_structured_response(intent: dict[str, Any], location_key: str) -> dict[str, str]:
    attribute = {
        "set_power": "电源",
        "adjust_volume": "音量",
        "set_mute": "静音",
        "change_channel": "频道",
        "set_channel": "频道",
        "set_source": "输入源",
        "playback": "播放",
        "set_picture_mode": "画面模式",
        "set_subtitles": "字幕",
    }.get(intent["action"], "控制")
    action = {
        "set_power": "设置电源",
        "adjust_volume": "调整音量",
        "set_mute": "设置静音",
        "change_channel": "切换频道",
        "set_channel": "设置频道",
        "set_source": "设置输入源",
        "playback": "播放控制",
        "set_picture_mode": "设置画面模式",
        "set_subtitles": "设置字幕",
    }.get(intent["action"], intent["action"])
    value = {
        "on": "开启",
        "off": "关闭",
        "up": "调高",
        "down": "调低",
        "next": "下一个",
        "previous": "上一个",
        "play": "播放",
        "pause": "暂停",
        "movie": "电影",
        "sports": "体育",
        "game": "游戏",
        "cast": "投屏",
    }.get(intent["value"], intent["value"])
    return {
        "device": "电视",
        "location": location_key,
        "action": action,
        "attribute": attribute,
        "value": value,
    }


def format_structured_response(structured_response: dict[str, str]) -> str:
    return "; ".join(f"{CHINESE_FIELDS[key]}={structured_response[key]}" for key in FIELD_ORDER)


def make_row(
    prompt: str,
    intent: dict[str, Any],
    location_key: str,
    command_type: str,
    difficulty: str,
) -> dict[str, Any]:
    structured_response = make_structured_response(intent, location_key)
    response = format_structured_response(structured_response)
    return {
        "prompt": prompt,
        "response": response,
        "structured_response": structured_response,
        "device": "电视",
        "command_type": command_type,
        "difficulty": difficulty,
        "scenic_metadata": {
            "domain": "smart_home",
            "formulation": "single_device_single_action",
            "intent": intent["intent"],
            "category": intent["category"],
            "location": location_key,
        },
    }


def generate_rows(seed: int, max_per_intent: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for intent in INTENTS:
        intent_rows: list[dict[str, Any]] = []
        for location_key, location in LOCATIONS.items():
            tv_name = f"{location['prefix']}电视" if location["prefix"] else "电视"

            for template in intent["direct"]:
                base = template.format(tv=tv_name)
                variants = {base, with_style(base, rng, command_type="direct")}
                for prompt in variants:
                    row = make_row(
                        prompt=prompt,
                        intent=intent,
                        location_key=location_key,
                        command_type="direct",
                        difficulty=difficulty_for("direct", intent, location_key, hard=False),
                    )
                    key = (row["prompt"], row["response"])
                    if key not in seen:
                        intent_rows.append(row)
                        seen.add(key)

            indirect_templates = list(intent["indirect"]) + list(intent["hard_indirect"])
            for template in indirect_templates:
                hard = template in intent["hard_indirect"]
                location_phrase = f"{location['cn']}里" if location_key != "未指定" else ""
                prompt_base = f"{location_phrase}{template}" if location_phrase else template
                variants = {prompt_base, with_style(prompt_base, rng, command_type="indirect")}
                for prompt in variants:
                    row = make_row(
                        prompt=prompt,
                        intent=intent,
                        location_key=location_key,
                        command_type="indirect",
                        difficulty=difficulty_for("indirect", intent, location_key, hard=hard),
                    )
                    key = (row["prompt"], row["response"])
                    if key not in seen:
                        intent_rows.append(row)
                        seen.add(key)

        if max_per_intent > 0 and len(intent_rows) > max_per_intent:
            rng.shuffle(intent_rows)
            intent_rows = intent_rows[:max_per_intent]
        rows.extend(intent_rows)

    attach_contrastive_fields(rows, rng)
    return rows


def attach_contrastive_fields(rows: list[dict[str, Any]], rng: random.Random) -> None:
    response_groups: dict[str, list[int]] = defaultdict(list)
    category_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        response_groups[row["response"]].append(index)
        category_groups[row["scenic_metadata"]["category"]].append(index)

    for index, row in enumerate(rows):
        same_response = [candidate for candidate in response_groups[row["response"]] if candidate != index]
        positive_index = rng.choice(same_response) if same_response else index

        same_category_negative = [
            candidate
            for candidate in category_groups[row["scenic_metadata"]["category"]]
            if rows[candidate]["response"] != row["response"]
        ]
        any_negative = [candidate for candidate, other in enumerate(rows) if other["response"] != row["response"]]
        negative_index = rng.choice(same_category_negative or any_negative)

        row["positive"] = rows[positive_index]["prompt"]
        row["negative"] = rows[negative_index]["prompt"]
        row["scenic_metadata"]["positive_response"] = rows[positive_index]["response"]
        row["scenic_metadata"]["negative_response"] = rows[negative_index]["response"]
        row["scenic_metadata"]["positive_source"] = "same_structured_response"
        row["scenic_metadata"]["negative_source"] = (
            "same_category_different_response" if same_category_negative else "different_response"
        )


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                return value[key]
        return [value]
    raise ValueError(f"{path} must contain a JSON object, array, or JSONL rows.")


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


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total_records": len(rows),
        "unique_responses": len({row["response"] for row in rows}),
        "unique_prompts": len({row["prompt"] for row in rows}),
        "difficulty": dict(sorted(Counter(row["difficulty"] for row in rows).items())),
        "command_type": dict(sorted(Counter(row["command_type"] for row in rows).items())),
        "category": dict(sorted(Counter(row["scenic_metadata"]["category"] for row in rows).items())),
        "location": dict(sorted(Counter(row["scenic_metadata"]["location"] for row in rows).items())),
        "positive_source": dict(sorted(Counter(row["scenic_metadata"]["positive_source"] for row in rows).items())),
        "negative_source": dict(sorted(Counter(row["scenic_metadata"]["negative_source"] for row in rows).items())),
        "schema": {
            "response": {
                "format": "设备=<device>; 位置=<location>; 动作=<action>; 属性=<attribute>; 值=<value>",
                "device": "电视",
                "location": "未指定|客厅|卧室|书房",
                "action": "single control operation",
                "attribute": "controlled attribute",
                "value": "operation value",
            }
        },
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    rows = generate_rows(seed=args.seed, max_per_intent=args.max_per_intent)
    fmt = output_format(output_path, args.format)
    write_records(output_path, rows, fmt)

    report = build_report(rows)
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote generated TV samples: {output_path}")
    print(f"wrote generation report: {report_path}")

    if args.merge_input:
        merge_rows = read_records(Path(args.merge_input).expanduser())
        merged_output = Path(args.merged_output or "data/scenic_plus_tv_structured.jsonl").expanduser()
        write_records(merged_output, merge_rows + rows, output_format(merged_output, None))
        print(f"wrote merged dataset: {merged_output} ({len(merge_rows) + len(rows)} rows)")


if __name__ == "__main__":
    main()
