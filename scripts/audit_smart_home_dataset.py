#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROMPT_FIELDS = ("prompt", "instruction", "input", "query", "user_command")
RESPONSE_FIELDS = ("response", "output", "answer", "target")
CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}
MAX_EXAMPLES_PER_TYPE = 25


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Audit smart-home prompt/response dataset quality without editing data.")
    parser.add_argument("--old", default=str(root / "data" / "619_Luke_fixed_dedup.json"))
    parser.add_argument("--new", default=str(root / "data" / "619_Luke_clean_plus_tv_natural_language_dedup.json"))
    parser.add_argument("--report", default=str(root / "reports" / "dataset_audit_report.md"))
    parser.add_argument("--suspicious", default=str(root / "reports" / "suspicious_examples.jsonl"))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def compact(text: str) -> str:
    return "".join(clean_text(text).split())


def read_json_list(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        for key in ("records", "data", "items", "examples"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list or a dict containing a list")

    rows: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{index} is not a JSON object")
        prompt = first_text(item, PROMPT_FIELDS)
        response = first_text(item, RESPONSE_FIELDS)
        rows.append({"prompt": prompt, "response": response})
    return rows


def first_text(record: dict[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        text = clean_text(record.get(field))
        if text:
            return text
    return ""


def response_style(response: str) -> str:
    response = clean_text(response)
    if not response:
        return "empty"
    if "设备=" in response or "动作=" in response:
        return "structured_nlp"
    if re.search(r"已播放.*播放", response):
        return "duplicated_playback_phrase"
    if any(term in response for term in ("半时", "一时", "二时", "三时", "关闭闭", "模式模式")):
        return "timer_or_wording_artifact"
    if "；" in response:
        return "multi_action_natural"
    if response.startswith("好的，已"):
        return "okay_completed_natural"
    if response.startswith("好的，"):
        return "okay_other_natural"
    if response.startswith("已"):
        return "bare_completed_natural"
    return "other_natural"


def top_responses(rows: list[dict[str, str]], n: int = 50) -> list[tuple[str, int]]:
    return Counter(row["response"] for row in rows).most_common(n)


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def extract_tv_value(text: str, choices: tuple[str, ...]) -> str:
    for choice in choices:
        if choice in text:
            return choice
    return ""


def infer_mode(prompt: str) -> str:
    for mode in ("制热", "制冷", "送风", "抽湿", "除湿"):
        if mode in prompt:
            return "抽湿模式" if mode in {"抽湿", "除湿"} else f"{mode}模式"
    return ""


def suggest_for_prompt(prompt: str, response: str) -> str:
    p = compact(prompt)
    r = clean_text(response)
    mode = infer_mode(p)
    if mode and "空调" in p:
        return f"好的，已将空调切换到{mode}。"
    if "电视" in p:
        if "字幕" in p:
            state = "打开" if contains_any(p, ("打开", "开启", "开字幕", "显示")) else "关闭"
            return f"好的，已{state}电视字幕。"
        if "静音" in p or "mute" in p.lower():
            state = "开启" if contains_any(p, ("打开", "开启", "静音", "禁音")) and not contains_any(p, ("关闭静音", "取消静音", "关静音")) else "关闭"
            return f"好的，已{state}电视静音。"
        if "投屏" in p or "HDMI1" in p.upper() or "HDMI2" in p.upper() or "输入源" in p:
            value = extract_tv_value(p.upper(), ("HDMI1", "HDMI2")) or ("投屏" if "投屏" in p else "输入源")
            return f"好的，已将电视输入源切换到{value}。"
        if "CCTV-1" in p.upper() or "CCTV1" in p.upper() or "频道" in p or "体育" in p:
            value = "CCTV-1" if "CCTV" in p.upper() else "体育" if "体育" in p else "指定"
            return f"好的，已将电视频道设为{value}。"
        if any(term in p for term in ("游戏", "电影", "体育", "画面模式")):
            value = extract_tv_value(p, ("游戏", "电影", "体育")) or "指定"
            return f"好的，已将电视画面模式设为{value}。"
        if "播放" in p:
            return "好的，已开始播放电视。"
        if "暂停" in p:
            return "好的，已暂停电视播放。"
    if contains_any(p, ("不要开灯", "别开灯", "不用开灯", "不要打开灯", "别打开灯")):
        return "好的，不打开灯。"
    if contains_any(p, ("不要关灯", "别关灯", "不用关灯", "不要关闭灯", "别关闭灯")):
        return "好的，不关闭灯。"
    if contains_any(r, ("半时", "一时", "二时", "三时", "关闭闭", "模式模式")):
        return normalize_response_artifacts(r)
    return r


def normalize_response_artifacts(response: str) -> str:
    replacements = {
        "半时": "半小时",
        "一时": "一小时",
        "二时": "二小时",
        "三时": "三小时",
        "关闭闭": "关闭",
        "开启启": "开启",
        "模式模式": "模式",
    }
    fixed = response
    for old, new in replacements.items():
        fixed = fixed.replace(old, new)
    fixed = re.sub(r"已播放(.+?)播放", r"已开始播放\1", fixed)
    return fixed


def add_suspicious(
    out: list[dict[str, Any]],
    dataset: str,
    index: int,
    prompt: str,
    response: str,
    error_type: str,
    suggested_response: str,
    confidence: str,
) -> None:
    out.append(
        {
            "dataset": dataset,
            "index": index,
            "prompt": prompt,
            "response": response,
            "error_type": error_type,
            "suggested_response": suggested_response,
            "confidence": confidence,
        }
    )


def detect_pattern_errors(dataset: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    suspicious: list[dict[str, Any]] = []
    suffix_dup_re = re.compile(r"(吧吧|谢谢谢谢|可以吗可以吗|嘛嘛|呀呀)$")
    timer_bad_re = re.compile(r"(半时|一时|二时|三时|关闭闭|模式模式)")

    for index, row in enumerate(rows):
        prompt = row["prompt"]
        response = row["response"]
        p = compact(prompt)
        r = compact(response)

        if re.search(r"已播放.*播放", response):
            add_suspicious(suspicious, dataset, index, prompt, response, "duplicated_playback_phrase", normalize_response_artifacts(response), "high")

        if timer_bad_re.search(response):
            add_suspicious(suspicious, dataset, index, prompt, response, "invalid_timer_wording", normalize_response_artifacts(response), "high")

        if infer_mode(p) and "空调" in p and r in {"好的，已打开空调。", "已打开空调。", "好的，已打开空调"}:
            add_suspicious(suspicious, dataset, index, prompt, response, "ac_mode_prompt_mapped_to_power_on", suggest_for_prompt(prompt, response), "high")

        if "电视" in p and contains_any(p, ("字幕", "静音", "mute")) and "已关闭电视" in response and not contains_any(p, ("关电视", "关闭电视", "关掉电视", "不看电视")):
            add_suspicious(suspicious, dataset, index, prompt, response, "tv_subtitle_or_mute_mapped_to_tv_off", suggest_for_prompt(prompt, response), "high")

        if "电视" in p and contains_any(p, ("字幕", "输入源", "HDMI", "hdmi", "投屏", "频道", "CCTV", "cctv", "画面", "游戏", "电影", "体育")) and "已打开电视" in response and not contains_any(p, ("开电视", "打开电视", "开启电视")):
            add_suspicious(suspicious, dataset, index, prompt, response, "tv_setting_mapped_to_tv_on", suggest_for_prompt(prompt, response), "high")

        if "电视" in p and "播放" in p and "已开始播放音箱" in response:
            add_suspicious(suspicious, dataset, index, prompt, response, "tv_playback_mapped_to_speaker", "好的，已开始播放电视。", "high")

        if contains_any(p, ("不要开灯", "别开灯", "不用开灯", "不要打开灯", "别打开灯")) and "已打开灯" in response:
            add_suspicious(suspicious, dataset, index, prompt, response, "contradictory_negation", "好的，不打开灯。", "high")

        if contains_any(p, ("不要关灯", "别关灯", "不用关灯", "不要关闭灯", "别关闭灯")) and "已关闭灯" in response:
            add_suspicious(suspicious, dataset, index, prompt, response, "contradictory_negation", "好的，不关闭灯。", "high")

        if suffix_dup_re.search(p):
            add_suspicious(suspicious, dataset, index, prompt, response, "prompt_suffix_duplication", response, "medium")

    return suspicious


def extract_number(text: str, suffix: str = "") -> str:
    pattern = rf"(\d+){re.escape(suffix)}" if suffix else r"(\d+)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def infer_intent_key(prompt: str, response: str) -> str | None:
    p = compact(prompt)
    r = compact(response)
    text = f"{p} {r}"

    # TV intents.
    if "电视" in text:
        if "字幕" in text:
            state = "on" if contains_any(text, ("打开字幕", "开启字幕", "字幕开启", "已打开电视字幕", "已开启电视字幕", "显示字幕")) else "off" if contains_any(text, ("关闭字幕", "字幕关闭", "已关闭电视字幕")) else ""
            return f"tv:subtitles:{state}" if state else None
        if "静音" in text:
            state = "off" if contains_any(text, ("关闭静音", "取消静音", "已关闭电视静音")) else "on" if contains_any(text, ("开启静音", "打开静音", "静音", "已开启电视静音")) else ""
            return f"tv:mute:{state}" if state else None
        if "暂停" in text:
            return "tv:playback:pause"
        if "播放" in text:
            return "tv:playback:play"
        if "HDMI1" in text.upper():
            return "tv:input:HDMI1"
        if "HDMI2" in text.upper():
            return "tv:input:HDMI2"
        if "投屏" in text:
            return "tv:input:投屏"
        if "CCTV-1" in text.upper() or "CCTV1" in text.upper():
            return "tv:channel:CCTV-1"
        if "频道" in text and "体育" in text:
            return "tv:channel:体育"
        if "画面" in text or "画面模式" in text or any(mode in text for mode in ("游戏", "电影", "体育")):
            value = extract_tv_value(text, ("游戏", "电影", "体育"))
            return f"tv:picture_mode:{value}" if value else None

    # Light intents.
    if any(term in text for term in ("灯", "灯光", "亮度", "色温")):
        location = extract_location(text)
        prefix = f"light:{location}:"
        if contains_any(text, ("关闭", "关灯", "已关闭")):
            return prefix + "power:off"
        if contains_any(text, ("打开", "开灯", "已打开")) and not contains_any(text, ("模式", "亮度", "色温", "冷色", "暖色", "自然光")):
            return prefix + "power:on"
        if contains_any(text, ("调高", "调亮", "亮一点")):
            return prefix + "brightness:increase"
        if contains_any(text, ("调低", "调暗", "暗一点")):
            return prefix + "brightness:decrease"
        if "亮度" in text:
            value = "最高" if "最高" in text or "最亮" in text else "最低" if "最低" in text or "最暗" in text else extract_number(text, "%")
            return prefix + f"brightness:set:{value}"
        for value in ("普通模式", "学习模式", "夜灯模式", "睡眠模式", "舒睡模式"):
            if value in text:
                return prefix + f"mode:{value}"
        for value in ("冷色光", "暖色光", "自然光"):
            if value in text:
                return prefix + f"color:{value}"

    # AC intents.
    if any(term in text for term in ("空调", "温度", "风速", "风向", "制冷", "制热", "送风", "抽湿")):
        location = extract_location(text)
        prefix = f"ac:{location}:"
        for mode in ("制冷模式", "制热模式", "送风模式", "抽湿模式", "自动模式", "环保模式", "静音模式", "睡眠模式", "强劲模式"):
            if mode in text:
                return prefix + f"mode:{mode}"
        short_mode = infer_mode(text)
        if short_mode:
            return prefix + f"mode:{short_mode}"
        if "温度" in text or re.search(r"\d+度", text):
            value = re.search(r"\d+度", text)
            if value:
                return prefix + f"temperature:{value.group(0)}"
            if contains_any(text, ("调高", "升高")):
                return prefix + "temperature:increase"
            if contains_any(text, ("调低", "降低")):
                return prefix + "temperature:decrease"
        if "风速" in text:
            if contains_any(text, ("调高", "提高", "更大", "最高")):
                return prefix + "fan_speed:increase_or_high"
            if contains_any(text, ("调低", "降低", "更小", "最低")):
                return prefix + "fan_speed:decrease_or_low"
            return prefix + "fan_speed:set"
        if "风向" in text:
            for value in ("上下左右", "上下", "左右", "左", "下", "不动"):
                if value in text:
                    return prefix + f"wind:{value}"
            return prefix + "wind:set"
        if contains_any(text, ("分钟后", "小时后", "点后", "定时")):
            return prefix + "timer"
        if contains_any(text, ("关闭", "关掉", "已关闭")):
            return prefix + "power:off"
        if contains_any(text, ("打开", "开启", "已打开")):
            return prefix + "power:on"

    return None


def extract_location(text: str) -> str:
    for location in ("客厅", "主卧室", "主卧", "次卧", "卧室", "厨房", "书房", "阳台", "餐厅", "浴室", "洗手间", "房间"):
        if location in text:
            return "主卧" if location == "主卧室" else location
    return "未指定"


def detect_inconsistent_mappings(dataset: str, rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = infer_intent_key(row["prompt"], row["response"])
        if key:
            groups[key].append((index, row))

    suspicious: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        response_counts = Counter(row["response"] for _, row in items)
        if len(response_counts) <= 1:
            continue
        canonical, canonical_count = response_counts.most_common(1)[0]
        group_summaries.append(
            {
                "dataset": dataset,
                "intent_key": key,
                "examples": len(items),
                "unique_responses": len(response_counts),
                "canonical_response": canonical,
                "top_responses": response_counts.most_common(8),
            }
        )
        for index, row in items:
            if row["response"] == canonical:
                continue
            confidence = "high" if canonical_count >= 3 else "medium"
            add_suspicious(
                suspicious,
                dataset,
                index,
                row["prompt"],
                row["response"],
                f"inconsistent_same_intent_mapping:{key}",
                canonical,
                confidence,
            )
    return suspicious, group_summaries


def dedupe_suspicious(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for item in items:
        key = (item["dataset"], item["index"], item["prompt"], item["error_type"])
        existing = by_key.get(key)
        if existing is None or CONFIDENCE_ORDER[item["confidence"]] < CONFIDENCE_ORDER[existing["confidence"]]:
            by_key[key] = item
    return sorted(by_key.values(), key=lambda item: (item["dataset"], item["error_type"], CONFIDENCE_ORDER[item["confidence"]], item["index"]))


def style_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(response_style(row["response"]) for row in rows).items()))


def write_suspicious_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    old_path: Path,
    new_path: Path,
    old_rows: list[dict[str, str]],
    new_rows: list[dict[str, str]],
    suspicious: list[dict[str, Any]],
    inconsistent_groups: list[dict[str, Any]],
) -> str:
    suspicious_counts = Counter(item["error_type"].split(":", 1)[0] for item in suspicious)
    dataset_counts = Counter(item["dataset"] for item in suspicious)

    lines: list[str] = []
    lines.append("# Smart-Home Dataset Audit Report")
    lines.append("")
    lines.append("This report is read-only: no dataset files were modified.")
    lines.append("")
    lines.append("## Files")
    lines.append(f"- older_reference: `{old_path}`")
    lines.append(f"- newer_dataset: `{new_path}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(
        markdown_table(
            [
                ["older_reference", len(old_rows), len({row["response"] for row in old_rows}), len({row["prompt"] for row in old_rows}), dataset_counts.get("older_reference", 0)],
                ["newer_dataset", len(new_rows), len({row["response"] for row in new_rows}), len({row["prompt"] for row in new_rows}), dataset_counts.get("newer_dataset", 0)],
            ],
            ["dataset", "total_examples", "unique_responses", "unique_prompts", "suspicious_examples"],
        )
    )
    lines.append("")
    lines.append("## Response Style Distribution")
    for name, rows in (("older_reference", old_rows), ("newer_dataset", new_rows)):
        lines.append(f"### {name}")
        lines.append(markdown_table([[key, value] for key, value in style_distribution(rows).items()], ["style", "count"]))
        lines.append("")
    lines.append("## Suspicious Count By Error Type")
    lines.append(markdown_table([[key, value] for key, value in sorted(suspicious_counts.items())], ["error_type", "count"]))
    lines.append("")
    lines.append("## Top 50 Responses")
    for name, rows in (("older_reference", old_rows), ("newer_dataset", new_rows)):
        lines.append(f"### {name}")
        lines.append(markdown_table([[response, count] for response, count in top_responses(rows)], ["response", "count"]))
        lines.append("")
    lines.append("## Likely Inconsistent Same-Intent Groups")
    if inconsistent_groups:
        for group in inconsistent_groups[:80]:
            lines.append(f"### {group['dataset']} :: `{group['intent_key']}`")
            lines.append(f"- examples: {group['examples']}")
            lines.append(f"- unique responses: {group['unique_responses']}")
            lines.append(f"- suggested canonical response: `{group['canonical_response']}`")
            lines.append(markdown_table([[response, count] for response, count in group["top_responses"]], ["response", "count"]))
            lines.append("")
    else:
        lines.append("No inconsistent same-intent groups detected.")
        lines.append("")
    lines.append("## Suspicious Examples By Error Type")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in suspicious:
        grouped[item["error_type"]].append(item)
    for error_type, items in sorted(grouped.items()):
        lines.append(f"### {error_type} ({len(items)})")
        rows = []
        for item in items[:MAX_EXAMPLES_PER_TYPE]:
            rows.append([item["dataset"], item["index"], item["confidence"], item["prompt"], item["response"], item["suggested_response"]])
        lines.append(markdown_table(rows, ["dataset", "index", "confidence", "prompt", "response", "suggested_response"]))
        if len(items) > MAX_EXAMPLES_PER_TYPE:
            lines.append(f"_Showing first {MAX_EXAMPLES_PER_TYPE} of {len(items)} examples._")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    old_path = Path(args.old).expanduser()
    new_path = Path(args.new).expanduser()
    report_path = Path(args.report).expanduser()
    suspicious_path = Path(args.suspicious).expanduser()

    old_rows = read_json_list(old_path)
    new_rows = read_json_list(new_path)

    suspicious: list[dict[str, Any]] = []
    inconsistent_groups: list[dict[str, Any]] = []
    for dataset, rows in (("older_reference", old_rows), ("newer_dataset", new_rows)):
        suspicious.extend(detect_pattern_errors(dataset, rows))
        inconsistent, groups = detect_inconsistent_mappings(dataset, rows)
        suspicious.extend(inconsistent)
        inconsistent_groups.extend(groups)
    suspicious = dedupe_suspicious(suspicious)

    write_suspicious_jsonl(suspicious_path, suspicious)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        build_markdown_report(old_path, new_path, old_rows, new_rows, suspicious, inconsistent_groups),
        encoding="utf-8",
    )

    summary = {
        "older_reference_total": len(old_rows),
        "older_reference_unique_responses": len({row["response"] for row in old_rows}),
        "newer_dataset_total": len(new_rows),
        "newer_dataset_unique_responses": len({row["response"] for row in new_rows}),
        "suspicious_examples": len(suspicious),
        "suspicious_by_dataset": dict(sorted(Counter(item["dataset"] for item in suspicious).items())),
        "suspicious_by_error_type": dict(sorted(Counter(item["error_type"].split(":", 1)[0] for item in suspicious).items())),
        "inconsistent_same_intent_groups": len(inconsistent_groups),
        "report": str(report_path),
        "suspicious_jsonl": str(suspicious_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
