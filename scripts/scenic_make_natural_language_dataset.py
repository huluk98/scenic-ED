#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FIELD_ORDER = ("device", "location", "action", "attribute", "value")
CHINESE_FIELDS = {
    "device": "设备",
    "location": "位置",
    "action": "动作",
    "attribute": "属性",
    "value": "值",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deduplicated SCENIC prompt/natural-language-response dataset.")
    parser.add_argument("--input", required=True, help="Clean SCENIC JSONL/JSON input with structured_response or structured NLP response.")
    parser.add_argument("--output", required=True, help="Output JSON file containing only prompt/response.")
    parser.add_argument("--report", default=None, help="Optional JSON validation report.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def normalize_key(text: str) -> str:
    return "".join(clean_text(text).split())


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
    raise ValueError(f"{path} must contain a JSON object or list of JSON objects")


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


def structured_ops(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("structured_response")
    items = raw if isinstance(raw, list) else [raw] if isinstance(raw, dict) else []
    ops: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ops.append({field: clean_text(item.get(field)) or "未知" for field in FIELD_ORDER})
    return ops or parse_structured_nlp(clean_text(row.get("response")))


def normalize_value(value: str) -> str:
    value = clean_text(value)
    value = value.replace("关闭闭", "关闭")
    value = value.replace("开启启", "开启")
    value = value.replace("舒睡模式一模式", "舒睡模式一")
    value = value.replace("舒睡模式二模式", "舒睡模式二")
    value = re.sub(r"模式模式$", "模式", value)
    return value


def target_name(op: dict[str, str]) -> str:
    device = clean_text(op.get("device")) or "设备"
    location = clean_text(op.get("location"))
    if location and location not in {"未指定", "未知", "none", "unspecified"}:
        return f"{location}{device}"
    return device


def operation_sentence(op: dict[str, str]) -> str:
    device = clean_text(op.get("device"))
    action = clean_text(op.get("action"))
    attribute = clean_text(op.get("attribute"))
    value = normalize_value(op.get("value", ""))
    target = target_name(op)

    if action in {"打开", "设置电源"} and value in {"开启", "打开", "on"}:
        return f"已打开{target}"
    if action in {"关闭", "设置电源"} and value in {"关闭", "关", "off"}:
        return f"已关闭{target}"
    if action == "打开":
        return f"已打开{target}"
    if action == "关闭":
        if attribute == "模式" and value:
            return f"已关闭{target}{value}"
        return f"已关闭{target}"

    if action == "设置温度":
        return f"已将{target}温度调到{value}"
    if action == "设置风速":
        return f"已将{target}风速调到{value}"
    if action == "设置风向":
        return f"已将{target}风向调到{value}"
    if action == "设置亮度":
        return f"已将{target}亮度调到{value}"
    if action == "设置模式":
        if attribute == "色温":
            return f"已将{target}调为{value}"
        return f"已将{target}切换到{value}"
    if action == "设置定时":
        return f"已为{target}设置{value}"

    if action in {"调高", "调低"}:
        direction = "调高" if action == "调高" else "调低"
        if value and value not in {"未知", "一档"}:
            return f"已将{target}{attribute}{direction}{value}"
        return f"已将{target}{attribute}{direction}一档"

    if device == "电视":
        if action == "调整音量":
            return f"已将{target}音量{value}"
        if action == "切换频道":
            return f"已将{target}切到{value}频道"
        if action == "设置频道":
            return f"已将{target}频道设为{value}"
        if action == "设置输入源":
            return f"已将{target}输入源切换到{value}"
        if action == "播放控制":
            return f"已{value}{target}播放"
        if action == "设置画面模式":
            return f"已将{target}画面模式设为{value}"
        if action == "设置字幕":
            state = "打开" if value == "开启" else "关闭" if value == "关闭" else value
            return f"已{state}{target}字幕"
        if action == "设置静音":
            state = "开启" if value == "开启" else "关闭" if value == "关闭" else value
            return f"已{state}{target}静音"

    if device == "音箱":
        if action == "播放":
            return f"已开始播放{target}"
        if action == "暂停":
            return f"已暂停{target}播放"
        if action == "下一首":
            return f"已将{target}切到下一首"

    return f"已完成{target}{attribute}{action}{value}"


def natural_response(ops: list[dict[str, str]]) -> str:
    pieces = [operation_sentence(op) for op in ops if isinstance(op, dict)]
    if not pieces:
        return ""
    return "好的，" + "；".join(pieces) + "。"


def make_dataset(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    by_prompt: dict[str, list[dict[str, str]]] = defaultdict(list)
    malformed = 0
    device_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()

    for row in rows:
        prompt = clean_text(row.get("prompt") or row.get("instruction") or row.get("input"))
        ops = structured_ops(row)
        response = natural_response(ops)
        if not prompt or not response:
            malformed += 1
            continue
        by_prompt[normalize_key(prompt)].append({"prompt": prompt, "response": response})
        for op in ops:
            device_counter[clean_text(op.get("device"))] += 1
            action_counter[clean_text(op.get("action"))] += 1

    output: list[dict[str, str]] = []
    prompt_conflicts = 0
    exact_duplicate_rows = 0
    for items in by_prompt.values():
        counts = Counter((item["prompt"], item["response"]) for item in items)
        exact_duplicate_rows += sum(count - 1 for count in counts.values())
        response_counts = Counter(item["response"] for item in items)
        if len(response_counts) > 1:
            prompt_conflicts += 1
            # Keep the majority response for deterministic recovery; ties keep the first observed response.
            winner_response = response_counts.most_common(1)[0][0]
            winner = next(item for item in items if item["response"] == winner_response)
        else:
            winner = items[0]
        output.append(winner)

    output.sort(key=lambda item: normalize_key(item["prompt"]))
    report = {
        "input_records": len(rows),
        "output_records": len(output),
        "malformed_records": malformed,
        "duplicate_prompt_rows_removed": sum(len(items) - 1 for items in by_prompt.values()),
        "exact_duplicate_prompt_response_rows_removed": exact_duplicate_rows,
        "prompt_conflicts_resolved": prompt_conflicts,
        "duplicate_prompts_after": len(output) - len({normalize_key(item["prompt"]) for item in output}),
        "duplicate_prompt_response_after": len(output) - len({(normalize_key(item["prompt"]), normalize_key(item["response"])) for item in output}),
        "duplicate_exact_rows_after": len(output) - len({json.dumps(item, ensure_ascii=False, sort_keys=True) for item in output}),
        "empty_response_after": sum(1 for item in output if not item["response"]),
        "response_style": "natural_language_only",
        "schema": {"prompt": "str", "response": "str"},
        "device_distribution": dict(sorted(device_counter.items())),
        "action_distribution": dict(sorted(action_counter.items())),
        "examples": output[:5],
    }
    return output, report


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    report_path = Path(args.report).expanduser() if args.report else None
    rows = read_records(input_path)
    output, report = make_dataset(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, ensure_ascii=False, indent=2))
    print(f"wrote natural-language dataset: {output_path}")
    if report_path:
        print(f"wrote report: {report_path}")


if __name__ == "__main__":
    main()
