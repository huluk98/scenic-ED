#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = PROJECT_ROOT / "data" / "619_Luke_clean_plus_tv_natural_language_REPAIRED.json"
DEFAULT_SINGLE = PROJECT_ROOT / "generated" / "iot_single_device_expansion.json"
DEFAULT_MULTI = PROJECT_ROOT / "generated" / "iot_multi_device_expansion.json"
DEFAULT_MULTI_BREAKDOWN = PROJECT_ROOT / "generated" / "iot_multi_device_expansion_breakdown.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200.json"
DEFAULT_BREAKDOWN = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200_breakdown.csv"
DEFAULT_REPORT = PROJECT_ROOT / "generated" / "iot_instruction_benchmark_200_report.json"

TARGETS = {"easy": 70, "medium": 65, "hard": 65}

DEVICE_TERMS = (
    "空调",
    "灯",
    "电视",
    "音箱",
    "窗帘",
    "扫地机器人",
    "空气净化器",
    "加湿器",
    "除湿机",
    "热水器",
    "风扇",
    "门锁",
    "摄像头",
    "智能插座",
)
INDIRECT_CUES = (
    "太",
    "有点",
    "准备",
    "睡觉",
    "出门",
    "客人",
    "洗澡",
    "地上",
    "空气",
    "热",
    "干",
    "潮",
    "闷",
    "晒",
    "需要",
    "想",
    "没人",
    "有人",
    "饭点",
    "会议",
    "电话",
    "电影",
    "游戏",
    "学习",
    "做饭",
    "晚安",
)
DIRECT_STARTS = ("打开", "关闭", "把", "让", "请", "帮我", "启动", "暂停", "停止", "开启", "切换", "调")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 200-item SCENIC IoT instruction benchmark with difficulty labels.")
    parser.add_argument("--original", default=str(DEFAULT_ORIGINAL))
    parser.add_argument("--single", default=str(DEFAULT_SINGLE))
    parser.add_argument("--multi", default=str(DEFAULT_MULTI))
    parser.add_argument("--multi-breakdown", default=str(DEFAULT_MULTI_BREAKDOWN))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--breakdown", default=str(DEFAULT_BREAKDOWN))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).replace("\ufeff", "").strip()


def compact(text: str) -> str:
    return re.sub(r"[\s。！？!?,，、；;：:]+", "", clean_text(text))


def load_json_rows(path: Path) -> list[dict[str, str]]:
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


def load_multi_breakdown(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["response"]: row for row in csv.DictReader(handle)}


def device_count_from_text(text: str) -> int:
    return sum(1 for term in DEVICE_TERMS if term in text)


def action_count(response: str) -> int:
    return len(response.removeprefix("好的，").removesuffix("。").split("；"))


def is_indirect(prompt: str) -> bool:
    return any(cue in prompt for cue in INDIRECT_CUES)


def is_direct(prompt: str) -> bool:
    return prompt.startswith(DIRECT_STARTS)


def infer_single_task_type(prompt: str) -> str:
    return "single_device_indirect" if is_indirect(prompt) and not is_direct(prompt) else "single_device_direct"


def group_rows(rows: list[dict[str, str]], source: str) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        enriched = {**row, "source": source}
        grouped[row["response"]].append(enriched)
    return grouped


def choose_easy(original_rows: list[dict[str, str]], single_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    original = [row for row in original_rows if "；" not in row["response"]]
    single = [row for row in single_rows if "；" not in row["response"]]
    groups = group_rows(single + original, "single_or_original")
    selected: list[dict[str, str]] = []
    used_prompts: set[str] = set()
    responses_by_device: dict[str, deque[str]] = defaultdict(deque)

    for response in sorted(groups):
        response_device = next((term for term in DEVICE_TERMS if term in response), "其他")
        if response_device == "其他":
            continue
        candidates = groups[response]
        direct = [row for row in candidates if is_direct(row["prompt"])]
        indirect = [row for row in candidates if is_indirect(row["prompt"]) and not is_direct(row["prompt"])]
        if direct and indirect:
            responses_by_device[response_device].append(response)

    device_cycle = deque(sorted(responses_by_device))
    while len(selected) < TARGETS["easy"] and device_cycle:
        device = device_cycle.popleft()
        if not responses_by_device[device]:
            continue
        response = responses_by_device[device].popleft()
        candidates = groups[response]
        direct = next(row for row in candidates if is_direct(row["prompt"]))
        indirect = next(row for row in candidates if is_indirect(row["prompt"]) and not is_direct(row["prompt"]))
        for row in (direct, indirect):
            key = compact(row["prompt"])
            if key not in used_prompts and len(selected) < TARGETS["easy"]:
                selected.append({**row, "task_type": infer_single_task_type(row["prompt"])})
                used_prompts.add(key)
        if responses_by_device[device]:
            device_cycle.append(device)

    if len(selected) != TARGETS["easy"]:
        raise ValueError(f"Only selected {len(selected)} easy examples")
    return selected


def prompt_rank(prompt: str, prefer_indirect: bool) -> tuple[int, int, str]:
    has_indirect = is_indirect(prompt)
    command_only = not has_indirect and ("，" not in prompt)
    if prefer_indirect:
        return (0 if has_indirect else 1, 1 if command_only else 0, len(prompt), prompt)
    return (0 if is_direct(prompt) else 1, 0 if "再" in prompt or "并" in prompt else 1, len(prompt), prompt)


def choose_multi(
    multi_rows: list[dict[str, str]],
    breakdown: dict[str, dict[str, str]],
    difficulty: str,
    target: int,
) -> list[dict[str, str]]:
    grouped = group_rows(multi_rows, "multi_expansion")
    eligible: list[tuple[str, str, int, int]] = []
    for response, rows in grouped.items():
        meta = breakdown.get(response)
        if not meta:
            continue
        fragments = action_count(response)
        devices = [part for part in meta["devices"].split("|") if part]
        if len(devices) < 2:
            continue
        if difficulty == "medium" and fragments == 2:
            eligible.append((meta["scenario"], response, len(devices), fragments))
        elif difficulty == "hard" and fragments >= 3:
            eligible.append((meta["scenario"], response, len(devices), fragments))

    by_scenario: dict[str, deque[str]] = defaultdict(deque)
    for scenario, response, _devices, fragments in sorted(eligible, key=lambda item: (item[0], item[3], item[1])):
        by_scenario[scenario].append(response)

    selected: list[dict[str, str]] = []
    used_prompts: set[str] = set()
    scenario_cycle = deque(sorted(by_scenario))
    prefer_indirect = difficulty == "hard"
    per_response_limit = 2

    while len(selected) < target and scenario_cycle:
        scenario = scenario_cycle.popleft()
        if not by_scenario[scenario]:
            continue
        response = by_scenario[scenario].popleft()
        candidates = sorted(grouped[response], key=lambda row: prompt_rank(row["prompt"], prefer_indirect))
        picked_for_response = 0
        for row in candidates:
            key = compact(row["prompt"])
            if key in used_prompts:
                continue
            selected.append(
                {
                    **row,
                    "task_type": "multi_device_two_action" if difficulty == "medium" else "multi_device_multi_action_indirect",
                    "scenario": scenario,
                }
            )
            used_prompts.add(key)
            picked_for_response += 1
            if len(selected) == target or picked_for_response == per_response_limit:
                break
        if by_scenario[scenario]:
            scenario_cycle.append(scenario)

    if len(selected) != target:
        raise ValueError(f"Only selected {len(selected)} {difficulty} examples")
    return selected


def make_benchmark(
    original_rows: list[dict[str, str]],
    single_rows: list[dict[str, str]],
    multi_rows: list[dict[str, str]],
    breakdown: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    pools = {
        "easy": choose_easy(original_rows, single_rows),
        "medium": choose_multi(multi_rows, breakdown, "medium", TARGETS["medium"]),
        "hard": choose_multi(multi_rows, breakdown, "hard", TARGETS["hard"]),
    }

    records: list[dict[str, Any]] = []
    used_prompts: set[str] = set()
    for difficulty in ("easy", "medium", "hard"):
        for row in pools[difficulty]:
            key = compact(row["prompt"])
            if key in used_prompts:
                raise ValueError(f"duplicate prompt in benchmark: {row['prompt']}")
            used_prompts.add(key)
            record = {
                "id": f"bench_{len(records) + 1:04d}",
                "difficulty": difficulty,
                "task_type": row["task_type"],
                "prompt": row["prompt"],
                "response": row["response"],
                "source": row["source"],
                "response_action_count": action_count(row["response"]),
                "device_term_count": device_count_from_text(row["response"]),
            }
            if "scenario" in row:
                record["scenario"] = row["scenario"]
            records.append(record)
    return records


def validate_benchmark(records: list[dict[str, Any]]) -> None:
    if len(records) != 200:
        raise ValueError(f"Expected 200 records, got {len(records)}")
    counts = Counter(record["difficulty"] for record in records)
    if counts != TARGETS:
        raise ValueError(f"Difficulty count mismatch: {counts}")
    if len({compact(record["prompt"]) for record in records}) != len(records):
        raise ValueError("Duplicate prompt detected")
    for record in records:
        if not record["response"].startswith("好的，"):
            raise ValueError(f"Bad response prefix: {record}")
        if record["difficulty"] == "easy" and "；" in record["response"]:
            raise ValueError(f"Easy example has multi-action response: {record}")
        if record["difficulty"] == "medium" and action_count(record["response"]) != 2:
            raise ValueError(f"Medium example should have exactly two actions: {record}")
        if record["difficulty"] == "hard" and action_count(record["response"]) < 3:
            raise ValueError(f"Hard example should have at least three actions: {record}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_breakdown(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["difficulty", "task_type", "scenario", "response_action_count", "device_term_count", "count"]
    counter: Counter[tuple[Any, ...]] = Counter(
        (
            record["difficulty"],
            record["task_type"],
            record.get("scenario", ""),
            record["response_action_count"],
            record["device_term_count"],
        )
        for record in records
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, count in sorted(counter.items()):
            writer.writerow(
                {
                    "difficulty": key[0],
                    "task_type": key[1],
                    "scenario": key[2],
                    "response_action_count": key[3],
                    "device_term_count": key[4],
                    "count": count,
                }
            )


def make_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "difficulty_distribution": dict(Counter(record["difficulty"] for record in records)),
        "task_type_distribution": dict(Counter(record["task_type"] for record in records)),
        "scenario_distribution": dict(Counter(record.get("scenario", "single_device") for record in records)),
        "action_count_distribution": dict(Counter(str(record["response_action_count"]) for record in records)),
        "unique_prompts": len({compact(record["prompt"]) for record in records}),
        "unique_responses": len({record["response"] for record in records}),
        "top_many_to_one_groups": [
            {"response": response, "count": count}
            for response, count in Counter(record["response"] for record in records).most_common(20)
            if count > 1
        ],
        "examples": records[:15],
    }


def main() -> None:
    args = parse_args()
    original_rows = load_json_rows(Path(args.original).expanduser())
    single_rows = load_json_rows(Path(args.single).expanduser())
    multi_rows = load_json_rows(Path(args.multi).expanduser())
    breakdown = load_multi_breakdown(Path(args.multi_breakdown).expanduser())

    records = make_benchmark(original_rows, single_rows, multi_rows, breakdown)
    validate_benchmark(records)

    output_path = Path(args.output).expanduser()
    breakdown_path = Path(args.breakdown).expanduser()
    report_path = Path(args.report).expanduser()
    write_json(output_path, records)
    write_breakdown(breakdown_path, records)
    write_json(report_path, make_report(records))

    report = make_report(records)
    print(f"total: {report['total']}")
    print(f"unique_prompts: {report['unique_prompts']}")
    print(f"unique_responses: {report['unique_responses']}")
    print(f"difficulty_distribution: {report['difficulty_distribution']}")
    print(f"task_type_distribution: {report['task_type_distribution']}")
    print(f"scenario_distribution: {report['scenario_distribution']}")
    print(f"output: {output_path}")
    print(f"breakdown: {breakdown_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
