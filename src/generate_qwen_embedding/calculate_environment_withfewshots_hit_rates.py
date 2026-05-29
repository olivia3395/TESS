#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 Environment/ver_generated_withfewshots 系列数据集在同步标签上的命中率。
基准：
- ver_synchronized            对应 ver_generated_withfewshots_trendstrength
- ver_synchronized_trendonly  对应 ver_generated_withfewshots_trendonly
- ver_synchronized_globalonly 对应 ver_generated_withfewshots_globalonly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple


def load_dataset(dataset_path: Path, split: str) -> List[dict]:
    """加载指定 split 的数据集。"""
    file_path = dataset_path / f"{split}.json"
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist")
        return []
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_news_to_steps(news_string: str) -> List[int]:
    """
    将 news 字符串解析为整数序列。
    例如 "-12, -13, -14" -> [-12, -13, -14]
    """
    if not news_string:
        return []
    try:
        parts = [part.strip() for part in news_string.split(",")]
        return [int(part) for part in parts if part]
    except ValueError:
        return []


def calculate_step_hit_rate(gt_dataset: Iterable[dict], pred_dataset: Iterable[dict]) -> Tuple[float, int, int]:
    """计算逐步趋势命中率（方向命中）。"""
    match_steps = 0
    total_steps = 0
    min_records = min(len(gt_dataset), len(pred_dataset))

    for i in range(min_records):
        gt_steps = parse_news_to_steps(gt_dataset[i].get("news", ""))
        pred_steps = parse_news_to_steps(pred_dataset[i].get("news", ""))

        compare_len = min(len(gt_steps), len(pred_steps))
        if compare_len == 0:
            continue

        total_steps += compare_len
        for j in range(compare_len):
            if gt_steps[j] == pred_steps[j]:
                match_steps += 1

    hit_rate = match_steps / total_steps if total_steps > 0 else 0.0
    return hit_rate, total_steps, match_steps


def calculate_strength_hit_rate(gt_dataset: Iterable[dict], pred_dataset: Iterable[dict]) -> Tuple[float, int, int]:
    """计算趋势强度命中率（取绝对值末位作为强度等级）。"""
    match_strengths = 0
    total_strengths = 0
    min_records = min(len(gt_dataset), len(pred_dataset))

    for i in range(min_records):
        gt_steps = parse_news_to_steps(gt_dataset[i].get("news", ""))
        pred_steps = parse_news_to_steps(pred_dataset[i].get("news", ""))

        compare_len = min(len(gt_steps), len(pred_steps))
        if compare_len == 0:
            continue

        total_strengths += compare_len
        for j in range(compare_len):
            gt_strength = abs(gt_steps[j]) % 10
            pred_strength = abs(pred_steps[j]) % 10
            if gt_strength == pred_strength:
                match_strengths += 1

    hit_rate = match_strengths / total_strengths if total_strengths > 0 else 0.0
    return hit_rate, total_strengths, match_strengths


def calculate_global_hit_rate(gt_dataset: Iterable[dict], pred_dataset: Iterable[dict]) -> Tuple[float, int, int]:
    """计算全局趋势命中率。"""
    match_records = 0
    total_records = 0
    min_records = min(len(gt_dataset), len(pred_dataset))

    for i in range(min_records):
        gt_steps = parse_news_to_steps(gt_dataset[i].get("news", ""))
        pred_steps = parse_news_to_steps(pred_dataset[i].get("news", ""))
        if gt_steps and pred_steps:
            total_records += 1
            if gt_steps[0] == pred_steps[0]:
                match_records += 1

    hit_rate = match_records / total_records if total_records > 0 else 0.0
    return hit_rate, total_records, match_records


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_dir = (script_dir / "../../dataset/Environment").resolve()

    comparisons = [
        ("ver_synchronized", "ver_generated_withfewshots_trendstrength", "trendstrength"),
        ("ver_synchronized_trendonly", "ver_generated_withfewshots_trendonly", "trendonly"),
    ]
    global_comparisons = [
        ("ver_synchronized_globalonly", "ver_generated_withfewshots_globalonly", "globalonly"),
    ]
    splits = ["test", "vali", "train"]

    print("Environment ver_generated_withfewshots 系列命中率：")
    print("=" * 72)

    results = []

    for split in splits:
        for gt_name, pred_name, kind in comparisons:
            gt_data = load_dataset(dataset_dir / gt_name, split)
            pred_data = load_dataset(dataset_dir / pred_name, split)

            step_hit_rate, total_steps, match_steps = calculate_step_hit_rate(gt_data, pred_data)
            result = {
                "description": f"{gt_name} vs {pred_name}",
                "split": split,
                "step_hit_rate": step_hit_rate,
                "total_steps": total_steps,
                "match_steps": match_steps,
            }

            if kind == "trendstrength":
                strength_hit_rate, total_strengths, match_strengths = calculate_strength_hit_rate(gt_data, pred_data)
                result.update(
                    {
                        "strength_hit_rate": strength_hit_rate,
                        "total_strengths": total_strengths,
                        "match_strengths": match_strengths,
                    }
                )

            results.append(result)

            print(f"{gt_name} vs {pred_name} ({split}):")
            print(f"  Step级别:  匹配 {match_steps:5d}/{total_steps:5d} 命中率: {step_hit_rate:.4f} ({step_hit_rate*100:.2f}%)")
            if kind == "trendstrength":
                print(
                    f"  强度级别: 匹配 {match_strengths:5d}/{total_strengths:5d} 命中率: {strength_hit_rate:.4f} ({strength_hit_rate*100:.2f}%)"
                )
            print()

    for split in splits:
        for gt_name, pred_name, _ in global_comparisons:
            gt_data = load_dataset(dataset_dir / gt_name, split)
            pred_data = load_dataset(dataset_dir / pred_name, split)

            hit_rate, total_records, match_records = calculate_global_hit_rate(gt_data, pred_data)
            results.append(
                {
                    "description": f"{gt_name} vs {pred_name}",
                    "split": split,
                    "record_hit_rate": hit_rate,
                    "total_records": total_records,
                    "match_records": match_records,
                }
            )

            print(f"{gt_name} vs {pred_name} ({split}):")
            print(f"  Global级别:匹配 {match_records:4d}/{total_records:4d} 命中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
            print()

    output_file = script_dir / "environment_withfewshots_hit_rate_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
