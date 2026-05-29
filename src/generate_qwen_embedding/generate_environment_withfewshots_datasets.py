#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 dataset/Environment/ver_generated_withfewshots 切分出三个新数据集：
1. ver_generated_withfewshots_trendstrength - 组合趋势与强度编码
2. ver_generated_withfewshots_trendonly - 只保留逐步趋势
3. ver_generated_withfewshots_globalonly - 只保留全局趋势

切分规则与 FNSPID/ver_8B_base 系列一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional


def load_dataset(input_dir: Path, split: str) -> Optional[List[dict]]:
    """加载指定切分的数据集。"""
    file_path = input_dir / f"{split}.json"
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist")
        return None
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: Iterable[dict], output_dir: Path, split: str) -> None:
    """保存新的切分数据集。"""
    data_list = list(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{split}.json"
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data_list, f, indent=2)
    print(f"Saved {len(data_list)} records to {file_path}")


def generate_trendstrength_dataset(input_data: List[dict]) -> List[dict]:
    """
    生成 trendstrength 数据集。
    news 字段格式: "12, -13, 11, -14, -15" (趋势 + 强度，直接拼接为字符串)。
    """
    output_data: List[dict] = []
    for record in input_data:
        historical_data = record.get("historical_data")
        ground_truth = record.get("ground_truth")
        step_trends = record.get("step_trends")
        step_strengths = record.get("step_strengths")

        if step_trends is None or step_strengths is None:
            continue

        news_parts = []
        for idx, trend in enumerate(step_trends):
            strength = step_strengths[idx] if idx < len(step_strengths) else 0
            news_parts.append(f"{trend}{strength}")

        output_data.append(
            {
                "historical_data": historical_data,
                "ground_truth": ground_truth,
                "news": ", ".join(news_parts),
            }
        )
    return output_data


def generate_trendonly_dataset(input_data: List[dict]) -> List[dict]:
    """生成只包含逐步趋势的 trendonly 数据集。"""
    output_data: List[dict] = []
    for record in input_data:
        historical_data = record.get("historical_data")
        ground_truth = record.get("ground_truth")
        step_trends = record.get("step_trends")

        if step_trends is None:
            continue

        output_data.append(
            {
                "historical_data": historical_data,
                "ground_truth": ground_truth,
                "news": ", ".join(str(trend) for trend in step_trends),
            }
        )
    return output_data


def generate_globalonly_dataset(input_data: List[dict]) -> List[dict]:
    """生成只包含全局趋势的 globalonly 数据集。"""
    output_data: List[dict] = []
    for record in input_data:
        historical_data = record.get("historical_data")
        ground_truth = record.get("ground_truth")
        global_trend = record.get("global_trend")

        if global_trend is None:
            continue

        output_data.append(
            {
                "historical_data": historical_data,
                "ground_truth": ground_truth,
                "news": str(global_trend),
            }
        )
    return output_data


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "../../dataset/Environment"

    input_dataset = "ver_generated_withfewshots"
    input_dir = (base_dir / input_dataset).resolve()

    output_configs = [
        ("ver_generated_withfewshots_trendstrength", generate_trendstrength_dataset),
        ("ver_generated_withfewshots_trendonly", generate_trendonly_dataset),
        ("ver_generated_withfewshots_globalonly", generate_globalonly_dataset),
    ]

    print(f"Input directory: {input_dir}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    splits = ["train", "vali", "test"]
    for split in splits:
        print(f"\nProcessing {split} split...")
        input_data = load_dataset(input_dir, split)
        if input_data is None:
            continue

        for output_name, generator in output_configs:
            output_dir = (base_dir / output_name).resolve()
            print(f"Generating {output_name}...")
            output_data = generator(input_data)
            save_dataset(output_data, output_dir, split)

    print("\nAll Environment withfewshots datasets generated successfully!")


if __name__ == "__main__":
    main()
