#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Environment 数据集生成 temporal_influence_shape 字段
基于新闻对时间序列的影响形态分析

Temporal Influence Shape 类型：
- immediate: 变化主要集中在早期（前两步）
- sustained: 变化持续多个时间步（总共>=4个显著变化，前两步>=2，后三步>=2）
- delayed: 其他情况（晚期变化、无显著变化等）

计算方法：
1. 计算增长率的绝对值（gt每个时间步减去上一个时间步，第一个减去历史数据最后一个）
2. 在训练集上计算7个增长率的0.75分位点作为阈值
3. 对每条样本统计显著变化的数量分布
4. 根据分布判断属于哪种类型
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def parse_time_series(value: str) -> np.ndarray:
    """解析时间序列字符串为numpy数组"""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=float)
    return np.asarray(value, dtype=float)


def calculate_growth_rates(historical: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """
    计算增长率的绝对值
    gt的每个时间步减去上一个时间步，第一个减去历史数据最后一个

    Args:
        historical: 历史数据 (7个时间步)
        ground_truth: 预测数据 (7个时间步)

    Returns:
        7个增长率的绝对值数组
    """
    if len(historical) != 7 or len(ground_truth) != 7:
        return np.array([])

    # 构造完整的序列：历史数据最后一个 + 预测数据7个时间步
    # 这样可以计算7个增长率：hist[-1]到gt[0], gt[0]到gt[1], ..., gt[5]到gt[6]
    extended_sequence = np.concatenate([[historical[-1]], ground_truth])

    # 计算7个连续的增长率
    growth_rates = []
    for i in range(len(extended_sequence) - 1):
        rate = abs(extended_sequence[i+1] - extended_sequence[i])
        growth_rates.append(rate)

    return np.array(growth_rates)


def analyze_temporal_influence_shape(growth_rates: np.ndarray, threshold: float) -> str:
    """
    分析时间影响形态

    Args:
        growth_rates: 7个增长率的绝对值
        threshold: 显著变化的阈值（0.75分位点）

    Returns:
        影响形态标签：immediate/sustained/delayed
    """
    if len(growth_rates) != 7:
        return "delayed"

    # 统计显著变化
    significant = growth_rates > threshold

    total = np.sum(significant)  # 变化显著数量
    early = np.sum(significant[:3])  # 前三步变化显著的数量
    late = np.sum(significant[3:])   # 后四步变化显著的数量

    # Sustained：total >= 4 且 early >= 2 且 late >= 2
    if total >= 4 and early >= 1 and late >= 1:
        return "Sustained"

    # Immediate：early >= 2 且 late == 0
    elif early >= 2 and late <= 1:
        return "Immediate"

    # Delayed：否则一律delayed
    else:
        return "Delayed"


def compute_threshold(growth_rates_list: List[np.ndarray]) -> float:
    """
    计算训练集上的0.75分位点作为阈值

    Args:
        growth_rates_list: 所有训练样本的增长率数组列表

    Returns:
        0.75分位点阈值
    """
    if not growth_rates_list:
        return 0.0

    # 收集所有增长率
    all_rates = []
    for rates in growth_rates_list:
        all_rates.extend(rates)

    if not all_rates:
        return 0.0

    # 计算0.75分位点
    return np.percentile(all_rates, 70)


def add_temporal_influence_shape(records: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """为每条记录添加 temporal_influence_shape 字段"""
    out: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        if "historical_data" not in rec or "ground_truth" not in rec:
            print(f"警告: 记录 {idx} 缺少必要字段")
            new_rec = dict(rec)
            new_rec["temporal_influence_shape"] = "delayed"
            out.append(new_rec)
            continue

        try:
            hist = parse_time_series(rec["historical_data"])
            gt = parse_time_series(rec["ground_truth"])

            growth_rates = calculate_growth_rates(hist, gt)
            shape = analyze_temporal_influence_shape(growth_rates, threshold)

            new_rec = dict(rec)
            new_rec["temporal_influence_shape"] = shape
            out.append(new_rec)

        except Exception as exc:
            print(f"警告: 处理记录 {idx} 时出错: {exc}")
            new_rec = dict(rec)
            new_rec["temporal_influence_shape"] = "delayed"
            out.append(new_rec)

    return out


def load_records(path: Path) -> List[Dict[str, Any]]:
    """从 path 读取 JSON list."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} in {path} is {type(item).__name__}, expected dict")
    return data


def write_json(data: List[Dict[str, Any]], path: Path) -> None:
    """写回 JSON 文件，末尾加换行方便 diff。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="为 Environment 数据集生成 temporal_influence_shape 字段"
    )
    parser.add_argument(
        "--dataset",
        default="Bitcoin",
        help="数据集文件夹名（默认：Environment）",
    )
    parser.add_argument(
        "--source-version",
        default="ver_camf",
        help="源版本文件夹名（默认：ver_camf）",
    )
    parser.add_argument(
        "--target-version",
        default="ver_synchronized_temporal_shape",
        help="目标版本文件夹名（默认：ver_synchronized_temporal_shape）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "vali", "test"],
        help="要处理的划分名（默认：train vali test）",
    )

    args = parser.parse_args()

    # 构建路径
    script_dir = Path(__file__).resolve().parents[1]  # MMTSF_LIB
    source_root = script_dir / "dataset" / args.dataset / args.source_version
    target_root = script_dir / "dataset" / args.dataset / args.target_version

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    print("=" * 80)
    print("Environment Synchronized Temporal Shape Analysis")
    print("=" * 80)
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    print(f"Splits: {args.splits}")
    print("=" * 80)

    # 第一步：计算训练集的阈值
    print("\n计算训练集阈值...")

    train_path = source_root / "train.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"Train file not found: {train_path}")

    train_records = load_records(train_path)

    # 计算所有训练样本的增长率
    train_growth_rates = []
    for rec in train_records:
        try:
            hist = parse_time_series(rec["historical_data"])
            gt = parse_time_series(rec["ground_truth"])
            rates = calculate_growth_rates(hist, gt)
            if len(rates) == 7:  # 修正：应该是7个增长率
                train_growth_rates.append(rates)
        except Exception as e:
            print(f"警告: 计算训练样本增长率时出错: {e}")
            continue

    # 计算阈值
    threshold = compute_threshold(train_growth_rates)
    print(f"  训练集样本数: {len(train_growth_rates)}")
    print(f"  训练集阈值: {threshold:.6f}")
    # 第二步：为所有数据集添加标签
    for split in args.splits:
        source_path = source_root / f"{split}.json"
        if not source_path.is_file():
            print(f"警告: 源文件不存在: {source_path}")
            continue

        print(f"\n处理 {split} 集...")

        # 加载数据
        records = load_records(source_path)

        # 添加temporal_influence_shape字段
        labeled = add_temporal_influence_shape(records, threshold)

        # 保存结果
        target_path = target_root / source_path.name
        write_json(labeled, target_path)

        print(f"  ✓ 写入 {len(labeled)} 条记录 -> {target_path.relative_to(script_dir)}")





if __name__ == "__main__":
    main()
