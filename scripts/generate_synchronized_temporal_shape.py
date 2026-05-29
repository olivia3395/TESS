#!/usr/bin/env python3
"""
生成 Environment ver_synchronized_temporal_shape 数据集
基于7个增长率的Temporal Influence Shape分析
"""

import json
import numpy as np
from pathlib import Path

def parse_time_series(value):
    """解析时间序列字符串"""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=float)
    return np.asarray(value, dtype=float)

def calculate_growth_rates(historical, ground_truth):
    """计算7个增长率的绝对值"""
    if len(historical) != 7 or len(ground_truth) != 7:
        return np.array([])

    # 历史数据最后一个 + 预测数据7个时间步 = 8个点，产生7个增长率
    extended_sequence = np.concatenate([[historical[-1]], ground_truth])

    growth_rates = []
    for i in range(len(extended_sequence) - 1):
        rate = abs(extended_sequence[i+1] - extended_sequence[i])
        growth_rates.append(rate)

    return np.array(growth_rates)

def analyze_temporal_influence_shape(growth_rates, threshold):
    """分析时间影响形态"""
    if len(growth_rates) != 7:
        return 'delayed'

    significant = growth_rates > threshold

    total = np.sum(significant)  # 总显著变化数
    early = np.sum(significant[:2])  # 前2个增长率显著变化数
    late = np.sum(significant[2:])   # 后5个增长率显著变化数

    # Sustained：total >= 4 且 early >= 2 且 late >= 2
    if total >= 4 and early >= 2 and late >= 2:
        return 'sustained'
    # Immediate：early >= 2 且 late == 0
    elif early >= 2 and late == 0:
        return 'immediate'
    # Delayed：其他情况
    else:
        return 'delayed'

def main():
    """主函数"""
    base_dir = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Environment")
    source_dir = base_dir / "ver_camf"
    target_dir = base_dir / "ver_synchronized_temporal_shape"
    target_dir.mkdir(exist_ok=True)

    print("生成 Environment ver_synchronized_temporal_shape 数据集")
    print("=" * 60)

    # 1. 计算训练集阈值
    print("步骤1: 计算训练集7个增长率的75分位点阈值...")

    train_data = json.load(open(source_dir / "train.json", 'r', encoding='utf-8'))

    train_growth_rates = []
    for rec in train_data:
        try:
            hist = parse_time_series(rec['historical_data'])
            gt = parse_time_series(rec['ground_truth'])
            rates = calculate_growth_rates(hist, gt)
            if len(rates) == 7:
                train_growth_rates.append(rates)
        except Exception as e:
            continue

    # 收集所有增长率
    all_rates = []
    for rates in train_growth_rates:
        all_rates.extend(rates)

    if not all_rates:
        print("错误: 没有收集到有效的增长率数据")
        return

    threshold = np.percentile(all_rates, 75)
    print(".6f"    print(f"  训练样本数: {len(train_growth_rates)}")
    print(f"  总增长率数: {len(all_rates)}")

    # 2. 生成所有数据集
    print("\n步骤2: 为所有数据集添加 temporal_influence_shape 字段...")

    for split in ['train', 'vali', 'test']:
        split_path = source_dir / f"{split}.json"
        if not split_path.exists():
            print(f"  跳过 {split}: 文件不存在")
            continue

        data = json.load(open(split_path, 'r', encoding='utf-8'))
        print(f"  处理 {split} 集: {len(data)} 条样本")

        labeled_data = []
        for rec in data:
            try:
                hist = parse_time_series(rec['historical_data'])
                gt = parse_time_series(rec['ground_truth'])
                rates = calculate_growth_rates(hist, gt)
                shape = analyze_temporal_influence_shape(rates, threshold)

                new_rec = rec.copy()
                new_rec['temporal_influence_shape'] = shape
                labeled_data.append(new_rec)
            except Exception as e:
                # 出错的样本标记为 delayed
                new_rec = rec.copy()
                new_rec['temporal_influence_shape'] = 'delayed'
                labeled_data.append(new_rec)

        # 保存结果
        output_path = target_dir / f"{split}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, ensure_ascii=False, indent=2)

        print(f"    ✓ 保存到 {output_path.name}")

    print(f"\n完成! 数据集已保存到: {target_dir}")
    print(f"使用的阈值: {threshold:.6f} (训练集增长率的75分位点)")

    # 3. 简要统计
    print("\n训练集分类统计:")
    train_labeled = json.load(open(target_dir / "train.json", 'r', encoding='utf-8'))
    shapes = [rec.get('temporal_influence_shape', 'unknown') for rec in train_labeled]

    from collections import Counter
    shape_counts = Counter(shapes)
    for shape, count in sorted(shape_counts.items()):
        pct = count / len(shapes) * 100
        print(".1f"
if __name__ == "__main__":
    main()








