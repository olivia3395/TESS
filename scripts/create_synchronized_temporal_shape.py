#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

# 简化的实现
def parse_time_series(value):
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=float)
    return np.asarray(value, dtype=float)

def calculate_growth_rates(historical, ground_truth):
    if len(historical) != 7 or len(ground_truth) != 7:
        return np.array([])

    full_sequence = np.concatenate([historical, ground_truth])

    growth_rates = []
    for i in range(7, 11):  # 5个增长率
        rate = abs(full_sequence[i+1] - full_sequence[i])
        growth_rates.append(rate)

    return np.array(growth_rates)

def analyze_temporal_influence_shape(growth_rates, threshold):
    if len(growth_rates) != 5:
        return 'delayed'

    significant = growth_rates > threshold

    total = np.sum(significant)
    early = np.sum(significant[:2])
    late = np.sum(significant[2:])

    if total >= 4 and early >= 2 and late >= 2:
        return 'sustained'
    elif early >= 2 and late == 0:
        return 'immediate'
    else:
        return 'delayed'

def compute_threshold(growth_rates_list):
    if not growth_rates_list:
        return 0.0

    all_rates = []
    for rates in growth_rates_list:
        all_rates.extend(rates)

    if not all_rates:
        return 0.0

    return np.percentile(all_rates, 75)

# 主逻辑
base_dir = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Environment")
source_dir = base_dir / "ver_camf"
target_dir = base_dir / "ver_synchronized_temporal_shape"
target_dir.mkdir(exist_ok=True)

print("开始处理...")

# 计算训练集阈值
train_data = json.load(open(source_dir / "train.json", 'r', encoding='utf-8'))
print(f"加载训练数据: {len(train_data)} 条")

train_growth_rates = []
for rec in train_data:
    try:
        hist = parse_time_series(rec['historical_data'])
        gt = parse_time_series(rec['ground_truth'])
        rates = calculate_growth_rates(hist, gt)
        if len(rates) == 5:
            train_growth_rates.append(rates)
    except:
        continue

threshold = compute_threshold(train_growth_rates)
print(f"计算阈值: {threshold:.6f}")

# 处理所有数据集
for split in ['train', 'vali', 'test']:
    data = json.load(open(source_dir / f"{split}.json", 'r', encoding='utf-8'))
    print(f"处理 {split} 集: {len(data)} 条")

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
        except:
            new_rec = rec.copy()
            new_rec['temporal_influence_shape'] = 'delayed'
            labeled_data.append(new_rec)

    # 保存
    output_path = target_dir / f"{split}.json"
    json.dump(labeled_data, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"保存到: {output_path}")

print("完成！")








