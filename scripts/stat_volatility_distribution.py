#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 Electricity/ver_volatility 数据集中 global_volatility 各类的占比
"""

import json
from collections import Counter
from pathlib import Path

# 数据集路径
dataset_path = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_volatility")

# 统计所有文件
all_volatilities = []
file_stats = {}

for split in ["train", "vali", "test"]:
    file_path = dataset_path / f"{split}.json"
    if not file_path.exists():
        print(f"⚠️  文件不存在: {file_path}")
        continue
    
    print(f"正在读取: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    volatilities = []
    for item in data:
        volatility = item.get("global_volatility", "Unknown")
        volatilities.append(volatility)
        all_volatilities.append(volatility)
    
    # 统计当前文件
    counter = Counter(volatilities)
    file_stats[split] = {
        'total': len(volatilities),
        'counter': counter
    }
    print(f"  - 总样本数: {len(volatilities)}")

# 统计总体
print("\n" + "=" * 60)
print("总体统计")
print("=" * 60)
total_counter = Counter(all_volatilities)
total_samples = len(all_volatilities)

print(f"\n总样本数: {total_samples}")
print(f"\nGlobal Volatility 分类统计:")
print("-" * 60)

# 按出现频率排序显示
for volatility_type, count in total_counter.most_common():
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"{volatility_type:15s}: {count:6d} ({percentage:6.2f}%)")

# 按数据集划分统计
print("\n" + "=" * 60)
print("按数据集划分统计")
print("=" * 60)

# 获取所有可能的 volatility 类型
all_types = sorted(set(all_volatilities))

for split in ["train", "vali", "test"]:
    if split not in file_stats:
        continue
    
    stats = file_stats[split]
    print(f"\n{split.upper()}:")
    print("-" * 60)
    print(f"总样本数: {stats['total']}")
    
    for vol_type in all_types:
        count = stats['counter'].get(vol_type, 0)
        percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {vol_type:15s}: {count:6d} ({percentage:6.2f}%)")

# 生成比例表格
print("\n" + "=" * 60)
print("比例汇总表")
print("=" * 60)
print(f"{'分类':<15} {'Train':<12} {'Vali':<12} {'Test':<12} {'Overall':<12}")
print("-" * 60)

for vol_type in all_types:
    train_pct = (file_stats.get('train', {}).get('counter', {}).get(vol_type, 0) / 
                 file_stats.get('train', {}).get('total', 1) * 100) if file_stats.get('train', {}).get('total', 0) > 0 else 0
    vali_pct = (file_stats.get('vali', {}).get('counter', {}).get(vol_type, 0) / 
                file_stats.get('vali', {}).get('total', 1) * 100) if file_stats.get('vali', {}).get('total', 0) > 0 else 0
    test_pct = (file_stats.get('test', {}).get('counter', {}).get(vol_type, 0) / 
                file_stats.get('test', {}).get('total', 1) * 100) if file_stats.get('test', {}).get('total', 0) > 0 else 0
    overall_pct = (total_counter.get(vol_type, 0) / total_samples * 100) if total_samples > 0 else 0
    
    print(f"{vol_type:<15} {train_pct:>10.2f}% {vali_pct:>10.2f}% {test_pct:>10.2f}% {overall_pct:>10.2f}%")

print("\n" + "=" * 60)

