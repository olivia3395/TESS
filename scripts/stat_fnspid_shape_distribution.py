#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计 FNSPID/label/ver_shape 数据集中 shape 五种分类的比例
"""

import json
from collections import Counter
from pathlib import Path

# 数据集路径
dataset_path = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/FNSPID/label/ver_shape")

# 定义五种 shape 分类
shape_types = ["Rise", "Fall", "Recover", "Oscillate", "Stable"]

# 统计所有文件
all_shapes = []
file_stats = {}

for split in ["train", "vali", "test"]:
    file_path = dataset_path / f"{split}.json"
    if not file_path.exists():
        print(f"⚠️  文件不存在: {file_path}")
        continue
    
    print(f"正在读取: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    shapes = []
    for item in data:
        shape = item.get("shape", "Unknown")
        shapes.append(shape)
        all_shapes.append(shape)
    
    # 统计当前文件
    counter = Counter(shapes)
    file_stats[split] = {
        'total': len(shapes),
        'counter': counter
    }
    print(f"  - 总样本数: {len(shapes)}")

# 统计总体
print("\n" + "=" * 60)
print("总体统计")
print("=" * 60)
total_counter = Counter(all_shapes)
total_samples = len(all_shapes)

print(f"\n总样本数: {total_samples}")
print(f"\nShape 分类统计:")
print("-" * 60)

# 按定义的顺序显示
for shape_type in shape_types:
    count = total_counter.get(shape_type, 0)
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"{shape_type:12s}: {count:6d} ({percentage:6.2f}%)")

# 显示其他未预期的分类
other_shapes = set(total_counter.keys()) - set(shape_types)
if other_shapes:
    print("\n其他分类:")
    for shape in sorted(other_shapes):
        count = total_counter[shape]
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{shape:12s}: {count:6d} ({percentage:6.2f}%)")

# 按数据集划分统计
print("\n" + "=" * 60)
print("按数据集划分统计")
print("=" * 60)

for split in ["train", "vali", "test"]:
    if split not in file_stats:
        continue
    
    stats = file_stats[split]
    print(f"\n{split.upper()}:")
    print("-" * 60)
    print(f"总样本数: {stats['total']}")
    
    for shape_type in shape_types:
        count = stats['counter'].get(shape_type, 0)
        percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {shape_type:12s}: {count:6d} ({percentage:6.2f}%)")
    
    # 其他分类
    other = set(stats['counter'].keys()) - set(shape_types)
    if other:
        for shape in sorted(other):
            count = stats['counter'][shape]
            percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  {shape:12s}: {count:6d} ({percentage:6.2f}%)")

# 生成比例表格
print("\n" + "=" * 60)
print("比例汇总表")
print("=" * 60)
print(f"{'分类':<12} {'Train':<12} {'Vali':<12} {'Test':<12} {'Overall':<12}")
print("-" * 60)

for shape_type in shape_types:
    train_pct = (file_stats.get('train', {}).get('counter', {}).get(shape_type, 0) / 
                 file_stats.get('train', {}).get('total', 1) * 100) if file_stats.get('train', {}).get('total', 0) > 0 else 0
    vali_pct = (file_stats.get('vali', {}).get('counter', {}).get(shape_type, 0) / 
                file_stats.get('vali', {}).get('total', 1) * 100) if file_stats.get('vali', {}).get('total', 0) > 0 else 0
    test_pct = (file_stats.get('test', {}).get('counter', {}).get(shape_type, 0) / 
                file_stats.get('test', {}).get('total', 1) * 100) if file_stats.get('test', {}).get('total', 0) > 0 else 0
    overall_pct = (total_counter.get(shape_type, 0) / total_samples * 100) if total_samples > 0 else 0
    
    print(f"{shape_type:<12} {train_pct:>10.2f}% {vali_pct:>10.2f}% {test_pct:>10.2f}% {overall_pct:>10.2f}%")

print("\n" + "=" * 60)

