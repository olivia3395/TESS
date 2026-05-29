#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计Electricity/ver_shape数据集的shape标签分布
"""

import json
import os
from collections import Counter
from pathlib import Path

def load_json_data(file_path):
    """加载JSON数据文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def count_shape_distribution(data_path):
    """统计shape标签分布"""
    splits = ['train', 'test', 'vali']
    all_shapes = []
    split_distributions = {}
    
    print("=" * 80)
    print("Electricity/ver_shape 数据集分布统计")
    print("=" * 80)
    
    for split in splits:
        file_path = os.path.join(data_path, f"{split}.json")
        if not os.path.exists(file_path):
            print(f"\n⚠️  {split}.json 文件不存在，跳过")
            continue
            
        data = load_json_data(file_path)
        shapes = []
        
        for item in data:
            shape = item.get('shape', '')
            if shape:
                shapes.append(shape)
                all_shapes.append(shape)
            else:
                shapes.append('Empty')
                all_shapes.append('Empty')
        
        # 统计该split的分布
        shape_counter = Counter(shapes)
        split_distributions[split] = shape_counter
        
        print(f"\n{split.upper()} 数据集:")
        print(f"  总样本数: {len(data)}")
        print(f"  有标签样本数: {len([s for s in shapes if s != 'Empty'])}")
        print(f"  无标签样本数: {len([s for s in shapes if s == 'Empty'])}")
        print(f"\n  Shape标签分布:")
        total = len(shapes)
        for shape, count in shape_counter.most_common():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"    {shape:15s}: {count:5d} ({percentage:5.2f}%)")
    
    # 统计总体分布
    if all_shapes:
        print("\n" + "=" * 80)
        print("总体分布统计:")
        print("=" * 80)
        overall_counter = Counter(all_shapes)
        total_samples = len(all_shapes)
        print(f"总样本数: {total_samples}")
        print(f"有标签样本数: {len([s for s in all_shapes if s != 'Empty'])}")
        print(f"无标签样本数: {len([s for s in all_shapes if s == 'Empty'])}")
        print(f"\n总体Shape标签分布:")
        for shape, count in overall_counter.most_common():
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            print(f"  {shape:15s}: {count:5d} ({percentage:5.2f}%)")
        
        # 统计有效标签（排除Empty）
        valid_shapes = [s for s in all_shapes if s != 'Empty']
        if valid_shapes:
            print(f"\n有效标签分布（排除Empty）:")
            valid_counter = Counter(valid_shapes)
            valid_total = len(valid_shapes)
            for shape, count in valid_counter.most_common():
                percentage = (count / valid_total * 100) if valid_total > 0 else 0
                print(f"  {shape:15s}: {count:5d} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 80)
    
    return split_distributions, overall_counter if all_shapes else None

if __name__ == "__main__":
    data_path = "/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_shape"
    count_shape_distribution(data_path)
