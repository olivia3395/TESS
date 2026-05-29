#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""统计 Electricity ver_temporal_shape 数据集中 temporal_influence_shape 字段的分布"""

import json
from collections import Counter
from pathlib import Path


def count_temporal_shape_distribution(data_path: str):
    """统计 temporal_influence_shape 字段的分布"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计所有标签
    labels = []
    empty_count = 0
    
    for item in data:
        label = item.get('temporal_influence_shape', '')
        if label:
            labels.append(label)
        else:
            empty_count += 1
    
    # 使用 Counter 统计
    label_counter = Counter(labels)
    total_count = len(data)
    non_empty_count = len(labels)
    
    # 打印统计结果
    print(f"=" * 60)
    print(f"数据集: {data_path}")
    print(f"=" * 60)
    print(f"总样本数: {total_count}")
    print(f"有效标签数: {non_empty_count}")
    print(f"空标签数: {empty_count}")
    print(f"\n标签分布:")
    print(f"-" * 60)
    
    # 按字母顺序排序显示
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[0])
    for label, count in sorted_labels:
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"  {label:15s}: {count:6d} ({percentage:6.2f}%)")
    
    if empty_count > 0:
        empty_percentage = (empty_count / total_count * 100) if total_count > 0 else 0
        print(f"  {'(空)':15s}: {empty_count:6d} ({empty_percentage:6.2f}%)")
    
    print(f"-" * 60)
    print(f"\n详细统计:")
    print(f"  有效标签占比: {non_empty_count/total_count*100:.2f}%")
    print(f"  空标签占比: {empty_count/total_count*100:.2f}%")
    
    # 计算平衡度（如果有多个标签）
    if len(label_counter) > 1:
        counts = list(label_counter.values())
        min_count = min(counts)
        max_count = max(counts)
        balance_ratio = min_count / max_count if max_count > 0 else 0
        print(f"  标签平衡度: {balance_ratio:.3f} (1.0为完全平衡)")
    
    return label_counter, empty_count, total_count


def main():
    """主函数"""
    base_path = Path("/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_temporal_shape")
    
    splits = ["train", "vali", "test"]
    
    print("\n" + "=" * 60)
    print("Electricity ver_temporal_shape 数据集标签分布统计")
    print("=" * 60 + "\n")
    
    all_labels = Counter()
    total_all = 0
    empty_all = 0
    
    for split in splits:
        split_path = base_path / f"{split}.json"
        if split_path.exists():
            print(f"\n[{split.upper()}]")
            label_counter, empty_count, total_count = count_temporal_shape_distribution(str(split_path))
            all_labels.update(label_counter)
            total_all += total_count
            empty_all += empty_count
        else:
            print(f"\n[{split.upper()}]")
            print(f"  文件不存在: {split_path}")
    
    # 汇总统计
    if total_all > 0:
        print(f"\n" + "=" * 60)
        print("汇总统计 (所有数据集)")
        print("=" * 60)
        print(f"总样本数: {total_all}")
        print(f"有效标签数: {sum(all_labels.values())}")
        print(f"空标签数: {empty_all}")
        print(f"\n标签分布:")
        print(f"-" * 60)
        
        sorted_labels = sorted(all_labels.items(), key=lambda x: x[0])
        for label, count in sorted_labels:
            percentage = (count / total_all * 100) if total_all > 0 else 0
            print(f"  {label:15s}: {count:6d} ({percentage:6.2f}%)")
        
        if empty_all > 0:
            empty_percentage = (empty_all / total_all * 100) if total_all > 0 else 0
            print(f"  {'(空)':15s}: {empty_all:6d} ({empty_percentage:6.2f}%)")
        
        print(f"-" * 60)


if __name__ == "__main__":
    main()
