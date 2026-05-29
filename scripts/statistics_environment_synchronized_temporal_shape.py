#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计 Environment ver_synchronized_temporal_shape 数据集中 temporal_influence_shape 字段的分布
"""

import json
from collections import Counter
from pathlib import Path


def count_temporal_influence_shape_distribution(data_path: str):
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
    print(f"=" * 80)
    print(f"数据集: {data_path}")
    print(f"=" * 80)
    print(f"总样本数: {total_count}")
    print(f"有效标签数: {non_empty_count}")
    print(f"空标签数: {empty_count}")
    print(f"\n影响形态分布:")
    print(f"-" * 80)

    # 按字母顺序排序显示
    sorted_labels = sorted(label_counter.items(), key=lambda x: x[0])
    for label, count in sorted_labels:
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"  {label:15s}: {count:6d} ({percentage:6.2f}%)")

    if empty_count > 0:
        empty_percentage = (empty_count / total_count * 100) if total_count > 0 else 0
        print(f"  {'(空)':15s}: {empty_count:6d} ({empty_percentage:6.2f}%)")

    print(f"-" * 80)
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
    base_path = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Environment/ver_synchronized_temporal_shape")

    splits = ["train", "vali", "test"]

    print("\n" + "=" * 80)
    print("Environment ver_synchronized_temporal_shape 数据集标签分布统计")
    print("=" * 80 + "\n")

    all_labels = Counter()
    total_all = 0
    empty_all = 0

    for split in splits:
        split_path = base_path / f"{split}.json"
        if split_path.exists():
            print(f"\n[{split.upper()}]")
            label_counter, empty_count, total_count = count_temporal_influence_shape_distribution(str(split_path))
            all_labels.update(label_counter)
            total_all += total_count
            empty_all += empty_count
        else:
            print(f"\n[{split.upper()}]")
            print(f"  文件不存在: {split_path}")

    # 汇总统计
    if total_all > 0:
        print(f"\n" + "=" * 80)
        print("汇总统计 (所有数据集)")
        print("=" * 80)
        print(f"总样本数: {total_all}")
        print(f"有效标签数: {sum(all_labels.values())}")
        print(f"空标签数: {empty_all}")
        print(f"\n影响形态分布:")
        print(f"-" * 80)

        sorted_labels = sorted(all_labels.items(), key=lambda x: x[0])
        for label, count in sorted_labels:
            percentage = (count / total_all * 100) if total_all > 0 else 0
            print(f"  {label:15s}: {count:6d} ({percentage:6.2f}%)")

        if empty_all > 0:
            empty_percentage = (empty_all / total_all * 100) if total_all > 0 else 0
            print(f"  {'(空)':15s}: {empty_all:6d} ({empty_percentage:6.2f}%)")

        print(f"-" * 80)
        print(f"\n总结:")
        print(f"  影响形态类别数: {len(all_labels)}")
        print(f"  最常见形态: {all_labels.most_common(1)[0][0] if all_labels else '无'}")

        # 分析影响形态的时效性分布
        immediate_count = all_labels.get('immediate', 0)
        sustained_count = all_labels.get('sustained', 0)
        delayed_count = all_labels.get('delayed', 0)

        print(f"  即时影响 (immediate): {immediate_count} ({immediate_count/total_all*100:.1f}%)")
        print(f"  持续影响 (sustained): {sustained_count} ({sustained_count/total_all*100:.1f}%)")
        print(f"  延迟影响 (delayed): {delayed_count} ({delayed_count/total_all*100:.1f}%)")

        # 分析不同形态的特点
        print(f"\n形态分析 (基于7个增长率分析):")
        if immediate_count > 0:
            print(f"  immediate: 前2个时间间隔变化显著，后5个无显著变化，市场快速反应")
        if sustained_count > 0:
            print(f"  sustained: 至少4个时间间隔变化显著，前2个和后5个都有至少2个显著变化，事件影响持续")
        if delayed_count > 0:
            print(f"  delayed: 不满足immediate或sustained条件的其他情况")


if __name__ == "__main__":
    main()
