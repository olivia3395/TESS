#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修改 Bitcoin ver_synchronized_globalonly 数据集
将所有文件的 news 字段改为 global_trend，并映射值：
- "-1" -> "Falling"
- "1" -> "Rising"
"""

import json
import os
from pathlib import Path


def modify_single_file(filepath):
    """修改单个文件"""
    print(f"处理文件: {filepath}")

    try:
        # 读取JSON文件
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"  读取了 {len(data)} 条记录")

        # 统计修改前的分布
        news_counts = {}
        for record in data:
            if 'news' in record:
                value = record['news']
                news_counts[value] = news_counts.get(value, 0) + 1

        print(f"  修改前 news 字段分布: {news_counts}")

        # 修改每条记录
        modified_count = 0
        for record in data:
            if 'news' in record:
                news_value = record['news']

                # 映射值
                if news_value == "-1":
                    global_trend_value = "Falling"
                elif news_value == "1":
                    global_trend_value = "Rising"
                else:
                    global_trend_value = news_value  # 保持其他值不变

                # 删除news字段，添加global_trend字段
                del record['news']
                record['global_trend'] = global_trend_value
                modified_count += 1

        print(f"  成功修改了 {modified_count} 条记录")

        # 统计修改后的分布
        trend_counts = {}
        for record in data:
            if 'global_trend' in record:
                value = record['global_trend']
                trend_counts[value] = trend_counts.get(value, 0) + 1

        print(f"  修改后 global_trend 字段分布: {trend_counts}")

        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 文件保存成功: {filepath}")
        return True

    except Exception as e:
        print(f"  ✗ 处理文件时出错: {e}")
        return False


def main():
    """主函数"""
    # 数据集路径
    dataset_path = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_synchronized_globalonly")

    print("=" * 80)
    print("批量修改 Bitcoin ver_synchronized_globalonly 数据集")
    print("=" * 80)
    print(f"数据集路径: {dataset_path}")
    print("修改规则:")
    print("  1. 'news' 字段 -> 'global_trend' 字段")
    print("  2. 值映射: '-1' -> 'Falling', '1' -> 'Rising'")
    print("=" * 80)

    # 要处理的三个文件
    files = ['train.json', 'vali.json', 'test.json']
    success_count = 0
    total_records = 0

    for filename in files:
        filepath = dataset_path / filename

        if filepath.exists():
            print(f"\n处理 {filename}:")
            if modify_single_file(filepath):
                success_count += 1
        else:
            print(f"\n跳过 {filename}: 文件不存在 ({filepath})")

    print("\n" + "=" * 80)
    print("批量修改完成!")
    print(f"成功处理: {success_count}/{len(files)} 个文件")

    if success_count == len(files):
        print("🎉 所有文件修改成功!")
        print("\n修改摘要:")
        print("- 字段名: news -> global_trend")
        print("- 值映射: '-1' -> 'Falling', '1' -> 'Rising'")
        print(f"- 数据集: {dataset_path}")
    else:
        print("⚠️  部分文件处理失败")

    print("=" * 80)


if __name__ == "__main__":
    main()
