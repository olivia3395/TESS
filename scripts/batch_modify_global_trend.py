#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修改 Bitcoin ver_synchronized_globalonly 数据集
将 news 字段改名为 global_trend，并映射值：
- "-1" -> "Falling"
- "1" -> "Rising"
"""

import json
import os
from pathlib import Path


def modify_single_file(filepath):
    """
    修改单个JSON文件
    """
    print(f"正在处理文件: {filepath}")

    try:
        # 读取JSON文件
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"  读取了 {len(data)} 条记录")

        # 统计修改前的分布
        original_counts = {}
        for record in data:
            if 'news' in record:
                value = record['news']
                original_counts[value] = original_counts.get(value, 0) + 1

        print(f"  修改前分布: {original_counts}")

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
                    # 如果是其他值，保持原样（虽然按理说只有-1和1）
                    global_trend_value = news_value
                    print(f"  警告: 发现意外的news值: {news_value}")

                # 删除news字段，添加global_trend字段
                del record['news']
                record['global_trend'] = global_trend_value
                modified_count += 1

        print(f"  修改了 {modified_count} 条记录")

        # 统计修改后的分布
        new_counts = {}
        for record in data:
            if 'global_trend' in record:
                value = record['global_trend']
                new_counts[value] = new_counts.get(value, 0) + 1

        print(f"  修改后分布: {new_counts}")

        # 验证修改结果
        if len(new_counts) == 2 and 'Falling' in new_counts and 'Rising' in new_counts:
            print("  ✓ 修改结果正确")
        else:
            print("  ⚠️ 修改结果可能有问题")

        # 保存修改后的文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 文件保存成功: {filepath}")
        return True

    except Exception as e:
        print(f"  ✗ 处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：批量处理所有文件
    """
    print("=" * 80)
    print("批量修改 Bitcoin ver_synchronized_globalonly 数据集")
    print("=" * 80)

    # 数据集路径
    dataset_path = Path("/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Bitcoin/ver_synchronized_globalonly")

    if not dataset_path.exists():
        print(f"错误: 数据集路径不存在: {dataset_path}")
        return

    print(f"数据集路径: {dataset_path}")
    print("修改规则:")
    print("  - 'news' -> 'global_trend'")
    print("  - '-1' -> 'Falling'")
    print("  - '1' -> 'Rising'")
    print("=" * 80)

    # 要处理的文件
    files = ['train.json', 'vali.json', 'test.json']
    success_count = 0
    total_records = 0

    for filename in files:
        filepath = dataset_path / filename

        if filepath.exists():
            print(f"\n处理文件: {filename}")
            print("-" * 40)

            if modify_single_file(filepath):
                success_count += 1

                # 读取修改后的文件统计总记录数
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    total_records += len(data)
                    print(f"  记录数: {len(data)}")
                except:
                    print("  无法读取修改后的文件进行统计")
            else:
                print(f"  文件处理失败: {filename}")
        else:
            print(f"\n文件不存在: {filepath}")

    # 最终统计
    print(f"\n" + "=" * 80)
    print("批量修改完成统计")
    print("=" * 80)
    print(f"成功处理文件数: {success_count}/{len(files)}")
    print(f"总记录数: {total_records}")

    if success_count == len(files):
        print("🎉 所有文件修改成功!")
        print("\n验证检查:")
        print("- news字段已改为global_trend")
        print("- '-1'已映射为'Falling'")
        print("- '1'已映射为'Rising'")
    else:
        print("⚠️ 部分文件修改失败，请检查上述错误信息")

    print("=" * 80)


if __name__ == "__main__":
    main()
