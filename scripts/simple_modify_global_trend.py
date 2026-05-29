#!/usr/bin/env python3
import json
import os

def modify_dataset(filepath):
    """修改单个数据集文件"""
    print(f"处理 {filepath}...")

    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 统计修改前的数据
    news_values = {}
    for record in data:
        if 'news' in record:
            value = record['news']
            news_values[value] = news_values.get(value, 0) + 1

    print(f"  修改前: {news_values}")

    # 修改数据
    for record in data:
        if 'news' in record:
            news_value = record['news']

            # 映射值
            if news_value == "-1":
                global_trend_value = "Falling"
            elif news_value == "1":
                global_trend_value = "Rising"
            else:
                global_trend_value = news_value

            # 删除news字段，添加global_trend字段
            del record['news']
            record['global_trend'] = global_trend_value

    # 统计修改后的数据
    trend_values = {}
    for record in data:
        if 'global_trend' in record:
            value = record['global_trend']
            trend_values[value] = trend_values.get(value, 0) + 1

    print(f"  修改后: {trend_values}")

    # 保存文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ 保存完成: {filepath}")

# 处理所有文件
dataset_path = "/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Bitcoin/ver_synchronized_globalonly"
files = ['train.json', 'vali.json', 'test.json']

print("=" * 60)
print("修改 Bitcoin ver_synchronized_globalonly 数据集")
print("=" * 60)

for filename in files:
    filepath = os.path.join(dataset_path, filename)
    if os.path.exists(filepath):
        modify_dataset(filepath)
    else:
        print(f"文件不存在: {filepath}")

print("=" * 60)
print("所有文件修改完成!")
print("=" * 60)








