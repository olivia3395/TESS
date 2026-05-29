#!/usr/bin/env python3
import json
import os

def modify_dataset_file(filepath):
    """修改单个数据集文件"""
    print(f"正在处理: {filepath}")

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
                    global_trend_value = news_value  # 保持其他值不变

                # 替换字段
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

        # 保存修改后的文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 保存成功: {filepath}")
        return True

    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        return False

def main():
    """主函数"""
    dataset_dir = "/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB/dataset/Bitcoin/ver_synchronized_globalonly"

    print("=" * 80)
    print("最终修改 Bitcoin ver_synchronized_globalonly 数据集")
    print("=" * 80)
    print(f"数据集目录: {dataset_dir}")
    print("修改规则:")
    print("  - 'news' -> 'global_trend'")
    print("  - '-1' -> 'Falling'")
    print("  - '1' -> 'Rising'")
    print("=" * 80)

    # 处理所有文件
    files = ['train.json', 'vali.json', 'test.json']
    success_count = 0

    for filename in files:
        filepath = os.path.join(dataset_dir, filename)

        if os.path.exists(filepath):
            if modify_dataset_file(filepath):
                success_count += 1
        else:
            print(f"文件不存在: {filepath}")

    print("=" * 80)
    print(f"处理完成! 成功修改了 {success_count}/{len(files)} 个文件")

    if success_count == len(files):
        print("🎉 所有文件修改成功!")
    else:
        print("⚠️  部分文件修改失败，请检查上述错误信息")

    print("=" * 80)

if __name__ == "__main__":
    main()








