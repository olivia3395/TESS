#!/usr/bin/env python
"""统计指定索引样本在三个数据源上的平均MSE"""

import json
from pathlib import Path
import numpy as np


def load_jsonl(file_path: Path):
    """Load records from a JSONL file."""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_mse(pred, gt):
    """计算MSE"""
    pred = np.array(pred)
    gt = np.array(gt)
    return np.mean((pred - gt) ** 2)


def main():
    # 路径配置
    base_dir = Path('/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB')

    # 读取diff_indices文件获取要筛选的索引列表
    diff_indices_path = base_dir / 'dataset' / 'FNSPID' / 'analysis' / 'diff_indices_global_volatility_camf_vs_sync_test.json'
    print(f"Reading diff indices from {diff_indices_path}...")

    with diff_indices_path.open('r', encoding='utf-8') as f:
        diff_data = json.load(f)

    # 提取索引集合
    target_indices = set(item['index'] for item in diff_data)
    print(f"筛选的索引: {sorted(target_indices)}")
    print(f"索引数量: {len(target_indices)}")

    # 三个模型的test_samples.jsonl路径 (FNSPID数据集)
    model_paths = {
        'UniModal_ver_camf': base_dir / 'saved' / 'UniModal_Baseline' / 'FNSPID' / 'ver_camf' / 'best_epoch_Oct-20-2025-18-24-46' / 'test_samples.jsonl',
        'MultiModal_ver_camf': base_dir / 'saved' / 'MultiModal_Baseline' / 'FNSPID' / 'ver_camf' / 'best' / 'test_samples.jsonl',
        'MultiModal_global_volatility': base_dir / 'saved' / 'MultiModal_Baseline' / 'FNSPID' / 'ver_global_shape_volatility_natural' / 'best' / 'test_samples.jsonl',
    }

    # 读取并筛选样本，计算MSE
    model_mses = {}
    print("\n" + "="*50)
    print("统计结果:")
    print("="*50)

    for model_name, jsonl_path in model_paths.items():
        print(f"\n处理 {model_name}...")
        print(f"  文件路径: {jsonl_path}")

        if not jsonl_path.exists():
            print(f"  错误: 文件不存在!")
            model_mses[model_name] = 0.0
            continue

        records = load_jsonl(jsonl_path)
        print(f"  总样本数: {len(records)}")

        # 筛选目标索引的样本
        filtered_mses = []
        found_indices = set()

        for record in records:
            # 从sample_id提取索引，格式为 "test_0", "test_1" 等
            sample_id = record.get('sample_id', '')
            if sample_id.startswith('test_'):
                try:
                    idx = int(sample_id.split('_')[1])
                    if idx in target_indices:
                        found_indices.add(idx)
                        pred = record.get('prediction')
                        gt = record.get('ground_truth')
                        if pred is not None and gt is not None:
                            mse = compute_mse(pred, gt)
                            filtered_mses.append(mse)
                except (ValueError, IndexError):
                    continue

        print(f"  找到的索引: {sorted(found_indices)}")
        print(f"  筛选样本数: {len(filtered_mses)}")

        if filtered_mses:
            avg_mse = np.mean(filtered_mses)
            model_mses[model_name] = avg_mse
            print(f"  平均MSE: {avg_mse:.8f}")
        else:
            print("  警告: 未找到匹配的样本!")
            model_mses[model_name] = 0.0

    # 输出最终结果
    print("\n" + "="*50)
    print("最终统计结果:")
    print("="*50)
    print("UniModal_ver_camf:     {:.8f}".format(model_mses.get('UniModal_ver_camf', 0.0)))
    print("MultiModal_ver_camf:   {:.8f}".format(model_mses.get('MultiModal_ver_camf', 0.0)))
    print("MultiModal_global_volatility: {:.8f}".format(model_mses.get('MultiModal_global_volatility', 0.0)))

    # 保存结果到文件
    output_file = base_dir / 'mse_statistics_results.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("MSE统计结果\n")
        f.write("="*30 + "\n")
        f.write(f"筛选索引: {sorted(target_indices)}\n")
        f.write(f"索引数量: {len(target_indices)}\n\n")

        f.write("各模型平均MSE:\n")
        for model_name, mse in model_mses.items():
            f.write("{}: {:.8f}\n".format(model_name, mse))
        f.write("\n")

        f.write(f"\n结果保存时间: {output_file}\n")

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
