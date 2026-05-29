"""根据索引文件筛选样本，计算三个模型的MSE并绘制柱状图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_jsonl(path: Path):
    """读取JSONL文件，返回字典列表。"""
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_mse(pred, gt):
    """计算MSE。"""
    pred = np.array(pred)
    gt = np.array(gt)
    return np.mean((pred - gt) ** 2)


def main():
    # 路径配置
    base_dir = Path('/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB')

    # 直接定义要筛选的索引列表
    # target_indices = {0, 1, 3, 4}  # shape字段不同的记录索引
    target_indices = {0, 2, 3, 4, 6, 7, 9, 10, 12}
    print(f"Using hardcoded indices: {sorted(target_indices)}")
    print(f"Found {len(target_indices)} indices to filter")
    
    # 三个模型的test_samples.jsonl路径
    model_paths = {
        'UniModal_ver_camf': base_dir / 'saved' / 'UniModal_Baseline' / 'Bitcoin' / 'ver_camf' / 'best' / 'test_samples.jsonl',
        'MultiModal_ver_camf': base_dir / 'saved' / 'MultiModal_Baseline' / 'Bitcoin' / 'ver_camf' / 'best' / 'test_samples.jsonl',
        'MultiModal_global_shape': base_dir / 'saved' / 'MultiModal_Baseline' / 'Bitcoin' / 'ver_shape_temporal_shape_volatility_structured' / 'best' / 'test_samples.jsonl',
    }
    
    # 读取并筛选样本，计算MSE
    model_mses = {}
    for model_name, jsonl_path in model_paths.items():
        print(f"\nProcessing {model_name}...")
        print(f"  Reading {jsonl_path}...")
        records = load_jsonl(jsonl_path)
        print(f"  Total records: {len(records)}")
        
        # 筛选目标索引的样本
        filtered_mses = []
        for record in records:
            # 从sample_id提取索引，格式为 "test_0", "test_1" 等
            sample_id = record.get('sample_id', '')
            if sample_id.startswith('test_'):
                try:
                    idx = int(sample_id.split('_')[1])
                    if idx in target_indices:
                        pred = record.get('prediction')
                        gt = record.get('ground_truth')
                        if pred is not None and gt is not None:
                            mse = compute_mse(pred, gt)
                            filtered_mses.append(mse)
                except (ValueError, IndexError):
                    continue
        
        if filtered_mses:
            avg_mse = np.mean(filtered_mses)
            model_mses[model_name] = avg_mse
            print(f"  Filtered samples: {len(filtered_mses)}")
            print(f"  Average MSE: {avg_mse:.6f}")
        else:
            print(f"  WARNING: No matching samples found!")
            model_mses[model_name] = 0.0
    
    # 绘制柱状图
    labels = ["UniModal", "Original_text", "Ours"]
    values = [
        model_mses.get('UniModal_ver_camf', 0.0),
        model_mses.get('MultiModal_ver_camf', 0.0),
        model_mses.get('MultiModal_global_shape', 0.0),
    ]
    
    print(f"\nFinal MSE values:")
    for label, value in zip(labels, values):
        print(f"  {label}: {value:.6f}")
    
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['tab:blue', 'tab:orange', 'tab:green'])
    plt.ylabel('Average MSE')
    plt.title('non-stationary volatility MSE Comparison(Bitcoin)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    
    # 保存图片
    out_dir = base_dir / 'analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'filtered_mse_comparison_bar1.png'
    plt.savefig(out_path, dpi=200)
    print(f"\nFigure saved to: {out_path}")


if __name__ == '__main__':
    main()

