#!/usr/bin/env python
"""绘制Bitcoin数据集两个MultiModal_Baseline模型的loss曲线"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_manifest_loss_data(manifest_path: Path):
    """从manifest.json文件中提取loss数据"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    epochs = []
    train_losses = []
    vali_losses = []

    epoch_metrics = data.get('epoch_metrics', {})

    # 提取0-24 epoch的loss数据
    for epoch in range(25):  # 0 to 24
        epoch_key = str(epoch)
        if epoch_key in epoch_metrics:
            metrics = epoch_metrics[epoch_key]

            # 提取train loss
            train_metrics = metrics.get('train_metrics', {})
            train_loss = train_metrics.get('loss')
            if train_loss is not None:
                epochs.append(epoch)
                train_losses.append(train_loss)

            # 提取vali loss
            vali_metrics = metrics.get('vali_metrics', {})
            vali_loss = vali_metrics.get('loss')
            if vali_loss is not None:
                vali_losses.append(vali_loss)

    return epochs, train_losses, vali_losses


def plot_loss_curves():
    """绘制loss曲线"""

    # 文件路径
    base_dir = Path('/public/home/maoyaoxin/llh/MMTSF/MMTSF_LIB')

    manifest_files = {
        'ver_shape_temporal_shape_volatility_structured': base_dir / 'output' / 'Bitcoin' / 'ver_shape_temporal_shape_volatility_structured' / 'MultiModal_Baseline' / 'Jan-20-2026-03-40-39' / 'manifest.json',
        'ver_camf': base_dir / 'output' / 'Bitcoin' / 'ver_camf' / 'MultiModal_Baseline' / 'Jan-20-2026-03-39-18' / 'manifest.json'
    }

    # 颜色设置
    colors = {
        'ver_shape_temporal_shape_volatility_structured': '#00008B',  # 深蓝色
        'ver_camf': '#006400'  # 深绿色
    }

    # 标签设置
    labels = {
        'ver_shape_temporal_shape_volatility_structured': 'Ours',
        'ver_camf': 'Original_text'
    }

    # 加载数据
    data = {}
    for model_name, manifest_path in manifest_files.items():
        print(f"Loading data from {manifest_path}")
        if manifest_path.exists():
            epochs, train_losses, vali_losses = load_manifest_loss_data(manifest_path)
            data[model_name] = {
                'epochs': epochs,
                'train_losses': train_losses,
                'vali_losses': vali_losses
            }
            print(f"  Loaded {len(epochs)} epochs for {model_name}")
        else:
            print(f"  ERROR: Manifest file not found: {manifest_path}")

    # 设置matplotlib参数
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 18
    })

    # 创建图表输出目录
    output_dir = base_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制训练loss曲线
    plt.figure(figsize=(12, 8))

    for model_name, model_data in data.items():
        if model_data['train_losses']:
            plt.plot(model_data['epochs'], model_data['train_losses'],
                    label=labels[model_name],
                    color=colors[model_name],
                    linewidth=2,
                    marker='o',
                    markersize=4)

    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Bitcoin - Training Loss Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存训练loss图
    train_output = output_dir / 'bitcoin_training_loss_comparison.png'
    plt.savefig(train_output, dpi=300, bbox_inches='tight')
    print(f"Saved training loss plot to: {train_output}")
    plt.close()

    # 绘制验证loss曲线
    plt.figure(figsize=(12, 8))

    for model_name, model_data in data.items():
        if model_data['vali_losses']:
            plt.plot(model_data['epochs'], model_data['vali_losses'],
                    label=labels[model_name],
                    color=colors[model_name],
                    linewidth=2,
                    marker='s',
                    markersize=4)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Bitcoin - Validation Loss Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存验证loss图
    vali_output = output_dir / 'bitcoin_validation_loss_comparison.png'
    plt.savefig(vali_output, dpi=300, bbox_inches='tight')
    print(f"Saved validation loss plot to: {vali_output}")
    plt.close()

    print("\nBitcoin数据集Loss曲线绘制完成！")
    print(f"训练loss图: {train_output}")
    print(f"验证loss图: {vali_output}")


if __name__ == "__main__":
    plot_loss_curves()
