#!/usr/bin/env python
"""Plot prediction and ground truth curves from test_samples.jsonl files for model comparison (with historical data, grid, and markers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_samples(
    original_file: Path,
    ours_file: Path,
    output: Path,
    sample_indices: List[int],
    show: bool = False,
) -> None:
    """Plot prediction and ground truth curves.
    
    Args:
        original_file: File containing Original model predictions and GT
        ours_file: File containing Ours model predictions and GT
        output: Output file path template
        sample_indices: List of sample indices to plot
        show: Whether to display plots
    """
    # Load data
    if not original_file.exists():
        raise FileNotFoundError(f"Original model file not found: {original_file}")
    if not ours_file.exists():
        raise FileNotFoundError(f"Ours model file not found: {ours_file}")
    
    original_records = load_jsonl(original_file)
    ours_records = load_jsonl(ours_file)
    
    # Check that files have the same number of samples
    num_samples_original = len(original_records)
    num_samples_ours = len(ours_records)
    min_samples = min(num_samples_original, num_samples_ours)
    
    if num_samples_original != num_samples_ours:
        print(f"Warning: Sample count mismatch - Original: {num_samples_original}, Ours: {num_samples_ours}")
        print(f"Will only plot samples that exist in both files (indices 0-{min_samples-1})")
    
    # Output directory
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Colors for model predictions (low saturation colors for better readability)
    original_color = "#D97777"  # Low saturation rose red for Original
    ours_color = "#77A3D9"      # Low saturation sky blue for Ours
    hist_color = "#9E9E9E"      # Gray color for historical data
    
    # Markers for different curves (to distinguish data points)
    hist_marker = "s"      # Square for historical data
    gt_marker = "o"        # Circle for ground truth
    original_marker = "^"  # Triangle up for Original
    ours_marker = "x"      # Cross/X for Ours
    
    # Plot each sample separately
    for sample_idx in sample_indices:
        # Check if sample exists in both files
        if sample_idx >= num_samples_original:
            print(f"Warning: Sample {sample_idx} is out of range in Original file (max: {num_samples_original-1})")
            continue
        if sample_idx >= num_samples_ours:
            print(f"Warning: Sample {sample_idx} is out of range in Ours file (max: {num_samples_ours-1})")
            continue
        
        # Create a new figure for each sample (narrower width, taller height)
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Enable grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Get ground truth and historical data from original records (both should have same GT and hist_data)
        original_record = original_records[sample_idx]
        gt_data = original_record.get("ground_truth", [])
        hist_data = original_record.get("hist_data", [])
        
        if not gt_data:
            ax.text(0.5, 0.5, f"Sample {sample_idx}\nNo ground truth data", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Sample {sample_idx}")
        else:
            gt_array = np.array(gt_data)
            pred_len = len(gt_array)
            
            # Plot historical data separately (no connection)
            hist_len = 0
            if hist_data:
                hist_array = np.array(hist_data)
                hist_len = len(hist_array)
                
                # Historical data time steps: from -hist_len to 0
                hist_time_steps = np.arange(-hist_len, 0)
                
                # Plot historical data as solid line with markers
                ax.plot(hist_time_steps, hist_array, "-", linewidth=2, 
                       label="Historical Data", color=hist_color, alpha=0.7,
                       marker=hist_marker, markersize=5, markevery=1)
            
            # Prediction/GT time steps: from 0 to pred_len - 1
            pred_time_steps = np.arange(pred_len)
            
            # Plot ground truth (solid line with circle markers)
            ax.plot(pred_time_steps, gt_array, "k-", linewidth=2, 
                   label="Ground Truth", alpha=0.8, marker=gt_marker, markersize=5, markevery=1)
            
            # Plot Original model's prediction (solid line with triangle up markers)
            original_pred = original_record.get("prediction", [])
            if original_pred:
                original_array = np.array(original_pred)
                if len(original_array) == len(gt_array):
                    ax.plot(pred_time_steps, original_array, "-", linewidth=1.5, 
                           label="Original", color=original_color, alpha=0.8,
                           marker=original_marker, markersize=5, markevery=1)
                else:
                    print(f"Warning: Sample {sample_idx} Original prediction length ({len(original_array)}) != GT length ({len(gt_array)})")
            else:
                print(f"Warning: Sample {sample_idx} Original has no prediction data")
            
            # Plot Ours model's prediction (solid line with cross markers)
            if sample_idx >= len(ours_records):
                print(f"Warning: Sample {sample_idx} not found in Ours file, skipping Ours prediction")
            else:
                ours_record = ours_records[sample_idx]
                ours_pred = ours_record.get("prediction", [])
                if ours_pred:
                    ours_array = np.array(ours_pred)
                    if len(ours_array) == len(gt_array):
                        ax.plot(pred_time_steps, ours_array, "-", linewidth=1.5, 
                               label="Ours", color=ours_color, alpha=0.8,
                               marker=ours_marker, markersize=5, markevery=1)
                    else:
                        print(f"Warning: Sample {sample_idx} Ours prediction length ({len(ours_array)}) != GT length ({len(gt_array)})")
                else:
                    print(f"Warning: Sample {sample_idx} Ours has no prediction data")
            
            # Draw vertical dashed line at the boundary between historical data and GT/prediction
            if hist_len > 0:
                ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_title(f"Sample {sample_idx}", fontsize=14)
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(loc="best", fontsize=10)
        
        # Save individual figure
        output_file = output.parent /f"{output.stem}_sample_{sample_idx}{output.suffix}"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    # 配置：数据集类型
    # 可选值: "test", "vali", "train"
    # 选择不同的数据集类型会加载对应的 samples.jsonl 文件
    dataset_type = "train"  # 修改这里来选择数据集类型
    
    if dataset_type not in ["test", "vali", "train"]:
        raise ValueError(f"dataset_type must be 'test', 'vali', or 'train', got '{dataset_type}'")
    
    # 根据数据集类型构建文件名
    samples_filename = f"{dataset_type}_samples.jsonl"
    
    # 配置：文件路径
    # Original (ver_global_shape_avg) 模型文件
    original_file = project_root / "saved" / "MultiModal_Baseline" / "FNSPID" / "ver_global_shape_avg" / "best" / samples_filename
    # Ours 模型文件
    ours_file = project_root / "saved" / "MultiModal_Baseline" / "FNSPID" / "ver_global_shape_volatility_natural" / "best" / samples_filename
    
    # 配置：要绘制的样本索引
    sample_indices = [5205, 2778, 6265, 2129, 799]  # 修改为要绘制的样本索引列表
    
    # 配置：输出文件路径（包含数据集类型）
    output = project_root / "log" / f"model_comparison_simple_fnspid_{dataset_type}.png"
    
    # 是否显示图片（False表示只保存，True表示保存后显示）
    show = False
    
    # 绘制两个模型的 GT 和预测曲线（同时显示）
    plot_samples(
        original_file=original_file,
        ours_file=ours_file,
        output=output,
        sample_indices=sample_indices,
        show=show,
    )


if __name__ == "__main__":
    main()

