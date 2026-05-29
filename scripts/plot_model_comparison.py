#!/usr/bin/env python
"""Plot prediction and ground truth curves from test_samples.jsonl files for model comparison."""

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
    plot_mode: str,
    output: Path,
    sample_indices: List[int],
    show: bool = False,
) -> None:
    """Plot prediction and ground truth curves.
    
    Args:
        original_file: File containing Original model predictions and GT
        ours_file: File containing Ours model predictions and GT
        plot_mode: Which mode to display ("original" or "ours")
        output: Output file path template
        sample_indices: List of sample indices to plot
        show: Whether to display plots
    """
    # Load data
    if not original_file.exists():
        raise FileNotFoundError(f"Original model file not found: {original_file}")
    if not ours_file.exists():
        raise FileNotFoundError(f"Ours model file not found: {ours_file}")
    
    if plot_mode not in ["original", "ours"]:
        raise ValueError(f"plot_mode must be 'original' or 'ours', got '{plot_mode}'")
    
    original_records = load_jsonl(original_file)
    ours_records = load_jsonl(ours_file)
    
    # Check that files have the same number of samples
    num_samples = len(original_records)
    if len(ours_records) != num_samples:
        raise ValueError(f"Ours file has {len(ours_records)} samples, expected {num_samples}")
    
    # Output directory
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Colors for model predictions (low saturation colors for better readability)
    original_color = "#D97777"  # Low saturation rose red for Original
    ours_color = "#77A3D9"      # Low saturation sky blue for Ours
    hidden_color = "white"      # White color to hide unselected mode
    hist_color = "#9E9E9E"      # Gray color for historical data
    
    # Determine which mode is selected and which is hidden
    if plot_mode == "original":
        selected_records = original_records
        selected_label = "Original"
        selected_color = original_color
        hidden_records = ours_records
        hidden_label = "Ours"
        hidden_model_color = hidden_color
    else:  # plot_mode == "ours"
        selected_records = ours_records
        selected_label = "Ours"
        selected_color = ours_color
        hidden_records = original_records
        hidden_label = "Original"
        hidden_model_color = hidden_color
    
    # Plot each sample separately
    for sample_idx in sample_indices:
        if sample_idx >= num_samples:
            print(f"Warning: Sample {sample_idx} is out of range (max: {num_samples-1})")
            continue
        
        # Create a new figure for each sample (width doubled to accommodate historical data)
        fig, ax = plt.subplots(figsize=(16, 5))
        
        # Get ground truth and historical data from selected mode's data source
        selected_record = selected_records[sample_idx]
        gt_data = selected_record.get("ground_truth", [])
        hist_data = selected_record.get("hist_data", [])
        
        if not gt_data:
            ax.text(0.5, 0.5, f"Sample {sample_idx}\nNo ground truth data", 
                   ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Sample {sample_idx}")
        else:
            gt_array = np.array(gt_data)
            pred_len = len(gt_array)
            
            # Calculate time steps to connect historical data and GT continuously
            hist_len = 0
            connection_value = None
            hist_array = None
            if hist_data:
                hist_array = np.array(hist_data)
                hist_len = len(hist_array)
                
                # Calculate interpolation value at connection point
                # Use average of last historical value and first GT value for smooth connection
                connection_value = (hist_array[-1] + gt_array[0]) / 2.0
                
                # Create connected historical data: extend to hist_len with interpolated value
                # This ensures all curves meet at the same point
                hist_time_steps = np.arange(hist_len + 1)  # Include connection point
                hist_values_connected = np.append(hist_array, connection_value)  # Use interpolated value
                
                # Plot historical data as solid line with different color
                ax.plot(hist_time_steps, hist_values_connected, "-", linewidth=2, 
                       label="Historical Data", color=hist_color, alpha=0.7)
            
            # Prediction/GT time steps: from hist_len to hist_len + pred_len - 1
            # Create connected GT: start from hist_len with interpolated connection value
            pred_time_steps = np.arange(hist_len, hist_len + pred_len)
            
            # Create connected GT values: prepend connection point if historical data exists
            if hist_len > 0 and connection_value is not None:
                # Use interpolation value at connection point
                gt_values_connected = np.append([connection_value], gt_array)
                gt_time_steps_connected = np.arange(hist_len, hist_len + pred_len + 1)
            else:
                gt_values_connected = gt_array
                gt_time_steps_connected = pred_time_steps
            
            # Plot ground truth from selected mode (always shown as solid line)
            ax.plot(gt_time_steps_connected, gt_values_connected, "k-", linewidth=2, label="Ground Truth", alpha=0.8)
            
            # Plot selected mode's prediction (dashed line, visible color)
            selected_pred = selected_record.get("prediction", [])
            if selected_pred:
                selected_array = np.array(selected_pred)
                if len(selected_array) == len(gt_array):
                    # Create connected prediction: prepend connection point if historical data exists
                    if hist_len > 0 and connection_value is not None and hist_array is not None:
                        # Use the same interpolation value for consistency
                        # Interpolate between hist_array[-1] and selected_array[0]
                        pred_connection_value = (hist_array[-1] + selected_array[0]) / 2.0
                        selected_array_connected = np.append([pred_connection_value], selected_array)
                        selected_time_steps_connected = np.arange(hist_len, hist_len + pred_len + 1)
                    else:
                        selected_array_connected = selected_array
                        selected_time_steps_connected = pred_time_steps
                    
                    ax.plot(selected_time_steps_connected, selected_array_connected, "--", linewidth=1.5, 
                           label=selected_label, color=selected_color, alpha=0.8)
                else:
                    print(f"Warning: Sample {sample_idx} {selected_label} prediction length ({len(selected_array)}) != GT length ({len(gt_array)})")
            else:
                print(f"Warning: Sample {sample_idx} {selected_label} has no prediction data")
            
            # Plot hidden mode's prediction (dashed line, white color to hide it)
            # This ensures data alignment but makes it invisible
            # Note: No label is set so it won't appear in the legend
            hidden_record = hidden_records[sample_idx]
            hidden_pred = hidden_record.get("prediction", [])
            if hidden_pred:
                hidden_array = np.array(hidden_pred)
                if len(hidden_array) == len(gt_array):
                    # Create connected prediction for hidden mode too
                    if hist_len > 0 and connection_value is not None and hist_array is not None:
                        # Use interpolation for hidden mode prediction too
                        pred_connection_value = (hist_array[-1] + hidden_array[0]) / 2.0
                        hidden_array_connected = np.append([pred_connection_value], hidden_array)
                        hidden_time_steps_connected = np.arange(hist_len, hist_len + pred_len + 1)
                    else:
                        hidden_array_connected = hidden_array
                        hidden_time_steps_connected = pred_time_steps
                    
                    ax.plot(hidden_time_steps_connected, hidden_array_connected, "--", linewidth=1.5, 
                           color=hidden_model_color, alpha=0.0)
                # Note: We don't print warnings for hidden mode to keep output clean
            
            # Draw vertical dashed line at the boundary between historical data and GT/prediction
            if hist_len > 0:
                ax.axvline(x=hist_len, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_title(f"Sample {sample_idx}", fontsize=14)
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.legend(loc="best", fontsize=10)
        
        # Save individual figure
        output_file = output.parent / f"{output.stem}_sample_{sample_idx}{output.suffix}"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_file}")
        
        if show:
            plt.show()
        else:
            plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    # 配置：文件路径
    # Original (ver_camf) 模型文件
    original_file = project_root / "saved" / "MultiModal_Baseline" / "Electricity" / "ver_camf" / "best" / "test_samples.jsonl"
    # Ours 模型文件
    ours_file = project_root / "saved" / "MultiModal_Baseline" / "Electricity" / "ver_global_temporal_shape_volatility_natural" / "best" / "test_samples.jsonl"
    
    # 配置：选择模式
    # 可选值: "original" 或 "ours"
    # 选择 "original" 时：显示 Original 的 GT 和预测曲线，Ours 的曲线用白色隐藏
    # 选择 "ours" 时：显示 Ours 的 GT 和预测曲线，Original 的曲线用白色隐藏
    plot_mode = "ours"  # 修改这里来选择要显示的模式
    
    # 配置：要绘制的样本索引
    sample_indices = [702]
    
    # 配置：输出文件路径
    output = project_root / f"model_comparison_{plot_mode}.png"
    
    # 是否显示图片（False表示只保存，True表示保存后显示）
    show = False
    
    # 绘制选中模式的 GT 和预测曲线，未选中模式的曲线用白色隐藏
    plot_samples(
        original_file=original_file,
        ours_file=ours_file,
        plot_mode=plot_mode,
        output=output,
        sample_indices=sample_indices,
        show=show,
    )


if __name__ == "__main__":
    main()

