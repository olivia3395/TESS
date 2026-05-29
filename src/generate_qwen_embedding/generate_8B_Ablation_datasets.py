#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 ver_8B_Ablation_onlynews_base 数据集生成三个新数据集：
1. ver_8B_Ablation_trendstrength - 包含趋势和强度信息
2. ver_8B_Ablation_trendonly - 只包含趋势信息
3. ver_8B_Ablation_globalonly - 只包含全局趋势信息

格式参考 ver_synchronized 系列数据集
"""

import json
import os
from pathlib import Path


def load_dataset(input_dir, split):
    """加载数据集"""
    file_path = os.path.join(input_dir, f"{split}.json")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def save_dataset(data, output_dir, split):
    """保存数据集"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{split}.json")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} records to {file_path}")


def generate_trendstrength_dataset(input_data):
    """
    生成 trendstrength 数据集
    格式: "12, -13, 11, -14, -15" (trend + strength)
    直接使用所有数据，包括包含无效强度0或无效趋势0的记录
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        step_trends = record.get('step_trends')
        step_strengths = record.get('step_strengths')
        
        # 即使有无效的强度0或趋势0也直接使用
        # 只有在字段完全缺失时才跳过
        if step_trends is None or step_strengths is None:
            continue
            
        # 生成 news 字段: trend + strength
        news_parts = []
        for i in range(len(step_trends)):
            trend = step_trends[i]
            strength = step_strengths[i] if i < len(step_strengths) else 0
            # 组合趋势和强度为两位数，直接使用所有值包括0
            combined = f"{trend}{strength}"
            news_parts.append(combined)
        
        news = ", ".join(news_parts)
        
        # 创建新记录
        new_record = {
            "historical_data": historical_data,
            "ground_truth": ground_truth,
            "news": news
        }
        
        output_data.append(new_record)
    
    return output_data


def generate_trendonly_dataset(input_data):
    """
    生成 trendonly 数据集
    格式: "1, -1, 1, -1, -1" (只有趋势)
    直接使用所有数据，包括包含无效趋势0的记录
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        step_trends = record.get('step_trends')
        
        # 即使有无效的趋势0也直接使用
        # 只有在字段完全缺失时才跳过
        if step_trends is None:
            continue
            
        # 生成 news 字段: 只有趋势，直接使用所有值包括0
        news_parts = [str(trend) for trend in step_trends]
        news = ", ".join(news_parts)
        
        # 创建新记录
        new_record = {
            "historical_data": historical_data,
            "ground_truth": ground_truth,
            "news": news
        }
        
        output_data.append(new_record)
    
    return output_data


def generate_globalonly_dataset(input_data):
    """
    生成 globalonly 数据集
    格式: "-1" 或 "1" (只有全局趋势)
    直接使用所有数据，包括包含无效全局趋势0的记录
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        global_trend = record.get('global_trend')
        
        # 即使有无效的全局趋势0也直接使用
        # 只有在字段完全缺失时才跳过
        if global_trend is None:
            continue
            
        # 生成 news 字段: 只有全局趋势，直接使用所有值包括0
        news = str(global_trend)
        
        # 创建新记录
        new_record = {
            "historical_data": historical_data,
            "ground_truth": ground_truth,
            "news": news
        }
        
        output_data.append(new_record)
    
    return output_data


def main():
    # 定义路径
    base_dir = "../../dataset/FNSPID"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_dataset = "ver_8B_Ablation_onlynews_base"
    input_dir = os.path.join(script_dir, base_dir, input_dataset)
    
    output_datasets = [
        ("ver_8B_Ablation_trendstrength", generate_trendstrength_dataset),
        ("ver_8B_Ablation_trendonly", generate_trendonly_dataset),
        ("ver_8B_Ablation_globalonly", generate_globalonly_dataset)
    ]
    
    print(f"Input directory: {input_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # 处理所有分割数据集 (train, vali, test)
    splits = ['train', 'vali', 'test']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # 加载输入数据
        input_data = load_dataset(input_dir, split)
        if input_data is None:
            continue
            
        # 生成所有输出数据集
        for output_dataset_name, generator_func in output_datasets:
            output_dir = os.path.join(script_dir, base_dir, output_dataset_name)
            print(f"Generating {output_dataset_name}...")
            
            output_data = generator_func(input_data)
            save_dataset(output_data, output_dir, split)
    
    print("\nAll datasets generated successfully!")


if __name__ == "__main__":
    main()