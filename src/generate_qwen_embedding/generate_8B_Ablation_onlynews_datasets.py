#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 ver_8B_Ablation_onlynews_base 数据集生成三个新数据集：
1. ver_8B_Ablation_onlynews_trendstrength - 包含趋势和强度信息
2. ver_8B_Ablation_onlynews_trendonly - 只包含趋势信息
3. ver_8B_Ablation_onlynews_globalonly - 只包含全局趋势信息

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
    格式: "12, -12, 11, -12, -13" (趋势值在前，强度值在后，组合为两位数)
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        step_trends = record.get('step_trends')
        step_strengths = record.get('step_strengths')
        
        # 检查字段是否存在
        if step_trends is None or step_strengths is None:
            continue
            
        # 确保长度为5
        if len(step_trends) != 5 or len(step_strengths) != 5:
            continue
            
        # 生成 news 字段: 趋势值在前，强度值在后，组合为两位数
        # 负数处理: -1, 2 -> -12
        news_parts = []
        for trend, strength in zip(step_trends, step_strengths):
            if trend == -1:
                # 负趋势，组合为负数
                combined = f"-{abs(trend)}{strength}"
            else:
                # 正趋势
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
    格式: "1, -1, 1, 1, -1" (只有趋势信息)
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        step_trends = record.get('step_trends')
        
        # 检查字段是否存在
        if step_trends is None:
            continue
            
        # 确保长度为5
        if len(step_trends) != 5:
            continue
            
        # 生成 news 字段: 只有趋势信息
        # 将-1转换为-1，将1转换为1
        news_parts = []
        for trend in step_trends:
            if trend == -1:
                news_parts.append("-1")
            else:
                news_parts.append(str(trend))
                
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
    """
    output_data = []
    
    for record in input_data:
        # 提取必要字段
        historical_data = record.get('historical_data')
        ground_truth = record.get('ground_truth')
        global_trend = record.get('global_trend')
        
        # 检查字段是否存在
        if global_trend is None:
            continue
            
        # 生成 news 字段: 只有全局趋势
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
        ("ver_8B_Ablation_onlynews_trendstrength", generate_trendstrength_dataset),
        ("ver_8B_Ablation_onlynews_trendonly", generate_trendonly_dataset),
        ("ver_8B_Ablation_onlynews_globalonly", generate_globalonly_dataset)
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