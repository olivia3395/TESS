#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算8B base系列数据集的命中率
"""

import json
import os
from pathlib import Path


def load_dataset(dataset_path, split):
    """加载数据集"""
    file_path = os.path.join(dataset_path, f"{split}.json")
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_news_to_steps(news_string):
    """
    将news字符串解析为step列表
    例如: "-12, -13, -13, -14, -13" -> [-12, -13, -13, -14, -13]
    或者: "-1, -1, -1, -1, -1" -> [-1, -1, -1, -1, -1]
    """
    if not news_string:
        return []
    
    try:
        # 移除空格并按逗号分割
        parts = [part.strip() for part in news_string.split(',')]
        # 转换为整数
        steps = [int(part) for part in parts if part]
        return steps
    except ValueError:
        # 如果转换失败，返回空列表
        return []


def calculate_step_hit_rate(gt_dataset, pred_dataset):
    """
    计算step级别的命中率
    
    Args:
        gt_dataset: ground truth 数据集
        pred_dataset: 预测数据集
    
    Returns:
        hit_rate: 命中率
        total_steps: 总step数
        match_steps: 匹配的step数
    """
    if gt_dataset is None or pred_dataset is None:
        return 0.0, 0, 0
    
    match_steps = 0
    total_steps = 0
    
    # 取两个数据集中较小的记录数
    min_records = min(len(gt_dataset), len(pred_dataset))
    
    for i in range(min_records):
        gt_news = gt_dataset[i].get("news", "")
        pred_news = pred_dataset[i].get("news", "")
        
        # 解析news为step列表
        gt_steps = parse_news_to_steps(gt_news)
        pred_steps = parse_news_to_steps(pred_news)
        
        # 确保都有5个step
        if len(gt_steps) == 5 and len(pred_steps) == 5:
            total_steps += 5
            # 计算匹配的step数
            for j in range(5):
                if gt_steps[j] == pred_steps[j]:
                    match_steps += 1
        else:
            # 如果step数不为5，按实际长度计算
            compare_length = min(len(gt_steps), len(pred_steps))
            total_steps += compare_length
            for j in range(compare_length):
                if gt_steps[j] == pred_steps[j]:
                    match_steps += 1
    
    hit_rate = match_steps / total_steps if total_steps > 0 else 0.0
    return hit_rate, total_steps, match_steps


def calculate_strength_hit_rate(gt_dataset, pred_dataset):
    """
    计算trendstrength数据集的强度级别命中率
    通过取每个两位数编码的最后一位来获取每个step的强度
    
    Args:
        gt_dataset: ground truth 数据集
        pred_dataset: 预测数据集
    
    Returns:
        hit_rate: 强度级别命中率
        total_strengths: 总强度数
        match_strengths: 匹配的强度数
    """
    if gt_dataset is None or pred_dataset is None:
        return 0.0, 0, 0
    
    match_strengths = 0
    total_strengths = 0
    
    # 取两个数据集中较小的记录数
    min_records = min(len(gt_dataset), len(pred_dataset))
    
    for i in range(min_records):
        gt_news = gt_dataset[i].get("news", "")
        pred_news = pred_dataset[i].get("news", "")
        
        # 解析news为step列表
        gt_steps = parse_news_to_steps(gt_news)
        pred_steps = parse_news_to_steps(pred_news)
        
        # 确保都有5个step
        if len(gt_steps) == 5 and len(pred_steps) == 5:
            total_strengths += 5
            # 计算匹配的强度数（取绝对值的最后一位）
            for j in range(5):
                gt_strength = abs(gt_steps[j]) % 10
                pred_strength = abs(pred_steps[j]) % 10
                if gt_strength == pred_strength:
                    match_strengths += 1
    
    hit_rate = match_strengths / total_strengths if total_strengths > 0 else 0.0
    return hit_rate, total_strengths, match_strengths


def calculate_global_hit_rate(gt_dataset, pred_dataset):
    """
    计算global-only数据集的命中率
    
    Args:
        gt_dataset: ground truth 数据集
        pred_dataset: 预测数据集
    
    Returns:
        hit_rate: 命中率
        total_records: 总记录数
        match_records: 匹配的记录数
    """
    if gt_dataset is None or pred_dataset is None:
        return 0.0, 0, 0
    
    match_records = 0
    total_records = 0
    
    # 取两个数据集中较小的记录数
    min_records = min(len(gt_dataset), len(pred_dataset))
    
    for i in range(min_records):
        gt_news = gt_dataset[i].get("news", "")
        pred_news = pred_dataset[i].get("news", "")
        
        # 解析news为global值
        gt_steps = parse_news_to_steps(gt_news)
        pred_steps = parse_news_to_steps(pred_news)
        
        # 确保都只有1个global值
        if len(gt_steps) >= 1 and len(pred_steps) >= 1:
            total_records += 1
            if gt_steps[0] == pred_steps[0]:
                match_records += 1
    
    hit_rate = match_records / total_records if total_records > 0 else 0.0
    return hit_rate, total_records, match_records


def main():
    # 定义数据集路径
    base_dir = "../../dataset/FNSPID"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, base_dir)
    
    # 定义需要比较的数据集组合（仅限8B base系列）
    comparisons = [
        # (ground_truth_dataset, prediction_dataset, split, description)
        ("ver_synchronized", "ver_8B_trendstrength", "test", "ver_synchronized vs ver_8B_trendstrength"),
        ("ver_synchronized", "ver_8B_trendstrength", "vali", "ver_synchronized vs ver_8B_trendstrength"),
        ("ver_synchronized", "ver_8B_trendstrength", "train", "ver_synchronized vs ver_8B_trendstrength"),
        
        ("ver_synchronized_trendonly", "ver_8B_trendonly", "test", "ver_synchronized_trendonly vs ver_8B_trendonly"),
        ("ver_synchronized_trendonly", "ver_8B_trendonly", "vali", "ver_synchronized_trendonly vs ver_8B_trendonly"),
        ("ver_synchronized_trendonly", "ver_8B_trendonly", "train", "ver_synchronized_trendonly vs ver_8B_trendonly"),
    ]
    
    # Global-only 数据集使用不同的计算方法
    global_comparisons = [
        ("ver_synchronized_globalonly", "ver_8B_globalonly", "test", "ver_synchronized_globalonly vs ver_8B_globalonly"),
        ("ver_synchronized_globalonly", "ver_8B_globalonly", "vali", "ver_synchronized_globalonly vs ver_8B_globalonly"),
        ("ver_synchronized_globalonly", "ver_8B_globalonly", "train", "ver_synchronized_globalonly vs ver_8B_globalonly"),
    ]
    
    # Trendstrength 数据集列表
    trendstrength_datasets = [
        ("ver_synchronized", "ver_8B_trendstrength"),
    ]
    
    print("8B Base系列数据集命中率计算结果:")
    print("=" * 80)
    
    results = []
    
    # 执行step级别比较
    for gt_dataset_name, pred_dataset_name, split, description in comparisons:
        gt_path = os.path.join(dataset_dir, gt_dataset_name)
        pred_path = os.path.join(dataset_dir, pred_dataset_name)
        
        # 加载数据集
        gt_data = load_dataset(gt_path, split)
        pred_data = load_dataset(pred_path, split)
        
        # 计算step命中率
        step_hit_rate, total_steps, match_steps = calculate_step_hit_rate(gt_data, pred_data)
        
        result = {
            "description": description,
            "split": split,
            "step_hit_rate": step_hit_rate,
            "total_steps": total_steps,
            "match_steps": match_steps
        }
        
        # 如果是trendstrength数据集，额外计算强度命中率
        if (gt_dataset_name, pred_dataset_name) in trendstrength_datasets:
            strength_hit_rate, total_strengths, match_strengths = calculate_strength_hit_rate(gt_data, pred_data)
            result["strength_hit_rate"] = strength_hit_rate
            result["total_strengths"] = total_strengths
            result["match_strengths"] = match_strengths
            
            print(f"{description} ({split}):")
            print(f"  Step级别:  匹配 {match_steps:5d}/{total_steps:5d} 命中率: {step_hit_rate:.4f} ({step_hit_rate*100:.2f}%)")
            print(f"  强度级别:  匹配 {match_strengths:5d}/{total_strengths:5d} 命中率: {strength_hit_rate:.4f} ({strength_hit_rate*100:.2f}%)")
        else:
            print(f"{description} ({split}):")
            print(f"  Step级别:  匹配 {match_steps:5d}/{total_steps:5d} 命中率: {step_hit_rate:.4f} ({step_hit_rate*100:.2f}%)")
        
        print()
        results.append(result)
    
    # 执行global-only比较
    for gt_dataset_name, pred_dataset_name, split, description in global_comparisons:
        gt_path = os.path.join(dataset_dir, gt_dataset_name)
        pred_path = os.path.join(dataset_dir, pred_dataset_name)
        
        # 加载数据集
        gt_data = load_dataset(gt_path, split)
        pred_data = load_dataset(pred_path, split)
        
        # 对于global-only数据集，使用专门的计算方法
        hit_rate, total_count, match_count = calculate_global_hit_rate(gt_data, pred_data)
        
        result = {
            "description": description,
            "split": split,
            "step_hit_rate": 0.0,  # global-only没有step级别数据
            "total_steps": 0,
            "match_steps": 0,
            "record_hit_rate": hit_rate,
            "total_records": total_count,
            "match_records": match_count
        }
        results.append(result)
        
        print(f"{description} ({split}):")
        print(f"  Global级别:匹配 {match_count:4d}/{total_count:4d} 命中率: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
        print()
    
    # 保存结果到文件
    output_file = os.path.join(script_dir, "8B_base_hit_rate_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    main()