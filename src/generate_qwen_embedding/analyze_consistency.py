#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析8B和8B_Ablation系列模型在ver_synchronized系列数据集上的一致性情况
统计8B对了Ablation没对，或者相反，或者都对的占比
"""

import json
import os


def load_results(file_path):
    """加载命中率结果文件"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} does not exist")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_consistency():
    """分析一致性"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载8B和8B_Ablation的结果
    base_results_file = os.path.join(script_dir, "8B_base_hit_rate_results.json")
    ablation_results_file = os.path.join(script_dir, "ablation_hit_rate_results.json")
    
    base_results = load_results(base_results_file)
    ablation_results = load_results(ablation_results_file)
    
    if base_results is None or ablation_results is None:
        print("无法加载结果文件")
        return
    
    # 创建结果字典便于查找
    base_dict = {(r["description"], r["split"]): r for r in base_results}
    ablation_dict = {(r["description"], r["split"]): r for r in ablation_results}
    
    # 定义要比较的数据集组合
    comparisons = [
        ("ver_synchronized vs ver_8B_trendstrength", "ver_synchronized vs ver_8B_Ablation_trendstrength", "trendstrength"),
        ("ver_synchronized_trendonly vs ver_8B_trendonly", "ver_synchronized_trendonly vs ver_8B_Ablation_trendonly", "trendonly"),
        ("ver_synchronized_globalonly vs ver_8B_globalonly", "ver_synchronized_globalonly vs ver_8B_Ablation_globalonly", "globalonly")
    ]
    
    splits = ["test", "vali", "train"]
    
    print("8B和8B_Ablation系列模型一致性分析")
    print("=" * 80)
    
    for base_desc, ablation_desc, dataset_type in comparisons:
        print(f"\n数据集类型: {dataset_type}")
        print("-" * 50)
        
        for split in splits:
            base_key = (base_desc, split)
            ablation_key = (ablation_desc, split)
            
            if base_key not in base_dict or ablation_key not in ablation_dict:
                print(f"  {split}: 数据缺失")
                continue
                
            base_result = base_dict[base_key]
            ablation_result = ablation_dict[ablation_key]
            
            if dataset_type == "trendstrength":
                # 对于trendstrength数据集，我们关注step级别命中率
                base_hit_rate = base_result["step_hit_rate"]
                ablation_hit_rate = ablation_result["step_hit_rate"]
                
                print(f"  {split} 数据集:")
                print(f"    8B系列模型命中率: {base_hit_rate:.4f} ({base_hit_rate*100:.2f}%)")
                print(f"    8B_Ablation系列模型命中率: {ablation_hit_rate:.4f} ({ablation_hit_rate*100:.2f}%)")
                
            elif dataset_type == "trendonly":
                # 对于trendonly数据集，我们关注step级别命中率
                base_hit_rate = base_result["step_hit_rate"]
                ablation_hit_rate = ablation_result["step_hit_rate"]
                
                print(f"  {split} 数据集:")
                print(f"    8B系列模型命中率: {base_hit_rate:.4f} ({base_hit_rate*100:.2f}%)")
                print(f"    8B_Ablation系列模型命中率: {ablation_hit_rate:.4f} ({ablation_hit_rate*100:.2f}%)")
                
            elif dataset_type == "globalonly":
                # 对于globalonly数据集，我们关注record级别命中率
                base_hit_rate = base_result["record_hit_rate"]
                ablation_hit_rate = ablation_result["record_hit_rate"]
                
                print(f"  {split} 数据集:")
                print(f"    8B系列模型命中率: {base_hit_rate:.4f} ({base_hit_rate*100:.2f}%)")
                print(f"    8B_Ablation系列模型命中率: {ablation_hit_rate:.4f} ({ablation_hit_rate*100:.2f}%)")
    
    # 详细分析trendstrength在test集上的表现差异
    print("\n\n详细分析: trendstrength在test集上的表现差异")
    print("-" * 50)
    
    base_key = ("ver_synchronized vs ver_8B_trendstrength", "test")
    ablation_key = ("ver_synchronized vs ver_8B_Ablation_trendstrength", "test")
    
    if base_key in base_dict and ablation_key in ablation_dict:
        base_result = base_dict[base_key]
        ablation_result = ablation_dict[ablation_key]
        
        base_step_rate = base_result["step_hit_rate"]
        ablation_step_rate = ablation_result["step_hit_rate"]
        base_strength_rate = base_result["strength_hit_rate"]
        ablation_strength_rate = ablation_result["strength_hit_rate"]
        
        print(f"Step级别命中率:")
        print(f"  8B系列模型: {base_step_rate:.4f} ({base_step_rate*100:.2f}%)")
        print(f"  8B_Ablation系列模型: {ablation_step_rate:.4f} ({ablation_step_rate*100:.2f}%)")
        print(f"  差异: {ablation_step_rate - base_step_rate:.4f}")
        
        print(f"\n强度级别命中率:")
        print(f"  8B系列模型: {base_strength_rate:.4f} ({base_strength_rate*100:.2f}%)")
        print(f"  8B_Ablation系列模型: {ablation_strength_rate:.4f} ({ablation_strength_rate*100:.2f}%)")
        print(f"  差异: {ablation_strength_rate - base_strength_rate:.4f}")
    
    # 详细分析trendonly在test集上的表现差异
    print("\n\n详细分析: trendonly在test集上的表现差异")
    print("-" * 50)
    
    base_key = ("ver_synchronized_trendonly vs ver_8B_trendonly", "test")
    ablation_key = ("ver_synchronized_trendonly vs ver_8B_Ablation_trendonly", "test")
    
    if base_key in base_dict and ablation_key in ablation_dict:
        base_result = base_dict[base_key]
        ablation_result = ablation_dict[ablation_key]
        
        base_step_rate = base_result["step_hit_rate"]
        ablation_step_rate = ablation_result["step_hit_rate"]
        
        print(f"Step级别命中率:")
        print(f"  8B系列模型: {base_step_rate:.4f} ({base_step_rate*100:.2f}%)")
        print(f"  8B_Ablation系列模型: {ablation_step_rate:.4f} ({ablation_step_rate*100:.2f}%)")
        print(f"  差异: {ablation_step_rate - base_step_rate:.4f}")
    
    # 详细分析globalonly在test集上的表现差异
    print("\n\n详细分析: globalonly在test集上的表现差异")
    print("-" * 50)
    
    base_key = ("ver_synchronized_globalonly vs ver_8B_globalonly", "test")
    ablation_key = ("ver_synchronized_globalonly vs ver_8B_Ablation_globalonly", "test")
    
    if base_key in base_dict and ablation_key in ablation_dict:
        base_result = base_dict[base_key]
        ablation_result = ablation_dict[ablation_key]
        
        base_record_rate = base_result["record_hit_rate"]
        ablation_record_rate = ablation_result["record_hit_rate"]
        
        print(f"Record级别命中率:")
        print(f"  8B系列模型: {base_record_rate:.4f} ({base_record_rate*100:.2f}%)")
        print(f"  8B_Ablation系列模型: {ablation_record_rate:.4f} ({ablation_record_rate*100:.2f}%)")
        print(f"  差异: {ablation_record_rate - base_record_rate:.4f}")


if __name__ == "__main__":
    analyze_consistency()