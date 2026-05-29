#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细分析8B和8B_Ablation系列模型在ver_synchronized系列数据集上的一致性情况
统计8B对了Ablation没对，或者相反，或者都对的占比
"""

import json
import os


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


def analyze_detailed_consistency():
    """详细分析一致性"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = "../../dataset/FNSPID"
    dataset_dir = os.path.join(script_dir, base_dir)
    
    # 定义数据集组合
    datasets = [
        # (ground_truth_dataset, model_8B_dataset, model_8B_Ablation_dataset, dataset_type)
        ("ver_synchronized", "ver_8B_trendstrength", "ver_8B_Ablation_trendstrength", "trendstrength"),
        ("ver_synchronized_trendonly", "ver_8B_trendonly", "ver_8B_Ablation_trendonly", "trendonly"),
        ("ver_synchronized_globalonly", "ver_8B_globalonly", "ver_8B_Ablation_globalonly", "globalonly"),
        # 新增onlyts系列数据集
        ("ver_synchronized", "ver_8B_trendstrength", "ver_8B_Ablation_onlyts_trendstrength", "trendstrength_onlyts"),
        ("ver_synchronized_trendonly", "ver_8B_trendonly", "ver_8B_Ablation_onlyts_trendonly", "trendonly_onlyts"),
        ("ver_synchronized_globalonly", "ver_8B_globalonly", "ver_8B_Ablation_onlyts_globalonly", "globalonly_onlyts")
    ]
    
    splits = ["test", "vali", "train"]
    
    print("8B和8B_Ablation系列模型详细一致性分析")
    print("=" * 80)
    
    for gt_dataset_name, model8B_dataset_name, ablation_dataset_name, dataset_type in datasets:
        print(f"\n数据集类型: {dataset_type}")
        print("-" * 50)
        
        for split in splits:
            print(f"  {split} 数据集:")
            
            # 加载数据集
            gt_path = os.path.join(dataset_dir, gt_dataset_name)
            model8B_path = os.path.join(dataset_dir, model8B_dataset_name)
            ablation_path = os.path.join(dataset_dir, ablation_dataset_name)
            
            gt_data = load_dataset(gt_path, split)
            model8B_data = load_dataset(model8B_path, split)
            ablation_data = load_dataset(ablation_path, split)
            
            if gt_data is None or model8B_data is None or ablation_data is None:
                print(f"    数据缺失")
                continue
            
            # 取三个数据集中最小的记录数
            min_records = min(len(gt_data), len(model8B_data), len(ablation_data))
            
            if "trendstrength" in dataset_type:
                # 对于trendstrength数据集，分析step和强度
                both_correct_steps = 0
                model8B_only_correct_steps = 0
                ablation_only_correct_steps = 0
                both_wrong_steps = 0
                
                both_correct_strengths = 0
                model8B_only_correct_strengths = 0
                ablation_only_correct_strengths = 0
                both_wrong_strengths = 0
                
                total_steps = 0
                total_strengths = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8B_news = model8B_data[i].get("news", "")
                    ablation_news = ablation_data[i].get("news", "")
                    
                    # 解析news为step列表
                    gt_steps = parse_news_to_steps(gt_news)
                    model8B_steps = parse_news_to_steps(model8B_news)
                    ablation_steps = parse_news_to_steps(ablation_news)
                    
                    # 确保都有5个step
                    if len(gt_steps) == 5 and len(model8B_steps) == 5 and len(ablation_steps) == 5:
                        total_steps += 5
                        total_strengths += 5
                        
                        # 计算step一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8B_step = model8B_steps[j]
                            ablation_step = ablation_steps[j]
                            
                            # Step比较
                            model8B_correct = (model8B_step == gt_step)
                            ablation_correct = (ablation_step == gt_step)
                            
                            if model8B_correct and ablation_correct:
                                both_correct_steps += 1
                            elif model8B_correct and not ablation_correct:
                                model8B_only_correct_steps += 1
                            elif not model8B_correct and ablation_correct:
                                ablation_only_correct_steps += 1
                            else:
                                both_wrong_steps += 1
                            
                            # 强度比较（取绝对值的最后一位）
                            gt_strength = abs(gt_step) % 10
                            model8B_strength = abs(model8B_step) % 10
                            ablation_strength = abs(ablation_step) % 10
                            
                            model8B_strength_correct = (model8B_strength == gt_strength)
                            ablation_strength_correct = (ablation_strength == gt_strength)
                            
                            if model8B_strength_correct and ablation_strength_correct:
                                both_correct_strengths += 1
                            elif model8B_strength_correct and not ablation_strength_correct:
                                model8B_only_correct_strengths += 1
                            elif not model8B_strength_correct and ablation_strength_correct:
                                ablation_only_correct_strengths += 1
                            else:
                                both_wrong_strengths += 1
                
                print(f"    Step级别统计:")
                print(f"      都正确: {both_correct_steps:5d}/{total_steps:5d} ({both_correct_steps/total_steps*100:6.2f}%)")
                print(f"      仅8B正确: {model8B_only_correct_steps:5d}/{total_steps:5d} ({model8B_only_correct_steps/total_steps*100:6.2f}%)")
                print(f"      仅Ablation正确: {ablation_only_correct_steps:5d}/{total_steps:5d} ({ablation_only_correct_steps/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_steps:5d}/{total_steps:5d} ({both_wrong_steps/total_steps*100:6.2f}%)")
                
                print(f"    强度级别统计:")
                print(f"      都正确: {both_correct_strengths:5d}/{total_strengths:5d} ({both_correct_strengths/total_strengths*100:6.2f}%)")
                print(f"      仅8B正确: {model8B_only_correct_strengths:5d}/{total_strengths:5d} ({model8B_only_correct_strengths/total_strengths*100:6.2f}%)")
                print(f"      仅Ablation正确: {ablation_only_correct_strengths:5d}/{total_strengths:5d} ({ablation_only_correct_strengths/total_strengths*100:6.2f}%)")
                print(f"      都错误: {both_wrong_strengths:5d}/{total_strengths:5d} ({both_wrong_strengths/total_strengths*100:6.2f}%)")
                
            elif "trendonly" in dataset_type:
                # 对于trendonly数据集，只分析step（趋势）
                both_correct_steps = 0
                model8B_only_correct_steps = 0
                ablation_only_correct_steps = 0
                both_wrong_steps = 0
                
                total_steps = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8B_news = model8B_data[i].get("news", "")
                    ablation_news = ablation_data[i].get("news", "")
                    
                    # 解析news为step列表
                    gt_steps = parse_news_to_steps(gt_news)
                    model8B_steps = parse_news_to_steps(model8B_news)
                    ablation_steps = parse_news_to_steps(ablation_news)
                    
                    # 确保都有5个step
                    if len(gt_steps) == 5 and len(model8B_steps) == 5 and len(ablation_steps) == 5:
                        total_steps += 5
                        
                        # 计算step一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8B_step = model8B_steps[j]
                            ablation_step = ablation_steps[j]
                            
                            # Step比较
                            model8B_correct = (model8B_step == gt_step)
                            ablation_correct = (ablation_step == gt_step)
                            
                            if model8B_correct and ablation_correct:
                                both_correct_steps += 1
                            elif model8B_correct and not ablation_correct:
                                model8B_only_correct_steps += 1
                            elif not model8B_correct and ablation_correct:
                                ablation_only_correct_steps += 1
                            else:
                                both_wrong_steps += 1
                
                print(f"    Step级别统计:")
                print(f"      都正确: {both_correct_steps:5d}/{total_steps:5d} ({both_correct_steps/total_steps*100:6.2f}%)")
                print(f"      仅8B正确: {model8B_only_correct_steps:5d}/{total_steps:5d} ({model8B_only_correct_steps/total_steps*100:6.2f}%)")
                print(f"      仅Ablation正确: {ablation_only_correct_steps:5d}/{total_steps:5d} ({ablation_only_correct_steps/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_steps:5d}/{total_steps:5d} ({both_wrong_steps/total_steps*100:6.2f}%)")
                
            elif "globalonly" in dataset_type:
                # 对于globalonly数据集，只分析record级别的全局趋势
                both_correct_records = 0
                model8B_only_correct_records = 0
                ablation_only_correct_records = 0
                both_wrong_records = 0
                
                total_records = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8B_news = model8B_data[i].get("news", "")
                    ablation_news = ablation_data[i].get("news", "")
                    
                    # 解析news为global值
                    gt_steps = parse_news_to_steps(gt_news)
                    model8B_steps = parse_news_to_steps(model8B_news)
                    ablation_steps = parse_news_to_steps(ablation_news)
                    
                    # 确保都只有1个global值
                    if len(gt_steps) >= 1 and len(model8B_steps) >= 1 and len(ablation_steps) >= 1:
                        total_records += 1
                        
                        gt_global = gt_steps[0]
                        model8B_global = model8B_steps[0]
                        ablation_global = ablation_steps[0]
                        
                        # Global比较
                        model8B_correct = (model8B_global == gt_global)
                        ablation_correct = (ablation_global == gt_global)
                        
                        if model8B_correct and ablation_correct:
                            both_correct_records += 1
                        elif model8B_correct and not ablation_correct:
                            model8B_only_correct_records += 1
                        elif not model8B_correct and ablation_correct:
                            ablation_only_correct_records += 1
                        else:
                            both_wrong_records += 1
                
                print(f"    Record级别统计:")
                print(f"      都正确: {both_correct_records:5d}/{total_records:5d} ({both_correct_records/total_records*100:6.2f}%)")
                print(f"      仅8B正确: {model8B_only_correct_records:5d}/{total_records:5d} ({model8B_only_correct_records/total_records*100:6.2f}%)")
                print(f"      仅Ablation正确: {ablation_only_correct_records:5d}/{total_records:5d} ({ablation_only_correct_records/total_records*100:6.2f}%)")
                print(f"      都错误: {both_wrong_records:5d}/{total_records:5d} ({both_wrong_records/total_records*100:6.2f}%)")


if __name__ == "__main__":
    analyze_detailed_consistency()