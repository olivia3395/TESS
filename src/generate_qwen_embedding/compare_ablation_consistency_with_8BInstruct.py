#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较onlynews、onlyts和8BInstruct在synchronized数据集上的表现一致性
统计两组数据，onlynews和8BInstruct的一致性，onlyts和8BInstruct一致性
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


def analyze_consistency():
    """分析一致性"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = "../../dataset/FNSPID"
    dataset_dir = os.path.join(script_dir, base_dir)
    
    # 定义数据集组合
    datasets = [
        # (ground_truth_dataset, model_8BInstruct_dataset, model_onlynews_dataset, model_onlyts_dataset, dataset_type)
        ("ver_synchronized", "ver_8BInstruct_trendstrength", "ver_8B_Ablation_onlynews_trendstrength", "ver_8B_Ablation_onlyts_trendstrength", "trendstrength"),
        ("ver_synchronized_trendonly", "ver_8BInstruct_trendonly", "ver_8B_Ablation_onlynews_trendonly", "ver_8B_Ablation_onlyts_trendonly", "trendonly"),
        ("ver_synchronized_globalonly", "ver_8BInstruct_globalonly", "ver_8B_Ablation_onlynews_globalonly", "ver_8B_Ablation_onlyts_globalonly", "globalonly")
    ]
    
    splits = ["test", "vali", "train"]
    
    print("8BInstruct、onlynews和onlyts模型一致性分析")
    print("=" * 80)
    
    for gt_dataset_name, model8BInstruct_dataset_name, onlynews_dataset_name, onlyts_dataset_name, dataset_type in datasets:
        print(f"\n数据集类型: {dataset_type}")
        print("-" * 50)
        
        for split in splits:
            print(f"  {split} 数据集:")
            
            # 加载数据集
            gt_path = os.path.join(dataset_dir, gt_dataset_name)
            model8BInstruct_path = os.path.join(dataset_dir, model8BInstruct_dataset_name)
            onlynews_path = os.path.join(dataset_dir, onlynews_dataset_name)
            onlyts_path = os.path.join(dataset_dir, onlyts_dataset_name)
            
            gt_data = load_dataset(gt_path, split)
            model8BInstruct_data = load_dataset(model8BInstruct_path, split)
            onlynews_data = load_dataset(onlynews_path, split)
            onlyts_data = load_dataset(onlyts_path, split)
            
            if gt_data is None or model8BInstruct_data is None or onlynews_data is None or onlyts_data is None:
                print(f"    数据缺失")
                continue
            
            # 取所有数据集中最小的记录数
            min_records = min(len(gt_data), len(model8BInstruct_data), len(onlynews_data), len(onlyts_data))
            
            if dataset_type == "trendstrength":
                # 对于trendstrength数据集，分析step和强度
                # 8BInstruct vs onlynews
                both_correct_8BInstruct_onlynews = 0
                only_8BInstruct_correct_8BInstruct_onlynews = 0
                only_onlynews_correct_8BInstruct_onlynews = 0
                both_wrong_8BInstruct_onlynews = 0
                
                # 8BInstruct vs onlyts
                both_correct_8BInstruct_onlyts = 0
                only_8BInstruct_correct_8BInstruct_onlyts = 0
                only_onlyts_correct_8BInstruct_onlyts = 0
                both_wrong_8BInstruct_onlyts = 0
                
                total_steps = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8BInstruct_news = model8BInstruct_data[i].get("news", "")
                    onlynews_news = onlynews_data[i].get("news", "")
                    onlyts_news = onlyts_data[i].get("news", "")
                    
                    # 解析news为step列表
                    gt_steps = parse_news_to_steps(gt_news)
                    model8BInstruct_steps = parse_news_to_steps(model8BInstruct_news)
                    onlynews_steps = parse_news_to_steps(onlynews_news)
                    onlyts_steps = parse_news_to_steps(onlyts_news)
                    
                    # 确保都有5个step
                    if len(gt_steps) == 5 and len(model8BInstruct_steps) == 5 and len(onlynews_steps) == 5 and len(onlyts_steps) == 5:
                        total_steps += 5
                        
                        # 计算8BInstruct vs onlynews一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8BInstruct_step = model8BInstruct_steps[j]
                            onlynews_step = onlynews_steps[j]
                            
                            # Step比较
                            model8BInstruct_correct = (model8BInstruct_step == gt_step)
                            onlynews_correct = (onlynews_step == gt_step)
                            
                            if model8BInstruct_correct and onlynews_correct:
                                both_correct_8BInstruct_onlynews += 1
                            elif model8BInstruct_correct and not onlynews_correct:
                                only_8BInstruct_correct_8BInstruct_onlynews += 1
                            elif not model8BInstruct_correct and onlynews_correct:
                                only_onlynews_correct_8BInstruct_onlynews += 1
                            else:
                                both_wrong_8BInstruct_onlynews += 1
                        
                        # 计算8BInstruct vs onlyts一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8BInstruct_step = model8BInstruct_steps[j]
                            onlyts_step = onlyts_steps[j]
                            
                            # Step比较
                            model8BInstruct_correct = (model8BInstruct_step == gt_step)
                            onlyts_correct = (onlyts_step == gt_step)
                            
                            if model8BInstruct_correct and onlyts_correct:
                                both_correct_8BInstruct_onlyts += 1
                            elif model8BInstruct_correct and not onlyts_correct:
                                only_8BInstruct_correct_8BInstruct_onlyts += 1
                            elif not model8BInstruct_correct and onlyts_correct:
                                only_onlyts_correct_8BInstruct_onlyts += 1
                            else:
                                both_wrong_8BInstruct_onlyts += 1
                
                print(f"    8BInstruct vs onlynews一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({both_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({only_8BInstruct_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      仅onlynews正确: {only_onlynews_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({only_onlynews_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlynews:5d}/{total_steps:5d} ({both_wrong_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                
                print(f"    8BInstruct vs onlyts一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({both_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({only_8BInstruct_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      仅onlyts正确: {only_onlyts_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({only_onlyts_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlyts:5d}/{total_steps:5d} ({both_wrong_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                
            elif dataset_type == "trendonly":
                # 对于trendonly数据集，只分析step（趋势）
                # 8BInstruct vs onlynews
                both_correct_8BInstruct_onlynews = 0
                only_8BInstruct_correct_8BInstruct_onlynews = 0
                only_onlynews_correct_8BInstruct_onlynews = 0
                both_wrong_8BInstruct_onlynews = 0
                
                # 8BInstruct vs onlyts
                both_correct_8BInstruct_onlyts = 0
                only_8BInstruct_correct_8BInstruct_onlyts = 0
                only_onlyts_correct_8BInstruct_onlyts = 0
                both_wrong_8BInstruct_onlyts = 0
                
                total_steps = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8BInstruct_news = model8BInstruct_data[i].get("news", "")
                    onlynews_news = onlynews_data[i].get("news", "")
                    onlyts_news = onlyts_data[i].get("news", "")
                    
                    # 解析news为step列表
                    gt_steps = parse_news_to_steps(gt_news)
                    model8BInstruct_steps = parse_news_to_steps(model8BInstruct_news)
                    onlynews_steps = parse_news_to_steps(onlynews_news)
                    onlyts_steps = parse_news_to_steps(onlyts_news)
                    
                    # 确保都有5个step
                    if len(gt_steps) == 5 and len(model8BInstruct_steps) == 5 and len(onlynews_steps) == 5 and len(onlyts_steps) == 5:
                        total_steps += 5
                        
                        # 计算8BInstruct vs onlynews一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8BInstruct_step = model8BInstruct_steps[j]
                            onlynews_step = onlynews_steps[j]
                            
                            # Step比较
                            model8BInstruct_correct = (model8BInstruct_step == gt_step)
                            onlynews_correct = (onlynews_step == gt_step)
                            
                            if model8BInstruct_correct and onlynews_correct:
                                both_correct_8BInstruct_onlynews += 1
                            elif model8BInstruct_correct and not onlynews_correct:
                                only_8BInstruct_correct_8BInstruct_onlynews += 1
                            elif not model8BInstruct_correct and onlynews_correct:
                                only_onlynews_correct_8BInstruct_onlynews += 1
                            else:
                                both_wrong_8BInstruct_onlynews += 1
                        
                        # 计算8BInstruct vs onlyts一致性
                        for j in range(5):
                            gt_step = gt_steps[j]
                            model8BInstruct_step = model8BInstruct_steps[j]
                            onlyts_step = onlyts_steps[j]
                            
                            # Step比较
                            model8BInstruct_correct = (model8BInstruct_step == gt_step)
                            onlyts_correct = (onlyts_step == gt_step)
                            
                            if model8BInstruct_correct and onlyts_correct:
                                both_correct_8BInstruct_onlyts += 1
                            elif model8BInstruct_correct and not onlyts_correct:
                                only_8BInstruct_correct_8BInstruct_onlyts += 1
                            elif not model8BInstruct_correct and onlyts_correct:
                                only_onlyts_correct_8BInstruct_onlyts += 1
                            else:
                                both_wrong_8BInstruct_onlyts += 1
                
                print(f"    8BInstruct vs onlynews一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({both_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({only_8BInstruct_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      仅onlynews正确: {only_onlynews_correct_8BInstruct_onlynews:5d}/{total_steps:5d} ({only_onlynews_correct_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlynews:5d}/{total_steps:5d} ({both_wrong_8BInstruct_onlynews/total_steps*100:6.2f}%)")
                
                print(f"    8BInstruct vs onlyts一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({both_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({only_8BInstruct_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      仅onlyts正确: {only_onlyts_correct_8BInstruct_onlyts:5d}/{total_steps:5d} ({only_onlyts_correct_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlyts:5d}/{total_steps:5d} ({both_wrong_8BInstruct_onlyts/total_steps*100:6.2f}%)")
                
            elif dataset_type == "globalonly":
                # 对于globalonly数据集，只分析record级别的全局趋势
                # 8BInstruct vs onlynews
                both_correct_8BInstruct_onlynews = 0
                only_8BInstruct_correct_8BInstruct_onlynews = 0
                only_onlynews_correct_8BInstruct_onlynews = 0
                both_wrong_8BInstruct_onlynews = 0
                
                # 8BInstruct vs onlyts
                both_correct_8BInstruct_onlyts = 0
                only_8BInstruct_correct_8BInstruct_onlyts = 0
                only_onlyts_correct_8BInstruct_onlyts = 0
                both_wrong_8BInstruct_onlyts = 0
                
                total_records = 0
                
                for i in range(min_records):
                    gt_news = gt_data[i].get("news", "")
                    model8BInstruct_news = model8BInstruct_data[i].get("news", "")
                    onlynews_news = onlynews_data[i].get("news", "")
                    onlyts_news = onlyts_data[i].get("news", "")
                    
                    # 解析news为global值
                    gt_steps = parse_news_to_steps(gt_news)
                    model8BInstruct_steps = parse_news_to_steps(model8BInstruct_news)
                    onlynews_steps = parse_news_to_steps(onlynews_news)
                    onlyts_steps = parse_news_to_steps(onlyts_news)
                    
                    # 确保都只有1个global值
                    if len(gt_steps) >= 1 and len(model8BInstruct_steps) >= 1 and len(onlynews_steps) >= 1 and len(onlyts_steps) >= 1:
                        total_records += 1
                        
                        gt_global = gt_steps[0]
                        model8BInstruct_global = model8BInstruct_steps[0]
                        onlynews_global = onlynews_steps[0]
                        onlyts_global = onlyts_steps[0]
                        
                        # Global比较
                        model8BInstruct_correct = (model8BInstruct_global == gt_global)
                        onlynews_correct = (onlynews_global == gt_global)
                        onlyts_correct = (onlyts_global == gt_global)
                        
                        # 8BInstruct vs onlynews
                        if model8BInstruct_correct and onlynews_correct:
                            both_correct_8BInstruct_onlynews += 1
                        elif model8BInstruct_correct and not onlynews_correct:
                            only_8BInstruct_correct_8BInstruct_onlynews += 1
                        elif not model8BInstruct_correct and onlynews_correct:
                            only_onlynews_correct_8BInstruct_onlynews += 1
                        else:
                            both_wrong_8BInstruct_onlynews += 1
                        
                        # 8BInstruct vs onlyts
                        if model8BInstruct_correct and onlyts_correct:
                            both_correct_8BInstruct_onlyts += 1
                        elif model8BInstruct_correct and not onlyts_correct:
                            only_8BInstruct_correct_8BInstruct_onlyts += 1
                        elif not model8BInstruct_correct and onlyts_correct:
                            only_onlyts_correct_8BInstruct_onlyts += 1
                        else:
                            both_wrong_8BInstruct_onlyts += 1
                
                print(f"    8BInstruct vs onlynews一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlynews:5d}/{total_records:5d} ({both_correct_8BInstruct_onlynews/total_records*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlynews:5d}/{total_records:5d} ({only_8BInstruct_correct_8BInstruct_onlynews/total_records*100:6.2f}%)")
                print(f"      仅onlynews正确: {only_onlynews_correct_8BInstruct_onlynews:5d}/{total_records:5d} ({only_onlynews_correct_8BInstruct_onlynews/total_records*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlynews:5d}/{total_records:5d} ({both_wrong_8BInstruct_onlynews/total_records*100:6.2f}%)")
                
                print(f"    8BInstruct vs onlyts一致性统计:")
                print(f"      都正确: {both_correct_8BInstruct_onlyts:5d}/{total_records:5d} ({both_correct_8BInstruct_onlyts/total_records*100:6.2f}%)")
                print(f"      仅8BInstruct正确: {only_8BInstruct_correct_8BInstruct_onlyts:5d}/{total_records:5d} ({only_8BInstruct_correct_8BInstruct_onlyts/total_records*100:6.2f}%)")
                print(f"      仅onlyts正确: {only_onlyts_correct_8BInstruct_onlyts:5d}/{total_records:5d} ({only_onlyts_correct_8BInstruct_onlyts/total_records*100:6.2f}%)")
                print(f"      都错误: {both_wrong_8BInstruct_onlyts:5d}/{total_records:5d} ({both_wrong_8BInstruct_onlyts/total_records*100:6.2f}%)")


if __name__ == "__main__":
    analyze_consistency()