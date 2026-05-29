#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成onlynews和onlyts消融实验命中率对比表格
"""

import json
import os


def load_hit_rate_results(file_path):
    """加载命中率结果"""
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def generate_comparison_table(onlynews_results, onlyts_results):
    """生成对比表格"""
    print("=" * 100)
    print("onlynews vs onlyts 消融实验命中率对比表")
    print("=" * 100)
    
    # 表头
    print(f"{'数据集类型':<15} {'分割集':<8} {'命中率(%)':<30} {'匹配数/总数':<30}")
    print(f"{'':<15} {'':<8} {'onlynews  onlyts':<30} {'onlynews    onlyts':<30}")
    print("-" * 100)
    
    # 按描述和分割集组织数据
    results_dict = {}
    
    # 处理onlynews结果
    for item in onlynews_results:
        desc = item['description']
        split = item['split']
        key = (desc, split)
        if key not in results_dict:
            results_dict[key] = {'onlynews': {}, 'onlyts': {}}
        results_dict[key]['onlynews'] = item
    
    # 处理onlyts结果
    for item in onlyts_results:
        desc = item['description']
        split = item['split']
        key = (desc, split)
        if key not in results_dict:
            results_dict[key] = {'onlynews': {}, 'onlyts': {}}
        results_dict[key]['onlyts'] = item
    
    # 按数据集类型排序并显示
    sorted_keys = sorted(results_dict.keys(), key=lambda x: (x[0], x[1]))
    
    for desc, split in sorted_keys:
        result = results_dict[(desc, split)]
        onlynews_data = result.get('onlynews', {})
        onlyts_data = result.get('onlyts', {})
        
        # 确定数据集类型名称
        if 'trendstrength' in desc:
            dataset_type = '趋势+强度'
        elif 'trendonly' in desc:
            dataset_type = '仅趋势'
        elif 'globalonly' in desc:
            dataset_type = '全局趋势'
        else:
            dataset_type = '未知'
        
        # 确定分割集名称
        split_name = {
            'test': '测试集',
            'vali': '验证集',
            'train': '训练集'
        }.get(split, split)
        
        # 根据数据集类型显示不同的命中率和匹配数
        if 'trendstrength' in desc:
            # 趋势+强度数据集显示step级别命中率
            onlynews_rate = onlynews_data.get('step_hit_rate', 0) * 100
            onlyts_rate = onlyts_data.get('step_hit_rate', 0) * 100
            onlynews_match = f"{onlynews_data.get('match_steps', 0):>5d}/{onlynews_data.get('total_steps', 0):>5d}"
            onlyts_match = f"{onlyts_data.get('match_steps', 0):>5d}/{onlyts_data.get('total_steps', 0):>5d}"
        elif 'trendonly' in desc:
            # 仅趋势数据集显示step级别命中率
            onlynews_rate = onlynews_data.get('step_hit_rate', 0) * 100
            onlyts_rate = onlyts_data.get('step_hit_rate', 0) * 100
            onlynews_match = f"{onlynews_data.get('match_steps', 0):>5d}/{onlynews_data.get('total_steps', 0):>5d}"
            onlyts_match = f"{onlyts_data.get('match_steps', 0):>5d}/{onlyts_data.get('total_steps', 0):>5d}"
        elif 'globalonly' in desc:
            # 全局趋势数据集显示record级别命中率
            onlynews_rate = onlynews_data.get('record_hit_rate', 0) * 100 if 'record_hit_rate' in onlynews_data else 0
            onlyts_rate = onlyts_data.get('record_hit_rate', 0) * 100 if 'record_hit_rate' in onlyts_data else 0
            onlynews_match = f"{onlynews_data.get('match_records', 0):>4d}/{onlynews_data.get('total_records', 0):>4d}" if 'match_records' in onlynews_data else "    0/    0"
            onlyts_match = f"{onlyts_data.get('match_records', 0):>4d}/{onlyts_data.get('total_records', 0):>4d}" if 'match_records' in onlyts_data else "    0/    0"
        else:
            onlynews_rate = 0
            onlyts_rate = 0
            onlynews_match = "    0/    0"
            onlyts_match = "    0/    0"
        
        print(f"{dataset_type:<15} {split_name:<8} {onlynews_rate:>7.2f}  {onlyts_rate:>7.2f}     {onlynews_match}  {onlyts_match}")
    
    print("=" * 100)


def generate_summary_table(onlynews_results, onlyts_results):
    """生成汇总表格，仅显示test集的数据"""
    print("\n" + "=" * 80)
    print("测试集命中率汇总表")
    print("=" * 80)
    
    # 表头
    print(f"{'数据集类型':<15} {'命中率(%)':<30} {'匹配数/总数':<30}")
    print(f"{'':<15} {'onlynews  onlyts':<30} {'onlynews    onlyts':<30}")
    print("-" * 80)
    
    # 创建字典以便查找
    onlynews_dict = {(item['description'], item['split']): item for item in onlynews_results}
    onlyts_dict = {(item['description'], item['split']): item for item in onlyts_results}
    
    # 合并所有的键
    all_keys = set(onlynews_dict.keys()) | set(onlyts_dict.keys())
    
    # 筛选出test集的数据
    test_keys = [key for key in all_keys if key[1] == 'test']
    
    # 显示每种数据集类型的结果
    for key in sorted(test_keys):
        onlynews_item = onlynews_dict.get(key, {})
        onlyts_item = onlyts_dict.get(key, {})
        
        desc, split = key
        
        # 确定数据集类型名称
        if 'trendstrength' in desc:
            dataset_type = '趋势+强度'
        elif 'trendonly' in desc:
            dataset_type = '仅趋势'
        elif 'globalonly' in desc:
            dataset_type = '全局趋势'
        else:
            dataset_type = '未知'
        
        # 根据数据集类型显示不同的命中率和匹配数
        if 'trendstrength' in desc:
            # 趋势+强度数据集显示step级别命中率
            onlynews_rate = onlynews_item.get('step_hit_rate', 0) * 100
            onlyts_rate = onlyts_item.get('step_hit_rate', 0) * 100
            onlynews_match = f"{onlynews_item.get('match_steps', 0):>5d}/{onlynews_item.get('total_steps', 0):>5d}"
            onlyts_match = f"{onlyts_item.get('match_steps', 0):>5d}/{onlyts_item.get('total_steps', 0):>5d}"
        elif 'trendonly' in desc:
            # 仅趋势数据集显示step级别命中率
            onlynews_rate = onlynews_item.get('step_hit_rate', 0) * 100
            onlyts_rate = onlyts_item.get('step_hit_rate', 0) * 100
            onlynews_match = f"{onlynews_item.get('match_steps', 0):>5d}/{onlynews_item.get('total_steps', 0):>5d}"
            onlyts_match = f"{onlyts_item.get('match_steps', 0):>5d}/{onlyts_item.get('total_steps', 0):>5d}"
        elif 'globalonly' in desc:
            # 全局趋势数据集显示record级别命中率
            onlynews_rate = onlynews_item.get('record_hit_rate', 0) * 100 if 'record_hit_rate' in onlynews_item else 0
            onlyts_rate = onlyts_item.get('record_hit_rate', 0) * 100 if 'record_hit_rate' in onlyts_item else 0
            onlynews_match = f"{onlynews_item.get('match_records', 0):>4d}/{onlynews_item.get('total_records', 0):>4d}" if 'match_records' in onlynews_item else "    0/    0"
            onlyts_match = f"{onlyts_item.get('match_records', 0):>4d}/{onlyts_item.get('total_records', 0):>4d}" if 'match_records' in onlyts_item else "    0/    0"
        else:
            onlynews_rate = 0
            onlyts_rate = 0
            onlynews_match = "    0/    0"
            onlyts_match = "    0/    0"
        
        print(f"{dataset_type:<15} {onlynews_rate:>7.2f}  {onlyts_rate:>7.2f}     {onlynews_match}  {onlyts_match}")
    
    print("=" * 80)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 加载onlynews结果
    onlynews_file = os.path.join(script_dir, "8B_Ablation_onlynews_hit_rate_results.json")
    onlynews_results = load_hit_rate_results(onlynews_file)
    
    # 加载onlyts结果
    onlyts_file = os.path.join(script_dir, "8B_Ablation_onlyts_hit_rate_results.json")
    onlyts_results = load_hit_rate_results(onlyts_file)
    
    if onlynews_results is None or onlyts_results is None:
        return
    
    # 生成对比表格
    generate_comparison_table(onlynews_results, onlyts_results)
    
    # 生成汇总表格
    generate_summary_table(onlynews_results, onlyts_results)


if __name__ == "__main__":
    main()