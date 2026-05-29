#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
生成消融实验一致性分析结果表格
"""

import json
import os
import re


def load_consistency_data(file_path):
    """加载一致性分析数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def parse_consistency_data(content):
    """解析一致性分析数据"""
    lines = content.strip().split('\n')
    
    # 存储解析结果
    results = {
        'trendstrength': {},
        'trendonly': {},
        'globalonly': {}
    }
    
    current_dataset_type = None
    current_split = None
    
    for line in lines:
        line = line.strip()
        
        # 检查数据集类型
        if line.startswith("数据集类型:"):
            current_dataset_type = line.split(":")[1].strip()
            results[current_dataset_type] = {}
            continue
            
        # 检查数据集分割类型
        if line.endswith("数据集:"):
            current_split = line.split(" ")[0]
            results[current_dataset_type][current_split] = {
                '8BInstruct_vs_onlynews': {},
                '8BInstruct_vs_onlyts': {}
            }
            continue
            
        # 解析8BInstruct vs onlynews一致性统计
        if line.startswith("8BInstruct vs onlynews一致性统计:"):
            continue
            
        if line.startswith("都正确:") and '8BInstruct_vs_onlynews' in results[current_dataset_type][current_split]:
            # 使用正则表达式提取数字
            match = re.search(r'都正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlynews']['both_correct'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("仅8BInstruct正确:") and '8BInstruct_vs_onlynews' in results[current_dataset_type][current_split]:
            match = re.search(r'仅8BInstruct正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlynews']['only_8BInstruct'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("仅onlynews正确:") and '8BInstruct_vs_onlynews' in results[current_dataset_type][current_split]:
            match = re.search(r'仅onlynews正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlynews']['only_onlynews'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("都错误:") and '8BInstruct_vs_onlynews' in results[current_dataset_type][current_split]:
            match = re.search(r'都错误:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlynews']['both_wrong'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        # 解析8BInstruct vs onlyts一致性统计
        if line.startswith("8BInstruct vs onlyts一致性统计:"):
            continue
            
        if line.startswith("都正确:") and '8BInstruct_vs_onlyts' in results[current_dataset_type][current_split]:
            match = re.search(r'都正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlyts']['both_correct'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("仅8BInstruct正确:") and '8BInstruct_vs_onlyts' in results[current_dataset_type][current_split]:
            match = re.search(r'仅8BInstruct正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlyts']['only_8BInstruct'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("仅onlyts正确:") and '8BInstruct_vs_onlyts' in results[current_dataset_type][current_split]:
            match = re.search(r'仅onlyts正确:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlyts']['only_onlyts'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
            
        if line.startswith("都错误:") and '8BInstruct_vs_onlyts' in results[current_dataset_type][current_split]:
            match = re.search(r'都错误:\s+(\d+)/\s*(\d+)\s+\(\s*([\d.]+)%\)', line)
            if match:
                count = int(match.group(1))
                total = int(match.group(2))
                percentage = float(match.group(3))
                results[current_dataset_type][current_split]['8BInstruct_vs_onlyts']['both_wrong'] = {
                    'count': count,
                    'total': total,
                    'percentage': percentage
                }
            continue
    
    return results


def generate_summary_table(results):
    """生成汇总表格"""
    print("=" * 100)
    print("消融实验一致性分析结果汇总表")
    print("=" * 100)
    
    # 表头
    print(f"{'数据集类型':<12} {'分割集':<8} {'都正确(%)':<20} {'仅完整模型正确(%)':<25} {'仅消融模型正确(%)':<25}")
    print("-" * 100)
    
    # 遍历每种数据集类型
    for dataset_type in ['trendstrength', 'trendonly', 'globalonly']:
        dataset_name = {
            'trendstrength': '趋势+强度',
            'trendonly': '仅趋势',
            'globalonly': '全局趋势'
        }[dataset_type]
        
        # 遍历每个分割集
        for split in ['test', 'vali', 'train']:
            split_name = {
                'test': '测试集',
                'vali': '验证集',
                'train': '训练集'
            }[split]
            
            # 8BInstruct vs onlynews
            if dataset_type in results and split in results[dataset_type]:
                both_correct_news = results[dataset_type][split]['8BInstruct_vs_onlynews']['both_correct']['percentage']
                only_8BInstruct_news = results[dataset_type][split]['8BInstruct_vs_onlynews']['only_8BInstruct']['percentage']
                only_onlynews = results[dataset_type][split]['8BInstruct_vs_onlynews']['only_onlynews']['percentage']
                
                print(f"{dataset_name:<12} {split_name:<8} {both_correct_news:<20.2f} {only_8BInstruct_news:<25.2f} {only_onlynews:<25.2f}")
            
            # 8BInstruct vs onlyts
            if dataset_type in results and split in results[dataset_type]:
                both_correct_ts = results[dataset_type][split]['8BInstruct_vs_onlyts']['both_correct']['percentage']
                only_8BInstruct_ts = results[dataset_type][split]['8BInstruct_vs_onlyts']['only_8BInstruct']['percentage']
                only_onlyts = results[dataset_type][split]['8BInstruct_vs_onlyts']['only_onlyts']['percentage']
                
                print(f"{dataset_name:<12} {split_name:<8} {both_correct_ts:<20.2f} {only_8BInstruct_ts:<25.2f} {only_onlyts:<25.2f}")
    
    print("=" * 100)


def generate_detailed_comparison_table(results):
    """生成详细对比表格"""
    print("\n" + "=" * 120)
    print("详细对比：移除文本信息 vs 移除时序信息")
    print("=" * 120)
    
    # 表头
    print(f"{'数据集类型':<12} {'分割集':<8} {'都正确(%)':<30} {'仅完整模型正确(%)':<30} {'仅消融模型正确(%)':<30}")
    print(f"{'':<12} {'':<8} {'文本  时序':<30} {'文本  时序':<30} {'文本  时序':<30}")
    print("-" * 120)
    
    # 遍历每种数据集类型
    for dataset_type in ['trendstrength', 'trendonly', 'globalonly']:
        dataset_name = {
            'trendstrength': '趋势+强度',
            'trendonly': '仅趋势',
            'globalonly': '全局趋势'
        }[dataset_type]
        
        # 遍历每个分割集
        for split in ['test', 'vali', 'train']:
            split_name = {
                'test': '测试集',
                'vali': '验证集',
                'train': '训练集'
            }[split]
            
            # 获取数据
            if dataset_type in results and split in results[dataset_type]:
                # 8BInstruct vs onlynews
                both_correct_news = results[dataset_type][split]['8BInstruct_vs_onlynews']['both_correct']['percentage']
                only_8BInstruct_news = results[dataset_type][split]['8BInstruct_vs_onlynews']['only_8BInstruct']['percentage']
                only_onlynews = results[dataset_type][split]['8BInstruct_vs_onlynews']['only_onlynews']['percentage']
                
                # 8BInstruct vs onlyts
                both_correct_ts = results[dataset_type][split]['8BInstruct_vs_onlyts']['both_correct']['percentage']
                only_8BInstruct_ts = results[dataset_type][split]['8BInstruct_vs_onlyts']['only_8BInstruct']['percentage']
                only_onlyts = results[dataset_type][split]['8BInstruct_vs_onlyts']['only_onlyts']['percentage']
                
                print(f"{dataset_name:<12} {split_name:<8} {both_correct_news:<6.2f}  {both_correct_ts:<6.2f}     {only_8BInstruct_news:<6.2f}  {only_8BInstruct_ts:<6.2f}     {only_onlynews:<6.2f}  {only_onlyts:<6.2f}")
    
    print("=" * 120)


def generate_key_insights_table(results):
    """生成关键洞察表格"""
    print("\n" + "=" * 80)
    print("关键洞察：移除不同模态对模型性能的影响")
    print("=" * 80)
    
    # 表头
    print(f"{'数据集类型':<12} {'分割集':<8} {'移除文本影响(%)':<20} {'移除时序影响(%)':<20} {'主要影响因素':<15}")
    print("-" * 80)
    
    # 遍历每种数据集类型
    for dataset_type in ['trendstrength', 'trendonly', 'globalonly']:
        dataset_name = {
            'trendstrength': '趋势+强度',
            'trendonly': '仅趋势',
            'globalonly': '全局趋势'
        }[dataset_type]
        
        # 遍历每个分割集
        for split in ['test', 'vali', 'train']:
            split_name = {
                'test': '测试集',
                'vali': '验证集',
                'train': '训练集'
            }[split]
            
            # 获取数据
            if dataset_type in results and split in results[dataset_type]:
                # 8BInstruct vs onlynews
                both_correct_news = results[dataset_type][split]['8BInstruct_vs_onlynews']['both_correct']['percentage']
                
                # 8BInstruct vs onlyts
                both_correct_ts = results[dataset_type][split]['8BInstruct_vs_onlyts']['both_correct']['percentage']
                
                # 计算影响程度（100% - 都正确的百分比）
                impact_news = 100 - both_correct_news
                impact_ts = 100 - both_correct_ts
                
                # 判断主要影响因素
                if impact_news > impact_ts:
                    main_factor = "文本信息"
                elif impact_ts > impact_news:
                    main_factor = "时序信息"
                else:
                    main_factor = "相当"
                
                print(f"{dataset_name:<12} {split_name:<8} {impact_news:<20.2f} {impact_ts:<20.2f} {main_factor:<15}")
    
    print("=" * 80)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "compare_ablation_consistency_with_8BInstruct.txt")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    # 加载数据
    content = load_consistency_data(file_path)
    
    # 解析数据
    results = parse_consistency_data(content)
    
    # 生成表格
    generate_summary_table(results)
    generate_detailed_comparison_table(results)
    generate_key_insights_table(results)


if __name__ == "__main__":
    main()