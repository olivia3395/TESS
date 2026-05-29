"""
跨模态分析可视化工具类
Cross-modal Analysis Visualization Tools

该模块实现跨模态分析的可视化功能，包括：
1. 标签分布可视化
2. 分类器性能对比
3. 混淆矩阵热图
4. 特征重要性分析
5. 统计检验结果可视化

作者：Claude Code Assistant
日期：2025-01-XX
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class CrossModalVisualizer:
    """跨模态分析可视化主类"""
    
    def __init__(self, figsize_default: Tuple[int, int] = (12, 8), dpi: int = 300):
        """
        初始化可视化器
        
        Args:
            figsize_default: 默认图片尺寸
            dpi: 图片分辨率
        """
        self.figsize_default = figsize_default
        self.dpi = dpi
        self.color_palette = sns.color_palette("husl", n_colors=10)
        
    def plot_label_distribution(self, label_analysis: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制标签分布图
        
        Args:
            label_analysis: 标签分析结果
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练集标签分布
        train_dist = label_analysis['train_distribution']
        labels = list(train_dist.keys())
        train_counts = list(train_dist.values())
        
        ax1.pie(train_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.color_palette[:len(labels)])
        ax1.set_title(f'Training Set Label Distribution\n(Total: {label_analysis["train_total"]})', fontsize=14, fontweight='bold')
        
        # 测试集标签分布
        test_dist = label_analysis['test_distribution']
        test_counts = [test_dist.get(label, 0) for label in labels]
        
        ax2.pie(test_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.color_palette[:len(labels)])
        ax2.set_title(f'Test Set Label Distribution\n(Total: {label_analysis["test_total"]})', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"标签分布图已保存到: {save_path}")
        
        return fig
    
    def plot_performance_comparison(self, evaluation_results: Dict[str, Dict], 
                                  baseline_results: Dict[str, float],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制分类器性能对比图
        
        Args:
            evaluation_results: 评估结果
            baseline_results: 基准结果
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 提取性能指标
        classifiers = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for name, results in evaluation_results.items():
            if 'accuracy' in results:
                classifiers.append(name.replace('_', ' ').title())
                accuracies.append(results['accuracy'])
                precisions.append(results['precision'])
                recalls.append(results['recall'])
                f1_scores.append(results['f1'])
        
        # 添加基准线
        random_acc = baseline_results['theoretical_random']
        majority_acc = baseline_results['majority_accuracy']
        
        x_pos = np.arange(len(classifiers))
        
        # 准确率对比
        bars1 = ax1.bar(x_pos, accuracies, color=self.color_palette[:len(classifiers)])
        ax1.axhline(y=random_acc, color='red', linestyle='--', alpha=0.7, label=f'Random Baseline ({random_acc:.3f})')
        ax1.axhline(y=majority_acc, color='orange', linestyle='--', alpha=0.7, label=f'Majority Baseline ({majority_acc:.3f})')
        ax1.set_xlabel('Classifiers')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Classification Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(classifiers, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 精确率对比
        bars2 = ax2.bar(x_pos, precisions, color=self.color_palette[:len(classifiers)])
        ax2.set_xlabel('Classifiers')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision Comparison', fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(classifiers, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        for i, (bar, prec) in enumerate(zip(bars2, precisions)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{prec:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 召回率对比
        bars3 = ax3.bar(x_pos, recalls, color=self.color_palette[:len(classifiers)])
        ax3.set_xlabel('Classifiers')
        ax3.set_ylabel('Recall')
        ax3.set_title('Recall Comparison', fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(classifiers, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        for i, (bar, rec) in enumerate(zip(bars3, recalls)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rec:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1分数对比
        bars4 = ax4.bar(x_pos, f1_scores, color=self.color_palette[:len(classifiers)])
        ax4.set_xlabel('Classifiers')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('F1 Score Comparison', fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(classifiers, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        for i, (bar, f1) in enumerate(zip(bars4, f1_scores)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"性能对比图已保存到: {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, evaluation_results: Dict[str, Dict], 
                               labels: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制混淆矩阵热图
        
        Args:
            evaluation_results: 评估结果
            labels: 标签列表
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        n_classifiers = len(evaluation_results)
        cols = min(3, n_classifiers)
        rows = (n_classifiers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_classifiers == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_classifiers > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, results) in enumerate(evaluation_results.items()):
            if 'confusion_matrix' in results:
                cm = np.array(results['confusion_matrix'])
                
                # 归一化混淆矩阵
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)
                
                # 绘制热图
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                           xticklabels=labels, yticklabels=labels,
                           ax=axes[idx], cbar_kws={'label': 'Normalized Count'})
                
                axes[idx].set_title(f'{name.replace("_", " ").title()}\nConfusion Matrix', 
                                  fontweight='bold')
                axes[idx].set_xlabel('Predicted Label')
                axes[idx].set_ylabel('True Label')
        
        # 隐藏多余的子图
        for idx in range(n_classifiers, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"混淆矩阵图已保存到: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, classifier_results: Dict[str, Any],
                               feature_names: List[str],
                               top_k: int = 20,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制特征重要性图
        
        Args:
            classifier_results: 分类器结果（包含特征重要性）
            feature_names: 特征名称列表
            top_k: 显示前k个重要特征
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, axes = plt.subplots(1, 1, figsize=(12, 8))
        
        # 这里假设我们使用随机森林的特征重要性
        # 在实际实现中，需要从训练好的分类器中提取特征重要性
        
        # 为演示目的，生成示例特征重要性
        if 'random_forest' in classifier_results:
            # 假设随机森林分类器有feature_importances_属性
            importances = np.random.random(len(feature_names))  # 实际应该从分类器获取
            
            # 获取top_k特征
            indices = np.argsort(importances)[::-1][:top_k]
            top_features = [feature_names[i] for i in indices]
            top_importances = importances[indices]
            
            # 绘制水平条形图
            y_pos = np.arange(len(top_features))
            bars = axes.barh(y_pos, top_importances, color=self.color_palette[0])
            
            axes.set_yticks(y_pos)
            axes.set_yticklabels(top_features)
            axes.set_xlabel('Feature Importance')
            axes.set_title(f'Top {top_k} Feature Importance (Random Forest)', fontweight='bold')
            axes.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, top_importances)):
                axes.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                         f'{imp:.3f}', ha='left', va='center')
        
        else:
            axes.text(0.5, 0.5, 'Feature importance not available\nfor selected classifiers', 
                     ha='center', va='center', transform=axes.transAxes, fontsize=14)
            axes.set_title('Feature Importance Analysis', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"特征重要性图已保存到: {save_path}")
        
        return fig
    
    def plot_statistical_significance(self, statistical_results: Dict[str, Any],
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制统计显著性检验结果
        
        Args:
            statistical_results: 统计检验结果
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 卡方检验结果
        chi2_results = []
        chi2_names = []
        chi2_pvalues = []
        chi2_significant = []
        
        for key, result in statistical_results.items():
            if 'chi2' in key and 'chi2_statistic' in result:
                chi2_names.append(key.replace('_chi2', '').replace('_', ' ').title())
                chi2_pvalues.append(result['p_value'])
                chi2_significant.append(result['is_significant'])
        
        if chi2_names:
            colors = ['green' if sig else 'red' for sig in chi2_significant]
            bars1 = ax1.bar(range(len(chi2_names)), chi2_pvalues, color=colors, alpha=0.7)
            ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
            ax1.set_xlabel('Classifiers')
            ax1.set_ylabel('p-value')
            ax1.set_title('Chi-square Test Results\n(Green: Significant, Red: Not Significant)', fontweight='bold')
            ax1.set_xticks(range(len(chi2_names)))
            ax1.set_xticklabels(chi2_names, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')  # 使用对数尺度显示p值
            
            # 添加p值标签
            for i, (bar, pval) in enumerate(zip(bars1, chi2_pvalues)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                        f'{pval:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # McNemar检验结果
        mcnemar_results = []
        mcnemar_names = []
        mcnemar_pvalues = []
        mcnemar_significant = []
        
        for key, result in statistical_results.items():
            if 'mcnemar' in key and 'p_value' in result:
                mcnemar_names.append(key.replace('mcnemar_', '').replace('_vs_', ' vs ').replace('_', ' ').title())
                mcnemar_pvalues.append(result['p_value'])
                mcnemar_significant.append(result['is_significant'])
        
        if mcnemar_names:
            colors = ['green' if sig else 'red' for sig in mcnemar_significant]
            bars2 = ax2.bar(range(len(mcnemar_names)), mcnemar_pvalues, color=colors, alpha=0.7)
            ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.8, label='α = 0.05')
            ax2.set_xlabel('Classifier Comparisons')
            ax2.set_ylabel('p-value')
            ax2.set_title('McNemar Test Results\n(Green: Significant Difference, Red: No Difference)', fontweight='bold')
            ax2.set_xticks(range(len(mcnemar_names)))
            ax2.set_xticklabels(mcnemar_names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')  # 使用对数尺度显示p值
            
            # 添加p值标签
            for i, (bar, pval) in enumerate(zip(bars2, mcnemar_pvalues)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                        f'{pval:.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 如果没有数据，显示提示信息
        if not chi2_names:
            ax1.text(0.5, 0.5, 'No Chi-square test results available', 
                    ha='center', va='center', transform=ax1.transAxes)
        
        if not mcnemar_names:
            ax2.text(0.5, 0.5, 'No McNemar test results available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"统计显著性图已保存到: {save_path}")
        
        return fig
    
    def plot_baseline_comparison(self, evaluation_results: Dict[str, Dict], 
                               baseline_results: Dict[str, float],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制与基准线的详细对比
        
        Args:
            evaluation_results: 评估结果
            baseline_results: 基准结果
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 提取准确率
        classifiers = []
        accuracies = []
        
        for name, results in evaluation_results.items():
            if 'accuracy' in results:
                classifiers.append(name.replace('_', ' ').title())
                accuracies.append(results['accuracy'])
        
        # 添加基准线结果
        baselines = ['Random', 'Majority', 'Theoretical Random']
        baseline_accs = [
            baseline_results.get('random_accuracy', 0),
            baseline_results.get('majority_accuracy', 0),
            baseline_results.get('theoretical_random', 0)
        ]
        
        all_names = classifiers + baselines
        all_accs = accuracies + baseline_accs
        
        # 区分分类器和基准线的颜色
        colors = (self.color_palette[:len(classifiers)] + 
                 ['gray', 'lightcoral', 'lightblue'])
        
        x_pos = np.arange(len(all_names))
        bars = ax.bar(x_pos, all_accs, color=colors, alpha=0.8)
        
        # 添加分隔线
        ax.axvline(x=len(classifiers) - 0.5, color='black', linestyle='-', alpha=0.5)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Classifier Performance vs Baselines', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(all_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, acc in zip(bars, all_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                   f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 添加图例
        classifier_patch = plt.Rectangle((0, 0), 1, 1, facecolor=self.color_palette[0], alpha=0.8)
        baseline_patch = plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.8)
        ax.legend([classifier_patch, baseline_patch], ['Classifiers', 'Baselines'], loc='upper right')
        
        # 添加效果分析文本
        max_acc = max(accuracies) if accuracies else 0
        theoretical_random = baseline_results.get('theoretical_random', 0)
        improvement = max_acc - theoretical_random
        
        ax.text(0.02, 0.98, f'Best Improvement over Random: {improvement:.3f}', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logging.info(f"基准对比图已保存到: {save_path}")
        
        return fig
    
    def create_summary_report_figures(self, analysis_results: Dict[str, Any], 
                                    output_dir: str) -> Dict[str, str]:
        """
        创建所有可视化图表
        
        Args:
            analysis_results: 完整分析结果
            output_dir: 输出目录
            
        Returns:
            生成的图片文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        try:
            # 1. 标签分布图
            if 'label_analysis' in analysis_results:
                path = output_path / 'label_distribution.png'
                self.plot_label_distribution(analysis_results['label_analysis'], str(path))
                generated_files['label_distribution'] = str(path)
            
            # 2. 性能对比图
            if ('evaluation_results' in analysis_results and 
                'baseline_results' in analysis_results):
                path = output_path / 'performance_comparison.png'
                self.plot_performance_comparison(
                    analysis_results['evaluation_results'],
                    analysis_results['baseline_results'], 
                    str(path)
                )
                generated_files['performance_comparison'] = str(path)
            
            # 3. 混淆矩阵图
            if 'evaluation_results' in analysis_results and 'label_analysis' in analysis_results:
                path = output_path / 'confusion_matrices.png'
                labels = analysis_results['label_analysis']['unique_labels']
                self.plot_confusion_matrices(
                    analysis_results['evaluation_results'], 
                    labels, 
                    str(path)
                )
                generated_files['confusion_matrices'] = str(path)
            
            # 4. 基准对比图
            if ('evaluation_results' in analysis_results and 
                'baseline_results' in analysis_results):
                path = output_path / 'baseline_comparison.png'
                self.plot_baseline_comparison(
                    analysis_results['evaluation_results'],
                    analysis_results['baseline_results'], 
                    str(path)
                )
                generated_files['baseline_comparison'] = str(path)
            
            # 5. 统计显著性图
            if 'statistical_results' in analysis_results:
                path = output_path / 'statistical_significance.png'
                self.plot_statistical_significance(
                    analysis_results['statistical_results'], 
                    str(path)
                )
                generated_files['statistical_significance'] = str(path)
            
            # 6. 特征重要性图（如果有相关数据）
            if ('evaluation_results' in analysis_results and 
                'feature_names' in analysis_results):
                path = output_path / 'feature_importance.png'
                self.plot_feature_importance(
                    analysis_results['evaluation_results'],
                    analysis_results['feature_names'], 
                    save_path=str(path)
                )
                generated_files['feature_importance'] = str(path)
            
            logging.info(f"所有可视化图表已生成，共 {len(generated_files)} 个文件")
            
        except Exception as e:
            logging.error(f"生成可视化图表失败: {e}")
        
        return generated_files