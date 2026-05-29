# ts_analyzer_utils.py

import json
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from typing import Dict, Optional

# 建议在非交互式脚本的开头设置Agg后端，防止没有图形界面的服务器报错
import matplotlib
matplotlib.use('Agg')

# 设置中文字体（如果需要在图表中显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')


def parse_series_from_entry(data_entry: Dict) -> Optional[pd.Series]:
    """从单个数据字典中解析出完整的时间序列。"""
    try:
        hist_str = data_entry['hist_data']
        gt_str = data_entry['ground_truth']
        
        history = [float(x.strip()) for x in hist_str.split(',')]
        ground_truth = [float(x.strip()) for x in gt_str.split(',')]
        
        full_series = pd.Series(history + ground_truth)
        return full_series
    except (KeyError, ValueError) as e:
        print(f"    [Warning] Skipping entry due to parsing error: {e}")
        return None

def perform_adf_test(ts_series: pd.Series) -> Dict:
    """执行ADF检验并返回一个结果字典。"""
    result = adfuller(ts_series.dropna())
    p_value = result[1]
    
    conclusion = "Non-Stationary (p > 0.05)" if p_value > 0.05 else "Stationary (p <= 0.05)"
    
    return {
        "adf_statistic": result[0],
        "p_value": p_value,
        "critical_values": result[4],
        "conclusion": conclusion
    }

def plot_timeseries(ts_series: pd.Series, title: str, output_path: str):
    """绘制时间序列图并保存到文件。"""
    plt.figure(figsize=(14, 7))
    ts_series.plot(marker='o', linestyle='-', label='观测值')
    
    num_hist_points = len(ts_series) // 2
    plt.axvline(x=num_hist_points - 0.5, color='r', linestyle='--', label='历史/未来分界线')
    
    plt.title(title, fontsize=16)
    plt.xlabel("时间点 (Time Step)")
    plt.ylabel("值 (Value)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close() # 关闭图形，释放内存

def plot_acf_pacf(ts_series: pd.Series, title: str, output_path: str):
    """绘制ACF/PACF图并保存到文件。"""
    lags = min(len(ts_series) // 2 - 1, 20)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    plot_acf(ts_series.dropna(), ax=axes[0], lags=lags)
    axes[0].set_title('自相关函数 (ACF)')
    
    plot_pacf(ts_series.dropna(), ax=axes[1], lags=lags, method='ywm')
    axes[1].set_title('偏自相关函数 (PACF)')
    
    fig.suptitle(title, fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, bbox_inches='tight')
    plt.close() # 关闭图形，释放内存
