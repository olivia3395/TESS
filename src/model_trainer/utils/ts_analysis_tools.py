"""
Time Series Analysis Tools
==========================

专用于时序数据分析的工具集，包含信息论分析、统计分析、复杂度分析等功能
符合软件工程规范的模块化设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入可选依赖
try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore')


class InformationTheoryAnalyzer:
    """信息论分析器"""
    
    @staticmethod
    def shannon_entropy(data: np.ndarray, bins: int = 50) -> float:
        """计算香农熵"""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # 移除零值
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    @staticmethod
    def conditional_entropy(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
        """计算条件熵 H(Y|X)"""
        # 离散化数据
        x_discrete = np.digitize(x, np.histogram_bin_edges(x, bins))
        y_discrete = np.digitize(y, np.histogram_bin_edges(y, bins))
        
        # 计算联合概率和边际概率
        joint_hist = np.histogram2d(x_discrete, y_discrete, bins=bins)[0]
        joint_prob = joint_hist / np.sum(joint_hist)
        x_prob = np.sum(joint_prob, axis=1)
        
        # 计算条件熵
        conditional_entropy = 0
        for i in range(len(x_prob)):
            if x_prob[i] > 0:
                conditional_prob = joint_prob[i, :] / x_prob[i]
                conditional_prob = conditional_prob[conditional_prob > 0]
                conditional_entropy += x_prob[i] * (-np.sum(conditional_prob * np.log2(conditional_prob + 1e-10)))
        
        return conditional_entropy
    
    @staticmethod
    def mutual_information_continuous(x: np.ndarray, y: np.ndarray) -> float:
        """计算连续变量的互信息"""
        if not SKLEARN_AVAILABLE:
            # 简化的互信息估算
            correlation = np.corrcoef(x, y)[0, 1]
            return -0.5 * np.log(1 - correlation**2) if abs(correlation) < 0.99 else 1.0
        
        # 标准化数据
        x_norm = (x - np.mean(x)) / np.std(x)
        y_norm = (y - np.mean(y)) / np.std(y)
        
        # 使用KDE估计
        return mutual_info_regression(x_norm.reshape(-1, 1), y_norm)[0]
    
    @staticmethod
    def information_content_ratio(historical_data: np.ndarray, future_data: np.ndarray) -> Dict[str, float]:
        """计算历史数据对未来数据的信息含量比例"""
        h_historical = InformationTheoryAnalyzer.shannon_entropy(historical_data)
        h_future = InformationTheoryAnalyzer.shannon_entropy(future_data)
        h_conditional = InformationTheoryAnalyzer.conditional_entropy(historical_data, future_data)
        
        # 信息增益
        information_gain = h_future - h_conditional
        
        # 信息含量比例
        info_ratio = information_gain / h_future if h_future > 0 else 0
        
        return {
            'historical_entropy': h_historical,
            'future_entropy': h_future,
            'conditional_entropy': h_conditional,
            'information_gain': information_gain,
            'information_content_ratio': info_ratio
        }


class StatisticalAnalyzer:
    """统计分析器"""
    
    @staticmethod
    def stationarity_test(data: np.ndarray) -> Dict[str, Union[float, bool, str]]:
        """ADF平稳性检验"""
        if not STATSMODELS_AVAILABLE:
            # 简化的平稳性检验：基于方差的滚动统计
            window_size = min(50, len(data) // 4)
            rolling_var = pd.Series(data).rolling(window=window_size).var().dropna()
            var_stability = np.std(rolling_var) / np.mean(rolling_var) if np.mean(rolling_var) > 0 else 1
            is_stationary = var_stability < 0.5
            
            return {
                'adf_statistic': -var_stability,
                'p_value': 0.01 if is_stationary else 0.1,
                'critical_values': {'1%': -3.43, '5%': -2.86, '10%': -2.57},
                'is_stationary': is_stationary,
                'conclusion': 'Stationary' if is_stationary else 'Non-Stationary'
            }
        
        result = adfuller(data)
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05,
            'conclusion': 'Stationary' if result[1] < 0.05 else 'Non-Stationary'
        }
    
    @staticmethod
    def autocorrelation_analysis(data: np.ndarray, max_lags: int = 40) -> Dict[str, np.ndarray]:
        """自相关分析"""
        # 确保max_lags不超过数据长度的一半
        max_lags = min(max_lags, len(data) // 2 - 1)
        
        if not STATSMODELS_AVAILABLE:
            # 简化的自相关计算
            acf_values = np.array([np.corrcoef(data[:-i], data[i:])[0,1] if i > 0 else 1.0 
                                  for i in range(max_lags + 1)])
            # 简化的偏自相关（使用递归公式近似）
            pacf_values = np.zeros(max_lags + 1)
            pacf_values[0] = 1.0
            if max_lags > 0:
                pacf_values[1] = acf_values[1]
                for k in range(2, max_lags + 1):
                    if k < len(acf_values):
                        pacf_values[k] = acf_values[k] - np.sum(pacf_values[1:k] * acf_values[k-1:0:-1])
        else:
            acf_values = acf(data, nlags=max_lags, fft=True)
            pacf_values = pacf(data, nlags=max_lags, method='ywm')
        
        # 处理NaN值
        acf_values = np.nan_to_num(acf_values, nan=0.0)
        pacf_values = np.nan_to_num(pacf_values, nan=0.0)
        
        # 计算有效记忆长度（ACF首次不显著的位置）
        # 显著性阈值约为 ±1.96/sqrt(n)
        significance_threshold = 1.96 / np.sqrt(len(data))
        significant_lags = np.where(np.abs(acf_values[1:]) > significance_threshold)[0]
        effective_memory_length = significant_lags[-1] + 1 if len(significant_lags) > 0 else 0
        
        return {
            'acf_values': acf_values,
            'pacf_values': pacf_values,
            'effective_memory_length': effective_memory_length,
            'significance_threshold': significance_threshold
        }
    
    @staticmethod
    def heteroscedasticity_test(residuals: np.ndarray, fitted_values: np.ndarray) -> Dict[str, float]:
        """异方差检验（简化版本）"""
        try:
            # 简化的异方差检验：比较不同区间的方差
            n = len(residuals)
            first_half = residuals[:n//2]
            second_half = residuals[n//2:]
            
            var1 = np.var(first_half)
            var2 = np.var(second_half)
            
            # F检验统计量
            f_stat = var1 / var2 if var2 > 0 else 1
            # 简化的p值估计
            p_value = 0.05 if abs(np.log(f_stat)) > 0.5 else 0.5
            
            return {
                'lm_statistic': f_stat,
                'lm_p_value': p_value,
                'f_statistic': f_stat,
                'f_p_value': p_value,
                'has_heteroscedasticity': p_value < 0.05
            }
        except:
            return {
                'lm_statistic': np.nan,
                'lm_p_value': np.nan,
                'f_statistic': np.nan,
                'f_p_value': np.nan,
                'has_heteroscedasticity': False
            }
    
    @staticmethod
    def change_point_detection(data: np.ndarray, window_size: int = 20) -> List[int]:
        """简单的变点检测"""
        change_points = []
        
        for i in range(window_size, len(data) - window_size):
            # 计算前后窗口的统计差异
            before_window = data[i-window_size:i]
            after_window = data[i:i+window_size]
            
            # 使用t检验检测均值变化
            _, p_value = stats.ttest_ind(before_window, after_window)
            
            # 如果显著不同，标记为变点
            if p_value < 0.01:  # 更严格的阈值
                change_points.append(i)
        
        # 移除过于接近的变点
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] > window_size:
                filtered_change_points.append(cp)
        
        return filtered_change_points


class ComplexityAnalyzer:
    """复杂度分析器"""
    
    @staticmethod
    def lempel_ziv_complexity(sequence: np.ndarray) -> float:
        """Lempel-Ziv复杂度"""
        # 将序列转换为二进制字符串
        binary_sequence = ''.join(['1' if x > np.median(sequence) else '0' for x in sequence])
        
        # 计算LZ复杂度
        i, k, l = 0, 1, 1
        k_max = 1
        n = len(binary_sequence)
        c = 1
        
        while k + l - 1 < n:
            if binary_sequence[i + l - 1] == binary_sequence[k + l - 1]:
                l += 1
            else:
                if l > k_max:
                    k_max = l
                i += 1
                if i == k:
                    c += 1
                    k += k_max
                    if k > n:
                        break
                    i = 0
                    k_max = 1
                l = 1
        
        if l > k_max:
            c += 1
        
        return c / (n / np.log2(n))  # 标准化复杂度
    
    @staticmethod
    def sample_entropy(data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """样本熵"""
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r * np.std(data):
                        C[i] += 1
            phi = np.mean(np.log(C / (N - m + 1.0)))
            return phi
        
        N = len(data)
        return _phi(m) - _phi(m + 1)
    
    @staticmethod
    def multiscale_entropy(data: np.ndarray, max_scale: int = 20) -> np.ndarray:
        """多尺度熵"""
        mse = np.zeros(max_scale)
        
        for scale in range(1, max_scale + 1):
            # 粗粒化
            coarse_grained = []
            for i in range(len(data) // scale):
                coarse_grained.append(np.mean(data[i*scale:(i+1)*scale]))
            
            coarse_grained = np.array(coarse_grained)
            
            # 计算样本熵
            if len(coarse_grained) > 10:  # 确保有足够的点
                mse[scale-1] = ComplexityAnalyzer.sample_entropy(coarse_grained)
            else:
                mse[scale-1] = np.nan
        
        return mse


class PredictabilityAnalyzer:
    """可预测性分析器"""
    
    @staticmethod
    def theoretical_predictability_bound(data: np.ndarray) -> Dict[str, float]:
        """理论可预测性上界"""
        # 基于信息熵的理论上界
        entropy = InformationTheoryAnalyzer.shannon_entropy(data)
        max_entropy = np.log2(len(np.unique(data)))  # 均匀分布的最大熵
        
        # 标准化可预测性
        predictability = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        # 基于自相关的可预测性
        acf_result = StatisticalAnalyzer.autocorrelation_analysis(data)
        acf_predictability = np.sum(np.abs(acf_result['acf_values'][1:10])) / 10  # 前10个滞后的平均ACF
        
        return {
            'entropy_based_predictability': predictability,
            'acf_based_predictability': acf_predictability,
            'combined_predictability': (predictability + acf_predictability) / 2
        }
    
    @staticmethod
    def forecast_horizon_analysis(data: np.ndarray, max_horizon: int = 20) -> Dict[str, np.ndarray]:
        """预测视野分析"""
        horizons = range(1, max_horizon + 1)
        mse_values = []
        
        # 使用简单的线性回归模型测试不同预测视野
        for h in horizons:
            X, y = [], []
            
            # 构造训练数据 - 适应短序列
            lag_length = min(3, len(data) // 3)  # 对于短序列使用更少的滞后
            for i in range(lag_length, len(data) - h):
                X.append(data[i-lag_length:i])
                y.append(data[i+h-1])
            
            if len(X) > 2:  # 降低最小训练数据要求
                X, y = np.array(X), np.array(y)
                
                # 训练测试分割
                split = len(X) // 2
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # 使用简单的线性回归（最小二乘法）
                try:
                    if SKLEARN_AVAILABLE:
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import mean_squared_error
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                    else:
                        # 手动实现简单线性回归
                        # 使用第一个滞后变量作为简单预测
                        y_pred = X_test[:, -1]  # 使用最后一个滞后值
                        mse = np.mean((y_test - y_pred) ** 2)
                    
                    mse_values.append(mse)
                except:
                    mse_values.append(np.nan)
            else:
                mse_values.append(np.nan)
        
        valid_mse = [mse for mse in mse_values if not np.isnan(mse)]
        effective_horizon = 1
        if valid_mse:
            min_idx = np.nanargmin(mse_values)
            effective_horizon = horizons[min_idx] if min_idx < len(horizons) else 1
        
        return {
            'horizons': np.array(horizons),
            'mse_values': np.array(mse_values),
            'effective_horizon': effective_horizon
        }


class TSVisualizationTools:
    """时序可视化工具"""
    
    @staticmethod
    def plot_comprehensive_analysis(data: np.ndarray, title: str, save_path: str) -> None:
        """综合分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 原始时序
        axes[0, 0].plot(data, 'b-', alpha=0.7)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].grid(True, alpha=0.3)
        
        # ACF图
        acf_result = StatisticalAnalyzer.autocorrelation_analysis(data)
        lags = range(len(acf_result['acf_values']))
        axes[0, 1].bar(lags, acf_result['acf_values'], alpha=0.7)
        axes[0, 1].axhline(y=acf_result['significance_threshold'], color='r', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=-acf_result['significance_threshold'], color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Autocorrelation Function (ACF)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # PACF图
        axes[0, 2].bar(lags, acf_result['pacf_values'], alpha=0.7, color='orange')
        axes[0, 2].axhline(y=acf_result['significance_threshold'], color='r', linestyle='--', alpha=0.7)
        axes[0, 2].axhline(y=-acf_result['significance_threshold'], color='r', linestyle='--', alpha=0.7)
        axes[0, 2].set_title('Partial Autocorrelation Function (PACF)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 分布直方图
        axes[1, 0].hist(data, bins=50, alpha=0.7, density=True)
        axes[1, 0].set_title('Distribution Histogram')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 差分序列
        diff_data = np.diff(data)
        axes[1, 1].plot(diff_data, 'g-', alpha=0.7)
        axes[1, 1].set_title('First Difference')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 多尺度熵
        mse = ComplexityAnalyzer.multiscale_entropy(data)
        valid_mse = mse[~np.isnan(mse)]
        valid_scales = np.arange(1, len(valid_mse) + 1)
        axes[1, 2].plot(valid_scales, valid_mse, 'ro-', alpha=0.7)
        axes[1, 2].set_title('Multiscale Entropy')
        axes[1, 2].set_xlabel('Scale')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_predictability_analysis(data: np.ndarray, title: str, save_path: str) -> None:
        """可预测性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 预测视野分析
        horizon_result = PredictabilityAnalyzer.forecast_horizon_analysis(data)
        valid_indices = ~np.isnan(horizon_result['mse_values'])
        valid_horizons = horizon_result['horizons'][valid_indices]
        valid_mse = horizon_result['mse_values'][valid_indices]
        
        axes[0, 0].plot(valid_horizons, valid_mse, 'bo-', alpha=0.7)
        axes[0, 0].set_title('Forecast Horizon vs MSE')
        axes[0, 0].set_xlabel('Forecast Horizon')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 滚动窗口统计
        window_size = 50
        rolling_mean = pd.Series(data).rolling(window_size).mean()
        rolling_std = pd.Series(data).rolling(window_size).std()
        
        axes[0, 1].plot(rolling_mean, 'b-', label='Rolling Mean', alpha=0.7)
        axes[0, 1].plot(rolling_std, 'r-', label='Rolling Std', alpha=0.7)
        axes[0, 1].set_title(f'Rolling Statistics (Window={window_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 变点检测
        change_points = StatisticalAnalyzer.change_point_detection(data)
        axes[1, 0].plot(data, 'b-', alpha=0.7, label='Time Series')
        for cp in change_points:
            axes[1, 0].axvline(x=cp, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_title(f'Change Point Detection ({len(change_points)} points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 信息内容分析
        if len(data) > 100:
            split_point = len(data) // 2
            historical = data[:split_point]
            future = data[split_point:]
            
            info_result = InformationTheoryAnalyzer.information_content_ratio(historical, future)
            
            labels = ['Historical\nEntropy', 'Future\nEntropy', 'Conditional\nEntropy', 'Information\nGain']
            values = [info_result['historical_entropy'], info_result['future_entropy'], 
                     info_result['conditional_entropy'], info_result['information_gain']]
            
            axes[1, 1].bar(labels, values, alpha=0.7, color=['blue', 'orange', 'green', 'red'])
            axes[1, 1].set_title('Information Content Analysis')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()