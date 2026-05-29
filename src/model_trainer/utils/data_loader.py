"""
Data Loading Utilities
======================

用于加载和预处理FNSPID数据集的工具模块
符合软件工程规范，提供统一的数据接口
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FNSPIDDataLoader:
    """FNSPID数据集加载器"""
    
    def __init__(self, data_dir: str = "dataset/FNSPID"):
        """
        初始化数据加载器
        
        Args:
            data_dir: 数据集目录路径
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist")
    
    def load_dataset(self, dataset_name: str, use_final_format: bool = False) -> List[Dict]:
        """
        加载指定数据集
        
        Args:
            dataset_name: 数据集名称 ('train', 'test', 'vali')
            use_final_format: 是否使用final格式的数据
            
        Returns:
            加载的数据列表
        """
        suffix = "_final.json" if use_final_format else ".json"
        file_path = self.data_dir / f"{dataset_name}{suffix}"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file {file_path} not found")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} samples from {file_path}")
            self.datasets[dataset_name] = data
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {file_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {file_path}: {e}")
    
    def load_all_datasets(self, use_final_format: bool = False) -> Dict[str, List[Dict]]:
        """
        加载所有数据集
        
        Args:
            use_final_format: 是否使用final格式的数据
            
        Returns:
            包含所有数据集的字典
        """
        dataset_names = ['train', 'test', 'vali']
        all_datasets = {}
        
        for name in dataset_names:
            try:
                all_datasets[name] = self.load_dataset(name, use_final_format)
            except FileNotFoundError:
                logger.warning(f"Dataset {name} not found, skipping...")
                continue
        
        return all_datasets
    
    def extract_time_series(self, data: List[Dict], 
                           historical_only: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        从数据中提取时序数据
        
        Args:
            data: 原始数据列表
            historical_only: 是否只返回历史数据
            
        Returns:
            (历史数据列表, 真实值列表) 如果historical_only=True，则真实值列表为空
        """
        historical_series = []
        ground_truth_series = []
        
        for item in data:
            try:
                # 解析历史数据
                if 'hist_data' in item:
                    hist_str = item['hist_data']
                elif 'historical_data' in item:
                    hist_str = item['historical_data']
                else:
                    logger.warning("No historical data field found in item, skipping...")
                    continue
                
                # 转换为数值数组
                historical = np.array([float(x.strip()) for x in hist_str.split(',')])
                historical_series.append(historical)
                
                # 解析真实值数据（如果需要）
                if not historical_only:
                    ground_truth_str = item['ground_truth']
                    ground_truth = np.array([float(x.strip()) for x in ground_truth_str.split(',')])
                    ground_truth_series.append(ground_truth)
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error parsing time series data: {e}, skipping item...")
                continue
        
        logger.info(f"Extracted {len(historical_series)} time series")
        return historical_series, ground_truth_series
    
    def combine_historical_and_future(self, data: List[Dict]) -> List[np.ndarray]:
        """
        将历史数据和未来数据合并为完整时序
        
        Args:
            data: 原始数据列表
            
        Returns:
            完整时序数据列表
        """
        complete_series = []
        
        historical_series, ground_truth_series = self.extract_time_series(data, historical_only=False)
        
        for hist, gt in zip(historical_series, ground_truth_series):
            complete = np.concatenate([hist, gt])
            complete_series.append(complete)
        
        logger.info(f"Combined {len(complete_series)} complete time series")
        return complete_series
    
    def get_dataset_statistics(self, dataset_name: str = None) -> Dict[str, any]:
        """
        获取数据集统计信息
        
        Args:
            dataset_name: 数据集名称，如果为None则统计所有数据集
            
        Returns:
            统计信息字典
        """
        if dataset_name is None:
            # 统计所有数据集
            all_stats = {}
            for name in self.datasets.keys():
                all_stats[name] = self._calculate_single_dataset_stats(self.datasets[name])
            return all_stats
        else:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset {dataset_name} not loaded")
            return self._calculate_single_dataset_stats(self.datasets[dataset_name])
    
    def _calculate_single_dataset_stats(self, data: List[Dict]) -> Dict[str, any]:
        """计算单个数据集的统计信息"""
        historical_series, ground_truth_series = self.extract_time_series(data)
        
        # 基本统计
        num_samples = len(historical_series)
        
        if num_samples == 0:
            return {"num_samples": 0, "error": "No valid samples found"}
        
        # 历史数据统计
        hist_lengths = [len(series) for series in historical_series]
        hist_values = np.concatenate(historical_series)
        
        # 未来数据统计
        future_lengths = [len(series) for series in ground_truth_series] if ground_truth_series else []
        future_values = np.concatenate(ground_truth_series) if ground_truth_series else np.array([])
        
        stats = {
            "num_samples": num_samples,
            "historical_data": {
                "length_stats": {
                    "mean": np.mean(hist_lengths),
                    "std": np.std(hist_lengths),
                    "min": np.min(hist_lengths),
                    "max": np.max(hist_lengths)
                },
                "value_stats": {
                    "mean": np.mean(hist_values),
                    "std": np.std(hist_values),
                    "min": np.min(hist_values),
                    "max": np.max(hist_values),
                    "median": np.median(hist_values)
                }
            }
        }
        
        if len(future_values) > 0:
            stats["future_data"] = {
                "length_stats": {
                    "mean": np.mean(future_lengths),
                    "std": np.std(future_lengths),
                    "min": np.min(future_lengths),
                    "max": np.max(future_lengths)
                },
                "value_stats": {
                    "mean": np.mean(future_values),
                    "std": np.std(future_values),
                    "min": np.min(future_values),
                    "max": np.max(future_values),
                    "median": np.median(future_values)
                }
            }
        
        return stats
    
    def sample_data(self, dataset_name: str, n_samples: int, 
                   random_state: int = 42) -> List[Dict]:
        """
        从数据集中随机采样
        
        Args:
            dataset_name: 数据集名称
            n_samples: 采样数量
            random_state: 随机种子
            
        Returns:
            采样后的数据列表
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not loaded")
        
        data = self.datasets[dataset_name]
        
        if n_samples >= len(data):
            return data
        
        np.random.seed(random_state)
        indices = np.random.choice(len(data), n_samples, replace=False)
        
        sampled_data = [data[i] for i in indices]
        logger.info(f"Sampled {n_samples} items from {dataset_name} dataset")
        
        return sampled_data


class TimeSeriesPreprocessor:
    """时序数据预处理器"""
    
    @staticmethod
    def remove_outliers(data: np.ndarray, method: str = 'iqr', 
                       factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        移除异常值
        
        Args:
            data: 输入数据
            method: 异常值检测方法 ('iqr', 'zscore')
            factor: 异常值阈值因子
            
        Returns:
            (处理后数据, 异常值掩码)
        """
        if method == 'iqr':
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            outlier_mask = z_scores > factor
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        # 使用线性插值填充异常值
        clean_data = data.copy()
        if np.any(outlier_mask):
            clean_data[outlier_mask] = np.interp(
                np.where(outlier_mask)[0],
                np.where(~outlier_mask)[0],
                data[~outlier_mask]
            )
        
        return clean_data, outlier_mask
    
    @staticmethod
    def normalize(data: np.ndarray, method: str = 'zscore') -> Tuple[np.ndarray, Dict[str, float]]:
        """
        数据标准化
        
        Args:
            data: 输入数据
            method: 标准化方法 ('zscore', 'minmax', 'robust')
            
        Returns:
            (标准化后数据, 标准化参数)
        """
        if method == 'zscore':
            mean, std = np.mean(data), np.std(data)
            normalized = (data - mean) / (std + 1e-8)
            params = {'mean': mean, 'std': std}
        elif method == 'minmax':
            min_val, max_val = np.min(data), np.max(data)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            params = {'min': min_val, 'max': max_val}
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            normalized = (data - median) / (mad + 1e-8)
            params = {'median': median, 'mad': mad}
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized, params
    
    @staticmethod
    def handle_missing_values(data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        处理缺失值
        
        Args:
            data: 输入数据
            method: 填充方法 ('linear', 'forward', 'backward', 'mean')
            
        Returns:
            处理后的数据
        """
        if not np.any(np.isnan(data)):
            return data
        
        if method == 'linear':
            # 线性插值
            valid_indices = ~np.isnan(data)
            if np.any(valid_indices):
                data_filled = np.interp(
                    np.arange(len(data)),
                    np.where(valid_indices)[0],
                    data[valid_indices]
                )
            else:
                data_filled = np.zeros_like(data)
        elif method == 'forward':
            data_filled = pd.Series(data).fillna(method='ffill').values
        elif method == 'backward':
            data_filled = pd.Series(data).fillna(method='bfill').values
        elif method == 'mean':
            mean_value = np.nanmean(data)
            data_filled = np.where(np.isnan(data), mean_value, data)
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
        
        return data_filled