"""
合成数据生成工具类
Synthetic Data Generator for Sanity Check Experiments

该模块实现"黄金数据"与可控降级实验的数据生成功能，包括：
1. 完美时序数据生成
2. 基于时序的文本embedding映射
3. 多样化文本模板生成
4. 数据质量降级模拟

作者：Claude Code Assistant
日期：2025-01-XX
"""

import numpy as np
import pandas as pd
import json
import random
import logging
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime

# 设置随机种子确保可重现性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


class PatchTSTEncoder(nn.Module):
    """简化的PatchTST编码器用于时序特征提取"""
    
    def __init__(self, seq_len: int = 5, d_model: int = 768, num_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 将每个时序点投影到高维空间
        self.point_projection = nn.Linear(1, d_model // seq_len)
        self.sequence_projection = nn.Linear(seq_len * (d_model // seq_len), d_model)
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead=8, batch_first=True)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (seq_len,) 时序数据
        Returns:
            embedding: (d_model,) 编码后的特征
        """
        if x.dim() == 1:
            batch_size = 1
            x = x.unsqueeze(0)  # (1, seq_len)
        else:
            batch_size = x.size(0)
            
        # 将每个时间点投影到子空间
        # x: (batch_size, seq_len) -> (batch_size, seq_len, 1) -> (batch_size, seq_len, d_model//seq_len)
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.point_projection(x)  # (batch_size, seq_len, d_model//seq_len)
        
        # 展平并投影到最终维度
        x = x.view(batch_size, -1)  # (batch_size, seq_len * d_model//seq_len)
        x = self.sequence_projection(x)  # (batch_size, d_model)
        
        # 为Transformer添加序列维度
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Transformer编码
        for layer in self.encoder_layers:
            x = layer(x)
            
        # 移除序列维度
        x = x.squeeze(1)  # (batch_size, d_model)
        
        if batch_size == 1:
            x = x.squeeze(0)  # (d_model,)
            
        return x


class TextEmbeddingMapper(nn.Module):
    """文本嵌入映射器"""
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 时序特征到语义概念的映射
        self.trend_mapper = nn.Linear(d_model, 3)  # bullish, bearish, sideways
        self.volatility_mapper = nn.Linear(d_model, 3)  # high, moderate, low  
        self.momentum_mapper = nn.Linear(d_model, 3)  # accelerating, decelerating, stable
        
        # 最终文本嵌入生成
        self.text_projection = nn.Linear(d_model + 9, d_model)  # 原特征+语义特征
        
        # 文本模板
        self.templates = [
            "Market shows {trend} momentum with {volatility} volatility levels",
            "Trading volume indicates {trend} sentiment amid {volatility} market conditions", 
            "Technical indicators suggest {momentum} price action in current session",
            "Market analysis reveals {trend} outlook with {volatility} risk assessment",
            "Financial reports show {trend} performance with {momentum} growth trajectory",
            "Investment climate demonstrates {trend} patterns with {volatility} fluctuations",
            "Economic indicators point to {trend} direction amid {volatility} uncertainty",
            "Analyst consensus indicates {trend} sentiment with {momentum} price dynamics"
        ]
        
        self.trend_words = ['bullish', 'bearish', 'sideways']
        self.volatility_words = ['high', 'moderate', 'low']
        self.momentum_words = ['accelerating', 'decelerating', 'stable']
        
    def forward(self, ts_embedding):
        """
        将时序嵌入映射为文本嵌入和文本内容
        
        Args:
            ts_embedding: (d_model,) 时序嵌入向量
            
        Returns:
            text_embedding: (d_model,) 文本嵌入向量
            event_texts: List[str] 生成的事件文本列表
        """
        # 映射到语义概念
        trend_logits = self.trend_mapper(ts_embedding)
        volatility_logits = self.volatility_mapper(ts_embedding)
        momentum_logits = self.momentum_mapper(ts_embedding)
        
        # 软选择语义概念
        trend_probs = torch.softmax(trend_logits, dim=-1)
        volatility_probs = torch.softmax(volatility_logits, dim=-1)  
        momentum_probs = torch.softmax(momentum_logits, dim=-1)
        
        # 组合语义特征
        semantic_features = torch.cat([trend_probs, volatility_probs, momentum_probs])
        
        # 生成最终文本嵌入
        combined_features = torch.cat([ts_embedding, semantic_features])
        text_embedding = self.text_projection(combined_features)
        
        # 生成文本内容
        trend_idx = torch.argmax(trend_probs).item()
        volatility_idx = torch.argmax(volatility_probs).item()
        momentum_idx = torch.argmax(momentum_probs).item()
        
        trend_word = self.trend_words[trend_idx]
        volatility_word = self.volatility_words[volatility_idx]
        momentum_word = self.momentum_words[momentum_idx]
        
        # 生成多个事件文本
        event_texts = []
        num_events = random.randint(3, 6)  # 3-6个事件
        
        for i in range(num_events):
            template = random.choice(self.templates)
            event_text = template.format(
                trend=trend_word,
                volatility=volatility_word,
                momentum=momentum_word
            )
            # 添加一些变化和随机性
            event_text = self._add_variation(event_text, i)
            event_texts.append(event_text)
            
        return text_embedding, event_texts
    
    def _add_variation(self, text: str, event_idx: int) -> str:
        """为文本添加变化，增加多样性"""
        # 添加公司名称变化
        companies = ['Technology Corp', 'Financial Group', 'Investment Partners', 
                    'Capital Management', 'Trading Systems', 'Market Solutions']
        
        # 添加数值变化
        numbers = ['15%', '8.5%', '$2.1B', '250M', '1.8x', '45 basis points']
        
        # 根据事件索引添加不同的修饰
        if event_idx == 0:
            text = f"Major {text.lower()}"
        elif event_idx == 1:
            text = f"{text} according to {random.choice(companies)}"
        elif event_idx == 2:
            text = f"{text} with {random.choice(numbers)} impact"
        else:
            text = f"Latest reports suggest {text.lower()}"
            
        return text


class SyntheticDataGenerator:
    """合成数据生成主类"""
    
    def __init__(self, d_model: int = 768):
        """
        初始化合成数据生成器
        
        Args:
            d_model: 模型维度
        """
        self.d_model = d_model
        self.ts_encoder = PatchTSTEncoder(d_model=d_model)
        self.text_mapper = TextEmbeddingMapper(d_model=d_model)
        
        # 设置为评估模式确保一致性
        self.ts_encoder.eval()
        self.text_mapper.eval()
        
    def generate_time_series(self, t_start: float, t_end: float, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成完美的时序数据
        
        Args:
            t_start: 开始时间
            t_end: 结束时间  
            dt: 时间步长
            
        Returns:
            t_values: 时间点数组
            y_values: 时序值数组
        """
        t_values = np.arange(t_start, t_end, dt)
        
        # 使用改进的生成函数：y = sin(t) + 0.1*t + 少量噪声
        y_values = np.sin(t_values) + 0.1 * t_values + np.random.normal(0, 0.01, len(t_values))
        
        return t_values, y_values
    
    def create_sample(self, t_center: float) -> Dict[str, Any]:
        """
        创建单个样本
        
        Args:
            t_center: 中心时间点
            
        Returns:
            sample: 包含hist_data, event, ground_truth的样本字典
        """
        # 生成历史和未来数据
        t_hist_start = t_center - 4 * 0.1
        t_future_end = t_center + 5 * 0.1
        
        t_values, y_values = self.generate_time_series(t_hist_start, t_future_end, 0.1)
        
        # 分割历史和未来数据
        hist_data = y_values[:5]  # 前5个点作为历史
        ground_truth = y_values[5:10]  # 后5个点作为真值
        
        # 使用时序编码器生成嵌入
        with torch.no_grad():
            hist_tensor = torch.FloatTensor(hist_data)
            ts_embedding = self.ts_encoder(hist_tensor)
            
            # 生成文本嵌入和内容
            text_embedding, event_texts = self.text_mapper(ts_embedding)
        
        # 构造事件字典
        events = {}
        for i, text in enumerate(event_texts):
            events[f"event{i+1}"] = text
            
        # 格式化数据为字符串（匹配FNSPID格式）
        hist_data_str = ", ".join([f"{x:.4f}" for x in hist_data])
        ground_truth_str = ", ".join([f"{x:.4f}" for x in ground_truth])
        
        sample = {
            "hist_data": hist_data_str,
            "event": events,
            "ground_truth": ground_truth_str,
            "_metadata": {
                "t_center": t_center,
                "ts_embedding": ts_embedding.numpy().tolist(),
                "text_embedding": text_embedding.numpy().tolist()
            }
        }
        
        return sample
    
    def generate_dataset(self, num_samples: int, t_start: float = 0, t_step: float = 0.5) -> List[Dict[str, Any]]:
        """
        生成完整数据集
        
        Args:
            num_samples: 样本数量
            t_start: 起始时间
            t_step: 时间步进
            
        Returns:
            dataset: 样本列表
        """
        dataset = []
        
        logging.info(f"开始生成 {num_samples} 个样本的黄金数据集...")
        
        for i in range(num_samples):
            t_center = t_start + i * t_step
            sample = self.create_sample(t_center)
            dataset.append(sample)
            
            if (i + 1) % 1000 == 0:
                logging.info(f"已生成 {i + 1}/{num_samples} 个样本")
        
        logging.info(f"黄金数据集生成完成，共 {len(dataset)} 个样本")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], save_path: str):
        """
        保存数据集
        
        Args:
            dataset: 数据集
            save_path: 保存路径
        """
        # 移除metadata以匹配原始格式
        clean_dataset = []
        for sample in dataset:
            clean_sample = {
                "hist_data": sample["hist_data"],
                "event": sample["event"], 
                "ground_truth": sample["ground_truth"]
            }
            clean_dataset.append(clean_sample)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(clean_dataset, f, ensure_ascii=False, indent=2)
            
        logging.info(f"数据集已保存到: {save_path}")


class DataDegradationSimulator:
    """数据降级模拟器"""
    
    def __init__(self):
        """初始化数据降级模拟器"""
        self.irrelevant_texts = [
            "The weather forecast shows sunny conditions for the upcoming week",
            "New restaurant chain announces expansion to major metropolitan areas",
            "Latest blockbuster movie receives positive reviews from critics nationwide", 
            "Professional sports team secures victory in championship tournament",
            "Local university announces new research partnership with technology firms",
            "City council approves infrastructure development project for downtown area",
            "Fashion industry reports emerging trends for the upcoming season",
            "Travel sector shows increased bookings for international destinations",
            "Entertainment venue hosts successful cultural festival attracting thousands",
            "Educational institution launches innovative online learning platform"
        ]
        
        self.ambiguity_mappings = {
            "bullish momentum": "market shows movement",
            "bearish momentum": "market exhibits changes",
            "sideways momentum": "market demonstrates activity",
            "bullish sentiment": "positive outlook", 
            "bearish sentiment": "negative perspective",
            "high volatility": "market fluctuations",
            "moderate volatility": "price variations",
            "low volatility": "stable conditions",
            "accelerating": "changing",
            "decelerating": "adjusting", 
            "stable": "consistent",
            "Technical indicators": "Market metrics",
            "Market analysis": "Financial review",
            "Trading volume": "Market activity",
            "Investment climate": "Financial environment"
        }
    
    def version_b_irrelevant_noise(self, dataset: List[Dict], noise_ratio: float = 0.3) -> List[Dict]:
        """
        版本B: 注入无关文本噪声
        
        Args:
            dataset: 原始数据集
            noise_ratio: 噪声比例
            
        Returns:
            degraded_dataset: 降级后的数据集
        """
        degraded_dataset = []
        
        for sample in dataset:
            new_sample = sample.copy()
            events = new_sample["event"].copy()
            
            # 随机替换部分事件为无关文本
            for event_key in events:
                if random.random() < noise_ratio:
                    events[event_key] = random.choice(self.irrelevant_texts)
                    
            new_sample["event"] = events
            degraded_dataset.append(new_sample)
            
        logging.info(f"版本B降级完成，噪声比例: {noise_ratio}")
        return degraded_dataset
    
    def version_c_time_delay(self, dataset: List[Dict], delay_steps: int = 2) -> List[Dict]:
        """
        版本C: 引入时延文本
        
        Args:
            dataset: 原始数据集
            delay_steps: 延迟步数
            
        Returns:
            degraded_dataset: 降级后的数据集
        """
        degraded_dataset = []
        
        # 创建延迟事件池
        event_pool = []
        for sample in dataset:
            event_pool.append(sample["event"])
        
        for i, sample in enumerate(dataset):
            new_sample = sample.copy()
            
            # 使用延迟的事件
            if i >= delay_steps:
                new_sample["event"] = event_pool[i - delay_steps]
            else:
                # 对于前几个样本，使用重复或空事件
                new_sample["event"] = {
                    "event1": "Market shows standard trading patterns",
                    "event2": "Regular market activity observed",
                    "event3": "Typical market conditions prevail"
                }
                
            degraded_dataset.append(new_sample)
            
        logging.info(f"版本C降级完成，延迟步数: {delay_steps}")
        return degraded_dataset
    
    def version_d_ambiguity(self, dataset: List[Dict]) -> List[Dict]:
        """
        版本D: 信息模糊化
        
        Args:
            dataset: 原始数据集
            
        Returns:
            degraded_dataset: 降级后的数据集
        """
        degraded_dataset = []
        
        for sample in dataset:
            new_sample = sample.copy()
            events = new_sample["event"].copy()
            
            # 应用模糊化映射
            for event_key in events:
                text = events[event_key]
                for clear_text, ambiguous_text in self.ambiguity_mappings.items():
                    text = text.replace(clear_text, ambiguous_text)
                events[event_key] = text
                
            new_sample["event"] = events
            degraded_dataset.append(new_sample)
            
        logging.info("版本D降级完成，应用信息模糊化")
        return degraded_dataset


class SanityCheckDataManager:
    """完整的实验数据管理器"""
    
    def __init__(self, output_dir: str = "dataset/synthetic_sanity_check"):
        """
        初始化数据管理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generator = SyntheticDataGenerator()
        self.degradator = DataDegradationSimulator()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        
    def generate_all_versions(self, train_size: int = 6842, test_size: int = 1568, vali_size: int = 1424):
        """
        生成所有版本的数据集
        
        Args:
            train_size: 训练集大小
            test_size: 测试集大小
            vali_size: 验证集大小
        """
        logging.info("开始生成完整的实验数据集...")
        
        # 1. 生成版本A (黄金数据)
        logging.info("生成版本A: 黄金数据...")
        train_golden = self.generator.generate_dataset(train_size, t_start=0, t_step=0.5)
        test_golden = self.generator.generate_dataset(test_size, t_start=train_size*0.5, t_step=0.5)  
        vali_golden = self.generator.generate_dataset(vali_size, t_start=(train_size+test_size)*0.5, t_step=0.5)
        
        # 保存版本A
        version_a_dir = self.output_dir / "version_a_golden"
        version_a_dir.mkdir(exist_ok=True)
        
        self.generator.save_dataset(train_golden, version_a_dir / "train.json")
        self.generator.save_dataset(test_golden, version_a_dir / "test.json")
        self.generator.save_dataset(vali_golden, version_a_dir / "vali.json")
        
        # 2. 生成版本B (无关文本噪声)
        logging.info("生成版本B: 无关文本噪声...")
        version_b_dir = self.output_dir / "version_b_noise"
        version_b_dir.mkdir(exist_ok=True)
        
        train_noise = self.degradator.version_b_irrelevant_noise(train_golden)
        test_noise = self.degradator.version_b_irrelevant_noise(test_golden)
        vali_noise = self.degradator.version_b_irrelevant_noise(vali_golden)
        
        self.generator.save_dataset(train_noise, version_b_dir / "train.json")
        self.generator.save_dataset(test_noise, version_b_dir / "test.json") 
        self.generator.save_dataset(vali_noise, version_b_dir / "vali.json")
        
        # 3. 生成版本C (时延文本)
        logging.info("生成版本C: 时延文本...")
        version_c_dir = self.output_dir / "version_c_delay"
        version_c_dir.mkdir(exist_ok=True)
        
        train_delay = self.degradator.version_c_time_delay(train_golden)
        test_delay = self.degradator.version_c_time_delay(test_golden)
        vali_delay = self.degradator.version_c_time_delay(vali_golden)
        
        self.generator.save_dataset(train_delay, version_c_dir / "train.json")
        self.generator.save_dataset(test_delay, version_c_dir / "test.json")
        self.generator.save_dataset(vali_delay, version_c_dir / "vali.json")
        
        # 4. 生成版本D (信息模糊化)
        logging.info("生成版本D: 信息模糊化...")
        version_d_dir = self.output_dir / "version_d_ambiguous"
        version_d_dir.mkdir(exist_ok=True)
        
        train_ambig = self.degradator.version_d_ambiguity(train_golden)
        test_ambig = self.degradator.version_d_ambiguity(test_golden)
        vali_ambig = self.degradator.version_d_ambiguity(vali_golden)
        
        self.generator.save_dataset(train_ambig, version_d_dir / "train.json")
        self.generator.save_dataset(test_ambig, version_d_dir / "test.json")
        self.generator.save_dataset(vali_ambig, version_d_dir / "vali.json")
        
        # 5. 生成数据集统计报告
        self._generate_statistics_report()
        
        logging.info("所有版本数据集生成完成！")
    
    def _generate_statistics_report(self):
        """生成数据集统计报告"""
        report = {
            "generation_time": datetime.now().isoformat(),
            "random_seed": RANDOM_SEED,
            "versions": {
                "version_a": "Golden data with perfect time-series to text correlation",
                "version_b": "30% irrelevant text noise injection", 
                "version_c": "2-step time delayed text information",
                "version_d": "Information ambiguity through vague language"
            },
            "data_sizes": {
                "train": 6842,
                "test": 1568, 
                "validation": 1424
            },
            "generation_parameters": {
                "time_series_function": "y = sin(t) + 0.1*t + noise(σ=0.01)",
                "embedding_dimension": 768,
                "events_per_sample": "3-6 (random)",
                "text_templates": len(self.generator.text_mapper.templates)
            }
        }
        
        report_path = self.output_dir / "dataset_statistics.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        logging.info(f"统计报告已保存到: {report_path}")


def main():
    """主函数：生成实验数据集"""
    # 创建数据管理器
    manager = SanityCheckDataManager()
    
    # 生成所有版本的数据集
    manager.generate_all_versions()
    
    print("✅ 合成数据集生成完成!")
    print(f"📁 输出目录: {manager.output_dir}")
    print("📊 包含版本:")
    print("   - 版本A: 黄金数据 (完美关联)")
    print("   - 版本B: 30%无关文本噪声")
    print("   - 版本C: 2步时延文本") 
    print("   - 版本D: 信息模糊化")


if __name__ == "__main__":
    main()