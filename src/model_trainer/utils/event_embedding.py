#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Event Embedding Module
处理事件文本数据，生成统一的嵌入向量
支持多种embedding模型
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Union, Optional
from sentence_transformers import SentenceTransformer
import logging
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

class EventEmbedder:
    """
    事件文本嵌入生成器
    支持多种预训练模型，专门针对金融事件文本优化
    """
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 device: str = None,
                 max_length: int = 10000,
                 use_mean_pooling: bool = True):
        """
        初始化嵌入器
        
        Args:
            model_name: 预训练模型名称
            device: 计算设备
            max_length: 最大序列长度
            use_mean_pooling: 是否使用mean pooling
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.use_mean_pooling = use_mean_pooling
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 优先使用sentence-transformers
            if 'sentence-transformers' in self.model_name or 'all-mpnet' in self.model_name:
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                self.model_type = 'sentence_transformer'
                self.logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            else:
                # 使用HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                self.model_type = 'huggingface'
                self.logger.info(f"Loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean Pooling - 考虑attention mask的平均池化"""
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def encode_event(self, event: Union[str, Dict, List]) -> np.ndarray:
        """
        编码单个事件
        
        Args:
            event: 事件数据，可以是字符串、字典或列表
            
        Returns:
            numpy数组形式的嵌入向量
        """
        # 1. 预处理事件文本
        event_text = self._preprocess_event(event)
        
        # 2. 生成嵌入
        if self.model_type == 'sentence_transformer':
            embedding = self.model.encode(event_text, convert_to_numpy=True)
        else:
            embedding = self._encode_with_transformers(event_text)
        
        return embedding
    
    def encode_events_batch(self, events: List[Union[str, Dict]], 
                           batch_size: int = 32) -> np.ndarray:
        """
        批量编码事件
        
        Args:
            events: 事件列表
            batch_size: 批处理大小
            
        Returns:
            嵌入矩阵 [num_events, embedding_dim]
        """
        # 预处理所有事件
        event_texts = [self._preprocess_event(event) for event in events]
        
        if self.model_type == 'sentence_transformer':
            # SentenceTransformer自带批处理
            embeddings = self.model.encode(
                event_texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
        else:
            # 手动批处理
            embeddings = []
            for i in range(0, len(event_texts), batch_size):
                batch = event_texts[i:i+batch_size]
                batch_embeddings = self._encode_with_transformers(batch)
                embeddings.append(batch_embeddings)
            embeddings = np.vstack(embeddings)
        
        return embeddings
    
    def _encode_with_transformers(self, texts: Union[str, List[str]]) -> np.ndarray:
        """使用transformers库编码文本"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Pooling
        if self.use_mean_pooling:
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        else:
            # Use CLS token
            embeddings = model_output.last_hidden_state[:, 0, :]
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def _preprocess_event(self, event: Union[str, Dict, List]) -> str:
        """
        预处理事件数据，转换为文本
        
        Args:
            event: 原始事件数据
            
        Returns:
            处理后的文本字符串
        """
        if isinstance(event, str):
            return event
        
        elif isinstance(event, dict):
            # 处理字典格式的事件
            if 'event' in event:
                # 处理嵌套的event字典
                sub_events = event['event']
                if isinstance(sub_events, dict):
                    # 合并所有子事件
                    texts = []
                    for key, value in sub_events.items():
                        if value:  # 跳过空值
                            texts.append(f"{value}")
                    return " [SEP] ".join(texts)
            
            # 处理普通字典
            text_parts = []
            for key, value in event.items():
                if value and isinstance(value, str):
                    text_parts.append(value)
            return " ".join(text_parts)
        
        elif isinstance(event, list):
            # 处理列表格式的事件
            return " [SEP] ".join([str(item) for item in event if item])
        
        else:
            return str(event)
    
    def aggregate_event_embeddings(self, 
                                  embeddings: np.ndarray,
                                  method: str = 'mean') -> np.ndarray:
        """
        聚合多个事件的嵌入
        
        Args:
            embeddings: 嵌入矩阵 [num_events, embedding_dim]
            method: 聚合方法 ('mean', 'max', 'weighted')
            
        Returns:
            聚合后的嵌入向量 [embedding_dim]
        """
        if method == 'mean':
            return np.mean(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        elif method == 'weighted':
            # 可以根据事件重要性加权
            # 这里简单使用递减权重
            weights = np.exp(-np.arange(len(embeddings)) * 0.1)
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.embedding_dim


class FinancialEventEmbedder(EventEmbedder):
    """
    金融事件专用嵌入器
    添加金融领域特定的预处理和后处理
    """
    
    def __init__(self, 
                 model_name: str = 'ProsusAI/finbert',
                 **kwargs):
        """
        初始化金融事件嵌入器
        
        Args:
            model_name: 预训练模型名称，默认使用FinBERT
        """
        super().__init__(model_name=model_name, **kwargs)
        
        # 金融关键词
        self.financial_keywords = {
            'positive': ['profit', 'growth', 'increase', 'gain', 'rise', 'up', 'buy', 'outperform'],
            'negative': ['loss', 'decline', 'decrease', 'fall', 'down', 'sell', 'underperform'],
            'neutral': ['maintain', 'stable', 'unchanged', 'flat']
        }
    
    def _preprocess_event(self, event: Union[str, Dict, List]) -> str:
        """
        金融事件预处理
        增加金融领域特定的处理逻辑
        """
        # 基础预处理
        text = super()._preprocess_event(event)
        
        # 添加金融上下文提示
        prompt = "Financial market event: "
        
        # 强调数字和百分比
        import re
        text = re.sub(r'(\d+\.?\d*%)', r'[PERCENT:\1]', text)
        text = re.sub(r'\$(\d+\.?\d*[BMK]?)', r'[DOLLAR:\1]', text)
        
        return prompt + text


def test_event_embedder():
    """测试函数"""
    print("="*50)
    print("Testing Event Embedder")
    print("="*50)
    
    # 1. 初始化嵌入器
    embedder = EventEmbedder(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # 使用小模型测试
        use_mean_pooling=True
    )
    
    print(f"Model loaded: {embedder.model_name}")
    print(f"Embedding dimension: {embedder.get_embedding_dim()}")
    
    # 2. 测试不同格式的事件
    test_events = [
        # 字符串格式
        "Micron Technology expects chip demand to grow in H2 2023.",
        
        # 字典格式
        {
            "event1": "China restricts use of foreign devices including Micron products.",
            "event2": "Micron experiences worst memory downturn since 2008."
        },
        
        # 嵌套字典格式
        {
            "event": {
                "event1": "Citigroup maintains Buy recommendation on Micron Technology.",
                "event2": "Micron must write down inventory due to falling chip prices.",
                "event3": "Analysts expect recovery in semiconductor market.",
            }
        },
        
        # 列表格式
        [
            "Tech stocks rally on AI optimism",
            "Federal Reserve hints at rate pause",
            "Unemployment rate remains stable"
        ]
    ]
    
    print("\n" + "="*30)
    print("Testing individual encoding:")
    print("="*30)
    
    for i, event in enumerate(test_events):
        print(f"\nEvent {i+1}:")
        print(f"Type: {type(event).__name__}")
        
        # 生成嵌入
        embedding = embedder.encode_event(event)
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"First 5 values: {embedding[:5]}")
    
    # 3. 测试批处理
    print("\n" + "="*30)
    print("Testing batch encoding:")
    print("="*30)
    
    batch_embeddings = embedder.encode_events_batch(test_events, batch_size=2)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # 4. 测试聚合
    print("\n" + "="*30)
    print("Testing aggregation:")
    print("="*30)
    
    for method in ['mean', 'max', 'weighted']:
        aggregated = embedder.aggregate_event_embeddings(batch_embeddings, method=method)
        print(f"{method} aggregation shape: {aggregated.shape}")
        print(f"{method} aggregation norm: {np.linalg.norm(aggregated):.4f}")
    
    # 5. 测试金融事件嵌入器
    print("\n" + "="*50)
    print("Testing Financial Event Embedder")
    print("="*50)
    
    fin_embedder = FinancialEventEmbedder(
        model_name='sentence-transformers/all-MiniLM-L6-v2'  # 使用小模型测试
    )
    
    financial_event = {
        "event": {
            "event1": "Company reports $2.5B profit, up 15% YoY",
            "event2": "Stock price increased by 8.5% following earnings",
            "event3": "CEO announces $500M share buyback program"
        }
    }
    
    fin_embedding = fin_embedder.encode_event(financial_event)
    print(f"Financial embedding shape: {fin_embedding.shape}")
    print(f"Financial embedding norm: {np.linalg.norm(fin_embedding):.4f}")
    
    print("\n" + "="*50)
    print("All tests passed successfully!")
    print("="*50)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    test_event_embedder()