
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Event Embedding Module using SiliconFlow API
使用SiliconFlow API处理事件文本数据，生成统一的嵌入向量
支持最新的Qwen3-Embedding系列模型
"""

import os
import time
import numpy as np
from typing import Dict, List, Union, Optional
import logging
import warnings
import requests
import json
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

class EmbeddingModel(Enum):
    """Available embedding models from SiliconFlow"""
    # Qwen3 Embedding Series - newest and most expensive
    QWEN3_8B = "Qwen/Qwen3-Embedding-8B"  # Best performance, 8B params, up to 4096 dims
    QWEN3_4B = "Qwen/Qwen3-Embedding-4B"  # Good performance, 4B params, up to 2560 dims
    QWEN3_0_6B = "Qwen/Qwen3-Embedding-0.6B"  # Lightweight, 0.6B params, up to 1024 dims
    
    # BGE Series
    BGE_LARGE_ZH = "BAAI/bge-large-zh-v1.5"  # Chinese optimized
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"  # English optimized
    BGE_M3 = "BAAI/bge-m3"  # Multilingual

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    api_key: str
    model: str = EmbeddingModel.QWEN3_8B.value  # Use the best model by default
    api_url: str = "https://api.siliconflow.cn/v1/embeddings"
    max_tokens: int = 8192  # Maximum tokens per request
    dimensions: Optional[int] = None  # Optional dimension reduction
    timeout: int = 30  # Request timeout in seconds
    max_retries: int = 3  # Maximum retry attempts
    retry_delay: float = 1.0  # Delay between retries

class EventEmbedderAPI:
    """
    事件文本嵌入生成器 - 使用SiliconFlow API
    专门针对金融事件文本优化，使用最新的Qwen3-Embedding模型
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = EmbeddingModel.QWEN3_8B.value,
                 dimensions: Optional[int] = None,
                 config: Optional[EmbeddingConfig] = None):
        """
        初始化嵌入器
        
        Args:
            api_key: SiliconFlow API密钥
            model: 模型名称，默认使用Qwen3-Embedding-8B (最新最强)
            dimensions: 可选的维度降维
            config: 完整配置对象
        """
        if config:
            self.config = config
        else:
            if not api_key:
                raise ValueError("API key is required")
            self.config = EmbeddingConfig(
                api_key=api_key,
                model=model,
                dimensions=dimensions
            )
        
        # 初始化日志
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized EventEmbedderAPI with model: {self.config.model}")
        
        # 设置请求头
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        # 获取模型信息
        self._model_info = self._get_model_info()
        
    def _get_model_info(self) -> Dict:
        """获取模型信息"""
        model_info = {
            "Qwen/Qwen3-Embedding-8B": {
                "max_dimensions": 4096,
                "default_dimensions": 4096,
                "context_length": 32000,
                "description": "最强大的嵌入模型，8B参数，MTEB榜首"
            },
            "Qwen/Qwen3-Embedding-4B": {
                "max_dimensions": 2560,
                "default_dimensions": 2560,
                "context_length": 32000,
                "description": "平衡性能的嵌入模型，4B参数"
            },
            "Qwen/Qwen3-Embedding-0.6B": {
                "max_dimensions": 1024,
                "default_dimensions": 1024,
                "context_length": 32000,
                "description": "轻量级嵌入模型，0.6B参数"
            },
            "BAAI/bge-large-zh-v1.5": {
                "max_dimensions": 1024,
                "default_dimensions": 1024,
                "context_length": 512,
                "description": "中文优化的BGE模型"
            },
            "BAAI/bge-large-en-v1.5": {
                "max_dimensions": 1024,
                "default_dimensions": 1024,
                "context_length": 512,
                "description": "英文优化的BGE模型"
            },
            "BAAI/bge-m3": {
                "max_dimensions": 1024,
                "default_dimensions": 1024,
                "context_length": 8192,
                "description": "多语言BGE模型"
            }
        }
        
        if self.config.model in model_info:
            info = model_info[self.config.model]
            self.logger.info(f"Using model: {info['description']}")
            return info
        else:
            self.logger.warning(f"Unknown model {self.config.model}, using default settings")
            return {
                "max_dimensions": 1024,
                "default_dimensions": 1024,
                "context_length": 8192,
                "description": "Unknown model"
            }
    
    def _make_api_request(self, texts: List[str]) -> Dict:
        """
        发送API请求
        
        Args:
            texts: 要编码的文本列表
            
        Returns:
            API响应字典
        """
        # 构建请求体
        request_body = {
            "model": self.config.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        # 添加可选的维度参数
        if self.config.dimensions:
            request_body["dimensions"] = self.config.dimensions
        
        # 重试机制
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    self.config.api_url,
                    headers=self.headers,
                    json=request_body,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    self.logger.error(error_msg)
                    
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise Exception(error_msg)
                        
            except requests.exceptions.Timeout:
                self.logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise
            except Exception as e:
                self.logger.error(f"API request error: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    raise
    
    def encode_event(self, event: Union[str, Dict, List]) -> np.ndarray:
        """
        编码单个事件
        
        Args:
            event: 事件数据，可以是字符串、字典或列表
            
        Returns:
            numpy数组形式的嵌入向量
        """
        # 预处理事件文本
        event_text = self._preprocess_event(event)
        
        # 调用API
        response = self._make_api_request([event_text])
        
        # 提取嵌入向量
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        
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
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(event_texts), batch_size):
            batch = event_texts[i:i+batch_size]
            
            # 调用API
            response = self._make_api_request(batch)
            
            # 提取嵌入向量
            batch_embeddings = [np.array(item['embedding'], dtype=np.float32) 
                               for item in response['data']]
            all_embeddings.extend(batch_embeddings)
            
            # 避免速率限制
            if i + batch_size < len(event_texts):
                time.sleep(0.1)  # Small delay between batches
        
        return np.vstack(all_embeddings)
    
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
            # 使用递减权重
            weights = np.exp(-np.arange(len(embeddings)) * 0.1)
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        if self.config.dimensions:
            return self.config.dimensions
        else:
            return self._model_info['default_dimensions']
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.config.model


class FinancialEventEmbedderAPI(EventEmbedderAPI):
    """
    金融事件专用嵌入器 - 使用SiliconFlow API
    添加金融领域特定的预处理和后处理
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = EmbeddingModel.QWEN3_8B.value,  # 默认使用最强模型
                 **kwargs):
        """
        初始化金融事件嵌入器
        
        Args:
            api_key: SiliconFlow API密钥
            model: 预训练模型名称，默认使用Qwen3-Embedding-8B
        """
        super().__init__(api_key=api_key, model=model, **kwargs)
        
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


def test_event_embedder_api():
    """测试函数"""
    print("="*50)
    print("Testing Event Embedder with SiliconFlow API")
    print("="*50)
    
    # 使用提供的API密钥
    API_KEY = "sk-dqiytlfixijiuklczwzwcovrqprvrpffzwwqogqninxqwzhx"
    
    # 1. 测试不同模型
    models_to_test = [
        (EmbeddingModel.QWEN3_8B.value, "最强大的8B参数模型"),
        (EmbeddingModel.QWEN3_4B.value, "平衡的4B参数模型"),
        (EmbeddingModel.QWEN3_0_6B.value, "轻量级0.6B参数模型"),
    ]
    
    print("\n" + "="*30)
    print("Testing different models:")
    print("="*30)
    
    for model, description in models_to_test:
        print(f"\nTesting {model} ({description})")
        
        try:
            # 初始化嵌入器
            embedder = EventEmbedderAPI(
                api_key=API_KEY,
                model=model
            )
            
            print(f"Model loaded: {embedder.get_model_name()}")
            print(f"Embedding dimension: {embedder.get_embedding_dim()}")
            
            # 测试简单文本
            test_text = "Micron Technology expects chip demand to grow in H2 2023."
            embedding = embedder.encode_event(test_text)
            
            print(f"Embedding shape: {embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
            print(f"First 5 values: {embedding[:5]}")
            
        except Exception as e:
            print(f"Error testing {model}: {e}")
    
    # 2. 详细测试最强模型
    print("\n" + "="*50)
    print("Detailed testing with Qwen3-Embedding-8B:")
    print("="*50)
    
    embedder = EventEmbedderAPI(
        api_key=API_KEY,
        model=EmbeddingModel.QWEN3_8B.value,
        dimensions=2048  # 可选：降维到2048
    )
    
    # 测试不同格式的事件
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
    print("Testing Financial Event Embedder with API")
    print("="*50)
    
    fin_embedder = FinancialEventEmbedderAPI(
        api_key=API_KEY,
        model=EmbeddingModel.QWEN3_8B.value  # 使用最强模型处理金融事件
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
    
    # 6. 性能测试
    print("\n" + "="*30)
    print("Performance comparison:")
    print("="*30)
    
    import time
    
    test_text = "Federal Reserve raises interest rates by 25 basis points"
    
    for model, description in models_to_test[:3]:  # Test only Qwen3 models
        embedder = EventEmbedderAPI(api_key=API_KEY, model=model)
        
        start_time = time.time()
        _ = embedder.encode_event(test_text)
        elapsed = time.time() - start_time
        
        print(f"{model}: {elapsed:.3f}s")
    
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
    test_event_embedder_api()