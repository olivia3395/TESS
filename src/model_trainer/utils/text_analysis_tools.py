"""
Text Analysis Tools for FNSPID Dataset
========================================

专用于文本模态分析的工具集，包含信息量分析、语义分析、情感分析等功能
符合软件工程规范的模块化设计
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import re
import gzip
import math
import warnings
import logging
from pathlib import Path

# 设置matplotlib
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, some features will be limited")

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    # 尝试下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except:
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using basic tokenization")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("VADER sentiment analyzer not available")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available, BERT features will be limited")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence-transformers not available")

warnings.filterwarnings('ignore')


class TextStatisticsAnalyzer:
    """文本基础统计分析器"""
    
    def __init__(self):
        """初始化分析器"""
        if NLTK_AVAILABLE:
            self.stop_words = set(stopwords.words('english'))
        else:
            # 基础停用词列表
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                                 'at', 'to', 'for', 'of', 'with', 'by', 'from', 
                                 'is', 'was', 'are', 'were', 'been', 'be'])
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        if NLTK_AVAILABLE:
            return word_tokenize(text.lower())
        else:
            # 基础分词
            return re.findall(r'\b\w+\b', text.lower())
    
    def analyze_event_statistics(self, events_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析事件基本统计信息"""
        results = {
            'event_count_stats': {},
            'event_length_stats': {},
            'vocabulary_stats': {},
            'detailed_distributions': {}
        }
        
        # 收集所有事件
        event_counts = []
        event_lengths = []
        all_words = []
        unique_words_per_sample = []
        
        for events in events_list:
            # 事件数量
            num_events = len(events)
            event_counts.append(num_events)
            
            # 每个事件的长度和词汇
            sample_words = []
            for event_text in events.values():
                words = self.tokenize(event_text)
                event_lengths.append(len(words))
                all_words.extend(words)
                sample_words.extend(words)
            
            # 每个样本的唯一词汇
            if sample_words:
                unique_words_per_sample.append(len(set(sample_words)))
        
        # 事件数量统计
        if event_counts:
            results['event_count_stats'] = {
                'mean': np.mean(event_counts),
                'median': np.median(event_counts),
                'std': np.std(event_counts),
                'min': np.min(event_counts),
                'max': np.max(event_counts),
                'q25': np.percentile(event_counts, 25),
                'q75': np.percentile(event_counts, 75)
            }
        else:
            results['event_count_stats'] = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'q25': 0, 'q75': 0
            }
        
        # 事件长度统计
        if event_lengths:
            results['event_length_stats'] = {
                'mean': np.mean(event_lengths),
                'median': np.median(event_lengths),
                'std': np.std(event_lengths),
                'min': np.min(event_lengths),
                'max': np.max(event_lengths),
                'q25': np.percentile(event_lengths, 25),
                'q75': np.percentile(event_lengths, 75)
            }
        else:
            results['event_length_stats'] = {
                'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0, 'q25': 0, 'q75': 0
            }
        
        # 词汇统计
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        results['vocabulary_stats'] = {
            'total_words': total_words,
            'unique_words': unique_words,
            'type_token_ratio': unique_words / total_words if total_words > 0 else 0,
            'avg_unique_words_per_sample': np.mean(unique_words_per_sample) if unique_words_per_sample else 0,
            'vocabulary_size': unique_words
        }
        
        # 词频分布
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        results['detailed_distributions']['top_20_words'] = top_words
        
        # 停用词占比
        stop_word_count = sum(1 for word in all_words if word in self.stop_words)
        results['vocabulary_stats']['stop_word_ratio'] = stop_word_count / total_words if total_words > 0 else 0
        
        return results
    
    def calculate_information_density(self, text: str) -> Dict[str, float]:
        """计算文本信息密度"""
        words = self.tokenize(text)
        
        # Shannon熵
        word_freq = Counter(words)
        total = sum(word_freq.values())
        entropy = 0
        for count in word_freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Kolmogorov复杂度（使用压缩近似）
        text_bytes = text.encode('utf-8')
        compressed = gzip.compress(text_bytes)
        compression_ratio = len(compressed) / len(text_bytes)
        
        # 信息密度
        info_density = entropy / len(words) if words else 0
        
        return {
            'shannon_entropy': entropy,
            'compression_ratio': compression_ratio,
            'kolmogorov_complexity': compression_ratio,  # 近似
            'information_density': info_density,
            'text_length': len(text),
            'word_count': len(words)
        }


class SemanticAnalyzer:
    """语义分析器"""
    
    def __init__(self):
        """初始化语义分析器"""
        self.tfidf_vectorizer = None
        self.lda_model = None
        self.sentence_model = None
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded Sentence-BERT model")
            except:
                logger.warning("Failed to load Sentence-BERT model")
    
    def extract_tfidf_features(self, texts: List[str], max_features: int = 100) -> Tuple[np.ndarray, List[str]]:
        """提取TF-IDF特征"""
        if not SKLEARN_AVAILABLE:
            return np.array([]), []
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        return tfidf_matrix.toarray(), feature_names.tolist()
    
    def calculate_semantic_similarity(self, texts: List[str]) -> np.ndarray:
        """计算文本间的语义相似度"""
        if not texts:
            return np.array([])
        
        if SKLEARN_AVAILABLE and self.tfidf_vectorizer is None:
            # 使用TF-IDF计算相似度
            tfidf_matrix, _ = self.extract_tfidf_features(texts)
            if tfidf_matrix.size > 0:
                similarity_matrix = cosine_similarity(tfidf_matrix)
                return similarity_matrix
        
        # 备用：简单的词汇重叠相似度
        similarity_matrix = np.zeros((len(texts), len(texts)))
        for i, text1 in enumerate(texts):
            words1 = set(text1.lower().split())
            for j, text2 in enumerate(texts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    words2 = set(text2.lower().split())
                    if words1 and words2:
                        similarity_matrix[i, j] = len(words1 & words2) / len(words1 | words2)
        
        return similarity_matrix
    
    def perform_topic_modeling(self, texts: List[str], n_topics: int = 10) -> Dict[str, Any]:
        """执行话题建模"""
        if not SKLEARN_AVAILABLE:
            return {'error': 'sklearn not available'}
        
        # 向量化
        vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        # LDA建模
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=10
        )
        
        lda_output = self.lda_model.fit_transform(doc_term_matrix)
        
        # 提取话题关键词
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = np.argsort(topic)[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': topic[top_indices].tolist()
            })
        
        return {
            'n_topics': n_topics,
            'topics': topics,
            'perplexity': self.lda_model.perplexity(doc_term_matrix),
            'topic_distribution': lda_output.mean(axis=0).tolist()
        }
    
    def extract_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """提取句子嵌入"""
        if self.sentence_model is not None:
            embeddings = self.sentence_model.encode(texts)
            return embeddings
        
        # 备用：使用平均词向量
        logger.warning("Using fallback: average word vectors")
        embeddings = []
        for text in texts:
            words = text.lower().split()
            if words:
                # 简单的哈希向量化
                vec = np.zeros(100)
                for word in words:
                    idx = hash(word) % 100
                    vec[idx] += 1
                vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
                embeddings.append(vec)
            else:
                embeddings.append(np.zeros(100))
        
        return np.array(embeddings)
    
    def analyze_semantic_redundancy(self, events_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析语义冗余度"""
        results = {
            'redundancy_stats': {},
            'similarity_distribution': [],
            'high_similarity_pairs': []
        }
        
        all_similarities = []
        
        for events in events_list[:100]:  # 限制样本数量
            if len(events) < 2:
                continue
            
            texts = list(events.values())
            sim_matrix = self.calculate_semantic_similarity(texts)
            
            # 提取上三角矩阵（排除对角线）
            for i in range(len(texts)):
                for j in range(i+1, len(texts)):
                    sim = sim_matrix[i, j]
                    all_similarities.append(sim)
                    
                    # 记录高相似度对
                    if sim > 0.8:
                        results['high_similarity_pairs'].append({
                            'similarity': sim,
                            'text1': texts[i][:100],  # 前100字符
                            'text2': texts[j][:100]
                        })
        
        if all_similarities:
            results['redundancy_stats'] = {
                'mean_similarity': np.mean(all_similarities),
                'median_similarity': np.median(all_similarities),
                'std_similarity': np.std(all_similarities),
                'high_similarity_ratio': sum(1 for s in all_similarities if s > 0.7) / len(all_similarities)
            }
            results['similarity_distribution'] = np.histogram(all_similarities, bins=10)[0].tolist()
        
        return results


class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        """初始化情感分析器"""
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """使用VADER进行情感分析"""
        if self.vader:
            scores = self.vader.polarity_scores(text)
            return scores
        
        # 备用：简单的基于词典的情感分析
        positive_words = {'good', 'great', 'excellent', 'positive', 'up', 'increase', 
                         'gain', 'profit', 'growth', 'strong', 'surge'}
        negative_words = {'bad', 'poor', 'negative', 'down', 'decrease', 'loss', 
                         'decline', 'weak', 'fall', 'drop'}
        
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = len(words)
        
        if total == 0:
            return {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
        
        return {
            'pos': pos_count / total,
            'neg': neg_count / total,
            'neu': 1 - (pos_count + neg_count) / total,
            'compound': (pos_count - neg_count) / total
        }
    
    def analyze_sentiment_distribution(self, events_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析情感分布"""
        results = {
            'sentiment_stats': {},
            'sentiment_distribution': {},
            'extreme_sentiments': []
        }
        
        all_sentiments = []
        
        for events in events_list:
            for event_text in events.values():
                sentiment = self.analyze_sentiment_vader(event_text)
                all_sentiments.append(sentiment)
                
                # 记录极端情感
                if abs(sentiment['compound']) > 0.8:
                    results['extreme_sentiments'].append({
                        'text': event_text[:100],
                        'compound_score': sentiment['compound']
                    })
        
        if all_sentiments:
            # 计算统计量
            compounds = [s['compound'] for s in all_sentiments]
            positives = [s['pos'] for s in all_sentiments]
            negatives = [s['neg'] for s in all_sentiments]
            neutrals = [s['neu'] for s in all_sentiments]
            
            results['sentiment_stats'] = {
                'compound': {
                    'mean': np.mean(compounds),
                    'std': np.std(compounds),
                    'median': np.median(compounds)
                },
                'positive': {
                    'mean': np.mean(positives),
                    'ratio': sum(1 for c in compounds if c > 0.05) / len(compounds)
                },
                'negative': {
                    'mean': np.mean(negatives),
                    'ratio': sum(1 for c in compounds if c < -0.05) / len(compounds)
                },
                'neutral': {
                    'mean': np.mean(neutrals),
                    'ratio': sum(1 for c in compounds if -0.05 <= c <= 0.05) / len(compounds)
                }
            }
            
            # 情感分布
            results['sentiment_distribution'] = {
                'positive': sum(1 for c in compounds if c > 0.05),
                'negative': sum(1 for c in compounds if c < -0.05),
                'neutral': sum(1 for c in compounds if -0.05 <= c <= 0.05)
            }
        
        return results
    
    def analyze_sentiment_consistency(self, events_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """分析情感一致性"""
        results = {
            'consistency_scores': [],
            'conflict_samples': []
        }
        
        for idx, events in enumerate(events_list[:100]):  # 限制样本数
            if len(events) < 2:
                continue
            
            sentiments = []
            for event_text in events.values():
                sentiment = self.analyze_sentiment_vader(event_text)
                sentiments.append(sentiment['compound'])
            
            # 计算一致性（标准差的倒数）
            if len(sentiments) > 1:
                consistency = 1 / (np.std(sentiments) + 1)
                results['consistency_scores'].append(consistency)
                
                # 检测情感冲突
                if max(sentiments) > 0.5 and min(sentiments) < -0.5:
                    results['conflict_samples'].append({
                        'sample_id': idx,
                        'sentiment_range': max(sentiments) - min(sentiments),
                        'sentiments': sentiments
                    })
        
        if results['consistency_scores']:
            results['consistency_stats'] = {
                'mean': np.mean(results['consistency_scores']),
                'std': np.std(results['consistency_scores']),
                'high_consistency_ratio': sum(1 for s in results['consistency_scores'] if s > 0.8) / len(results['consistency_scores'])
            }
        
        return results


class TextVisualizationTools:
    """文本分析可视化工具"""
    
    @staticmethod
    def plot_event_statistics(stats: Dict[str, Any], save_path: str) -> None:
        """绘制事件统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 事件数量分布
        event_counts = stats.get('event_count_distribution', [])
        if event_counts:
            axes[0, 0].hist(event_counts, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Event Count Distribution')
            axes[0, 0].set_xlabel('Number of Events')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(stats['event_count_stats']['mean'], color='red', 
                              linestyle='--', label=f"Mean: {stats['event_count_stats']['mean']:.2f}")
            axes[0, 0].legend()
        
        # 事件长度分布
        event_lengths = stats.get('event_length_distribution', [])
        if event_lengths:
            axes[0, 1].hist(event_lengths, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('Event Length Distribution')
            axes[0, 1].set_xlabel('Number of Words')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(stats['event_length_stats']['mean'], color='red',
                              linestyle='--', label=f"Mean: {stats['event_length_stats']['mean']:.2f}")
            axes[0, 1].legend()
        
        # 词频分布（前20词）
        if 'detailed_distributions' in stats and 'top_20_words' in stats['detailed_distributions']:
            words, counts = zip(*stats['detailed_distributions']['top_20_words'])
            axes[1, 0].barh(range(len(words)), counts)
            axes[1, 0].set_yticks(range(len(words)))
            axes[1, 0].set_yticklabels(words)
            axes[1, 0].set_title('Top 20 Most Frequent Words')
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].invert_yaxis()
        
        # 词汇统计
        vocab_stats = stats.get('vocabulary_stats', {})
        if vocab_stats:
            labels = ['Unique Words', 'Stop Words', 'Content Words']
            sizes = [
                vocab_stats.get('unique_words', 0),
                vocab_stats.get('total_words', 0) * vocab_stats.get('stop_word_ratio', 0),
                vocab_stats.get('total_words', 0) * (1 - vocab_stats.get('stop_word_ratio', 0))
            ]
            axes[1, 1].pie([s for s in sizes if s > 0], 
                          labels=[l for l, s in zip(labels, sizes) if s > 0],
                          autopct='%1.1f%%')
            axes[1, 1].set_title('Vocabulary Composition')
        
        plt.suptitle('Text Statistics Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_semantic_analysis(semantic_results: Dict[str, Any], save_path: str) -> None:
        """绘制语义分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 相似度分布
        if 'similarity_distribution' in semantic_results:
            axes[0, 0].bar(range(len(semantic_results['similarity_distribution'])),
                          semantic_results['similarity_distribution'])
            axes[0, 0].set_title('Semantic Similarity Distribution')
            axes[0, 0].set_xlabel('Similarity Range')
            axes[0, 0].set_ylabel('Count')
        
        # 话题分布
        if 'topic_distribution' in semantic_results:
            topic_dist = semantic_results['topic_distribution']
            axes[0, 1].bar(range(len(topic_dist)), topic_dist)
            axes[0, 1].set_title('Topic Distribution')
            axes[0, 1].set_xlabel('Topic ID')
            axes[0, 1].set_ylabel('Probability')
        
        # 冗余度统计
        if 'redundancy_stats' in semantic_results:
            stats = semantic_results['redundancy_stats']
            metrics = ['Mean Similarity', 'High Similarity Ratio']
            values = [
                stats.get('mean_similarity', 0),
                stats.get('high_similarity_ratio', 0)
            ]
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Semantic Redundancy Metrics')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_ylim(0, 1)
        
        # 话题词云（简化版）
        if 'topics' in semantic_results and semantic_results['topics']:
            # 显示前5个话题的关键词
            topic_text = []
            for i, topic in enumerate(semantic_results['topics'][:5]):
                words = ' '.join(topic['top_words'][:5])
                topic_text.append(f"Topic {i}: {words}")
            
            axes[1, 1].text(0.1, 0.5, '\n\n'.join(topic_text), 
                           fontsize=10, verticalalignment='center')
            axes[1, 1].set_title('Top Topics Keywords')
            axes[1, 1].axis('off')
        
        plt.suptitle('Semantic Analysis Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_sentiment_analysis(sentiment_results: Dict[str, Any], save_path: str) -> None:
        """绘制情感分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 情感分布饼图
        if 'sentiment_distribution' in sentiment_results:
            dist = sentiment_results['sentiment_distribution']
            labels = list(dist.keys())
            sizes = list(dist.values())
            colors = ['green', 'red', 'gray']
            axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0, 0].set_title('Overall Sentiment Distribution')
        
        # 情感得分分布
        if 'sentiment_scores' in sentiment_results:
            scores = sentiment_results['sentiment_scores']
            axes[0, 1].hist(scores, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('Compound Sentiment Score Distribution')
            axes[0, 1].set_xlabel('Compound Score')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(0, color='black', linestyle='-', alpha=0.3)
        
        # 情感统计
        if 'sentiment_stats' in sentiment_results:
            stats = sentiment_results['sentiment_stats']
            if 'compound' in stats:
                metrics = ['Mean', 'Median', 'Std Dev']
                values = [
                    stats['compound'].get('mean', 0),
                    stats['compound'].get('median', 0),
                    stats['compound'].get('std', 0)
                ]
                axes[1, 0].bar(metrics, values)
                axes[1, 0].set_title('Compound Sentiment Statistics')
                axes[1, 0].set_ylabel('Value')
        
        # 情感一致性
        if 'consistency_stats' in sentiment_results:
            cons_stats = sentiment_results['consistency_stats']
            labels = ['Mean Consistency', 'High Consistency Ratio']
            values = [
                cons_stats.get('mean', 0),
                cons_stats.get('high_consistency_ratio', 0)
            ]
            axes[1, 1].bar(labels, values)
            axes[1, 1].set_title('Sentiment Consistency Metrics')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_ylim(0, 1)
        
        plt.suptitle('Sentiment Analysis Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()