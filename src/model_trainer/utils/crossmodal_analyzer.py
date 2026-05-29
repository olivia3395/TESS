"""
跨模态关联预检验工具类
Cross-modal Correlation Pre-screening Tools

该模块实现跨模态关联预检验的核心功能，包括：
1. 时序数据离散化
2. 文本特征提取
3. 简单分类器训练和评估
4. 基准对比分析

作者：Claude Code Assistant
日期：2025-01-XX
"""

import numpy as np
import pandas as pd
import json
from typing import List, Dict, Tuple, Any, Optional
import logging
from pathlib import Path

# 机器学习相关
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier

# 文本处理相关
import re
from collections import Counter

# 情感分析
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# 统计分析
from scipy.stats import chi2_contingency, chi2
import scipy.stats as stats

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns


class TimeSeriesDiscretizer:
    """时序离散化工具类"""
    
    def __init__(self, up_threshold: float = 2.0, down_threshold: float = -2.0):
        """
        初始化时序离散化器
        
        Args:
            up_threshold: 上涨阈值（百分比）
            down_threshold: 下跌阈值（百分比）
        """
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        
    def compute_change_rate(self, hist_data: str, ground_truth: str) -> float:
        """
        计算价格变化率
        
        Args:
            hist_data: 历史数据字符串，如 "86.3146, 91.6460, 88.8138, 83.9487, 80.8501"
            ground_truth: 未来数据字符串，如 "81.8567, 79.6561, 80.8896, 77.6824, 78.2252"
            
        Returns:
            变化率（百分比）
        """
        try:
            # 解析历史数据和未来数据
            hist_values = [float(x.strip()) for x in hist_data.split(',')]
            future_values = [float(x.strip()) for x in ground_truth.split(',')]
            
            # 计算平均值（或使用最后一个值）
            current_price = hist_values[-1]  # 使用最后一个历史值
            future_price = future_values[0]   # 使用第一个未来值
            
            # 计算变化率
            change_rate = (future_price - current_price) / current_price * 100
            return change_rate
            
        except Exception as e:
            logging.error(f"计算变化率失败: {e}")
            return 0.0
    
    def discretize_3class(self, change_rate: float) -> str:
        """
        三分类离散化
        
        Args:
            change_rate: 变化率（百分比）
            
        Returns:
            分类标签："Up", "Down", "Stable"
        """
        if change_rate > self.up_threshold:
            return "Up"
        elif change_rate < self.down_threshold:
            return "Down"
        else:
            return "Stable"
    
    def discretize_5class(self, change_rate: float) -> str:
        """
        五分类离散化
        
        Args:
            change_rate: 变化率（百分比）
            
        Returns:
            分类标签："Strong_Up", "Up", "Stable", "Down", "Strong_Down"
        """
        if change_rate > 5.0:
            return "Strong_Up"
        elif change_rate > 2.0:
            return "Up"
        elif change_rate > -2.0:
            return "Stable"
        elif change_rate > -5.0:
            return "Down"
        else:
            return "Strong_Down"


class TextFeatureExtractor:
    """文本特征提取工具类"""
    
    def __init__(self):
        """初始化文本特征提取器"""
        self.tfidf_vectorizer = None
        self.sentiment_analyzer = None
        
        # 初始化情感分析器
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        if not text or pd.isna(text):
            return ""
            
        # 转换为小写
        text = text.lower()
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
        
        return text.strip()
    
    def extract_events_text(self, event_dict: Dict[str, str]) -> str:
        """
        从事件字典中提取并合并文本
        
        Args:
            event_dict: 事件字典，如 {"event1": "...", "event2": "..."}
            
        Returns:
            合并后的文本字符串
        """
        if not event_dict:
            return ""
            
        texts = []
        for key, value in event_dict.items():
            if value and not pd.isna(value):
                texts.append(str(value))
        
        return " ".join(texts)
    
    def extract_basic_features(self, text: str) -> Dict[str, float]:
        """
        提取基础文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        features = {}
        
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'sentence_count': 0,
                'exclamation_count': 0,
                'question_count': 0
            }
        
        # 基础统计
        features['char_count'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        features['sentence_count'] = len(re.findall(r'[.!?]+', text))
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        
        return features
    
    def extract_sentiment_features(self, text: str) -> Dict[str, float]:
        """
        提取情感特征
        
        Args:
            text: 输入文本
            
        Returns:
            情感特征字典
        """
        features = {}
        
        if not text or not self.sentiment_analyzer:
            return {
                'sentiment_pos': 0.0,
                'sentiment_neg': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_compound': 0.0
            }
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            features['sentiment_pos'] = scores['pos']
            features['sentiment_neg'] = scores['neg']
            features['sentiment_neu'] = scores['neu']
            features['sentiment_compound'] = scores['compound']
        except:
            features = {
                'sentiment_pos': 0.0,
                'sentiment_neg': 0.0,
                'sentiment_neu': 0.0,
                'sentiment_compound': 0.0
            }
        
        return features
    
    def fit_tfidf(self, texts: List[str], max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        训练TF-IDF向量化器
        
        Args:
            texts: 文本列表
            max_features: 最大特征数
            ngram_range: n-gram范围
        """
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.tfidf_vectorizer.fit(texts)
    
    def extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """
        提取TF-IDF特征
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF向量化器未训练，请先调用fit_tfidf方法")
        
        return self.tfidf_vectorizer.transform(texts).toarray()


class CrossModalClassifier:
    """跨模态分类器工具类"""
    
    def __init__(self):
        """初始化分类器"""
        self.classifiers = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self.trained_classifiers = {}
        self.feature_names = []
    
    def prepare_features(self, tfidf_features: np.ndarray, 
                        basic_features: List[Dict], 
                        sentiment_features: List[Dict]) -> np.ndarray:
        """
        合并所有特征
        
        Args:
            tfidf_features: TF-IDF特征矩阵
            basic_features: 基础特征列表
            sentiment_features: 情感特征列表
            
        Returns:
            合并后的特征矩阵
        """
        # 转换基础特征为矩阵
        basic_df = pd.DataFrame(basic_features)
        sentiment_df = pd.DataFrame(sentiment_features)
        
        # 合并所有特征
        all_features = np.hstack([
            tfidf_features,
            basic_df.values,
            sentiment_df.values
        ])
        
        # 记录特征名称
        tfidf_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
        self.feature_names = tfidf_names + list(basic_df.columns) + list(sentiment_df.columns)
        
        return all_features
    
    def train_classifiers(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        训练所有分类器
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            训练结果字典
        """
        results = {}
        
        for name, classifier in self.classifiers.items():
            try:
                # 训练分类器
                classifier.fit(X, y)
                self.trained_classifiers[name] = classifier
                
                # 交叉验证
                cv_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
                
                results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                logging.info(f"{name} - CV准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logging.error(f"训练{name}失败: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def evaluate_classifiers(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        评估分类器性能
        
        Args:
            X_test: 测试特征矩阵
            y_test: 测试标签
            
        Returns:
            评估结果字典
        """
        results = {}
        
        for name, classifier in self.trained_classifiers.items():
            try:
                # 预测
                y_pred = classifier.predict(X_test)
                y_proba = None
                if hasattr(classifier, 'predict_proba'):
                    y_proba = classifier.predict_proba(X_test)
                
                # 计算指标
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                # 混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': cm.tolist(),
                    'predictions': y_pred.tolist(),
                    'probabilities': y_proba.tolist() if y_proba is not None else None,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                logging.info(f"{name} - 测试准确率: {accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"评估{name}失败: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def get_baseline_performance(self, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        计算基准性能
        
        Args:
            y_train: 训练标签
            y_test: 测试标签
            
        Returns:
            基准性能字典
        """
        # 随机基准
        dummy_random = DummyClassifier(strategy='uniform', random_state=42)
        dummy_random.fit(np.zeros((len(y_train), 1)), y_train)
        random_accuracy = dummy_random.score(np.zeros((len(y_test), 1)), y_test)
        
        # 多数类基准
        dummy_majority = DummyClassifier(strategy='most_frequent')
        dummy_majority.fit(np.zeros((len(y_train), 1)), y_train)
        majority_accuracy = dummy_majority.score(np.zeros((len(y_test), 1)), y_test)
        
        # 理论随机准确率
        unique_labels = np.unique(y_train)
        theoretical_random = 1.0 / len(unique_labels)
        
        return {
            'random_accuracy': random_accuracy,
            'majority_accuracy': majority_accuracy,
            'theoretical_random': theoretical_random,
            'n_classes': len(unique_labels),
            'class_distribution': {str(label): np.sum(y_train == label) for label in unique_labels}
        }


class StatisticalAnalyzer:
    """统计分析工具类"""
    
    @staticmethod
    def chi_square_test(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        卡方独立性检验
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            检验结果字典
        """
        try:
            # 构建列联表
            contingency_table = pd.crosstab(y_true, y_pred)
            
            # 执行卡方检验
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # 计算Cramér's V (效应量)
            n = contingency_table.sum().sum()
            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
            
            return {
                'chi2_statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'is_significant': p_value < 0.05,
                'contingency_table': contingency_table.to_dict()
            }
            
        except Exception as e:
            logging.error(f"卡方检验失败: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, float]:
        """
        McNemar检验比较两个分类器
        
        Args:
            y_true: 真实标签
            y_pred1: 分类器1预测
            y_pred2: 分类器2预测
            
        Returns:
            检验结果字典
        """
        try:
            # 计算正确/错误分类
            correct1 = (y_true == y_pred1)
            correct2 = (y_true == y_pred2)
            
            # 构建2x2表
            n01 = np.sum(~correct1 & correct2)  # 分类器1错，分类器2对
            n10 = np.sum(correct1 & ~correct2)  # 分类器1对，分类器2错
            
            # McNemar统计量
            if n01 + n10 == 0:
                mcnemar_stat = 0
                p_value = 1.0
            else:
                mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
                p_value = 1 - chi2.cdf(mcnemar_stat, 1)
            
            return {
                'mcnemar_statistic': mcnemar_stat,
                'p_value': p_value,
                'n01': n01,
                'n10': n10,
                'is_significant': p_value < 0.05
            }
            
        except Exception as e:
            logging.error(f"McNemar检验失败: {e}")
            return {'error': str(e)}


class CrossModalAnalyzer:
    """跨模态分析主类"""
    
    def __init__(self, up_threshold: float = 2.0, down_threshold: float = -2.0):
        """
        初始化跨模态分析器
        
        Args:
            up_threshold: 上涨阈值
            down_threshold: 下跌阈值
        """
        self.discretizer = TimeSeriesDiscretizer(up_threshold, down_threshold)
        self.feature_extractor = TextFeatureExtractor()
        self.classifier = CrossModalClassifier()
        self.statistical_analyzer = StatisticalAnalyzer()
        
        self.analysis_results = {}
        
    def load_data(self, data_path: str) -> List[Dict]:
        """
        加载数据
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            数据列表
        """
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logging.info(f"成功加载数据: {len(data)} 个样本")
            return data
        except Exception as e:
            logging.error(f"加载数据失败: {e}")
            return []
    
    def analyze_dataset(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """
        完整的跨模态分析流程
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            
        Returns:
            分析结果字典
        """
        logging.info("开始跨模态分析...")
        
        # 1. 数据预处理和标签生成
        train_processed = self._process_data(train_data)
        test_processed = self._process_data(test_data)
        
        # 2. 分析标签分布
        label_analysis = self._analyze_labels(train_processed, test_processed)
        
        # 3. 特征提取
        X_train, y_train = self._extract_features(train_processed, fit_tfidf=True)
        X_test, y_test = self._extract_features(test_processed, fit_tfidf=False)
        
        # 4. 训练分类器
        training_results = self.classifier.train_classifiers(X_train, y_train)
        
        # 5. 评估分类器
        evaluation_results = self.classifier.evaluate_classifiers(X_test, y_test)
        
        # 6. 基准性能分析
        baseline_results = self.classifier.get_baseline_performance(y_train, y_test)
        
        # 7. 统计显著性检验
        statistical_results = self._perform_statistical_tests(y_test, evaluation_results)
        
        # 8. 整合结果
        self.analysis_results = {
            'data_info': {
                'train_size': len(train_data),
                'test_size': len(test_data),
                'feature_dim': X_train.shape[1]
            },
            'label_analysis': label_analysis,
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'baseline_results': baseline_results,
            'statistical_results': statistical_results,
            'feature_names': self.classifier.feature_names
        }
        
        logging.info("跨模态分析完成!")
        return self.analysis_results
    
    def _process_data(self, data: List[Dict]) -> List[Dict]:
        """
        处理单个数据集
        
        Args:
            data: 原始数据列表
            
        Returns:
            处理后的数据列表
        """
        processed = []
        
        for item in data:
            try:
                # 提取文本
                event_text = self.feature_extractor.extract_events_text(item.get('event', {}))
                processed_text = self.feature_extractor.preprocess_text(event_text)
                
                # 计算价格变化率
                change_rate = self.discretizer.compute_change_rate(
                    item['hist_data'], 
                    item['ground_truth']
                )
                
                # 生成标签
                label = self.discretizer.discretize_3class(change_rate)
                
                processed.append({
                    'text': processed_text,
                    'change_rate': change_rate,
                    'label': label,
                    'original_item': item
                })
                
            except Exception as e:
                logging.warning(f"处理数据项失败: {e}")
                continue
        
        return processed
    
    def _analyze_labels(self, train_data: List[Dict], test_data: List[Dict]) -> Dict[str, Any]:
        """
        分析标签分布
        
        Args:
            train_data: 训练数据
            test_data: 测试数据
            
        Returns:
            标签分析结果
        """
        train_labels = [item['label'] for item in train_data]
        test_labels = [item['label'] for item in test_data]
        
        train_dist = Counter(train_labels)
        test_dist = Counter(test_labels)
        
        return {
            'train_distribution': dict(train_dist),
            'test_distribution': dict(test_dist),
            'train_total': len(train_labels),
            'test_total': len(test_labels),
            'unique_labels': sorted(set(train_labels + test_labels)),
            'is_balanced': min(train_dist.values()) / max(train_dist.values()) > 0.5 if train_dist else False
        }
    
    def _extract_features(self, data: List[Dict], fit_tfidf: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取特征
        
        Args:
            data: 处理后的数据
            fit_tfidf: 是否训练TF-IDF向量化器
            
        Returns:
            特征矩阵和标签数组
        """
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        
        # TF-IDF特征
        if fit_tfidf:
            self.feature_extractor.fit_tfidf(texts)
        tfidf_features = self.feature_extractor.extract_tfidf_features(texts)
        
        # 基础特征
        basic_features = [self.feature_extractor.extract_basic_features(text) for text in texts]
        
        # 情感特征
        sentiment_features = [self.feature_extractor.extract_sentiment_features(text) for text in texts]
        
        # 合并特征
        X = self.classifier.prepare_features(tfidf_features, basic_features, sentiment_features)
        y = np.array(labels)
        
        return X, y
    
    def _perform_statistical_tests(self, y_test: np.ndarray, 
                                  evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        执行统计显著性检验
        
        Args:
            y_test: 测试标签
            evaluation_results: 评估结果
            
        Returns:
            统计检验结果
        """
        results = {}
        
        # 对每个分类器进行卡方检验
        for name, eval_result in evaluation_results.items():
            if 'predictions' in eval_result:
                y_pred = np.array(eval_result['predictions'])
                chi2_result = self.statistical_analyzer.chi_square_test(y_test, y_pred)
                results[f'{name}_chi2'] = chi2_result
        
        # McNemar检验比较分类器
        classifier_names = list(evaluation_results.keys())
        for i, name1 in enumerate(classifier_names):
            for name2 in classifier_names[i+1:]:
                if ('predictions' in evaluation_results[name1] and 
                    'predictions' in evaluation_results[name2]):
                    
                    y_pred1 = np.array(evaluation_results[name1]['predictions'])
                    y_pred2 = np.array(evaluation_results[name2]['predictions'])
                    
                    mcnemar_result = self.statistical_analyzer.mcnemar_test(y_test, y_pred1, y_pred2)
                    results[f'mcnemar_{name1}_vs_{name2}'] = mcnemar_result
        
        return results
    
    def save_results(self, output_path: str):
        """
        保存分析结果
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=str)
            logging.info(f"结果已保存到: {output_path}")
        except Exception as e:
            logging.error(f"保存结果失败: {e}")