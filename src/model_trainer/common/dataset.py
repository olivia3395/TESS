import json
from typing import List, Optional, Sequence
import ipdb
import torch
from torch.utils.data import Dataset
from logging import getLogger
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from model_trainer.utils.lag_sampling import (
    apply_lag_to_sequence,
    config_lookup,
    filter_by_mask,
    parse_lag_policy,
    replay_lag_with_mapping,
)


def _config_lookup(config_obj, key: str, default=None):
    """兼容旧版调用的配置查询包装器。"""
    return config_lookup(config_obj, key, default)


def _legacy_apply_news_lag(
    news_items: Sequence,
    config,
    *,
    value_type: str = "text",
) -> List:
    """向后兼容的滞后函数，保证长度不变并固定为 clamp 策略。"""
    policy = parse_lag_policy(config)
    if not policy.enabled:
        return list(news_items)
    policy.drop_border = False
    policy.clamp_border = True
    lagged, keep_mask, _ = apply_lag_to_sequence(
        news_items,
        policy,
        value_type=value_type,
    )
    if not all(keep_mask):
        raise ValueError("Legacy 模式下不允许丢弃样本，请检查滞后配置")
    return lagged


def _apply_availability_lag(
    sequence: Sequence,
    config,
    *,
    value_type: str,
    logger,
    stage_tag: str,
):
    """根据配置对序列执行滞后，并返回重排结果与统计信息。"""
    policy = parse_lag_policy(config)
    lagged, keep_mask, stats = apply_lag_to_sequence(
        sequence,
        policy,
        value_type=value_type,
        logger=logger,
        log_prefix=stage_tag,
    )
    return lagged, keep_mask, stats


def _apply_drop_mask_to_records(records: Sequence, keep_mask: Sequence[bool]):
    """仅保留掩码标记的样本，确保 drop 策略下特征与标签数量一致。"""
    if not records:
        return list(records)
    if len(records) != len(keep_mask):
        raise ValueError("记录长度与掩码长度不一致，无法执行过滤")
    return [item for item, keep in zip(records, keep_mask) if keep]


def _finalize_tensor_sequence(lagged_sequence: Sequence[Optional[torch.Tensor]], keep_mask: Sequence[bool]) -> torch.Tensor:
    """将滞后后的张量序列整理为新的二维张量。"""
    filtered = []
    for tensor_item, keep in zip(lagged_sequence, keep_mask):
        if not keep:
            continue
        if tensor_item is None:
            raise ValueError("存在被保留但缺失内容的样本，滞后逻辑异常")
        filtered.append(tensor_item)
    if not filtered:
        raise ValueError("滞后后的张量序列为空，无法继续训练")
    return torch.stack(filtered, dim=0)


'''
Legacy ElectricityDataset（保留为注释，便于参考旧格式逻辑）
class ElectricityDataset(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        \"\"\"
        加载多模态时序数据集 并进行归一化处理 
        \"\"\"
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
     
        if flag == \"train\":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config[\"train_meta_file\"]
            news_path = dataset_path+config[\"train_news_file\"]
        elif flag == \"vali\":
            self.file_path = dataset_path+config[\"vali_file\"]
            meta_domain_path = dataset_path+config[\"vali_meta_file\"]
            news_path = dataset_path+config[\"vali_news_file\"]

        elif flag == \"test\":
            self.file_path = dataset_path+config[\"test_file\"]
            meta_domain_path = dataset_path+config[\"test_meta_file\"]
            news_path = dataset_path+config[\"test_news_file\"]

   
        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        self.meta_feats = None
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path, allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), \"样本数与Embedding数量不匹配\"

        self.news_feats = None
        if os.path.isfile(news_path):
            raw_news_feats = torch.from_numpy(np.load(news_path, allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(raw_news_feats), \"样本数与Embedding数量不匹配\"
            lagged_feats, keep_mask, _ = _apply_availability_lag(
                [feat.clone() for feat in raw_news_feats],
                config,
                value_type=\"tensor\",
                logger=self.logger,
                stage_tag=f\"[{self.__class__.__name__}-{flag}-news]\",
            )
            if not all(keep_mask):
                keep_tensor = torch.tensor(keep_mask, dtype=torch.bool)
                self.data = _apply_drop_mask_to_records(self.data, keep_mask)
                if self.meta_feats is not None:
                    self.meta_feats = self.meta_feats[keep_tensor]
                raw_news_feats = raw_news_feats[keep_tensor]
            self.news_feats = _finalize_tensor_sequence(lagged_feats, keep_mask).to(raw_news_feats.dtype)
        else:
            raise FileNotFoundError(f\"缺少新闻嵌入文件: {news_path}\")

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news_feats': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        \"\"\"获取归一化参数\"\"\"
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news_feats'],sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean
'''




# Legacy BitcoinDataset (注释保留兼容历史参考)
# class BitcoinDataset(Dataset):
#     def __init__(self, config,flag="train",scaler=None):
#         """
#         加载多模态时序数据集 并进行归一化处理 
#         """
#    
#         self.config = config 
#         self.logger = getLogger()
#         data_path = config['data_path']
#         dataset_path =os.path.abspath(data_path+config['dataset'])
#      
#         if flag == "train":
#             self.file_path = dataset_path+config['train_file']
#             meta_domain_path = dataset_path+config["train_meta_file"]
#             news_path = dataset_path+config["train_news_file"]
#         elif flag == "vali":
#             self.file_path = dataset_path+config["vali_file"]
#             meta_domain_path = dataset_path+config["vali_meta_file"]
#             news_path = dataset_path+config["vali_news_file"]
#
#         elif flag == "test":
#             meta_domain_path = dataset_path+config["test_meta_file"]
#             self.file_path = dataset_path+config["test_file"]
#             news_path = dataset_path+config["test_news_file"]
#
#         with open(self.file_path, 'r') as f:
#             self.data = json.load(f)
#         if os.path.isfile(news_path):
#             self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
#             assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
#             lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
#             if lag_strength > 0:
#                 lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
#                 self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
#         if os.path.isfile(meta_domain_path):
#             self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
#             assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"
#
#         if scaler is None:
#             self.scaler = StandardScaler()
#             all_data = []
#             for sample in self.data:
#                 hist = list(map(float, sample['historical_data'].split(',')))
#                 all_data.extend(hist)
#   
#             all_data = np.array(all_data).reshape(-1, 1)
#             self.scaler.fit(all_data)
#
#         else:
#             self.scaler = scaler 
#             self.mean = scaler.mean_[0]
#             self.std = np.sqrt(scaler.var_[0])
# 
#         self.samples = []
#         for idx,sample in enumerate(self.data):
#             hist_data = self._normalize_str(sample['historical_data'])
#             gt_data = self._normalize_str(sample['ground_truth'])
#             self.samples.append({
#                 'x': torch.tensor(hist_data, dtype=torch.float32),
#                 'meta_feats': self.meta_feats[idx],
#                 'news': self.news_feats[idx],
#                 'y': torch.tensor(gt_data, dtype=torch.float32)
#             })
#     
#     def _normalize_str(self, data_str):
#         values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
#         normalized = self.scaler.transform(values).flatten()
#         return normalized.tolist()
#     
#     def get_scaler(self):
#         """获取归一化参数"""
#         return self.scaler
#     
#     def __len__(self):
#         return len(self.samples)
#     
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
#     
#     def inverse_transform(self, normalized_data):
#         if isinstance(normalized_data,torch.Tensor):
#             return normalized_data*self.std + self.mean
#         return normalized_data * self.std + self.mean



class TrafficDataset(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"


        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            # news = str(sample['prompt'])
            # news = str(sample['news'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean
    
# Legacy EnvironmentDataset (kept for reference)
# class EnvironmentDataset(Dataset):
#     def __init__(self, config,flag="train",scaler=None):
#         """
#         加载多模态时序数据集 并进行归一化处理 
#         """
#    
#         self.config = config 
#         self.logger = getLogger()
#         data_path = config['data_path']
#         dataset_path =os.path.abspath(data_path+config['dataset'])
#      
#         if flag == "train":
#             self.file_path = dataset_path+config['train_file']
#             meta_domain_path = dataset_path+config["train_meta_file"]
#             news_path = dataset_path+config["train_news_file"]
#         elif flag == "vali":
#             self.file_path = dataset_path+config["vali_file"]
#             meta_domain_path = dataset_path+config["vali_meta_file"]
#             news_path = dataset_path+config["vali_news_file"]
#
#         elif flag == "test":
#             meta_domain_path = dataset_path+config["test_meta_file"]
#             self.file_path = dataset_path+config["test_file"]
#             news_path = dataset_path+config["test_news_file"]
#
#         with open(self.file_path, 'r') as f:
#             self.data = json.load(f)
#         if os.path.isfile(news_path):
#             self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
#             assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
#             lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
#             if lag_strength > 0:
#                 lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
#                 self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
#         if os.path.isfile(meta_domain_path):
#             self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
#             assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"
#
#             
#         if scaler is None:
#             self.scaler = StandardScaler()
#             all_data = []
#             for sample in self.data:
#                 hist = list(map(float, sample['historical_data'].split(',')))
#                 all_data.extend(hist)
#   
#             all_data = np.array(all_data).reshape(-1, 1)
#             self.scaler.fit(all_data)
#
#             
#                 
#             
#         else:
#             self.scaler = scaler 
#             self.mean = scaler.mean_[0]
#             self.std = np.sqrt(scaler.var_[0])
#         
#         
#
#         self.samples = []
#         for idx,sample in enumerate(self.data):
#             hist_data = self._normalize_str(sample['historical_data'])
#             
#             gt_data = self._normalize_str(sample['ground_truth'])
#             
#             self.samples.append({
#                 'x': torch.tensor(hist_data, dtype=torch.float32),
#                 'meta_feats': self.meta_feats[idx],
#                 'news': self.news_feats[idx],
#                 'y': torch.tensor(gt_data, dtype=torch.float32)
#             })
#     
#     
#     def _normalize_str(self, data_str):
#
#         values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
#         normalized = self.scaler.transform(values).flatten()
#         
#         return normalized.tolist()
#     
#     def get_scaler(self):
#         """获取归一化参数"""
#         return self.scaler
#     
#     def __len__(self):
#         return len(self.samples)
#     
#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
#     
#     def inverse_transform(self, normalized_data):
#         if isinstance(normalized_data,torch.Tensor):
#             return normalized_data*self.std + self.mean
#         return normalized_data * self.std + self.mean


class EnvironmentDataset(Dataset):
    def __init__(self, config, flag: str = "train", scaler=None):
        """Environment 数据集，兼容注册表样式嵌入与可选 meta。"""
        self.config = config
        self.logger = getLogger()

        base_multimodal = bool(_config_lookup(config, 'use_multimodal', False))
        self.use_text_news = bool(_config_lookup(config, 'use_text_news', base_multimodal))
        self.use_news_embedding = bool(_config_lookup(config, 'use_news_embedding', False))
        self.use_llm_hidden = bool(_config_lookup(config, 'use_llm_hidden', False))
        self.use_multimodal = (
            base_multimodal or self.use_text_news or self.use_news_embedding or self.use_llm_hidden
        )

        data_path = _config_lookup(config, 'data_path', os.getcwd())
        dataset_name = _config_lookup(config, 'dataset', '')
        dataset_root = _config_lookup(config, 'dataset_root', None)
        if dataset_root:
            dataset_path = os.path.abspath(dataset_root)
        else:
            dataset_path = os.path.abspath(os.path.join(data_path, dataset_name)) if dataset_name else os.path.abspath(data_path)

        def _resolve_relative(rel_path: Optional[str]) -> str:
            if rel_path is None:
                raise ValueError("Missing dataset split path")
            rel = rel_path.lstrip('/')
            return os.path.join(dataset_path, rel)

        if flag == "train":
            self.file_path = _resolve_relative(_config_lookup(config, 'train_file'))
        elif flag == "vali":
            self.file_path = _resolve_relative(_config_lookup(config, 'vali_file'))
        elif flag == "test":
            self.file_path = _resolve_relative(_config_lookup(config, 'test_file'))
        else:
            raise ValueError(f"Unknown flag: {flag}")

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)
            self.mean = float(self.scaler.mean_[0])
            self.std = float(np.sqrt(self.scaler.var_[0]))
        else:
            self.scaler = scaler
            self.mean = float(scaler.mean_[0])
            self.std = float(np.sqrt(scaler.var_[0]))

        original_len = len(self.data)
        raw_news_source = [sample.get('news', '') or '' for sample in self.data] if self.use_text_news else None
        index_sequence = list(range(original_len))
        _, keep_mask, lag_stats = _apply_availability_lag(
            index_sequence,
            config,
            value_type="text",
            logger=self.logger,
            stage_tag=f"[EnvironmentDataset-{flag}-index]",
        )
        assignments = lag_stats.get("assignments", list(range(original_len)))
        valid_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        if len(valid_indices) != original_len:
            drop_ratio = (original_len - len(valid_indices)) / max(original_len, 1)
            self.logger.info(
                f"[EnvironmentDataset-{flag}] drop_ratio={drop_ratio:.4f}, kept={len(valid_indices)}/{original_len}"
            )
        self.data = [self.data[idx] for idx in valid_indices]

        lagged_news_filtered: Optional[List[str]] = None
        if self.use_text_news and raw_news_source is not None:
            lagged_news_full = replay_lag_with_mapping(
                raw_news_source,
                assignments,
                keep_mask,
                value_type="text",
            )
            lagged_news_filtered = [lagged_news_full[idx] for idx in valid_indices]

        self.meta_feats: Optional[torch.Tensor] = None
        meta_path_key = f"{flag}_meta_file"
        meta_rel_path = _config_lookup(config, meta_path_key, None)
        if meta_rel_path:
            meta_path = os.path.abspath(os.path.join(dataset_path, meta_rel_path.lstrip('/')))
            if os.path.isfile(meta_path):
                raw_meta = np.load(meta_path, allow_pickle=True)
                meta_tensor = torch.from_numpy(raw_meta).type(torch.FloatTensor)
                if len(meta_tensor) != len(assignments):
                    raise ValueError("元特征数量与样本数不一致")
                keep_tensor = torch.tensor(keep_mask, dtype=torch.bool)
                self.meta_feats = meta_tensor[keep_tensor]
                if self.meta_feats.shape[0] != len(self.data):
                    raise ValueError("滞后后的元特征数量与有效样本数不符")

        self.news_embeddings: Optional[torch.Tensor] = None
        embed_path_key = f"{flag}_news_embed_file"
        embed_rel_path = _config_lookup(config, embed_path_key, None)
        if self.use_news_embedding:
            if not embed_rel_path:
                raise ValueError(f"配置缺少 {embed_path_key}，无法加载新闻嵌入")
            embed_rel_actual = embed_rel_path
            embed_tensor_key = None
            if '::' in embed_rel_path:
                embed_rel_actual, embed_tensor_key = embed_rel_path.split('::', 1)
            embed_path = os.path.abspath(os.path.join(dataset_path, embed_rel_actual.lstrip('/')))
            if not os.path.isfile(embed_path):
                raise FileNotFoundError(f"缺少新闻嵌入文件: {embed_path}")

            if embed_path.endswith(('.pt', '.pth')):
                embed_dict = torch.load(embed_path, map_location='cpu')
                if embed_tensor_key is None:
                    default_keys = {'train': 'train_news', 'vali': 'vali_news', 'test': 'test_news'}
                    embed_tensor_key = _config_lookup(config, f"{flag}_news_embed_key", None) or default_keys.get(flag)
                if embed_tensor_key not in embed_dict:
                    raise KeyError(f"在 {embed_path} 中未找到键 {embed_tensor_key}")
                raw_embeds = embed_dict[embed_tensor_key]
                if isinstance(raw_embeds, np.ndarray):
                    raw_embeds = torch.from_numpy(raw_embeds)
                raw_embeds = raw_embeds.detach().clone().cpu().type(torch.FloatTensor)
            else:
                raw_np = np.load(embed_path, allow_pickle=True)
                raw_embeds = torch.from_numpy(raw_np).type(torch.FloatTensor)

            if len(raw_embeds) != len(assignments):
                raise ValueError("新闻嵌入数量与样本数不一致")
            lagged_embed_full = replay_lag_with_mapping(
                [feat.clone() for feat in raw_embeds],
                assignments,
                keep_mask,
                value_type="tensor",
            )
            self.news_embeddings = _finalize_tensor_sequence(lagged_embed_full, keep_mask).to(raw_embeds.dtype)
            if self.news_embeddings.shape[0] != len(self.data):
                raise ValueError("滞后后的新闻嵌入数量与有效样本数不符")

        self.news_hidden: Optional[torch.Tensor] = None
        self.news_hidden_mask: Optional[torch.Tensor] = None
        hidden_path_key = f"{flag}_news_hidden_file"
        hidden_rel_path = _config_lookup(config, hidden_path_key, None)
        if self.use_llm_hidden:
            if not hidden_rel_path:
                raise ValueError(f"配置缺少 {hidden_path_key}，无法加载预编码 LLM hidden")
            hidden_rel_actual = hidden_rel_path
            hidden_tensor_key = None
            hidden_mask_key = None
            if '::' in hidden_rel_path:
                parts = hidden_rel_path.split('::')
                hidden_rel_actual = parts[0]
                if len(parts) >= 2:
                    hidden_tensor_key = parts[1]
                if len(parts) >= 3:
                    hidden_mask_key = parts[2]
            hidden_path = os.path.abspath(os.path.join(dataset_path, hidden_rel_actual.lstrip('/')))
            if not os.path.isfile(hidden_path):
                raise FileNotFoundError(f"缺少预编码 LLM hidden 文件: {hidden_path}")

            hidden_dict = torch.load(hidden_path, map_location='cpu')
            if not isinstance(hidden_dict, dict):
                raise ValueError(f"预编码 LLM hidden 文件 {hidden_path} 需要是包含张量的字典")

            default_hidden_keys = {'train': 'train_hidden', 'vali': 'vali_hidden', 'test': 'test_hidden'}
            default_mask_keys = {'train': 'train_mask', 'vali': 'vali_mask', 'test': 'test_mask'}
            hidden_key = hidden_tensor_key or _config_lookup(config, f"{flag}_news_hidden_key", None) or default_hidden_keys.get(flag)
            mask_key = hidden_mask_key or _config_lookup(config, f"{flag}_news_hidden_mask_key", None) or default_mask_keys.get(flag)
            if hidden_key not in hidden_dict:
                raise KeyError(f"在 {hidden_path} 中未找到键 {hidden_key}")
            raw_hidden = hidden_dict[hidden_key]
            if isinstance(raw_hidden, np.ndarray):
                raw_hidden = torch.from_numpy(raw_hidden)
            raw_hidden = raw_hidden.detach().clone().cpu().type(torch.FloatTensor)

            if mask_key:
                if mask_key not in hidden_dict:
                    raise KeyError(f"在 {hidden_path} 中未找到 mask 键 {mask_key}")
                raw_mask = hidden_dict[mask_key]
                if isinstance(raw_mask, np.ndarray):
                    raw_mask = torch.from_numpy(raw_mask)
                raw_mask = raw_mask.detach().clone().cpu()
                if raw_mask.dtype != torch.bool:
                    raw_mask = raw_mask.bool()
            else:
                raw_mask = torch.ones(raw_hidden.shape[:2], dtype=torch.bool)

            if len(raw_hidden) != len(assignments):
                raise ValueError("预编码 LLM hidden 数量与样本数不一致")

            lagged_hidden_full = replay_lag_with_mapping(
                [feat.clone() for feat in raw_hidden],
                assignments,
                keep_mask,
                value_type="tensor",
            )
            self.news_hidden = _finalize_tensor_sequence(lagged_hidden_full, keep_mask).to(raw_hidden.dtype)

            lagged_mask_full = replay_lag_with_mapping(
                [mask.clone() for mask in raw_mask],
                assignments,
                keep_mask,
                value_type="tensor",
            )
            self.news_hidden_mask = _finalize_tensor_sequence(lagged_mask_full, keep_mask)
            if self.news_hidden.shape[0] != len(self.data):
                raise ValueError("滞后后的 LLM hidden 数量与有效样本数不符")

        self.lagged_news = lagged_news_filtered if lagged_news_filtered is not None else None
        self.lag_stats = lag_stats
        self.lag_assignments = assignments

        self.samples = []
        for new_idx, sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            gt_data = self._normalize_str(sample['ground_truth'])

            item = {
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'y': torch.tensor(gt_data, dtype=torch.float32)
            }

            if self.meta_feats is not None and new_idx < len(self.meta_feats):
                item['meta_feats'] = self.meta_feats[new_idx]
                item['meta_tensor'] = self.meta_feats[new_idx]

            if self.lagged_news is not None:
                item['news'] = self.lagged_news[new_idx]

            if self.news_embeddings is not None:
                item['news_embedding'] = self.news_embeddings[new_idx]

            if self.news_hidden is not None:
                item['news_hidden'] = self.news_hidden[new_idx]
                if self.news_hidden_mask is not None:
                    item['news_hidden_mask'] = self.news_hidden_mask[new_idx]

            self.samples.append(item)
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.use_llm_hidden and ('news_hidden' in sample):
            return sample

        meta_tensor = sample.get('meta_tensor')
        news_embed = sample.get('news_embedding')
        news_text = sample.get('news')
        news_attention_mask = sample.get('news_attention_mask')

        if meta_tensor is not None:
            return {
                'x': sample['x'],
                'y': sample['y'],
                'meta_tensor': meta_tensor,
                'news_embedding': news_embed,
                'news_attention_mask': news_attention_mask,
                'news': news_text,
                'news_hidden': sample.get('news_hidden'),
                'news_hidden_mask': sample.get('news_hidden_mask')
            }

        has_text = self.use_text_news and (news_text is not None)
        has_embed = self.use_news_embedding and (news_embed is not None)
        if has_text and has_embed:
            return sample['x'], sample['news_embedding'], sample['news'], sample['y']
        if has_embed:
            return sample['x'], sample['news_embedding'], sample['y']
        if has_text:
            return sample['x'], sample['news'], sample['y']
        return sample['x'], sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class FnspidDataset(Dataset):
    def __init__(self, config, flag: str = "train", scaler=None):
        """
        FNSPID 数据集
        - 当 use_multimodal=True 时，返回 (x, news, y)，其中 news 为新闻文本字符串
        - 当 use_multimodal=False 时，返回 (x, y)
        """

        self.config = config
        self.logger = getLogger()
        base_multimodal = bool(_config_lookup(config, 'use_multimodal', False))
        self.use_text_news = bool(_config_lookup(config, 'use_text_news', base_multimodal))
        self.use_news_embedding = bool(_config_lookup(config, 'use_news_embedding', False))
        self.use_llm_hidden = bool(_config_lookup(config, 'use_llm_hidden', False))
        # 综合两类模态控制，后续 Trainer 可根据返回的字段做出适配
        self.use_multimodal = base_multimodal or self.use_text_news or self.use_news_embedding or self.use_llm_hidden
        # price_mode 控制价格序列信息量，news_mode 控制新闻向量是否保留
        self.price_mode = str(_config_lookup(config, 'price_mode', 'normal')).lower()
        self.news_mode = str(_config_lookup(config, 'news_mode', 'normal')).lower()

        data_path = _config_lookup(config, 'data_path', os.getcwd())
        dataset_name = _config_lookup(config, 'dataset', '')
        dataset_root = _config_lookup(config, 'dataset_root', None)
        if dataset_root:
            dataset_path = os.path.abspath(dataset_root)
        else:
            dataset_path = os.path.abspath(os.path.join(data_path, dataset_name)) if dataset_name else os.path.abspath(data_path)

        def _resolve_relative(rel_path: Optional[str]) -> str:
            if rel_path is None:
                raise ValueError("Missing dataset split path")
            rel = rel_path.lstrip('/')
            return os.path.join(dataset_path, rel)

        if flag == "train":
            self.file_path = _resolve_relative(_config_lookup(config, 'train_file'))
        elif flag == "vali":
            self.file_path = _resolve_relative(_config_lookup(config, 'vali_file'))
        elif flag == "test":
            self.file_path = _resolve_relative(_config_lookup(config, 'test_file'))
        else:
            raise ValueError(f"Unknown flag: {flag}")

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)

        # 归一化拟合
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                # ver_camf 数据使用 historical_data 字段
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)
            self.mean = float(self.scaler.mean_[0])
            self.std = float(np.sqrt(self.scaler.var_[0]))
        else:
            self.scaler = scaler
            self.mean = float(scaler.mean_[0])
            self.std = float(np.sqrt(scaler.var_[0]))

        # 构建滞后映射：先对样本索引施加滞后，以便后续在多种模态间复用同一映射
        # === 滞后映射构建 ===
        original_len = len(self.data)
        # 记录原始新闻文本，后续复用相同滞后映射即可得到滞后后的文本
        raw_news_source = [sample.get('news', '') or '' for sample in self.data] if self.use_text_news else None
        index_sequence = list(range(original_len))
        # 先对样本索引执行滞后，确保后续不同模态共享同一随机映射
        _, keep_mask, lag_stats = _apply_availability_lag(
            index_sequence,
            config,
            value_type="text",
            logger=self.logger,
            stage_tag=f"[FnspidDataset-{flag}-index]",
        )
        assignments = lag_stats.get("assignments", list(range(original_len)))
        valid_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
        if len(valid_indices) != original_len:
            drop_ratio = (original_len - len(valid_indices)) / max(original_len, 1)
            self.logger.info(
                f"[FnspidDataset-{flag}] drop_ratio={drop_ratio:.4f}, kept={len(valid_indices)}/{original_len}"
            )
        filtered_data = [self.data[idx] for idx in valid_indices]
        self.data = filtered_data
        self.keep_mask = keep_mask
        self.valid_indices = valid_indices

        lagged_news_filtered: Optional[List[str]] = None
        if self.use_text_news and raw_news_source is not None:
            # 对新闻文本执行与索引相同的滞后重排，避免重复采样
            lagged_news_full = replay_lag_with_mapping(
                raw_news_source,
                assignments,
                keep_mask,
                value_type="text",
            )
            lagged_news_filtered = [lagged_news_full[idx] for idx in valid_indices]
        
        self.news_embeddings: Optional[torch.Tensor] = None
        self.news_attention_mask: Optional[torch.Tensor] = None
        self.news_hidden: Optional[torch.Tensor] = None
        self.news_hidden_mask: Optional[torch.Tensor] = None

        embed_path_key = f"{flag}_news_embed_file"
        embed_rel_path = _config_lookup(config, embed_path_key, None)
        if self.use_news_embedding:
            if not embed_rel_path:
                raise ValueError(f"配置缺少 {embed_path_key}，无法加载新闻嵌入")
            embed_rel_actual = embed_rel_path
            embed_tensor_key = None
            if '::' in embed_rel_path:
                embed_rel_actual, embed_tensor_key = embed_rel_path.split('::', 1)
            rel_embed_path = embed_rel_actual.lstrip('/')
            embed_path = os.path.abspath(os.path.join(dataset_path, rel_embed_path))
            if not os.path.isfile(embed_path):
                raise FileNotFoundError(f"缺少新闻嵌入文件: {embed_path}")

            if embed_path.endswith(('.pt', '.pth')):
                embed_dict = torch.load(embed_path, map_location='cpu')
                if embed_tensor_key is None:
                    default_keys = {'train': 'train', 'vali': 'vali', 'test': 'test'}
                    embed_tensor_key = _config_lookup(config, f"{flag}_news_embed_key", None) or default_keys.get(flag)

                if embed_tensor_key not in embed_dict:
                    raise KeyError(f"在 {embed_path} 中未找到键 {embed_tensor_key}")

                split_data = embed_dict[embed_tensor_key]
                if not isinstance(split_data, dict):
                    # 兼容旧格式：直接是tensor
                    split_data = {'embeddings': split_data}

                # 加载embeddings
                raw_embeds = split_data['embeddings']
                if isinstance(raw_embeds, np.ndarray):
                    raw_embeds = torch.from_numpy(raw_embeds)
                raw_embeds = raw_embeds.detach().clone().cpu().type(torch.FloatTensor)

                # 加载attention mask (如果存在)
                attention_mask = None
                if 'attention_mask' in split_data:
                    attention_mask = split_data['attention_mask']
                    if isinstance(attention_mask, np.ndarray):
                        attention_mask = torch.from_numpy(attention_mask)
                    attention_mask = attention_mask.detach().clone().cpu()
                    if attention_mask.dtype != torch.bool:
                        attention_mask = attention_mask.bool()

                # 处理不同维度的情况
                if raw_embeds.dim() == 2:
                    # 句子级别embeddings (兼容旧格式)
                    assert len(raw_embeds) == len(assignments), "新闻嵌入数量与样本数不一致"
                    lagged_embed_full = replay_lag_with_mapping(
                        [feat.clone() for feat in raw_embeds],
                        assignments,
                        keep_mask,
                        value_type="tensor",
                    )
                    self.news_embeddings = _finalize_tensor_sequence(lagged_embed_full, keep_mask).to(raw_embeds.dtype)

                    if attention_mask is not None:
                        lagged_mask_full = replay_lag_with_mapping(
                            [mask.clone() for mask in attention_mask],
                            assignments,
                            keep_mask,
                            value_type="tensor",
                        )
                        self.news_attention_mask = _finalize_tensor_sequence(lagged_mask_full, keep_mask).to(attention_mask.dtype)

                elif raw_embeds.dim() == 3:
                    # token级别embeddings
                    assert raw_embeds.shape[0] == len(assignments), f"Token级别嵌入数量 {raw_embeds.shape[0]} 与样本数 {len(assignments)} 不一致"
                    lagged_embed_full = replay_lag_with_mapping(
                        [feat.clone() for feat in raw_embeds],  # [seq_len, emb_dim]
                        assignments,
                        keep_mask,
                        value_type="tensor",
                    )
                    self.news_embeddings = _finalize_tensor_sequence(lagged_embed_full, keep_mask).to(raw_embeds.dtype)

                    if attention_mask is not None:
                        assert attention_mask.shape[0] == len(assignments), f"Attention mask数量 {attention_mask.shape[0]} 与样本数 {len(assignments)} 不一致"
                        lagged_mask_full = replay_lag_with_mapping(
                            [mask.clone() for mask in attention_mask],  # [seq_len]
                            assignments,
                            keep_mask,
                            value_type="tensor",
                        )
                        self.news_attention_mask = _finalize_tensor_sequence(lagged_mask_full, keep_mask).to(attention_mask.dtype)
                    else:
                        self.logger.warning(f"Token级别embeddings检测到但缺少attention_mask，可能影响padding处理")

            else:
                # .npy文件处理 (保持兼容)
                raw_np = np.load(embed_path, allow_pickle=True)
                raw_embeds = torch.from_numpy(raw_np).type(torch.FloatTensor)
                assert len(raw_embeds) == len(assignments), "新闻嵌入数量与样本数不一致"
                lagged_embed_full = replay_lag_with_mapping(
                    [feat.clone() for feat in raw_embeds],
                    assignments,
                    keep_mask,
                    value_type="tensor",
                )
                self.news_embeddings = _finalize_tensor_sequence(lagged_embed_full, keep_mask).to(raw_embeds.dtype)

            # 应用news_mode设置
            if self.news_mode == 'zero':
                self.news_embeddings = torch.zeros_like(self.news_embeddings)

            if self.news_embeddings.shape[0] != len(valid_indices):
                raise ValueError(f"滞后后的新闻嵌入数量 {self.news_embeddings.shape[0]} 与有效样本数 {len(valid_indices)} 不符")
        else:
            self.news_embeddings = None

        # 加载 GT embeddings（用于损失函数）
        self.gt_embeddings: Optional[torch.Tensor] = None
        self.gt_attention_mask: Optional[torch.Tensor] = None

        gt_embed_path_key = f"{flag}_gt_embed_file"
        gt_embed_rel_path = _config_lookup(config, gt_embed_path_key, None)
        if gt_embed_rel_path:
            # 获取 GT 数据集根目录（如果配置了）
            gt_dataset_root = _config_lookup(config, 'gt_dataset_root', None)
            if gt_dataset_root:
                gt_dataset_path = os.path.abspath(gt_dataset_root)
            else:
                # 如果没有配置，使用当前数据集路径（向后兼容）
                gt_dataset_path = dataset_path
            
            # 解析路径和键名（格式：路径::键名）
            gt_embed_rel_actual = gt_embed_rel_path
            gt_embed_tensor_key = None
            if '::' in gt_embed_rel_path:
                gt_embed_rel_actual, gt_embed_tensor_key = gt_embed_rel_path.split('::', 1)
            
            rel_gt_embed_path = gt_embed_rel_actual.lstrip('/')
            gt_embed_path = os.path.abspath(os.path.join(gt_dataset_path, rel_gt_embed_path))
            
            if not os.path.isfile(gt_embed_path):
                raise FileNotFoundError(f"缺少 GT embedding 文件: {gt_embed_path}")
            
            if gt_embed_path.endswith(('.pt', '.pth')):
                gt_embed_dict = torch.load(gt_embed_path, map_location='cpu')
                if gt_embed_tensor_key is None:
                    default_keys = {'train': 'train', 'vali': 'vali', 'test': 'test'}
                    gt_embed_tensor_key = _config_lookup(config, f"{flag}_gt_embed_key", None) or default_keys.get(flag)
                
                if gt_embed_tensor_key not in gt_embed_dict:
                    raise KeyError(f"在 {gt_embed_path} 中未找到键 {gt_embed_tensor_key}")
                
                gt_split_data = gt_embed_dict[gt_embed_tensor_key]
                if not isinstance(gt_split_data, dict):
                    gt_split_data = {'embeddings': gt_split_data}
                
                # 加载 GT embeddings
                raw_gt_embeds = gt_split_data['embeddings']
                if isinstance(raw_gt_embeds, np.ndarray):
                    raw_gt_embeds = torch.from_numpy(raw_gt_embeds)
                raw_gt_embeds = raw_gt_embeds.detach().clone().cpu().type(torch.FloatTensor)
                
                # 加载 GT attention mask
                gt_attention_mask = None
                if 'attention_mask' in gt_split_data:
                    gt_attention_mask = gt_split_data['attention_mask']
                    if isinstance(gt_attention_mask, np.ndarray):
                        gt_attention_mask = torch.from_numpy(gt_attention_mask)
                    gt_attention_mask = gt_attention_mask.detach().clone().cpu()
                    if gt_attention_mask.dtype != torch.bool:
                        gt_attention_mask = gt_attention_mask.bool()
                
                # 处理 token 级别 embeddings（当前格式）
                if raw_gt_embeds.dim() == 3:
                    # token级别embeddings [N, L, D]
                    assert raw_gt_embeds.shape[0] == len(assignments), f"GT嵌入数量 {raw_gt_embeds.shape[0]} 与样本数 {len(assignments)} 不一致"
                    lagged_gt_embed_full = replay_lag_with_mapping(
                        [feat.clone() for feat in raw_gt_embeds],
                        assignments,
                        keep_mask,
                        value_type="tensor",
                    )
                    self.gt_embeddings = _finalize_tensor_sequence(lagged_gt_embed_full, keep_mask).to(raw_gt_embeds.dtype)
                    
                    if gt_attention_mask is not None:
                        assert gt_attention_mask.shape[0] == len(assignments), f"GT attention mask数量 {gt_attention_mask.shape[0]} 与样本数 {len(assignments)} 不一致"
                        lagged_gt_mask_full = replay_lag_with_mapping(
                            [mask.clone() for mask in gt_attention_mask],
                            assignments,
                            keep_mask,
                            value_type="tensor",
                        )
                        self.gt_attention_mask = _finalize_tensor_sequence(lagged_gt_mask_full, keep_mask).to(gt_attention_mask.dtype)
                    else:
                        self.logger.warning(f"GT token级别embeddings检测到但缺少attention_mask，可能影响损失计算")
                elif raw_gt_embeds.dim() == 2:
                    # 句子级别embeddings（兼容旧格式）
                    assert len(raw_gt_embeds) == len(assignments), "GT嵌入数量与样本数不一致"
                    lagged_gt_embed_full = replay_lag_with_mapping(
                        [feat.clone() for feat in raw_gt_embeds],
                        assignments,
                        keep_mask,
                        value_type="tensor",
                    )
                    self.gt_embeddings = _finalize_tensor_sequence(lagged_gt_embed_full, keep_mask).to(raw_gt_embeds.dtype)
                    
                    if gt_attention_mask is not None:
                        lagged_gt_mask_full = replay_lag_with_mapping(
                            [mask.clone() for mask in gt_attention_mask],
                            assignments,
                            keep_mask,
                            value_type="tensor",
                        )
                        self.gt_attention_mask = _finalize_tensor_sequence(lagged_gt_mask_full, keep_mask).to(gt_attention_mask.dtype)
                
                if self.gt_embeddings.shape[0] != len(valid_indices):
                    raise ValueError(f"滞后后的 GT 嵌入数量 {self.gt_embeddings.shape[0]} 与有效样本数 {len(valid_indices)} 不符")

        hidden_path_key = f"{flag}_news_hidden_file"
        hidden_rel_path = _config_lookup(config, hidden_path_key, None)
        if self.use_llm_hidden:
            if not hidden_rel_path:
                raise ValueError(f"配置缺少 {hidden_path_key}，无法加载预编码 LLM hidden")
            hidden_rel_actual = hidden_rel_path
            hidden_tensor_key = None
            hidden_mask_key = None
            if '::' in hidden_rel_path:
                parts = hidden_rel_path.split('::')
                hidden_rel_actual = parts[0]
                if len(parts) >= 2:
                    hidden_tensor_key = parts[1]
                if len(parts) >= 3:
                    hidden_mask_key = parts[2]
            hidden_path = os.path.abspath(os.path.join(dataset_path, hidden_rel_actual.lstrip('/')))
            if not os.path.isfile(hidden_path):
                raise FileNotFoundError(f"缺少预编码 LLM hidden 文件: {hidden_path}")

            hidden_dict = torch.load(hidden_path, map_location='cpu')
            if not isinstance(hidden_dict, dict):
                raise ValueError(f"预编码 LLM hidden 文件 {hidden_path} 需要是包含张量的字典")

            default_hidden_keys = {'train': 'train_hidden', 'vali': 'vali_hidden', 'test': 'test_hidden'}
            default_mask_keys = {'train': 'train_mask', 'vali': 'vali_mask', 'test': 'test_mask'}
            hidden_key = hidden_tensor_key or _config_lookup(config, f"{flag}_news_hidden_key", None) or default_hidden_keys.get(flag)
            mask_key = hidden_mask_key or _config_lookup(config, f"{flag}_news_hidden_mask_key", None) or default_mask_keys.get(flag)
            if hidden_key not in hidden_dict:
                raise KeyError(f"在 {hidden_path} 中未找到键 {hidden_key}")
            raw_hidden = hidden_dict[hidden_key]
            if isinstance(raw_hidden, np.ndarray):
                raw_hidden = torch.from_numpy(raw_hidden)
            raw_hidden = raw_hidden.detach().clone().cpu().type(torch.FloatTensor)

            if mask_key:
                if mask_key not in hidden_dict:
                    raise KeyError(f"在 {hidden_path} 中未找到 mask 键 {mask_key}")
                raw_mask = hidden_dict[mask_key]
                if isinstance(raw_mask, np.ndarray):
                    raw_mask = torch.from_numpy(raw_mask)
                raw_mask = raw_mask.detach().clone().cpu()
                if raw_mask.dtype != torch.bool:
                    raw_mask = raw_mask.bool()
            else:
                raw_mask = torch.ones(raw_hidden.shape[:2], dtype=torch.bool)

            if len(raw_hidden) != len(assignments):
                raise ValueError("预编码 LLM hidden 数量与样本数不一致")

            lagged_hidden_full = replay_lag_with_mapping(
                [feat.clone() for feat in raw_hidden],
                assignments,
                keep_mask,
                value_type="tensor",
            )
            self.news_hidden = _finalize_tensor_sequence(lagged_hidden_full, keep_mask).to(raw_hidden.dtype)

            lagged_mask_full = replay_lag_with_mapping(
                [mask.clone() for mask in raw_mask],
                assignments,
                keep_mask,
                value_type="tensor",
            )
            self.news_hidden_mask = _finalize_tensor_sequence(lagged_mask_full, keep_mask)
            if self.news_hidden.shape[0] != len(valid_indices):
                raise ValueError("滞后后的 LLM hidden 数量与有效样本数不符")
        else:
            self.news_hidden = None
            self.news_hidden_mask = None

        if lagged_news_filtered is not None:
            self.lagged_news = lagged_news_filtered
        else:
            self.lagged_news = None
        self.lag_stats = lag_stats
        self.lag_assignments = assignments

        self.samples = []
        for new_idx, sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            # 当 price_mode 设置为 "zero" 时，将价格序列置零以模拟纯新闻输入
            if self.price_mode == 'zero':
                hist_data = [0.0] * len(hist_data)

            gt_data = self._normalize_str(sample['ground_truth'])

            item = {
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'y': torch.tensor(gt_data, dtype=torch.float32)
            }

            if self.use_multimodal and self.lagged_news is not None:
                # 保存滞后后的新闻文本，供文本模型使用
                item['news'] = self.lagged_news[new_idx]

            if self.news_embeddings is not None:
                # 同步保存滞后后的新闻嵌入，供仅向量或混合模型使用
                item['news_embedding'] = self.news_embeddings[new_idx]
                # 修复：同时存储 attention_mask（如果存在）
                if self.news_attention_mask is not None:
                    item['news_attention_mask'] = self.news_attention_mask[new_idx]

            # 添加：存储 GT embeddings
            if self.gt_embeddings is not None:
                item['gt_embedding'] = self.gt_embeddings[new_idx]
                if self.gt_attention_mask is not None:
                    item['gt_attention_mask'] = self.gt_attention_mask[new_idx]

            if self.news_hidden is not None:
                item['news_hidden'] = self.news_hidden[new_idx]
                if self.news_hidden_mask is not None:
                    item['news_hidden_mask'] = self.news_hidden_mask[new_idx]

            self.samples.append(item)
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # print(f"Sample {idx} keys: {sample.keys()}")
   
        # 如果有 attention_mask 或 GT embedding，返回 dict 格式以便 collate_fn 处理
        if 'news_attention_mask' in sample or 'gt_embedding' in sample:
            return sample  # 返回完整 dict，让 collate_fn 处理
        
        if self.use_llm_hidden and ('news_hidden' in sample):
            return sample
        has_text = self.use_text_news and ('news' in sample)
        has_embed = self.use_news_embedding and ('news_embedding' in sample)
        if has_text and has_embed:
            # 返回顺序遵循 (x, 嵌入, 文本, y)，便于 collate 统一处理
            return sample['x'], sample['news_embedding'], sample['news'], sample['y']
        if has_embed:
            return sample['x'], sample['news_embedding'], sample['y']
        if has_text:
            return sample['x'], sample['news'], sample['y']
        return sample['x'], sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class ElectricityDataset(FnspidDataset):
    """Electricity 数据集（新格式），复用 Fnspid/Bitcoin 的多模态与滞后逻辑。"""

    def __init__(self, config, flag: str = "train", scaler=None):
        super().__init__(config, flag=flag, scaler=scaler)


class BitcoinDataset(FnspidDataset):
    """兼容新训练设置的 Bitcoin 数据集，复用 FnspidDataset 多模态/滞后逻辑。"""

    def __init__(self, config, flag: str = "train", scaler=None):
        super().__init__(config, flag=flag, scaler=scaler)


class Electricity_wo_disen(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']

        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]


        elif flag == "test":
            self.file_path = dataset_path+config["test_file"]


        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
   
            
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class Electricity_best(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])

            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean

class Bitcoin_best(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

            
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class Bitcoin_wo_disen(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])

            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean
    
class Environment_best(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

            
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class Environment_wo_disen(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])

            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean
    
class FNSPID_best(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

            
        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])
            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean


class FNSPID_wo_disen(Dataset):
    def __init__(self, config,flag="train",scaler=None):
        """
        加载多模态时序数据集 并进行归一化处理 
        """
   
        self.config = config 
        self.logger = getLogger()
        data_path = config['data_path']
        dataset_path =os.path.abspath(data_path+config['dataset'])
        dataset_name = os.path.basename(dataset_path).split('_')[0]
        dataset_path =os.path.abspath(data_path+dataset_name)
     
        if flag == "train":
            self.file_path = dataset_path+config['train_file']
            meta_domain_path = dataset_path+config["train_meta_file"]
            news_path = dataset_path+config["train_news_file"]
        elif flag == "vali":
            self.file_path = dataset_path+config["vali_file"]
            meta_domain_path = dataset_path+config["vali_meta_file"]
            news_path = dataset_path+config["vali_news_file"]

        elif flag == "test":
            meta_domain_path = dataset_path+config["test_meta_file"]
            self.file_path = dataset_path+config["test_file"]
            news_path = dataset_path+config["test_news_file"]

        with open(self.file_path, 'r') as f:
            self.data = json.load(f)
        if os.path.isfile(news_path):
            self.news_feats = torch.from_numpy(np.load(news_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.news_feats), "样本数与Embedding数量不匹配"
            lag_strength = int(_config_lookup(config, "news_lag_strength", _config_lookup(config, "news_lag", 0)) or 0)
            if lag_strength > 0:
                lagged_feats = _legacy_apply_news_lag([feat.clone() for feat in self.news_feats], config, value_type="tensor")
                self.news_feats = torch.stack(lagged_feats, dim=0).to(self.news_feats.dtype)
        if os.path.isfile(meta_domain_path):
            self.meta_feats = torch.from_numpy(np.load(meta_domain_path,allow_pickle=True)).type(torch.FloatTensor)
            assert len(self.data) == len(self.meta_feats), "样本数与Embedding数量不匹配"

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample['historical_data'].split(',')))
                all_data.extend(hist)
  
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)

            
                
            
        else:
            self.scaler = scaler 
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])
        
        

        self.samples = []
        for idx,sample in enumerate(self.data):
            hist_data = self._normalize_str(sample['historical_data'])
            
            gt_data = self._normalize_str(sample['ground_truth'])

            
            self.samples.append({
                'x': torch.tensor(hist_data, dtype=torch.float32),
                'meta_feats': self.meta_feats[idx],
                'news': self.news_feats[idx],
                'y': torch.tensor(gt_data, dtype=torch.float32)
            })
    
    
    def _normalize_str(self, data_str):

        values = np.array(list(map(float, data_str.split(',')))).reshape(-1, 1) 
        normalized = self.scaler.transform(values).flatten()
        
        return normalized.tolist()
    
    def get_scaler(self):
        """获取归一化参数"""
        return self.scaler
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample['x'],sample['meta_feats'],sample['news'] ,sample['y']
    
    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data,torch.Tensor):
            return normalized_data*self.std + self.mean
        return normalized_data * self.std + self.mean
    
