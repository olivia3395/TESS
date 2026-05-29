import copy
import torch
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from time import time
from logging import getLogger
from typing import Any, Dict, List
import torch.optim as optim
import inspect
from torch.utils.data.distributed import DistributedSampler
from model_trainer.utils.evaluator import TemporalEvaluator
import numpy as np  
from model_trainer.utils.utils import get_local_time, early_stopping, dict2str
import torch.nn as nn
from model_trainer.utils.utils import EarlyStopping, adjust_learning_rate
from model_trainer.utils.metrics import metric
from model_trainer.utils.artifact_manager import ArtifactManager
import os 
class Trainer:
    def __init__(self, model,config):
        self.logger = getLogger()
        self.config = config
        
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric']
        self.model = model
        self.device = config["device"]
        self.inverse = False

        self.distributed = bool(config.get('distributed', False)) and dist.is_available() and dist.is_initialized()
        self.rank = int(config.get('rank', 0)) if self.distributed else 0
        self.world_size = int(config.get('world_size', 1)) if self.distributed else 1
        self.is_main = (not self.distributed) or self.rank == 0

        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.req_training = config['req_training']
        self.start_epoch = 0
        self.cur_step = 0
        
        self.best_valid_score = float('inf')
        self.train_loss_dict = dict()
        self.best_state_dict = None
        self.best_metrics = None
        self.best_epoch = None

        self.optimizer = self._build_optimizer()

        self.evaluator = TemporalEvaluator(config)
        self.patience = config['patience']
        self.path = './checkpoints/'
        self.use_multimodal = bool(config['use_multimodal']) if 'use_multimodal' in config else False
        self.artifacts = ArtifactManager(config) if self.is_main else None
        
        
        
    def _prepare_batch(self, batch_data):
        """兼容新旧 collate 输出，统一整理成字典。"""
        if isinstance(batch_data, dict):
            # 优化：.to() 和 .float() 是幂等的，不会对已正确类型/设备上的张量重复操作
            batch_x = batch_data['x'].float().to(self.device)
            batch_y = batch_data['y'].float().to(self.device)

            meta_tensor = batch_data.get('meta_tensor')
            if isinstance(meta_tensor, torch.Tensor):
                meta_tensor = meta_tensor.float().to(self.device)

            news_embed = batch_data.get('news_embed')
            if isinstance(news_embed, torch.Tensor):
                news_embed = news_embed.float().to(self.device)

            news_hidden = batch_data.get('news_hidden')
            if isinstance(news_hidden, torch.Tensor):
                news_hidden = news_hidden.to(self.device)

            news_hidden_mask = batch_data.get('news_hidden_mask')
            if isinstance(news_hidden_mask, torch.Tensor):
                news_hidden_mask = news_hidden_mask.to(self.device)

            meta_misc = batch_data.get('meta_misc')
            meta = meta_tensor if meta_tensor is not None else meta_misc

            news_text = batch_data.get('news_text')
            news_events = batch_data.get('news_events')

            # 处理外部质量标签
            text_quality = batch_data.get('text_quality')
            if isinstance(text_quality, torch.Tensor):
                text_quality = text_quality.float().to(self.device)

            # 处理 GT embedding
            gt_embed = batch_data.get('gt_embed')
            if isinstance(gt_embed, torch.Tensor):
                gt_embed = gt_embed.float().to(self.device)

            gt_attention_mask = batch_data.get('gt_attention_mask')
            if isinstance(gt_attention_mask, torch.Tensor):
                gt_attention_mask = gt_attention_mask.to(self.device)

            news = None

            if news_text:
                news = news_text
            elif news_embed is not None:
                news = news_embed
            elif news_events:
                news = news_events


            return {
                'x': batch_x,
                'y': batch_y,
                'meta': meta,
                'meta_tensor': meta_tensor,
                'news': news,
                'news_text': news_text,
                'news_embed': news_embed,
                'news_events': news_events,
                'news_hidden': news_hidden,
                'news_hidden_mask': news_hidden_mask,
                'text_quality': text_quality,  # 外部质量标签
                'gt_embed': gt_embed,  # GT embeddings for loss
                'gt_attention_mask': gt_attention_mask,  # GT attention masks
                'batch_size': batch_x.size(0)
            }

        # 兼容旧版 tuple/list 模式
        if isinstance(batch_data, (tuple, list)):

            sample_len = len(batch_data)
            if sample_len == 4:
                batch_x, batch_meta, batch_news, batch_y = batch_data
            elif sample_len == 3:
                batch_x, batch_news, batch_y = batch_data
                batch_meta = None
            elif sample_len == 2:
                batch_x, batch_y = batch_data
                batch_meta = None
                batch_news = None
            else:
                raise ValueError(f"无法解析长度为 {sample_len} 的 batch 数据")

            # 优化：直接使用 .to() 和 .float()，PyTorch 会处理已正确类型/设备的情况
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            meta_tensor = batch_meta.float().to(self.device) if isinstance(batch_meta, torch.Tensor) else None
            news_embed = batch_news.float().to(self.device) if isinstance(batch_news, torch.Tensor) else None
            news_text = batch_news if isinstance(batch_news, list) and batch_news and isinstance(batch_news[0], str) else None

            return {
                'x': batch_x,
                'y': batch_y,
                'meta': meta_tensor if meta_tensor is not None else batch_meta,
                'meta_tensor': meta_tensor,
                'news': batch_news,
                'news_text': news_text,
                'news_embed': news_embed,
                'news_events': None,
                'news_hidden': None,
                'news_hidden_mask': None,
                'batch_size': batch_x.size(0)
            }

        raise TypeError(f"不支持的 batch_data 类型: {type(batch_data)}")

    def _compute_outputs(self, batch: Dict[str, torch.Tensor], mode: str):
        batch_x = batch['x']
        batch_y = batch['y']
        batch_meta_tensor = batch.get('meta_tensor')
        batch_news_text = batch.get('news_text')
        batch_news_embed = batch.get('news_embed')
        batch_news_hidden = batch.get('news_hidden')
        batch_news_hidden_mask = batch.get('news_hidden_mask')
        batch_size = batch.get('batch_size', batch_x.size(0))
        
  

        news_feat = batch_news_embed if batch_news_embed is not None else batch_meta_tensor
        model_name = self.model.__class__.__name__

    

        if isinstance(batch_news_hidden, torch.Tensor):
            batch_news_hidden = batch_news_hidden.to(self.device)
        if isinstance(batch_news_hidden_mask, torch.Tensor):
            batch_news_hidden_mask = batch_news_hidden_mask.to(self.device)

        if model_name == 'TimeLLM':
            news_inputs = batch_news_text if batch_news_text is not None else [""] * batch_size
            outputs = self.model(batch_x, news=news_inputs, flag=mode)
        elif model_name == 'MultiModal_Baseline_SelfAttention':
            if batch_news_text is None and batch_news_hidden is None:
                raise ValueError("MultiModal_Baseline_SelfAttention 模型需要新闻文本输入，但批次缺少 news_text")
            news_inputs = batch_news_text if batch_news_text is not None else [""] * batch_size
            outputs = self.model(batch_x, news_inputs, news_hidden=batch_news_hidden, news_mask=batch_news_hidden_mask, flag=mode)
        elif model_name == 'MultiModal_Baseline_Token_Level':
            news_inputs = batch_news_text if batch_news_text is not None else [""] * batch_size
            outputs = self.model(batch_x, news_inputs, news_hidden=batch_news_hidden, news_mask=batch_news_hidden_mask, flag=mode)
        elif model_name == 'CMA':
            if news_feat is None:
                raise ValueError("CMA 模型需要新闻 embedding，但批次缺少相关张量")
            outputs = self.model(batch_x, news_feat, flag=mode)
        elif model_name == 'MultiModal_Baseline':

            if news_feat is None:
                raise ValueError("MultiModal_Baseline 模型需要新闻 embedding，但批次缺少相关张量")
            if 'flag' in self.model.forward.__code__.co_varnames:
                outputs = self.model(batch_x, news_feat, flag=mode)
            else:
                outputs = self.model(batch_x, news_feat)
        elif model_name == 'MultiModal_Baseline_QualityAware':

            if news_feat is None:
                raise ValueError("MultiModal_Baseline_QualityAware 模型需要新闻 embedding，但批次缺少相关张量")

            # 检查是否有外部质量标签
            batch_text_quality = batch.get('text_quality')
            if batch_text_quality is not None and mode == 'train':
                # 训练时传递质量标签
                outputs = self.model(batch_x, news_feat, text_quality_gt=batch_text_quality, flag=mode)
            else:
                # 评估时不传递质量标签，只返回预测损失
                outputs = self.model(batch_x, news_feat, flag=mode)
        elif model_name == 'MultiModal_Baseline_Dynamic_Dropout':
            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Dynamic_Dropout 模型需要新闻 embedding，但批次缺少相关张量")
            # 获取attention mask用于过滤padding tokens
            attention_mask = batch.get('news_attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(batch_x, news_feat, flag=mode, attention_mask=attention_mask)
        elif model_name == 'MultiModal_Baseline_Dynamic_Dropout_BERT':
            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Dynamic_Dropout_BERT 模型需要新闻 embedding，但批次缺少相关张量")
            # 获取attention mask用于过滤padding tokens
            attention_mask = batch.get('news_attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(batch_x, news_feat, flag=mode, attention_mask=attention_mask)
        elif model_name == 'MultiModal_Baseline_Dynamic_Dropout_BERT_Hard':
            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Dynamic_Dropout_BERT_Hard 模型需要新闻 embedding，但批次缺少相关张量")
            # 获取attention mask用于过滤padding tokens
            attention_mask = batch.get('news_attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(batch_x, news_feat, flag=mode, attention_mask=attention_mask)
        elif model_name == 'MultiModal_Baseline_Dynamic_Dropout_MaxPool_Hard':
            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Dynamic_Dropout_MaxPool_Hard 模型需要新闻 embedding，但批次缺少相关张量")
            # 获取attention mask用于过滤padding tokens
            attention_mask = batch.get('news_attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(batch_x, news_feat, flag=mode, attention_mask=attention_mask)
        elif model_name == 'MultiModal_Baseline_Dynamic_Dropout_Hard':
            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Dynamic_Dropout_Hard 模型需要新闻 embedding，但批次缺少相关张量")
            # 获取attention mask用于过滤padding tokens
            attention_mask = batch.get('news_attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            outputs = self.model(batch_x, news_feat, flag=mode, attention_mask=attention_mask)
        elif model_name == 'MultiModal_Baseline_Token_Level':

            if news_feat is None:
                raise ValueError("MultiModal_Baseline_Token_Level 模型需要新闻 embedding，但批次缺少相关张量")
            outputs = self.model(batch_x, batch_news_text, flag=mode)
        elif self.use_multimodal:
            # if isinstance(news_feat,torch.Tensor):
            #     print("news_feat isinstance(torch.Tensor)")
            outputs=self.model(batch_x, news_feat, flag=mode)
            # outputs = self.model(batch_x, batch_news_text, flag=mode)
            # outputs = self.model(batch_x, flag=mode) if 'flag' in self.model.forward.__code__.co_varnames else self.model(batch_x)
        else:
            outputs = self.model(batch_x)
            

        return outputs


    def _config_snapshot(self) -> Dict:
        snapshot = {}
        for key, value in self.config.final_config_dict.items():
            if isinstance(value, torch.device):
                snapshot[key] = str(value)
            else:
                snapshot[key] = value
        return snapshot

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
   
    def train_epoch(self, train_loader, epoch_idx, loss_func=None):
        """单个训练步骤"""
        if not self.req_training:
            return 0.0, []

        train_loss = []
        running_avg_loss = 0.0  # 使用运行平均值，避免每次计算 sum()
        with tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx + 1}/{self.epochs}",
            leave=False,
            disable=not self.is_main,
        ) as progress:
            for batch_idx, batch_data in enumerate(progress):
                self.optimizer.zero_grad()

                batch = self._prepare_batch(batch_data)
                outputs = self._compute_outputs(batch, mode='train')
                batch_y = batch['y']

                if hasattr(self.model, 'calculate_loss'):
                    # 为质量感知模型传递text_quality_gt（如果有的话）
                    text_quality_gt = batch.get('text_quality')
                    gt_embeddings = batch.get('gt_embed')  # 获取GT embeddings

                    # 检查模型的calculate_loss方法支持哪些参数
                    calculate_loss_sig = inspect.signature(self.model.calculate_loss)
                    param_names = set(calculate_loss_sig.parameters.keys())

                    # 动态构建参数字典
                    loss_kwargs = {'batch_y': batch_y}

                    if 'gt_embeddings' in param_names and gt_embeddings is not None:
                        loss_kwargs['gt_embeddings'] = gt_embeddings
                    if 'text_quality_gt' in param_names and text_quality_gt is not None:
                        loss_kwargs['text_quality_gt'] = text_quality_gt
                    if 'mode' in param_names:
                        loss_kwargs['mode'] = 'train'

                    loss = self.model.calculate_loss(**loss_kwargs)
                else:
                    loss = loss_func(outputs, batch_y)

                loss_value = loss.item()
                train_loss.append(loss_value)
                loss.backward()
                self.optimizer.step()

                if self.is_main:
                    # 使用增量平均值计算，避免每次 O(n) 的 sum() 操作
                    running_avg_loss = (running_avg_loss * batch_idx + loss_value) / (batch_idx + 1)
                    progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{running_avg_loss:.4f}")

        train_loss = np.average(train_loss) if train_loss else 0.0

        return train_loss
        
    def vali(self, vali_loader, loss_func):
        total_loss = []
        if vali_loader is None:
            return float('inf')
        self.model.eval()
        with torch.no_grad():
            with tqdm(
                vali_loader,
                desc="Validation",
                leave=False,
                disable=not self.is_main,
            ) as progress:
                for batch_idx, batch_data in enumerate(progress):
                    batch = self._prepare_batch(batch_data)
                    outputs = self._compute_outputs(batch, mode='test')
                    batch_y = batch['y']
                    # 在 GPU 上计算 loss，避免不必要的 CPU-GPU 数据传输
                    loss = loss_func(outputs, batch_y)
                    total_loss.append(float(loss.item()))

        total_loss = np.average(total_loss) if total_loss else 0.0
        self.model.train()
        return total_loss
 
    def _evaluate_split(self, loader, split: str, return_raw: bool = False, include_details: bool = False):
        preds = []
        trues = []
        dataset = loader.dataset
        hist_samples: List[np.ndarray] = []
        news_samples: List[Any] = []

        self.model.eval()
        with torch.no_grad():
            with tqdm(
                loader,
                desc=f"{split.capitalize()} Eval",
                leave=False,
                disable=not self.is_main,
            ) as progress:
                for batch_idx, batch_data in enumerate(progress):
                    batch = self._prepare_batch(batch_data)
                    outputs = self._compute_outputs(batch, mode='test')
                    batch_y = batch['y']
                    batch_x = batch['x']
                    outputs = outputs.float().detach().cpu().numpy()
                    batch_y_np = batch_y.float().detach().cpu().numpy()
                    if self.inverse:
                        pred = dataset.inverse_transform(outputs)
                        gt = dataset.inverse_transform(batch_y_np)
                    else:
                        pred = outputs
                        gt = batch_y_np

                    preds.append(pred)
                    trues.append(gt)
                    if include_details:
                        hist_batch = batch_x.float().detach().cpu().numpy()
                        hist_samples.extend(list(hist_batch))
                        news_batch = self._extract_news_batch(batch.get('news'), pred.shape[0])
                        news_samples.extend(news_batch)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # if self.is_main:
        #     print(f'{split} shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "MSPE": mspe,
        }

        if return_raw and include_details:
            return metrics_dict, preds, trues, hist_samples, news_samples
        if return_raw:
            return metrics_dict, preds, trues
        return metrics_dict

    def _extract_news_batch(self, news, batch_size: int) -> List[Any]:
        if isinstance(news, torch.Tensor):
            news_np = news.float().detach().cpu().numpy()
            items = [news_np[idx] for idx in range(min(news_np.shape[0], batch_size))]
            if len(items) < batch_size:
                items.extend([None] * (batch_size - len(items)))
            return items
        if isinstance(news, np.ndarray):
            items = [news[idx] for idx in range(min(news.shape[0], batch_size))]
            if len(items) < batch_size:
                items.extend([None] * (batch_size - len(items)))
            return items
        if isinstance(news, list):
            if len(news) == batch_size:
                return list(news)
            return [news] * batch_size
        if news is None:
            return [None] * batch_size
        return [news] * batch_size

    @staticmethod
    def _to_serializable(value: Any):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [Trainer._to_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [Trainer._to_serializable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _build_sample_records(
        self,
        split: str,
        preds: np.ndarray,
        trues: np.ndarray,
        hist_samples: List[np.ndarray],
        news_samples: List[Any],
    ) -> List[Dict[str, Any]]:
        total = len(hist_samples)
        records: List[Dict[str, Any]] = []
        for idx in range(total):
            record = {
                'sample_id': f'{split}_{idx}',
                'hist_data': self._to_serializable(hist_samples[idx]),
                'ground_truth': self._to_serializable(trues[idx]),
                'prediction': self._to_serializable(preds[idx]),
                'news': self._to_serializable(news_samples[idx]) if idx < len(news_samples) else None,
            }
            records.append(record)
        return records

    def test(self, test_loader, return_raw: bool = False):
        """评估模型，必要时返回逐样本预测用于滞后分析。"""
        return self._evaluate_split(test_loader, 'test', return_raw=return_raw, include_details=return_raw)

    

    def fit(self, train_loader, valid_loader, test_loader=None, saved=False, verbose=True):
        """执行完整训练流程"""

        os.makedirs(self.path, exist_ok=True)

        metrics_dict = None

        for epoch_idx in range(self.start_epoch, self.epochs):
            if self.distributed:
                sampler = getattr(train_loader, 'sampler', None)
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch_idx)

            self.model.train()
            loss_func = nn.MSELoss()

            train_loss = self.train_epoch(train_loader, epoch_idx, loss_func=loss_func)
            if self.distributed:
                loss_tensor = torch.tensor([train_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                train_loss = loss_tensor.item()

            if self.artifacts:
                self.artifacts.write_epoch_metrics(epoch_idx, {'loss': float(train_loss)}, 'train')

            vali_score = float('inf')
            if valid_loader is not None:
                valid_start_time = time()
                # 在分布式训练中，所有进程都需要参与验证步骤的同步
                vali_score = self.vali(valid_loader, loss_func)
                if self.artifacts and self.is_main:
                    self.artifacts.write_epoch_metrics(epoch_idx, {'loss': float(vali_score)}, 'vali')
                valid_end_time = time()
                if self.is_main:
                    valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % (
                        epoch_idx,
                        valid_end_time - valid_start_time,
                        vali_score,
                    )
                    self.logger.info(valid_score_output)

            # 确保在分布式训练中所有进程都获得相同的vali_score
            if self.distributed:
                vali_tensor = torch.tensor([vali_score], device=self.device)
                dist.broadcast(vali_tensor, src=0)
                vali_score = vali_tensor.item()

            self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                vali_score,
                self.best_valid_score,
                self.cur_step,
                max_step=self.patience,
            )

            adjust_learning_rate(self.optimizer, epoch_idx + 1, self.learning_rate)

            metrics_dict = None
            # 只在 eval_step 指定的 epoch 评估 test 集，提高训练速度
            if test_loader is not None and self.is_main and (epoch_idx + 1) % self.eval_step == 0:
                metrics_dict = self.test(test_loader=test_loader)
                self.logger.info(
                    '\n Current Epoch {} Test Result: MAE: {}, MSE: {}, RMSE: {}, MAPE: {}, MSPE: {}'.format(
                        epoch_idx,
                        metrics_dict['MAE'],
                        metrics_dict['MSE'],
                        metrics_dict['RMSE'],
                        metrics_dict['MAPE'],
                        metrics_dict['MSPE'],
                    )
                )

            if update_flag and self.is_main:
                update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                self.logger.info(update_output)
                metrics_payload = metrics_dict or {}
                self.best_test_upon_valid = metrics_payload
                model_to_save = self.model.module if self.distributed else self.model
                self.best_state_dict = {k: v.detach().cpu() for k, v in model_to_save.state_dict().items()}
                self.best_metrics = metrics_payload if metrics_payload else None
                self.best_epoch = epoch_idx + 1

                should_promote = (
                    metrics_dict is not None
                    and self.artifacts is not None
                    and self.artifacts.should_promote(vali_score, metrics_dict)
                )
                if should_promote:
                    split_loaders = [
                        ('train', train_loader),
                        ('vali', valid_loader),
                        ('test', test_loader),
                    ]
                    split_metrics = {}
                    split_sample_records = {}
                    for split_name, loader in split_loaders:
                        if loader is None:
                            continue
                        metrics_split, preds_split, trues_split, hist_samples, news_samples = self._evaluate_split(
                            loader,
                            split_name,
                            return_raw=True,
                            include_details=True,
                        )
                        split_metrics[split_name] = metrics_split
                        split_sample_records[split_name] = self._build_sample_records(
                            split_name,
                            preds_split,
                            trues_split,
                            hist_samples,
                            news_samples,
                        )

                    if self.artifacts:
                        self.artifacts.save_best_model(model_to_save)
                        self.artifacts.save_config_snapshot(
                            self._config_snapshot(),
                            best_epoch=self.best_epoch,
                            valid_score=vali_score,
                            test_metrics=metrics_dict,
                        )
                        self.artifacts.write_epoch_metrics(epoch_idx, metrics_dict, 'test')

                        for split_name, records in split_sample_records.items():
                            self.artifacts.save_split_samples(split_name, records)
                            self.artifacts.write_epoch_metrics('best', split_metrics[split_name], split_name)

            if stop_flag:
                if self.is_main:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % (
                        epoch_idx - self.cur_step * self.eval_step
                    )
                    self.logger.info(stop_output)
                break

        if self.is_main and self.best_state_dict is not None:
            model_to_load = self.model.module if self.distributed else self.model
            model_to_load.load_state_dict(self.best_state_dict)

        if self.is_main and self.artifacts and self.artifacts.export_samples:
            split_loaders = [
                ('train', train_loader),
                ('vali', valid_loader),
                ('test', test_loader),
            ]
            for split, loader in split_loaders:
                if loader is None:
                    continue
                metrics_dict, preds, trues = self._evaluate_split(loader, split, return_raw=True)
                self.artifacts.write_sample_scores(split, preds, trues)
                self.artifacts.write_epoch_metrics('best', metrics_dict, split)

        if self.is_main and self.artifacts:
            if self.best_metrics is not None:
                self.artifacts.manifest['best_metrics'] = self.artifacts._sanitize_metrics(self.best_metrics)
            self.artifacts.write_manifest()

        if self.best_test_upon_valid is None and metrics_dict is not None:
            self.best_test_upon_valid = metrics_dict

        return self.best_valid_score, self.best_test_upon_valid
                # self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
                # training_end_time = time()
                # train_loss_output = \
                #     self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
             
                # if verbose:
                #     self.logger.info(train_loss_output)
                # if (epoch_idx + 1) % self.eval_step == 0:
                #     valid_start_time = time()
                #     valid_score, valid_result = self._valid_epoch(valid_loader)
                #     self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                #         valid_score, self.best_valid_score, self.cur_step,
                #         max_step=self.stopping_step)
                #     valid_end_time = time()
                #     valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                #                         (epoch_idx, valid_end_time - valid_start_time, valid_score)
                #     valid_result_output = 'valid result: \n' + dict2str(valid_result)
                #     # test
                #     _, test_result = self._valid_epoch(test_loader)
                #     if verbose:
                #         self.logger.info(valid_score_output)
                #         self.logger.info(valid_result_output)
                #         self.logger.info('test result: \n' + dict2str(test_result))
                #     if update_flag:
                     
                        
                #         update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                #         if verbose:
                #             self.logger.info(update_output)
                #         self.best_valid_result = valid_result
                #         self.best_test_upon_valid = test_result

                #     if stop_flag:
                #         stop_output = '+++++Finished training, best eval result in epoch %d' % \
                #                     (epoch_idx - self.cur_step * self.eval_step)
                #         if verbose:
                #             self.logger.info(stop_output)
                #         break


    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'
