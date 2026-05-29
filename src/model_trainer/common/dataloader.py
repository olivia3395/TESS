from model_trainer.common.dataset import (
    BitcoinDataset,
    Bitcoin_best,
    Bitcoin_wo_disen,
    ElectricityDataset,
    Electricity_best,
    Electricity_wo_disen,
    EnvironmentDataset,
    Environment_best,
    Environment_wo_disen,
    FNSPID_best,
    FNSPID_wo_disen,
    FnspidDataset,
    TrafficDataset,
)
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from logging import getLogger

from model_trainer.utils.embedding_checker import ensure_embeddings

def custom_collate_fn(batch):
    """将不同模态组合的样本统一组装成结构化批次。"""
    if not batch:
        raise ValueError("custom_collate_fn 收到空 batch，无法继续")

    x_list, y_list = [], []
    meta_tensors, meta_misc = [], []
    news_embed_tensors, news_texts, news_events = [], [], []
    news_attention_masks = []  # attention masks for token-level embeddings
    news_hidden_tensors, news_hidden_masks = [], []
    text_qualities = []  # 外部质量标签
    gt_embed_tensors = []  # GT embeddings for loss computation
    gt_attention_masks = []  # GT attention masks

    for sample in batch:
        if isinstance(sample, dict):
            x = sample.get('x')
            y = sample.get('y')
            if x is None or y is None:
                raise ValueError('字典样本缺少 x 或 y 字段')

            # 使用 is not None 检查，避免 Tensor 的布尔转换错误
            meta_val = sample.get('meta_tensor')
            if meta_val is None:
                meta_val = sample.get('meta_feats')
            if meta_val is None:
                meta_val = sample.get('meta')
            
            embed_val = sample.get('news_embedding')
            if embed_val is None:
                embed_val = sample.get('news_embed')
            
            embed_mask_val = sample.get('news_attention_mask')  # attention mask for embeddings
            hidden_val = sample.get('news_hidden')
            hidden_mask_val = sample.get('news_hidden_mask')
            
            news_text_val = sample.get('news')
            if news_text_val is None:
                news_text_val = sample.get('news_text')
            news_event_val = sample.get('news_events')
            text_quality_val = sample.get('text_quality')  # 外部质量标签

            x_list.append(torch.as_tensor(x))
            y_list.append(torch.as_tensor(y))

            if text_quality_val is not None:
                text_qualities.append(torch.as_tensor(text_quality_val))

            if meta_val is not None:
                if torch.is_tensor(meta_val):
                    meta_tensors.append(meta_val)
                else:
                    meta_misc.append(meta_val)

            if embed_val is not None:
                news_embed_tensors.append(torch.as_tensor(embed_val))
                if embed_mask_val is not None:
                    mask_tensor = torch.as_tensor(embed_mask_val)
                    if mask_tensor.dtype != torch.bool:
                        mask_tensor = mask_tensor.bool()
                    news_attention_masks.append(mask_tensor)
                else:
                    # 如果没有提供attention_mask，根据embeddings形状推断
                    embed_tensor = torch.as_tensor(embed_val)
                    if embed_tensor.dim() == 2:
                        # 句子级别，全部为有效
                        news_attention_masks.append(torch.ones(embed_tensor.shape[0], dtype=torch.bool))
                    elif embed_tensor.dim() == 3:
                        # token级别，全部为有效（可能需要后续处理padding）
                        news_attention_masks.append(torch.ones(embed_tensor.shape[:2], dtype=torch.bool))

            if hidden_val is not None:
                hidden_tensor = torch.as_tensor(hidden_val)
                news_hidden_tensors.append(hidden_tensor)
                if hidden_mask_val is not None:
                    mask_tensor = torch.as_tensor(hidden_mask_val)
                    if mask_tensor.dtype != torch.bool:
                        mask_tensor = mask_tensor.bool()
                else:
                    mask_tensor = torch.ones(hidden_tensor.shape[:2], dtype=torch.bool)
                news_hidden_masks.append(mask_tensor)

            if news_text_val is not None:
                news_texts.append(news_text_val)

            if news_event_val is not None:
                news_events.append(news_event_val)

            # 收集 GT embeddings
            gt_embed_val = sample.get('gt_embedding')
            gt_embed_mask_val = sample.get('gt_attention_mask')
            if gt_embed_val is not None:
                gt_embed_tensors.append(torch.as_tensor(gt_embed_val))
                if gt_embed_mask_val is not None:
                    gt_mask_tensor = torch.as_tensor(gt_embed_mask_val)
                    if gt_mask_tensor.dtype != torch.bool:
                        gt_mask_tensor = gt_mask_tensor.bool()
                    gt_attention_masks.append(gt_mask_tensor)
                else:
                    # 如果没有提供mask，根据embeddings形状推断
                    gt_embed_tensor = torch.as_tensor(gt_embed_val)
                    if gt_embed_tensor.dim() == 3:
                        gt_attention_masks.append(torch.ones(gt_embed_tensor.shape[:2], dtype=torch.bool))
            continue

        sample_len = len(sample)
        embed_val = None
        meta_val = None
        news_text_val = None
        news_event_val = None

        if sample_len == 4:
            x, second, third, y = sample
            if torch.is_tensor(second):
                embed_val = second
            else:
                meta_val = second
            if torch.is_tensor(third):
                embed_val = third if embed_val is None else embed_val
            elif isinstance(third, str):
                news_text_val = third
            else:
                news_event_val = third
        elif sample_len == 3:
            x, second, y = sample
            if torch.is_tensor(second):
                embed_val = second
            elif isinstance(second, str):
                news_text_val = second
            else:
                meta_val = second
        elif sample_len == 2:
            x, y = sample
        else:
            raise ValueError(f"不支持的样本长度: {sample_len}")

        x_list.append(x)
        y_list.append(y)

        if meta_val is not None:
            if torch.is_tensor(meta_val):
                meta_tensors.append(meta_val)
            else:
                meta_misc.append(meta_val)

        if embed_val is not None:
            news_embed_tensors.append(embed_val)

        if news_text_val is not None:
            news_texts.append(news_text_val)

        if news_event_val is not None:
            news_events.append(news_event_val)

    batch_dict = {
        'x': torch.stack(x_list),
        'y': torch.stack(y_list),
        'meta_tensor': torch.stack(meta_tensors) if meta_tensors else None,
        'meta_misc': meta_misc or None,
        'news_embed': torch.stack(news_embed_tensors) if news_embed_tensors else None,
        'news_attention_mask': torch.stack(news_attention_masks) if news_attention_masks else None,
        'news_text': news_texts or None,
        'news_events': news_events or None,
        'news_hidden': torch.stack(news_hidden_tensors) if news_hidden_tensors else None,
        'news_hidden_mask': torch.stack(news_hidden_masks) if news_hidden_masks else None,
        'text_quality': torch.stack(text_qualities) if text_qualities else None,  # 外部质量标签
        'gt_embed': torch.stack(gt_embed_tensors) if gt_embed_tensors else None,  # GT embeddings for loss
        'gt_attention_mask': torch.stack(gt_attention_masks) if gt_attention_masks else None,  # GT attention masks
    }

    return batch_dict

data_dict = {
    'Electricity':ElectricityDataset,
    'Bitcoin':BitcoinDataset,
    'Traffic':TrafficDataset,
    'Environment':EnvironmentDataset,
    'FNSPID':FnspidDataset,
    'Electricity_best': Electricity_best,
    'Electricity_wo_disen':Electricity_wo_disen,
    'Bitcoin_best': Bitcoin_best,
    'Bitcoin_wo_disen': Bitcoin_wo_disen,
    'Environment_best': Environment_best,
    'Environment_wo_disen': Environment_wo_disen,
    'FNSPID_best': FNSPID_best,
    'FNSPID_wo_disen': FNSPID_wo_disen,

}

def data_loader(config):
    logger = getLogger()
    config_dict = config.final_config_dict if hasattr(config, 'final_config_dict') else config
    legacy_loader = bool(config_dict.get('legacy_loader', False))
    if legacy_loader:
        logger.warning('Legacy loader enabled: skipping embedding auto-checks and using new collate pipeline.')
    else:
        ensure_embeddings(config_dict, logger=logger)
    data_class = data_dict[config['dataset']]
    train_dataset = data_class(config,flag="train")
    scaler = train_dataset.get_scaler()
    
    vali_dataset = data_class(config,flag="vali",scaler = scaler)
    test_dataset = data_class(config,flag="test",scaler = scaler)
    pin_memory = bool(config['use_gpu']) if 'use_gpu' in config else True

    distributed = bool(config_dict.get('distributed', False))
    rank = int(config_dict.get('rank', 0))
    world_size = int(config_dict.get('world_size', 1))

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
    # num_workers > 0 启用多进程数据加载，显著提升训练速度（20-40倍）
    # 之前在并发改造时被误设为 0，导致数据加载变为同步，GPU 大部分时间在等待
    # 默认值设为 4，平衡单脚本性能和 multi-script 并发时的 CPU 资源占用
    # 可根据实际 CPU 负载和数据加载复杂度进一步调整
    num_workers = config_dict.get('num_workers', 4) if not distributed else 0
    # 分布式训练时使用 num_workers=0 以避免进程间冲突
    # persistent_workers=True 保持 worker 进程存活，避免每个 epoch 重新启动 worker 导致的延迟
    # prefetch_factor=8 大幅增加预取 batch 数量，最大化减少 GPU 等待时间（默认是 2）
    train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=(train_sampler is None),
                collate_fn=custom_collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
                persistent_workers=(num_workers > 0),  # 仅在非分布式且 num_workers > 0 时启用
                prefetch_factor=8 if num_workers > 0 else None,  # 大幅增加预取数量，最大化数据加载速度
            )
    # validation 和 test 数据集通常较小，但仍可使用较多 worker 以提升速度
    # 但如果 num_workers=0，也会导致每个 epoch 开始时数据加载延迟
    valid_num_workers = min(num_workers, 4) if not distributed else 0  # 验证使用 4 个 worker
    test_num_workers = min(num_workers, 4) if not distributed else 0  # 测试使用 4 个 worker
    valid_loader = DataLoader(
            vali_dataset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            num_workers=valid_num_workers,
            persistent_workers=(valid_num_workers > 0),  # 保持 worker 进程存活
            prefetch_factor=4 if valid_num_workers > 0 else None,  # 增加预取以加速验证
        )
    test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate_fn,
            pin_memory=pin_memory,
            num_workers=test_num_workers,
            persistent_workers=(test_num_workers > 0),  # 保持 worker 进程存活
            prefetch_factor=4 if test_num_workers > 0 else None,  # 增加预取以加速测试
        )
    if (not distributed) or rank == 0:
        logger.info('\n====Current Dataset====\n'+str(config['dataset']))
        logger.info('\n====Training====\n' + str(len(train_dataset)))
        logger.info('\n====Validation====\n' + str(len(vali_dataset)))
        logger.info('\n====Testing====\n' + str(len(test_dataset)))
    return train_loader,valid_loader,test_loader
