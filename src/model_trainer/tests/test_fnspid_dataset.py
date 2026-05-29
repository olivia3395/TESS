import itertools
from pathlib import Path

import torch

from model_trainer.common.dataloader import data_loader
from model_trainer.common.dataset import FnspidDataset
from model_trainer.utils.configurator import Config


def _build_base_config():
    """构造针对 FNSPID 的基础配置，统一采用仓库内的数据路径。"""
    repo_root = Path(__file__).resolve().parents[3]
    data_path = repo_root / "dataset"
    config = Config(
        config_dict={
            'data_path': f"{data_path.as_posix()}/",
            # 显式指定启用多模态，确保同时返回文本与 embedding
            'use_multimodal': True,
            'use_text_news': True,
            'use_news_embedding': True,
        },
        model='PatchTST',
        dataset='FNSPID',
    )
    return config.final_config_dict


def test_fnspid_dataset_modalities():
    """验证 FNSPID 数据集单条样本是否同时提供价格、新闻 embedding 与新闻文本。"""
    cfg = _build_base_config()
    dataset = FnspidDataset(cfg, flag='train')
    sample = dataset[0]
    # 期望返回顺序：(price, embedding, text, target)
    assert len(sample) == 4

    price_series, news_embedding, news_text, target_series = sample
    assert isinstance(price_series, torch.Tensor)
    assert isinstance(news_embedding, torch.Tensor)
    assert isinstance(news_text, str)
    assert isinstance(target_series, torch.Tensor)

    assert price_series.ndim == 1
    assert news_embedding.ndim == 1
    assert target_series.ndim == 1
    assert price_series.shape[0] == cfg['seq_len']
    assert target_series.shape[0] == cfg['pred_len']
    assert news_embedding.shape[0] > 0  # embedding 维度应为正


def test_fnspid_dataloader_parallel_batch():
    """验证 data_loader 在并行 worker 设置下可正常批量聚合数据。"""
    cfg = _build_base_config()
    # 设置并行 worker，验证自定义聚合逻辑与 pin_memory/persistent_workers 兼容
    cfg_for_loader = dict(cfg)
    cfg_for_loader.update({
        'num_workers': 2,
        'valid_num_workers': 0,
        'test_num_workers': 0,
        'prefetch_factor': 2,
        'batch_size': 16,
    })

    train_loader, _, _ = data_loader(cfg_for_loader)
    try:
        batch_iter = iter(train_loader)
        batch = next(batch_iter)
    finally:
        # 主动清理 DataLoader，避免 pytest 结束时存在存活的 worker
        if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
            train_loader._iterator._shutdown_workers()  # type: ignore[attr-defined]

    price_batch, embedding_batch, news_batch, target_batch = batch

    # 批次维度应与 batch_size 对齐（或因尾批略小）
    batch_size = price_batch.shape[0]
    assert 0 < batch_size <= cfg_for_loader['batch_size']
    assert embedding_batch.shape[0] == batch_size
    assert target_batch.shape[0] == batch_size

    # price/target 为张量，embedding 为张量，news 为字符串列表
    assert price_batch.ndim == 2  # (batch, seq_len)
    assert embedding_batch.ndim == 2
    assert target_batch.ndim == 2
    assert isinstance(news_batch, list)
    assert all(isinstance(item, str) for item in itertools.islice(news_batch, batch_size))

    # 校验价格与目标序列长度
    assert price_batch.shape[1] == cfg['seq_len']
    assert target_batch.shape[1] == cfg['pred_len']
