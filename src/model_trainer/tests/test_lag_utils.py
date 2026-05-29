import numpy as np
import torch

from model_trainer.utils.lag_sampling import (
    apply_lag_to_sequence,
    parse_lag_policy,
    replay_lag_with_mapping,
)


def test_apply_lag_uniform_clamp_deterministic():
    """校验 clamp 策略在固定 seed 下具有确定性。"""
    config = {
        'news_lag': {
            'max_lag': 2,
            'min_lag': 0,
            'mode': 'uniform',
            'drop_border': False,
            'clamp_border': True,
            'seed': 42,
        }
    }
    seq = [f"news_{i}" for i in range(6)]
    policy = parse_lag_policy(config)
    lagged, keep_mask, stats = apply_lag_to_sequence(seq, policy, value_type='text')

    assert len(lagged) == len(seq)
    assert all(keep_mask)
    assert stats['drop_ratio'] == 0.0

    lagged_repeat, keep_mask_repeat, _ = apply_lag_to_sequence(seq, policy, value_type='text')
    assert lagged == lagged_repeat
    assert keep_mask == keep_mask_repeat


def test_apply_lag_drop_border():
    """启用 drop_border 时应出现样本丢弃。"""
    config = {
        'news_lag': {
            'max_lag': 3,
            'min_lag': 1,
            'mode': 'uniform',
            'drop_border': True,
            'clamp_border': False,
            'seed': 123,
        }
    }
    seq = [f"item_{i}" for i in range(5)]
    policy = parse_lag_policy(config)
    _, keep_mask, stats = apply_lag_to_sequence(seq, policy, value_type='text')

    assert len(keep_mask) == len(seq)
    assert stats['drop_ratio'] >= 0.0
    assert any(not flag for flag in keep_mask)


def test_replay_lag_mapping_tensor():
    """验证 replay_lag_with_mapping 对张量的映射与原滞后结果一致。"""
    config = {
        'news_lag': {
            'max_lag': 2,
            'mode': 'uniform',
            'drop_border': False,
            'clamp_border': True,
            'seed': 7,
        }
    }
    import torch

    seq = [torch.tensor([i, i + 1], dtype=torch.float32) for i in range(6)]
    policy = parse_lag_policy(config)
    lagged, keep_mask, stats = apply_lag_to_sequence(seq, policy, value_type='tensor')

    replayed = replay_lag_with_mapping(seq, stats['assignments'], keep_mask, value_type='tensor')
    kept_original = [lagged[i] for i, flag in enumerate(keep_mask) if flag]
    kept_replayed = [replayed[i] for i, flag in enumerate(keep_mask) if flag]

    assert len(kept_original) == len(kept_replayed)
    for a, b in zip(kept_original, kept_replayed):
        assert np.allclose(a.numpy(), b.numpy())
