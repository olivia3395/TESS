"""滞后采样与序列重排工具。

该模块提供通用的可用性滞后处理能力，包括：
1. 从配置中解析滞后策略；
2. 基于不同分布（均匀 / 截断几何）为每个样本采样滞后步数；
3. 按需对文本或向量序列执行滞后重排，同时统计滞后直方图与边界处理情况。

所有函数均默认使用中文注释，方便在实验记录或复现时快速理解处理逻辑。
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch


@dataclass
class LagPolicy:
    """封装滞后策略超参数，便于在不同组件之间传递。"""

    max_lag: int = 0
    min_lag: int = 0
    mode: str = "uniform"
    geometric_p: float = 0.5
    drop_border: bool = False
    clamp_border: bool = True
    seed: Optional[int] = None

    @property
    def enabled(self) -> bool:
        """辅助属性：判断是否真正启用滞后机制。"""
        return self.max_lag > 0


def config_lookup(config_obj: Any, key: str, default: Any = None) -> Any:
    """兼容不同配置对象类型的统一取值函数。"""
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    if hasattr(config_obj, "final_config_dict"):
        return config_obj.final_config_dict.get(key, default)
    if hasattr(config_obj, "get"):
        try:
            return config_obj.get(key, default)
        except TypeError:
            pass
    try:
        return config_obj[key]
    except Exception:
        return default


def parse_lag_policy(config_obj: Any, section: str = "news_lag") -> LagPolicy:
    """从配置中读取滞后相关超参数。

    参数
    -----
    config_obj: dict-like
        任意可索引的配置对象。
    section: str
        滞后参数所在的小节名称，默认为 `news_lag`。
    """
    section_cfg = {}
    raw_section = config_lookup(config_obj, section, None)
    if isinstance(raw_section, dict):
        section_cfg = raw_section
    elif isinstance(config_obj, dict) and section in config_obj and isinstance(config_obj[section], dict):
        section_cfg = config_obj[section]

    # 兼容旧版平铺式配置
    fallback_section = config_lookup(config_obj, "news_lag", {})
    max_lag = config_lookup(section_cfg, "max_lag", config_lookup(config_obj, "news_lag_strength", config_lookup(fallback_section, "max_lag", 0)))
    min_lag = config_lookup(section_cfg, "min_lag", config_lookup(config_obj, "news_lag_min", config_lookup(fallback_section, "min_lag", 0)))
    mode = config_lookup(section_cfg, "mode", config_lookup(config_obj, "news_lag_mode", config_lookup(fallback_section, "mode", "uniform")))
    geometric_p = float(config_lookup(section_cfg, "geometric_p", config_lookup(config_obj, "news_lag_geometric_p", config_lookup(fallback_section, "geometric_p", 0.5))))
    drop_border = bool(config_lookup(section_cfg, "drop_border", config_lookup(config_obj, "news_lag_drop_border", config_lookup(fallback_section, "drop_border", False))))
    clamp_border = bool(config_lookup(section_cfg, "clamp_border", config_lookup(config_obj, "news_lag_clamp_border", config_lookup(fallback_section, "clamp_border", True))))
    seed = config_lookup(section_cfg, "seed", config_lookup(config_obj, "news_lag_seed", config_lookup(fallback_section, "seed", config_lookup(config_obj, "seed", None))))

    # 清洗取值，确保安全边界
    max_lag = int(max(0, max_lag or 0))
    min_lag = int(max(0, min_lag or 0))
    if max_lag == 0:
        min_lag = 0
    else:
        min_lag = min(min_lag, max_lag)
    mode = str(mode or "uniform").lower()
    geometric_p = float(min(max(geometric_p, 1e-6), 1.0))
    if drop_border and clamp_border:
        # drop 与 clamp 互斥，若用户同时设置则默认优先 clamp 以保持数据完整性
        drop_border = False
    return LagPolicy(
        max_lag=max_lag,
        min_lag=min_lag,
        mode=mode,
        geometric_p=geometric_p,
        drop_border=drop_border,
        clamp_border=clamp_border,
        seed=seed,
    )


def _sample_offset(idx: int, policy: LagPolicy, rng: random.Random) -> Optional[int]:
    """根据策略为样本采样滞后步数。

    若需要丢弃样本则返回 ``None``。
    """
    if not policy.enabled:
        return 0

    max_shift = policy.max_lag
    if max_shift <= 0:
        return 0

    lower = min(policy.min_lag, max_shift)

    if policy.mode == "geometric":
        # 采样截断几何分布，概率质量函数 p(ℓ) ∝ (1-p)^ℓ * p
        probs = []
        norm = 0.0
        for lag in range(lower, max_shift + 1):
            prob = (1 - policy.geometric_p) ** lag * policy.geometric_p
            probs.append(prob)
            norm += prob
        # 归一化
        u = rng.random()
        cum = 0.0
        for offset, prob in zip(range(lower, max_shift + 1), probs):
            cum += prob / norm
            if u <= cum:
                return offset
        return max_shift

    # 默认均匀分布
    return rng.randint(lower, max_shift)


def _combine_bucket(bucket: List[Any], value_type: str) -> Any:
    """根据类型聚合同一窗口的多个新闻。"""
    if not bucket:
        return None
    if value_type == "text":
        return "\n".join(map(str, bucket))
    if value_type == "tensor":
        if len(bucket) == 1:
            return bucket[0]
        stacked = torch.stack(bucket, dim=0)
        return stacked.mean(dim=0)
    # 兜底：直接返回首个元素
    return bucket[0]


def apply_lag_to_sequence(
    sequence: Sequence[Any],
    policy: LagPolicy,
    *,
    value_type: str = "text",
    logger=None,
    log_prefix: str = "",
) -> Tuple[List[Any], List[bool], Dict[str, Any]]:
    """对给定序列施加滞后并返回重排后的结果及统计信息。

    返回
    -----
    lagged_sequence: List[Any]
        滞后后的序列，长度与原序列一致；若样本被丢弃则对应位置为 ``None``。
    keep_mask: List[bool]
        与序列等长的布尔掩码，标记哪些样本仍被保留。
    stats: Dict[str, Any]
        包含滞后直方图、丢弃比例、策略配置等信息，可用于实验日志。
    """
    seq_len = len(sequence)
    keep_mask = [True] * seq_len
    lagged_sequence: List[Any] = [None] * seq_len
    assignments: List[Optional[int]] = [None] * seq_len
    buckets: List[List[Any]] = [[] for _ in range(seq_len)]
    offsets = Counter()
    dropped_indices: List[int] = []

    rng = random.Random()
    if policy.seed is not None:
        rng.seed(policy.seed)

    for idx, item in enumerate(sequence):
        offset = _sample_offset(idx, policy, rng) if policy.enabled else 0
        target_idx = idx - offset
        if target_idx < 0:
            if policy.drop_border:
                keep_mask[idx] = False
                dropped_indices.append(idx)
                assignments[idx] = None
                continue
            if policy.clamp_border:
                target_idx = 0
            else:
                target_idx = idx
        assignments[idx] = target_idx
        offsets[offset] += 1
        if value_type == "tensor" and isinstance(item, torch.Tensor):
            buckets[target_idx].append(item.clone())
        else:
            buckets[target_idx].append(item)

    for idx in range(seq_len):
        if not keep_mask[idx]:
            continue
        if buckets[idx]:
            combined = _combine_bucket(buckets[idx], value_type)
        else:
            elem = sequence[idx]
            combined = elem.clone() if isinstance(elem, torch.Tensor) else elem
        lagged_sequence[idx] = combined

    stats = {
        "enabled": policy.enabled,
        "offset_hist": dict(offsets),
        "dropped_indices": dropped_indices,
        "drop_ratio": len(dropped_indices) / seq_len if seq_len > 0 else 0.0,
        "assignments": assignments,
        "policy": policy,
    }

    if logger is None:
        logger = getLogger()
    if policy.enabled:
        msg = f"[Lag] policy={policy}, kept={seq_len - len(dropped_indices)}, drop_ratio={stats['drop_ratio']:.3f}, offsets={dict(offsets)}"
        if log_prefix:
            msg = f"{log_prefix} {msg}"
        logger.info(msg)

    return lagged_sequence, keep_mask, stats


def filter_by_mask(sequence: Sequence[Any], keep_mask: Sequence[bool]) -> List[Any]:
    """按照掩码过滤序列，在 drop 策略下使用。"""
    return [item for item, keep in zip(sequence, keep_mask) if keep]


def replay_lag_with_mapping(
    sequence: Sequence[Any],
    assignments: Sequence[Optional[int]],
    keep_mask: Sequence[bool],
    *,
    value_type: str = "text",
) -> List[Any]:
    """复用既有映射对其他序列执行相同的滞后重排。"""
    seq_len = len(sequence)
    if len(assignments) != seq_len or len(keep_mask) != seq_len:
        raise ValueError("序列、映射与掩码长度必须一致")
    buckets: Dict[int, List[Any]] = {}
    for src_idx, target_idx in enumerate(assignments):
        if not keep_mask[src_idx]:
            continue
        if target_idx is None:
            raise ValueError("存在被保留但缺失目标索引的样本，无法重放滞后")
        buckets.setdefault(target_idx, [])
        item = sequence[src_idx]
        if value_type == "tensor" and isinstance(item, torch.Tensor):
            buckets[target_idx].append(item.clone())
        else:
            buckets[target_idx].append(item)
    lagged_sequence = [None] * seq_len
    for idx in range(seq_len):
        if not keep_mask[idx]:
            continue
        if idx in buckets and buckets[idx]:
            lagged_sequence[idx] = _combine_bucket(buckets[idx], value_type)
        else:
            elem = sequence[idx]
            lagged_sequence[idx] = elem.clone() if isinstance(elem, torch.Tensor) else elem
    return lagged_sequence
