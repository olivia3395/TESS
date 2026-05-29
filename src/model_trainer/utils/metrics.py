import numpy as np
from typing import Dict, List, Optional, Sequence


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    

    return mae, mse, rmse, mape, mspe


METRIC_FUNC_MAP = {
    "mae": MAE,
    "mse": MSE,
    "rmse": RMSE,
    "mape": MAPE,
    "mspe": MSPE,
    "rse": RSE,
    "corr": CORR,
}


def compute_suffix_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    start_idx: int,
    *,
    fixed_len: Optional[int] = None,
    metric_keys: Sequence[str] = ("MAE", "RMSE", "MSE", "MAPE", "MSPE"),
) -> Dict[str, Dict[str, Sequence[float]]]:
    """计算窗口后缀的逐步与宏平均指标。

    参数均提供中文注释便于实验记录：
    - pred/true: 形状为 (样本数, 预测步数) 的数组。
    - start_idx: 评估起始下标，对应影响滞后的 \ell。
    - fixed_len: 若指定，仅截取固定长度窗口，便于等长对比。
    - metric_keys: 需要输出的指标列表，默认涵盖 MAE/RMSE/MSE/MAPE/MSPE。
    """
    if pred.shape != true.shape:
        raise ValueError("预测与真实标签的形状必须一致")
    if pred.ndim != 2:
        raise ValueError("compute_suffix_metrics 仅支持二维数组输入")
    horizon = pred.shape[1]
    if start_idx < 0 or start_idx >= horizon:
        raise ValueError("start_idx 超出预测范围")
    end_idx = horizon if fixed_len is None else min(horizon, start_idx + fixed_len)
    if end_idx <= start_idx:
        raise ValueError("固定窗口长度过短，无法计算后缀指标")

    suffix_pred = pred[:, start_idx:end_idx]
    suffix_true = true[:, start_idx:end_idx]
    step_count = suffix_pred.shape[1]

    per_step: Dict[str, List[float]] = {}
    macro: Dict[str, float] = {}

    for key in metric_keys:
        key_lower = key.lower()
        if key_lower not in METRIC_FUNC_MAP:
            raise ValueError(f"不支持的指标: {key}")
        metric_func = METRIC_FUNC_MAP[key_lower]
        step_values: List[float] = []
        for step in range(step_count):
            step_pred = suffix_pred[:, step]
            step_true = suffix_true[:, step]
            value = metric_func(step_pred, step_true)
            step_values.append(float(value))
        per_step[key.upper()] = step_values
        macro_value = metric_func(suffix_pred, suffix_true)
        macro[key.upper()] = float(macro_value)

    return {
        "per_step": per_step,
        "macro": macro,
        "context": {
            "start_idx": start_idx,
            "effective_steps": step_count,
            "fixed_len": fixed_len,
        },
    }
