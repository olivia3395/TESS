"""Audit late-additive primitive gate experiments.

This script is intentionally read-only for trained runs: it loads existing
checkpoints, evaluates one split, and writes compact diagnostic summaries. It
does not train models and does not implement primitive-wise contribution P6.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tess_basic import ScaledTESSDataset, model_kwargs
from models.simple_tess import build_model
from primitive_inference_rc2.tess_dataset import DEFAULT_PRIMITIVE_ORDER, TESS_Dataset


EXPERIMENTS = (
    "additive_text_scale0",
    "additive_text_scale1",
    "additive_gt_scale1",
    "additive_text_gate_scale1",
    "additive_gt_gate_scale1",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_dataset_root(data_root: str | Path) -> Path:
    """Map requested data path conventions to the repo root expected by TESS_Dataset."""
    path = Path(data_root).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if path.name == "dataset":
        return path.parent
    if (path / "dataset").exists():
        return path
    return path


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.asarray(mask, dtype=bool)]


def safe_mean(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return finite_or_none(float(values.mean()))


def safe_std(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return None
    return finite_or_none(float(values.std()))


def describe(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "p99": None,
        }
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "mean": finite_or_none(float(values.mean())),
        "std": finite_or_none(float(values.std())),
        "min": finite_or_none(float(values.min())),
        "max": finite_or_none(float(values.max())),
        "p01": finite_or_none(float(quantiles[0])),
        "p05": finite_or_none(float(quantiles[1])),
        "p25": finite_or_none(float(quantiles[2])),
        "p50": finite_or_none(float(quantiles[3])),
        "p75": finite_or_none(float(quantiles[4])),
        "p95": finite_or_none(float(quantiles[5])),
        "p99": finite_or_none(float(quantiles[6])),
    }


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return finite_or_none(float(np.corrcoef(x, y)[0, 1]))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def roc_auc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid] > 0.5
    pos = int(targets.sum())
    neg = int((~targets).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = average_ranks(scores)
    pos_rank_sum = float(ranks[targets].sum())
    auc = (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return finite_or_none(float(auc))


def bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> float | None:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    valid = np.isfinite(logits) & np.isfinite(targets)
    logits = logits[valid]
    targets = targets[valid]
    if logits.size == 0:
        return None
    losses = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return finite_or_none(float(losses.mean()))


def sample_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(axis=1)


def primitive_metrics(
    values: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    logits: np.ndarray | None = None,
) -> dict[str, Any]:
    values_flat = masked_values(values, mask)
    targets_flat = masked_values(targets, mask)
    logits_flat = masked_values(logits, mask) if logits is not None else None
    pos = targets_flat > 0.5
    neg = ~pos
    result: dict[str, Any] = {
        "target_positive_rate": safe_mean(targets_flat),
        "positive_mean_gate_weight": safe_mean(values_flat[pos]),
        "negative_mean_gate_weight": safe_mean(values_flat[neg]),
        "pos_neg_difference": None,
        "pearson": pearson(values_flat, targets_flat),
        "spearman": spearman(values_flat, targets_flat),
        "roc_auc": roc_auc(values_flat, targets_flat),
    }
    if result["positive_mean_gate_weight"] is not None and result["negative_mean_gate_weight"] is not None:
        result["pos_neg_difference"] = (
            result["positive_mean_gate_weight"] - result["negative_mean_gate_weight"]
        )
    if logits_flat is not None:
        result["bce_with_logits"] = bce_with_logits(logits_flat, targets_flat)
    return result


def margin_metrics(margins: np.ndarray, targets: np.ndarray, mask: np.ndarray, bins: int) -> dict[str, Any]:
    margins_flat = masked_values(margins, mask)
    targets_flat = masked_values(targets, mask)
    correct = targets_flat > 0.5
    incorrect = ~correct
    bucket_rows = []
    if margins_flat.size:
        quantiles = np.quantile(margins_flat, np.linspace(0.0, 1.0, bins + 1))
        for idx in range(bins):
            lo = quantiles[idx]
            hi = quantiles[idx + 1]
            if idx == bins - 1:
                in_bucket = (margins_flat >= lo) & (margins_flat <= hi)
            else:
                in_bucket = (margins_flat >= lo) & (margins_flat < hi)
            bucket_rows.append(
                {
                    "bin": idx,
                    "lo": finite_or_none(float(lo)),
                    "hi": finite_or_none(float(hi)),
                    "count": int(in_bucket.sum()),
                    "correct_rate": safe_mean(targets_flat[in_bucket]),
                }
            )
    correct_mean = safe_mean(margins_flat[correct])
    incorrect_mean = safe_mean(margins_flat[incorrect])
    return {
        "margin_mean": safe_mean(margins_flat),
        "margin_std": safe_std(margins_flat),
        "correct_margin_mean": correct_mean,
        "incorrect_margin_mean": incorrect_mean,
        "correct_incorrect_margin_difference": (
            correct_mean - incorrect_mean
            if correct_mean is not None and incorrect_mean is not None
            else None
        ),
        "pearson_margin_target": pearson(margins_flat, targets_flat),
        "roc_auc_margin_predicts_target": roc_auc(margins_flat, targets_flat),
        "bucketed_correctness_by_margin_quantile": bucket_rows,
    }


def make_config_namespace(config: dict[str, Any]) -> SimpleNamespace:
    defaults = {
        "dropout": None,
        "dropout2": 0.3,
        "dropout3": 0.1,
        "embedding_dropout": 0.1,
        "primitive_emb_dim": 32,
        "margin_emb_dim": 8,
        "time_gate_dim": 16,
        "delta_gate_beta": 0.3,
        "text_delta_scale": 1.0,
        "d_model": 128,
        "hidden_dim": None,
        "n_heads": 4,
        "num_layers": 2,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "dim_feedforward": 128,
        "pooling": "flatten",
        "depth": 10,
    }
    payload = {**defaults, **config}
    return SimpleNamespace(**payload)


def build_and_load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module | None, dict[str, Any], str | None]:
    config = read_json(run_dir / "config.json")
    checkpoint_path = run_dir / "best.pt"
    if not config or not checkpoint_path.exists():
        return None, config, "missing config.json or best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if isinstance(checkpoint_config, dict):
        config = {**config, **checkpoint_config}
    model = build_model(
        config["model"],
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        **model_kwargs(make_config_namespace(config)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, None


def call_model_components(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    primitive_source: str,
) -> torch.Tensor | dict[str, torch.Tensor]:
    if primitive_source == "text":
        try:
            return model(
                batch["x"],
                batch["text_primitive_ids"],
                batch["text_primitive_mask"],
                batch["text_primitive_margins"],
                return_components=True,
            )
        except TypeError:
            return model(
                batch["x"],
                batch["text_primitive_ids"],
                batch["text_primitive_mask"],
                return_components=True,
            )
    if primitive_source == "gt":
        return model(
            batch["x"],
            batch["gt_primitive_ids"],
            batch["gt_primitive_mask"],
            return_components=True,
        )
    raise ValueError(f"Unsupported primitive_source={primitive_source}")


def collect_run(
    name: str,
    run_dir: Path,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    model, config, error = build_and_load_model(run_dir, device)
    metadata = {
        "experiment_name": name,
        "run_dir": str(run_dir),
        "present": error is None,
        "error": error,
        "config": config,
    }
    if model is None:
        return metadata, None

    primitive_source = str(config.get("primitive_source", "text"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    arrays: dict[str, list[np.ndarray]] = {
        "sample_id": [],
        "y_true": [],
        "text_primitive_ids": [],
        "text_primitive_mask": [],
        "text_primitive_margins": [],
        "gt_primitive_ids": [],
        "gt_primitive_mask": [],
        "gate_targets": [],
        "input_primitive_ids": [],
        "input_primitive_mask": [],
        "y_hat": [],
    }
    optional_keys = (
        "y_num",
        "y_primitive_delta",
        "y_num_norm",
        "y_primitive_delta_norm",
        "gate_logits",
        "gate_weights",
        "dynamic_scale",
    )
    optional_arrays: dict[str, list[np.ndarray]] = {key: [] for key in optional_keys}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            output = call_model_components(model, batch, primitive_source)
            if not isinstance(output, dict):
                output = {"y_hat": output}
            arrays["sample_id"].append(tensor_to_numpy(batch["sample_id"]))
            arrays["y_true"].append(tensor_to_numpy(batch["y"]))
            arrays["text_primitive_ids"].append(tensor_to_numpy(batch["text_primitive_ids"]))
            arrays["text_primitive_mask"].append(tensor_to_numpy(batch["text_primitive_mask"]))
            arrays["text_primitive_margins"].append(tensor_to_numpy(batch["text_primitive_margins"]))
            arrays["gt_primitive_ids"].append(tensor_to_numpy(batch["gt_primitive_ids"]))
            arrays["gt_primitive_mask"].append(tensor_to_numpy(batch["gt_primitive_mask"]))
            arrays["gate_targets"].append(tensor_to_numpy(batch["gate_targets"]))
            if primitive_source == "gt":
                arrays["input_primitive_ids"].append(tensor_to_numpy(batch["gt_primitive_ids"]))
                arrays["input_primitive_mask"].append(tensor_to_numpy(batch["gt_primitive_mask"]))
            else:
                arrays["input_primitive_ids"].append(tensor_to_numpy(batch["text_primitive_ids"]))
                arrays["input_primitive_mask"].append(tensor_to_numpy(batch["text_primitive_mask"]))
            arrays["y_hat"].append(tensor_to_numpy(output["y_hat"]))
            for key in optional_keys:
                if key in output:
                    optional_arrays[key].append(tensor_to_numpy(output[key]))

    collected = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    for key, value in optional_arrays.items():
        if value:
            collected[key] = np.concatenate(value, axis=0)
    return metadata, collected


def gate_target_semantics(
    data: dict[str, np.ndarray],
    primitive_source: str,
    is_gate_model: bool,
) -> dict[str, Any]:
    text_ids = data["text_primitive_ids"]
    gt_ids = data["gt_primitive_ids"]
    text_mask = data["text_primitive_mask"].astype(bool)
    gt_mask = data["gt_primitive_mask"].astype(bool)
    input_ids = data["input_primitive_ids"]
    input_mask = data["input_primitive_mask"].astype(bool)
    gate_targets = data["gate_targets"]
    text_eq_gt = (text_ids == gt_ids).astype(np.float64)
    text_gt_valid = text_mask & gt_mask
    gate_valid = input_mask
    target_matches_text_correctness = bool(
        np.array_equal(gate_targets[text_gt_valid] > 0.5, text_eq_gt[text_gt_valid] > 0.5)
    )
    input_eq_gt_rate = safe_mean((input_ids[input_mask & gt_mask] == gt_ids[input_mask & gt_mask]).astype(float))
    warning = None
    if (
        is_gate_model
        and primitive_source == "gt"
        and target_matches_text_correctness
        and safe_mean(gate_targets[gate_valid]) != 1.0
    ):
        warning = (
            "WARNING: GT primitive input is being gated by text-vs-GT correctness targets. "
            "This may suppress valid oracle primitive information."
        )
    per_primitive = {}
    for idx, primitive in enumerate(DEFAULT_PRIMITIVE_ORDER):
        valid = gate_valid[:, idx]
        per_primitive[primitive] = {
            "gate_target_positive_rate": safe_mean(gate_targets[:, idx][valid]),
            "input_primitive_equals_gt_rate": safe_mean(
                (input_ids[:, idx][valid & gt_mask[:, idx]] == gt_ids[:, idx][valid & gt_mask[:, idx]]).astype(float)
            ),
            "text_primitive_equals_gt_rate": safe_mean(text_eq_gt[:, idx][text_gt_valid[:, idx]]),
        }
    return {
        "primitive_source_used_by_model": primitive_source,
        "gate_target_source_currently_used": "gate_targets from grouped cache",
        "gate_targets_defined_as_text_primitive_equals_gt_primitive": target_matches_text_correctness,
        "gate_target_positive_rate": safe_mean(gate_targets[gate_valid]),
        "input_primitive_equals_gt_rate": input_eq_gt_rate,
        "per_primitive": per_primitive,
        "warning": warning,
    }


def gate_dynamicity(data: dict[str, np.ndarray], threshold: float) -> dict[str, Any]:
    if "dynamic_scale" not in data:
        return {}
    result: dict[str, Any] = {
        "dynamic_scale": describe(data["dynamic_scale"]),
    }
    std = result["dynamic_scale"]["std"]
    result["is_near_constant"] = bool(std is not None and std < threshold)
    if "gate_weights" in data:
        gate_weights = data["gate_weights"]
        mask = data["input_primitive_mask"].astype(bool)
        result["gate_weights_global"] = describe(masked_values(gate_weights, mask))
        result["gate_weights_per_primitive"] = {}
        for idx, primitive in enumerate(DEFAULT_PRIMITIVE_ORDER):
            result["gate_weights_per_primitive"][primitive] = describe(gate_weights[:, idx][mask[:, idx]])
    if "gate_logits" in data:
        logits = data["gate_logits"]
        mask = data["input_primitive_mask"].astype(bool)
        result["gate_logits_per_primitive"] = {}
        for idx, primitive in enumerate(DEFAULT_PRIMITIVE_ORDER):
            result["gate_logits_per_primitive"][primitive] = describe(logits[:, idx][mask[:, idx]])
    return result


def gate_vs_correctness(data: dict[str, np.ndarray]) -> dict[str, Any]:
    if "gate_weights" not in data:
        return {}
    weights = data["gate_weights"]
    targets = data["gate_targets"]
    mask = data["input_primitive_mask"].astype(bool)
    logits = data.get("gate_logits")
    result = {
        "global": primitive_metrics(weights, targets, mask, logits),
        "per_primitive": {},
    }
    for idx, primitive in enumerate(DEFAULT_PRIMITIVE_ORDER):
        result["per_primitive"][primitive] = primitive_metrics(
            weights[:, idx],
            targets[:, idx],
            mask[:, idx],
            logits[:, idx] if logits is not None else None,
        )
    return result


def margin_vs_correctness(data: dict[str, np.ndarray], bins: int) -> dict[str, Any]:
    margins = data["text_primitive_margins"]
    targets = data["gate_targets"]
    mask = data["text_primitive_mask"].astype(bool)
    result = {
        "global": margin_metrics(margins, targets, mask, bins),
        "per_primitive": {},
    }
    for idx, primitive in enumerate(DEFAULT_PRIMITIVE_ORDER):
        result["per_primitive"][primitive] = margin_metrics(
            margins[:, idx],
            targets[:, idx],
            mask[:, idx],
            bins,
        )
    return result


def forecast_benefit(
    name: str,
    data: dict[str, np.ndarray],
    all_data: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    if "y_num" not in data:
        return {}
    y_true = data["y_true"]
    y_num = data["y_num"]
    y_hat = data["y_hat"]
    loss_num = sample_mse(y_num, y_true)
    loss_hat = sample_mse(y_hat, y_true)
    benefit = loss_num - loss_hat
    result: dict[str, Any] = {
        "mean_loss_num_sample": safe_mean(loss_num),
        "mean_loss_hat_sample": safe_mean(loss_hat),
        "mean_benefit_sample": safe_mean(benefit),
        "fraction_improves_over_y_num": safe_mean((benefit > 0.0).astype(float)),
    }
    if "dynamic_scale" in data:
        dynamic_scale = data["dynamic_scale"].reshape(-1)
        high = dynamic_scale >= np.median(dynamic_scale)
        result.update(
            {
                "corr_dynamic_scale_benefit": pearson(dynamic_scale, benefit),
                "spearman_dynamic_scale_benefit": spearman(dynamic_scale, benefit),
                "low_scale_mean_benefit": safe_mean(benefit[~high]),
                "high_scale_mean_benefit": safe_mean(benefit[high]),
            }
        )
    pair_name = None
    if name == "additive_text_gate_scale1":
        pair_name = "additive_text_scale1"
    elif name == "additive_gt_gate_scale1":
        pair_name = "additive_gt_scale1"
    if pair_name and pair_name in all_data:
        pair = all_data[pair_name]
        gate_loss = loss_hat
        nogate_loss = sample_mse(pair["y_hat"], pair["y_true"])
        gate_minus_nogate = gate_loss - nogate_loss
        result.update(
            {
                "paired_nogate_experiment": pair_name,
                "mean_gate_minus_nogate_loss": safe_mean(gate_minus_nogate),
                "fraction_gate_improves_over_nogate": safe_mean((gate_minus_nogate < 0.0).astype(float)),
                "corr_dynamic_scale_gate_minus_nogate": pearson(data.get("dynamic_scale", np.zeros_like(gate_loss)), gate_minus_nogate),
            }
        )
        if "y_num" in pair:
            nogate_benefit = sample_mse(pair["y_num"], pair["y_true"]) - nogate_loss
            result["fraction_nogate_residual_improves_over_y_num"] = safe_mean(
                (nogate_benefit > 0.0).astype(float)
            )
            if "dynamic_scale" in data:
                suppressed = (data["dynamic_scale"].reshape(-1) < np.median(data["dynamic_scale"])) & (nogate_benefit > 0.0)
                result["fraction_low_scale_samples_where_nogate_helped"] = safe_mean(suppressed.astype(float))
    return result


def oracle_scale(data: dict[str, np.ndarray]) -> dict[str, Any]:
    if "y_num" not in data or "y_primitive_delta" not in data:
        return {}
    y_true = data["y_true"].reshape(data["y_true"].shape[0], -1)
    y_num = data["y_num"].reshape(data["y_num"].shape[0], -1)
    delta = data["y_primitive_delta"].reshape(data["y_primitive_delta"].shape[0], -1)
    residual = y_true - y_num
    denom = (delta * delta).sum(axis=1) + 1e-8
    alpha = (residual * delta).sum(axis=1) / denom
    loss_num = sample_mse(data["y_num"], data["y_true"])
    out: dict[str, Any] = {
        "alpha_star_unclamped": describe(alpha),
        "zero_delta_fraction": safe_mean((denom <= 1e-8).astype(float)),
        "loss_num": safe_mean(loss_num),
    }
    for label, clipped in (
        ("unclamped", alpha),
        ("clamped_0_1", np.clip(alpha, 0.0, 1.0)),
        ("clamped_0_2", np.clip(alpha, 0.0, 2.0)),
    ):
        pred = y_num + clipped[:, None] * delta
        loss = ((pred - y_true) ** 2).mean(axis=1)
        out[label] = {
            "mean_loss": safe_mean(loss),
            "mean_benefit_vs_y_num": safe_mean(loss_num - loss),
            "fraction_improves_vs_y_num": safe_mean(((loss_num - loss) > 0.0).astype(float)),
        }
    return out


def build_dataset(args: argparse.Namespace, config: dict[str, Any] | None) -> Dataset:
    dataset_name = args.dataset.lower()
    root = resolve_dataset_root(args.data_root)
    base = TESS_Dataset(root, dataset_name, args.split)
    if args.scale == "raw":
        return base
    scaler_mean = None if config is None else config.get("scaler_mean")
    scaler_std = None if config is None else config.get("scaler_std")
    if scaler_mean is None or scaler_std is None:
        raise ValueError("legacy_standard audit requires scaler_mean/scaler_std from a config.json")
    return ScaledTESSDataset(base, float(scaler_mean), float(scaler_std))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    experiments_root = Path(args.experiments_root)
    first_config = None
    for name in EXPERIMENTS:
        config = read_json(experiments_root / name / "config.json")
        if config:
            first_config = config
            break
    dataset = build_dataset(args, first_config)
    device = torch.device(args.device)
    metadata: dict[str, Any] = {}
    collected: dict[str, dict[str, np.ndarray]] = {}
    for name in EXPERIMENTS:
        run_dir = experiments_root / name
        run_metadata, run_data = collect_run(
            name,
            run_dir,
            dataset,
            args.batch_size,
            device,
            args.max_batches,
        )
        metadata[name] = run_metadata
        if run_data is not None:
            collected[name] = run_data

    report: dict[str, Any] = {
        "args": vars(args),
        "primitive_order": list(DEFAULT_PRIMITIVE_ORDER),
        "experiments": {},
        "warnings": [],
    }
    for name, data in collected.items():
        config = metadata[name]["config"]
        primitive_source = str(config.get("primitive_source", "text"))
        semantic = gate_target_semantics(
            data,
            primitive_source,
            config.get("model") == "legacy_multimodal_primitive_additive_gate",
        )
        if semantic.get("warning"):
            report["warnings"].append({"experiment_name": name, "warning": semantic["warning"]})
        result = {
            "metadata": metadata[name],
            "p0_gate_target_semantics": semantic,
            "p1_gate_dynamicity": gate_dynamicity(data, args.constant_gate_std_threshold),
            "p2_gate_vs_correctness": gate_vs_correctness(data),
            "p3_margin_vs_correctness": margin_vs_correctness(data, args.margin_bins),
            "p4_dynamic_scale_vs_forecast_benefit": forecast_benefit(name, data, collected),
            "p5_oracle_scale_upper_bound": (
                oracle_scale(data)
                if config.get("model") == "legacy_multimodal_primitive_additive"
                else {}
            ),
        }
        report["experiments"][name] = result
    for name, run_metadata in metadata.items():
        if name not in report["experiments"]:
            report["experiments"][name] = {"metadata": run_metadata}
    return report


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_text_summary(path: Path, report: dict[str, Any]) -> None:
    lines = ["Additive gate audit summary", ""]
    for warning in report.get("warnings", []):
        lines.append(f"{warning['experiment_name']}: {warning['warning']}")
    if report.get("warnings"):
        lines.append("")
    header = (
        "experiment",
        "source",
        "target_pos",
        "input_eq_gt",
        "dyn_mean",
        "dyn_std",
        "near_const",
        "benefit",
        "gate_auc",
    )
    lines.append("  ".join(header))
    lines.append("  ".join("-" * len(item) for item in header))
    for name in EXPERIMENTS:
        exp = report["experiments"].get(name, {})
        if "p0_gate_target_semantics" not in exp:
            lines.append(f"{name}  missing  NA  NA  NA  NA  NA  NA  NA")
            continue
        p0 = exp["p0_gate_target_semantics"]
        p1 = exp.get("p1_gate_dynamicity", {})
        p2 = exp.get("p2_gate_vs_correctness", {})
        p4 = exp.get("p4_dynamic_scale_vs_forecast_benefit", {})
        dyn = p1.get("dynamic_scale", {})
        gate_global = p2.get("global", {})
        row = (
            name,
            p0.get("primitive_source_used_by_model"),
            format_value(p0.get("gate_target_positive_rate")),
            format_value(p0.get("input_primitive_equals_gt_rate")),
            format_value(dyn.get("mean")),
            format_value(dyn.get("std")),
            format_value(p1.get("is_near_constant")),
            format_value(p4.get("mean_benefit_sample")),
            format_value(gate_global.get("roc_auc")),
        )
        lines.append("  ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit additive primitive gate checkpoints.")
    parser.add_argument("--data_root", "--root", default="/home/lyc/workspace/TESS-RC2")
    parser.add_argument(
        "--cache_root",
        default=None,
        help="Accepted for CLI compatibility. Current TESS_Dataset derives cache paths from data_root/root.",
    )
    parser.add_argument("--dataset", default="fnspid")
    parser.add_argument("--split", choices=("train", "vali", "test"), default="test")
    parser.add_argument("--scale", choices=("raw", "legacy_standard"), default="legacy_standard")
    parser.add_argument(
        "--experiments_root",
        default="/home/lyc/workspace/TESS-RC2/outputs/tess_basic/real_fnspid_legacy_standard",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/lyc/workspace/TESS-RC2/outputs/tess_basic/real_fnspid_legacy_standard/gate_audit",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--constant_gate_std_threshold", type=float, default=0.02)
    parser.add_argument("--margin_bins", type=int, default=5)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional partial-audit limit for smoke tests; omit for full split evaluation.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_report(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "gate_audit_report.json", report)
    write_text_summary(output_dir / "gate_audit_summary.txt", report)
    print(f"wrote {output_dir / 'gate_audit_report.json'}")
    print(f"wrote {output_dir / 'gate_audit_summary.txt'}")
    for warning in report.get("warnings", []):
        print(f"{warning['experiment_name']}: {warning['warning']}")


if __name__ == "__main__":
    main()
