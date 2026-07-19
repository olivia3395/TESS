"""Minimal training loop for numeric and no-gate TESS smoke experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.simple_tess import build_model
from primitive_inference_rc2.tess_dataset import TESS_Dataset

LEGACY_METRIC_SCALE = "legacy_train_history_standardized"


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def validate_args(args: argparse.Namespace) -> None:
    if args.model in {"numeric_mlp", "tiny_temporal", "legacy_timesnet"} and args.primitive_source != "none":
        raise ValueError(f"{args.model} requires --primitive-source none")
    if args.model in {
        "tess_nogate",
        "tiny_temporal_tess",
        "legacy_multimodal_primitive",
        "legacy_multimodal_primitive_additive",
        "legacy_multimodal_primitive_additive_soft",
        "legacy_multimodal_primitive_additive_gate",
    } and args.primitive_source not in {"text", "gt"}:
        raise ValueError(f"{args.model} requires --primitive-source text or gt")
    if args.model in {"legacy_multimodal_primitive_gate", "legacy_multimodal_primitive_delta_gate"} and args.primitive_source != "text":
        raise ValueError(f"{args.model} requires --primitive-source text")
    if args.model not in {
        "legacy_multimodal_primitive_additive",
        "legacy_multimodal_primitive_additive_soft",
        "legacy_multimodal_primitive_additive_gate",
    } and args.text_delta_scale != 1.0:
        raise ValueError(
            "--text-delta-scale is only supported for legacy_multimodal_primitive_additive "
            "and legacy_multimodal_primitive_additive_gate"
        )


class ScaledTESSDataset(Dataset):
    """Apply a shared affine scaler to x/y while preserving primitive fields."""

    def __init__(self, base_dataset: Dataset, mean: float, std: float) -> None:
        self.base_dataset = base_dataset
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        item = dict(self.base_dataset[index])
        item["x"] = (item["x"] - self.mean) / self.std
        item["y"] = (item["y"] - self.mean) / self.std
        return item


def fit_legacy_standard_scaler(train_dataset: Dataset) -> dict[str, Any]:
    """Fit sklearn-compatible StandardScaler stats on flattened train x only."""
    xs = [train_dataset[idx]["x"].reshape(-1).to(torch.float64) for idx in range(len(train_dataset))]
    if not xs:
        raise ValueError("Cannot fit legacy_standard scaler on an empty train split.")
    values = torch.cat(xs)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if std <= 0.0:
        raise ValueError(
            "legacy_standard scaler fitted zero standard deviation from train historical_data."
        )
    return {
        "mean": mean,
        "std": std,
        "fit_source": "train_historical_data",
    }


def maybe_scale_datasets(
    args: argparse.Namespace,
    train_ds: Dataset,
    vali_ds: Dataset,
    test_ds: Dataset,
) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    if args.scale == "raw":
        scaler_info = {
            "metric_scale": "raw",
            "scaler_mean": None,
            "scaler_std": None,
            "scaler_fit_source": None,
            "train_x_scaled_mean": None,
            "train_x_scaled_std": None,
        }
        return train_ds, vali_ds, test_ds, scaler_info

    scaler = fit_legacy_standard_scaler(train_ds)
    scaled_train = ScaledTESSDataset(train_ds, scaler["mean"], scaler["std"])
    scaled_vali = ScaledTESSDataset(vali_ds, scaler["mean"], scaler["std"])
    scaled_test = ScaledTESSDataset(test_ds, scaler["mean"], scaler["std"])
    train_x_stats = compute_dataset_x_stats(scaled_train)
    scaler_info = {
        "metric_scale": LEGACY_METRIC_SCALE,
        "scaler_mean": scaler["mean"],
        "scaler_std": scaler["std"],
        "scaler_fit_source": scaler["fit_source"],
        "train_x_scaled_mean": train_x_stats["mean"],
        "train_x_scaled_std": train_x_stats["std"],
    }
    return scaled_train, scaled_vali, scaled_test, scaler_info


def compute_dataset_x_stats(dataset: Dataset) -> dict[str, float]:
    xs = [dataset[idx]["x"].reshape(-1).to(torch.float64) for idx in range(len(dataset))]
    values = torch.cat(xs)
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
    }


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict(model: nn.Module, batch: dict[str, torch.Tensor], primitive_source: str) -> torch.Tensor | dict[str, torch.Tensor]:
    if primitive_source == "none":
        return model(batch["x"])
    if primitive_source == "text":
        if getattr(model, "uses_soft_primitive_probs", False) and "text_primitive_probs" in batch:
            return model(
                batch["x"],
                batch["text_primitive_ids"],
                batch["text_primitive_mask"],
                batch["text_primitive_probs"],
                batch["text_primitive_prob_mask"],
            )
        try:
            return model(
                batch["x"],
                batch["text_primitive_ids"],
                batch["text_primitive_mask"],
                batch["text_primitive_margins"],
            )
        except TypeError:
            return model(batch["x"], batch["text_primitive_ids"], batch["text_primitive_mask"])
    if primitive_source == "gt":
        return model(batch["x"], batch["gt_primitive_ids"], batch["gt_primitive_mask"])
    raise ValueError(f"Unknown primitive_source={primitive_source}")


def prediction_tensor(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["y_hat"]
    return output


def compute_metrics(y_hat: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """Return element-weighted regression metrics for one prediction tensor."""
    diff = y_hat - y
    return {
        "sse": float((diff ** 2).sum().item()),
        "sae": float(diff.abs().sum().item()),
        "count": int(diff.numel()),
    }


def compute_gate_loss_and_metrics(
    output: torch.Tensor | dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    primitive_source: str,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    if not isinstance(output, dict) or "gate_logits" not in output:
        return None, {}
    logits = output["gate_logits"]
    targets = batch["gate_targets"].to(dtype=logits.dtype, device=logits.device)
    mask = batch.get("gt_primitive_mask" if primitive_source == "gt" else "text_primitive_mask")
    if mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = mask.to(device=logits.device, dtype=torch.bool)
    if valid.sum().item() == 0:
        loss = logits.sum() * 0.0
        return loss, {
            "gate_bce": 0.0,
            "gate_acc": 0.0,
            "mean_gate_weight": 0.0,
            "gate_pred_pos_rate": 0.0,
            "gate_pos_mean_weight": 0.0,
            "gate_neg_mean_weight": 0.0,
            "gate_count": 0,
            "gate_pos_count": 0,
            "gate_neg_count": 0,
        }

    element_loss = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    loss = element_loss[valid].mean()
    probs = torch.sigmoid(logits)
    pred = probs >= 0.5
    valid_probs = probs[valid]
    valid_targets = targets[valid].bool()
    valid_pred = pred[valid]
    correct = (valid_pred == valid_targets).to(dtype=torch.float32)
    pos_mask = valid_targets
    neg_mask = ~valid_targets
    pos_count = int(pos_mask.sum().item())
    neg_count = int(neg_mask.sum().item())
    return loss, {
        "gate_bce": float(loss.detach().item()),
        "gate_acc": float(correct.mean().detach().item()),
        "mean_gate_weight": float(valid_probs.mean().detach().item()),
        "gate_pred_pos_rate": float(valid_pred.to(dtype=torch.float32).mean().detach().item()),
        "gate_pos_mean_weight": float(valid_probs[pos_mask].mean().detach().item()) if pos_count else 0.0,
        "gate_neg_mean_weight": float(valid_probs[neg_mask].mean().detach().item()) if neg_count else 0.0,
        "gate_count": int(valid.sum().item()),
        "gate_pos_count": pos_count,
        "gate_neg_count": neg_count,
    }


def output_diagnostics(output: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, float]:
    if not isinstance(output, dict) or "dynamic_scale" not in output:
        return {}
    dynamic_scale = output["dynamic_scale"].detach().to(dtype=torch.float32)
    return {
        "mean_dynamic_scale": float(dynamic_scale.mean().item()),
        "dynamic_scale_count": int(dynamic_scale.numel()),
    }


def finalize_metrics(accumulator: dict[str, float]) -> dict[str, float]:
    count = max(int(accumulator["count"]), 1)
    return {
        "mse": accumulator["sse"] / count,
        "mae": accumulator["sae"] / count,
    }


def inverse_transform(tensor: torch.Tensor, scaler_info: dict[str, Any]) -> torch.Tensor:
    return tensor * float(scaler_info["scaler_std"]) + float(scaler_info["scaler_mean"])


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    primitive_source: str,
    scaler_info: dict[str, Any],
) -> dict[str, float]:
    model.eval()
    totals = {"sse": 0.0, "sae": 0.0, "count": 0}
    raw_totals = {"sse": 0.0, "sae": 0.0, "count": 0}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            output = predict(model, batch, primitive_source)
            pred = prediction_tensor(output)
            batch_metrics = compute_metrics(pred, batch["y"])
            for key, value in batch_metrics.items():
                totals[key] += value
            _, gate_metrics = compute_gate_loss_and_metrics(output, batch, primitive_source)
            if gate_metrics:
                raw_totals.setdefault("gate_bce_sum", 0.0)
                raw_totals.setdefault("gate_acc_sum", 0.0)
                raw_totals.setdefault("mean_gate_weight_sum", 0.0)
                raw_totals.setdefault("gate_pred_pos_rate_sum", 0.0)
                raw_totals.setdefault("gate_pos_mean_weight_sum", 0.0)
                raw_totals.setdefault("gate_neg_mean_weight_sum", 0.0)
                raw_totals.setdefault("gate_metric_count", 0)
                raw_totals.setdefault("gate_pos_count", 0)
                raw_totals.setdefault("gate_neg_count", 0)
                raw_totals["gate_bce_sum"] += gate_metrics["gate_bce"] * gate_metrics["gate_count"]
                raw_totals["gate_acc_sum"] += gate_metrics["gate_acc"] * gate_metrics["gate_count"]
                raw_totals["mean_gate_weight_sum"] += gate_metrics["mean_gate_weight"] * gate_metrics["gate_count"]
                raw_totals["gate_pred_pos_rate_sum"] += gate_metrics["gate_pred_pos_rate"] * gate_metrics["gate_count"]
                raw_totals["gate_pos_mean_weight_sum"] += gate_metrics["gate_pos_mean_weight"] * gate_metrics["gate_pos_count"]
                raw_totals["gate_neg_mean_weight_sum"] += gate_metrics["gate_neg_mean_weight"] * gate_metrics["gate_neg_count"]
                raw_totals["gate_metric_count"] += gate_metrics["gate_count"]
                raw_totals["gate_pos_count"] += gate_metrics["gate_pos_count"]
                raw_totals["gate_neg_count"] += gate_metrics["gate_neg_count"]
            diagnostics = output_diagnostics(output)
            if diagnostics:
                raw_totals.setdefault("mean_dynamic_scale_sum", 0.0)
                raw_totals.setdefault("dynamic_scale_count", 0)
                raw_totals["mean_dynamic_scale_sum"] += (
                    diagnostics["mean_dynamic_scale"] * diagnostics["dynamic_scale_count"]
                )
                raw_totals["dynamic_scale_count"] += diagnostics["dynamic_scale_count"]
            if scaler_info["metric_scale"] == LEGACY_METRIC_SCALE:
                raw_metrics = compute_metrics(
                    inverse_transform(pred, scaler_info),
                    inverse_transform(batch["y"], scaler_info),
                )
                for key, value in raw_metrics.items():
                    raw_totals[key] += value
    metrics = finalize_metrics(totals)
    if scaler_info["metric_scale"] == LEGACY_METRIC_SCALE:
        raw_metrics = finalize_metrics(raw_totals)
        metrics["raw_mse"] = raw_metrics["mse"]
        metrics["raw_mae"] = raw_metrics["mae"]
    gate_count = int(raw_totals.get("gate_metric_count", 0))
    if gate_count:
        metrics["gate_bce"] = raw_totals["gate_bce_sum"] / gate_count
        metrics["gate_acc"] = raw_totals["gate_acc_sum"] / gate_count
        metrics["mean_gate_weight"] = raw_totals["mean_gate_weight_sum"] / gate_count
        metrics["gate_pred_pos_rate"] = raw_totals["gate_pred_pos_rate_sum"] / gate_count
        gate_pos_count = int(raw_totals.get("gate_pos_count", 0))
        gate_neg_count = int(raw_totals.get("gate_neg_count", 0))
        metrics["gate_pos_mean_weight"] = (
            raw_totals["gate_pos_mean_weight_sum"] / gate_pos_count if gate_pos_count else 0.0
        )
        metrics["gate_neg_mean_weight"] = (
            raw_totals["gate_neg_mean_weight_sum"] / gate_neg_count if gate_neg_count else 0.0
        )
    dynamic_scale_count = int(raw_totals.get("dynamic_scale_count", 0))
    if dynamic_scale_count:
        metrics["mean_dynamic_scale"] = raw_totals["mean_dynamic_scale_sum"] / dynamic_scale_count
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    primitive_source: str,
    gate_loss_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"sse": 0.0, "sae": 0.0, "count": 0}
    gate_sums = {
        "gate_bce_sum": 0.0,
        "gate_acc_sum": 0.0,
        "mean_gate_weight_sum": 0.0,
        "gate_pred_pos_rate_sum": 0.0,
        "gate_pos_mean_weight_sum": 0.0,
        "gate_neg_mean_weight_sum": 0.0,
        "gate_metric_count": 0,
        "gate_pos_count": 0,
        "gate_neg_count": 0,
    }
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = predict(model, batch, primitive_source)
        pred = prediction_tensor(output)
        forecast_loss = nn.functional.mse_loss(pred, batch["y"])
        gate_loss, gate_metrics = compute_gate_loss_and_metrics(output, batch, primitive_source)
        loss = forecast_loss if gate_loss is None else forecast_loss + gate_loss_weight * gate_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            batch_metrics = compute_metrics(pred, batch["y"])
            for key, value in batch_metrics.items():
                totals[key] += value
            if gate_metrics:
                gate_sums["gate_bce_sum"] += gate_metrics["gate_bce"] * gate_metrics["gate_count"]
                gate_sums["gate_acc_sum"] += gate_metrics["gate_acc"] * gate_metrics["gate_count"]
                gate_sums["mean_gate_weight_sum"] += gate_metrics["mean_gate_weight"] * gate_metrics["gate_count"]
                gate_sums["gate_pred_pos_rate_sum"] += gate_metrics["gate_pred_pos_rate"] * gate_metrics["gate_count"]
                gate_sums["gate_pos_mean_weight_sum"] += gate_metrics["gate_pos_mean_weight"] * gate_metrics["gate_pos_count"]
                gate_sums["gate_neg_mean_weight_sum"] += gate_metrics["gate_neg_mean_weight"] * gate_metrics["gate_neg_count"]
                gate_sums["gate_metric_count"] += gate_metrics["gate_count"]
                gate_sums["gate_pos_count"] += gate_metrics["gate_pos_count"]
                gate_sums["gate_neg_count"] += gate_metrics["gate_neg_count"]
    metrics = finalize_metrics(totals)
    gate_count = int(gate_sums["gate_metric_count"])
    if gate_count:
        metrics["gate_bce"] = gate_sums["gate_bce_sum"] / gate_count
        metrics["gate_acc"] = gate_sums["gate_acc_sum"] / gate_count
        metrics["mean_gate_weight"] = gate_sums["mean_gate_weight_sum"] / gate_count
        metrics["gate_pred_pos_rate"] = gate_sums["gate_pred_pos_rate_sum"] / gate_count
        metrics["gate_pos_mean_weight"] = (
            gate_sums["gate_pos_mean_weight_sum"] / gate_sums["gate_pos_count"]
            if gate_sums["gate_pos_count"]
            else 0.0
        )
        metrics["gate_neg_mean_weight"] = (
            gate_sums["gate_neg_mean_weight_sum"] / gate_sums["gate_neg_count"]
            if gate_sums["gate_neg_count"]
            else 0.0
        )
    return metrics


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {output_dir}. "
                "Pass --overwrite to replace old training artifacts."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    epoch: int,
    best_vali_mse: float,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
        "best_vali_mse": best_vali_mse,
        "model_name": config["model"],
        "primitive_source": config["primitive_source"],
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
    }


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    default_dropout = 0.1 if args.dropout is None else args.dropout
    if args.model in {"numeric_mlp", "tess_nogate"}:
        return {
            "d_model": args.d_model,
            "hidden_dim": args.hidden_dim,
            "dropout": default_dropout,
        }
    if args.model in {"tiny_temporal", "tiny_temporal_tess"}:
        return {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "num_layers": args.num_layers,
            "dim_feedforward": args.dim_feedforward,
            "dropout": default_dropout,
            "pooling": args.pooling,
        }
    if args.model == "legacy_timesnet":
        return {
            "d_model": args.d_model,
            "e_layers": args.e_layers,
            "top_k": args.top_k,
            "num_kernels": args.num_kernels,
            "dropout": default_dropout,
        }
    if args.model in {
        "legacy_multimodal_primitive",
        "legacy_multimodal_primitive_additive",
        "legacy_multimodal_primitive_additive_soft",
        "legacy_multimodal_primitive_additive_gate",
        "legacy_multimodal_primitive_gate",
        "legacy_multimodal_primitive_delta_gate",
    }:
        legacy_dropout = 0.3 if args.dropout is None else args.dropout
        kwargs = {
            "d_model": args.d_model,
            "primitive_emb_dim": args.primitive_emb_dim,
            "dropout": legacy_dropout,
            "dropout2": args.dropout2,
            "embedding_dropout": args.embedding_dropout,
            "depth": args.depth,
        }
        if args.model not in {
            "legacy_multimodal_primitive_additive",
            "legacy_multimodal_primitive_additive_soft",
            "legacy_multimodal_primitive_additive_gate",
        }:
            kwargs["dropout3"] = args.dropout3
        if args.model in {
            "legacy_multimodal_primitive_additive_gate",
            "legacy_multimodal_primitive_gate",
            "legacy_multimodal_primitive_delta_gate",
        }:
            kwargs["margin_emb_dim"] = args.margin_emb_dim
        if args.model == "legacy_multimodal_primitive_additive_gate":
            kwargs["time_gate_dim"] = args.time_gate_dim
        if args.model in {
            "legacy_multimodal_primitive_additive",
            "legacy_multimodal_primitive_additive_soft",
            "legacy_multimodal_primitive_additive_gate",
        }:
            kwargs["text_delta_scale"] = args.text_delta_scale
        if args.model == "legacy_multimodal_primitive_delta_gate":
            kwargs["time_gate_dim"] = args.time_gate_dim
            kwargs["delta_gate_beta"] = args.delta_gate_beta
        return kwargs
    raise ValueError(f"Unknown model={args.model}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    device = torch.device(args.device)

    include_probs = args.model == "legacy_multimodal_primitive_additive_soft"
    train_ds = TESS_Dataset(args.root, args.dataset, "train", include_probs=include_probs)
    vali_ds = TESS_Dataset(args.root, args.dataset, "vali", include_probs=include_probs)
    test_ds = TESS_Dataset(args.root, args.dataset, "test", include_probs=include_probs)
    train_ds, vali_ds, test_ds, scaler_info = maybe_scale_datasets(args, train_ds, vali_ds, test_ds)
    first = train_ds[0]
    seq_len = int(first["x"].shape[0])
    pred_len = int(first["y"].shape[0])
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    eval_loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=False),
        "vali": DataLoader(vali_ds, batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False),
    }

    model = build_model(
        args.model,
        seq_len=seq_len,
        pred_len=pred_len,
        **model_kwargs(args),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = vars(args).copy()
    config.update(
        {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "dataset_sizes": {
                "train": len(train_ds),
                "vali": len(vali_ds),
                "test": len(test_ds),
            },
            "num_parameters": count_parameters(model),
            "oracle": args.primitive_source == "gt",
            **scaler_info,
        }
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"dataset sizes train={len(train_ds)} vali={len(vali_ds)} test={len(test_ds)} "
        f"seq_len={seq_len} pred_len={pred_len}"
    )
    print(
        f"model={args.model} primitive_source={args.primitive_source} "
        f"params={config['num_parameters']} metric_scale={config['metric_scale']}"
    )
    if args.scale == "legacy_standard":
        print(
            "legacy_standard scaler "
            f"mean={config['scaler_mean']:.12g} std={config['scaler_std']:.12g} "
            f"train_x_scaled_mean={config['train_x_scaled_mean']:.12g} "
            f"train_x_scaled_std={config['train_x_scaled_std']:.12g}"
        )
    if args.primitive_source == "gt":
        print("ORACLE primitive source: using gt_primitive_ids derived from ground_truth.")
        print("This is for upper-bound/sanity testing only.")

    history = []
    best_vali_mse = float("inf")
    best_epoch = 0
    best_metrics: dict[str, dict[str, float]] | None = None
    for epoch in range(1, args.epochs + 1):
        train_update_metrics = train_one_epoch(
            model, train_loader, optimizer, device, args.primitive_source, args.gate_loss_weight
        )
        metrics = {
            split: evaluate(model, loader, device, args.primitive_source, scaler_info)
            for split, loader in eval_loaders.items()
        }
        train_mse_loss = metrics["train"]["mse"]
        row = {
            "epoch": epoch,
            "train": metrics["train"],
            "vali": metrics["vali"],
            "test": metrics["test"],
            "train_mse_loss": train_mse_loss,
            "train_update_mse": train_update_metrics["mse"],
            "train_update_mae": train_update_metrics["mae"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if metrics["vali"]["mse"] < best_vali_mse:
            best_vali_mse = metrics["vali"]["mse"]
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                checkpoint_payload(model, optimizer, config, epoch, best_vali_mse),
                output_dir / "best.pt",
            )
        torch.save(
            checkpoint_payload(model, optimizer, config, epoch, best_vali_mse),
            output_dir / "last.pt",
        )
        print(
            f"epoch={epoch:03d} train_mse_loss={train_mse_loss:.6f} "
            f"train_mse={metrics['train']['mse']:.6f} train_mae={metrics['train']['mae']:.6f} "
            f"vali_mse={metrics['vali']['mse']:.6f} vali_mae={metrics['vali']['mae']:.6f} "
            f"test_mse={metrics['test']['mse']:.6f} test_mae={metrics['test']['mae']:.6f}"
            + (
                f" gate_bce={metrics['vali'].get('gate_bce', 0.0):.6f} "
                f"gate_acc={metrics['vali'].get('gate_acc', 0.0):.6f} "
                f"mean_gate_weight={metrics['vali'].get('mean_gate_weight', 0.0):.6f}"
                if "gate_bce" in metrics["vali"]
                else ""
            )
        )

    last_metrics = {
        split: evaluate(model, loader, device, args.primitive_source, scaler_info)
        for split, loader in eval_loaders.items()
    }
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    best_reloaded = {
        split: evaluate(model, loader, device, args.primitive_source, scaler_info)
        for split, loader in eval_loaders.items()
    }
    best_reload_abs_diff = abs(best_reloaded["vali"]["mse"] - best_vali_mse)
    best_reload_tolerance = 1e-7
    best_reload_check_passed = best_reload_abs_diff <= best_reload_tolerance
    if not best_reload_check_passed:
        print(
            "warning: reloaded best vali MSE differs from recorded best "
            f"by {best_reload_abs_diff:.12g}"
        )

    result = {
        "config": config,
        "metric_scale": config["metric_scale"],
        "scaler_mean": config["scaler_mean"],
        "scaler_std": config["scaler_std"],
        "scaler_fit_source": config["scaler_fit_source"],
        "best_epoch": best_epoch,
        "best_vali_mse": best_vali_mse,
        "history": history,
        "last": last_metrics,
        "best_reloaded": best_reloaded,
        "best_recorded": best_metrics,
        "best_reload_check": {
            "passed": best_reload_check_passed,
            "abs_diff": best_reload_abs_diff,
            "tolerance": best_reload_tolerance,
        },
        "train_loss_decreased": history[-1]["train_mse_loss"] < history[0]["train_mse_loss"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "model": args.model,
        "primitive_source": args.primitive_source,
        "seed": args.seed,
        "dataset_sizes": config["dataset_sizes"],
        "seq_len": seq_len,
        "pred_len": pred_len,
        "num_parameters": config["num_parameters"],
        "metric_scale": config["metric_scale"],
        "scaler_mean": config["scaler_mean"],
        "scaler_std": config["scaler_std"],
        "scaler_fit_source": config["scaler_fit_source"],
        "train_x_scaled_mean": config["train_x_scaled_mean"],
        "train_x_scaled_std": config["train_x_scaled_std"],
        "best_epoch": best_epoch,
        "best_vali_mse": best_vali_mse,
        "final_test_mse": last_metrics["test"]["mse"],
        "best_reloaded_test_mse": best_reloaded["test"]["mse"],
        "best_reloaded_test_mae": best_reloaded["test"]["mae"],
        "final_test_mae": last_metrics["test"]["mae"],
        "train_loss_decreased": result["train_loss_decreased"],
        "best_reload_check": result["best_reload_check"],
    }
    for key in (
        "gate_bce",
        "gate_acc",
        "mean_gate_weight",
        "gate_pos_mean_weight",
        "gate_neg_mean_weight",
        "gate_pred_pos_rate",
        "mean_dynamic_scale",
    ):
        if key in best_reloaded["test"]:
            summary[f"best_reloaded_test_{key}"] = best_reloaded["test"][key]
        if key in last_metrics["test"]:
            summary[f"final_test_{key}"] = last_metrics["test"][key]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal TESS smoke model.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--model",
        choices=(
            "numeric_mlp",
            "tess_nogate",
            "tiny_temporal",
            "tiny_temporal_tess",
            "legacy_timesnet",
            "legacy_multimodal_primitive",
            "legacy_multimodal_primitive_additive",
            "legacy_multimodal_primitive_additive_soft",
            "legacy_multimodal_primitive_additive_gate",
            "legacy_multimodal_primitive_gate",
            "legacy_multimodal_primitive_delta_gate",
        ),
        required=True,
    )
    parser.add_argument("--primitive-source", choices=("none", "text", "gt"), required=True)
    parser.add_argument(
        "--scale",
        choices=("raw", "legacy_standard"),
        default="raw",
        help="Value scale for x/y and main metrics. legacy_standard matches legacy FNSPID StandardScaler.",
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dropout2", type=float, default=0.3)
    parser.add_argument("--dropout3", type=float, default=0.1)
    parser.add_argument("--embedding-dropout", type=float, default=0.1)
    parser.add_argument("--primitive-emb-dim", type=int, default=32)
    parser.add_argument("--margin-emb-dim", type=int, default=8)
    parser.add_argument("--time-gate-dim", type=int, default=16)
    parser.add_argument("--delta-gate-beta", type=float, default=0.3)
    parser.add_argument(
        "--text-delta-scale",
        "--text_delta_scale",
        type=float,
        default=1.0,
        help="Scale primitive residuals for legacy_multimodal_primitive_additive.",
    )
    parser.add_argument("--gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-kernels", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--pooling", default="flatten")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable PyTorch deterministic algorithms. On CUDA this may require CUBLAS_WORKSPACE_CONFIG.",
    )
    return parser


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
