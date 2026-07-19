"""Teacher-student distillation for the late-additive primitive model.

The teacher uses GT primitives only as privileged training-time supervision.
The deployed student is always evaluated with text primitive IDs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.train_tess_basic import (  # noqa: E402
    LEGACY_METRIC_SCALE,
    compute_metrics,
    count_parameters,
    maybe_scale_datasets,
    model_kwargs,
    move_batch,
    prediction_tensor,
)
from models.simple_tess import build_model  # noqa: E402
from primitive_inference_rc2.tess_dataset import TESS_Dataset  # noqa: E402


TEACHER_MODEL_NAME = "legacy_multimodal_primitive_additive"


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is non-empty: {output_dir}. "
                "Pass --overwrite to replace old artifacts."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_namespace(config: dict[str, Any]) -> SimpleNamespace:
    defaults = {
        "model": TEACHER_MODEL_NAME,
        "d_model": 128,
        "hidden_dim": None,
        "dropout": None,
        "dropout2": 0.3,
        "dropout3": 0.1,
        "embedding_dropout": 0.1,
        "primitive_emb_dim": 32,
        "margin_emb_dim": 8,
        "time_gate_dim": 16,
        "delta_gate_beta": 0.3,
        "text_delta_scale": 1.0,
        "depth": 10,
        "n_heads": 4,
        "num_layers": 2,
        "e_layers": 2,
        "top_k": 5,
        "num_kernels": 6,
        "dim_feedforward": 128,
        "pooling": "flatten",
    }
    return SimpleNamespace(**{**defaults, **config})


def load_teacher(
    checkpoint_path: Path,
    device: torch.device,
    teacher_text_delta_scale: float,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    if not config:
        config_path = checkpoint_path.parent / "config.json"
        config = read_json(config_path)
    if not config:
        raise ValueError(f"Cannot load teacher config from {checkpoint_path}")
    config = dict(config)
    config["model"] = TEACHER_MODEL_NAME
    config["text_delta_scale"] = float(teacher_text_delta_scale)
    teacher = build_model(
        TEACHER_MODEL_NAME,
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        **model_kwargs(config_namespace(config)),
    ).to(device)
    teacher.load_state_dict(checkpoint["model_state"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher, config


def build_student(args: argparse.Namespace, seq_len: int, pred_len: int, device: torch.device) -> nn.Module:
    cfg = vars(args).copy()
    cfg["model"] = args.student_model
    cfg["text_delta_scale"] = args.student_text_delta_scale
    student = build_model(
        args.student_model,
        seq_len=seq_len,
        pred_len=pred_len,
        **model_kwargs(config_namespace(cfg)),
    ).to(device)
    return student


def copy_module(student: nn.Module, teacher: nn.Module, name: str) -> None:
    getattr(student, name).load_state_dict(getattr(teacher, name).state_dict())


def initialize_student_from_teacher(
    student: nn.Module,
    teacher: nn.Module,
    init_primitive: bool,
) -> list[str]:
    copied = []
    for name in ("enc_embedding", "temporal", "mlp_flatten", "numerical_decoder"):
        copy_module(student, teacher, name)
        copied.append(name)
    if init_primitive:
        for name in ("primitive_embeddings", "dynamic_fc", "primitive_decoder"):
            copy_module(student, teacher, name)
            copied.append(name)
    return copied


def freeze_student_numerical(student: nn.Module) -> list[str]:
    frozen = []
    for name in ("enc_embedding", "temporal", "mlp_flatten", "numerical_decoder"):
        module = getattr(student, name)
        for param in module.parameters():
            param.requires_grad_(False)
        frozen.append(name)
    return frozen


def student_forward(student: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if getattr(student, "uses_soft_primitive_probs", False):
        output = student(
            batch["x"],
            batch["text_primitive_ids"],
            batch["text_primitive_mask"],
            batch["text_primitive_probs"],
            batch["text_primitive_prob_mask"],
            return_components=True,
        )
        if not isinstance(output, dict):
            raise TypeError("Expected additive student to return components.")
        return output
    output = student(
        batch["x"],
        batch["text_primitive_ids"],
        batch["text_primitive_mask"],
        return_components=True,
    )
    if not isinstance(output, dict):
        raise TypeError("Expected additive student to return components.")
    return output


def teacher_forward(teacher: nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = teacher(
        batch["x"],
        batch["gt_primitive_ids"],
        batch["gt_primitive_mask"],
        return_components=True,
    )
    if not isinstance(output, dict):
        raise TypeError("Expected additive teacher to return components.")
    return output


def finalize_metric_totals(totals: dict[str, float]) -> dict[str, float]:
    count = max(int(totals["count"]), 1)
    return {
        "mse": totals["sse"] / count,
        "mae": totals["sae"] / count,
    }


def evaluate_student(
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    student.eval()
    totals = {"sse": 0.0, "sae": 0.0, "count": 0}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = move_batch(batch, device)
            if getattr(student, "uses_soft_primitive_probs", False):
                output = student(
                    batch["x"],
                    batch["text_primitive_ids"],
                    batch["text_primitive_mask"],
                    batch["text_primitive_probs"],
                    batch["text_primitive_prob_mask"],
                )
            else:
                output = student(
                    batch["x"],
                    batch["text_primitive_ids"],
                    batch["text_primitive_mask"],
                )
            metrics = compute_metrics(prediction_tensor(output), batch["y"])
            for key, value in metrics.items():
                totals[key] += value
    return finalize_metric_totals(totals)


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, float]:
    student.train()
    totals = {
        "loss_sum": 0.0,
        "forecast_loss_sum": 0.0,
        "delta_distill_loss_sum": 0.0,
        "pred_distill_loss_sum": 0.0,
        "count": 0,
    }
    for batch_idx, batch in enumerate(loader):
        if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        student_out = student_forward(student, batch)
        with torch.no_grad():
            teacher_out = teacher_forward(teacher, batch)

        loss_forecast = nn.functional.mse_loss(student_out["y_hat"], batch["y"])
        student_delta_norm = args.student_text_delta_scale * student_out["y_primitive_delta_norm"]
        teacher_delta_norm = args.teacher_text_delta_scale * teacher_out["y_primitive_delta_norm"].detach()
        loss_delta_distill = nn.functional.mse_loss(student_delta_norm, teacher_delta_norm)

        student_y_hat_norm = student_out["y_num_norm"] + student_delta_norm
        teacher_y_hat_norm = (
            teacher_out["y_num_norm"] + args.teacher_text_delta_scale * teacher_out["y_primitive_delta_norm"]
        ).detach()
        loss_pred_distill = nn.functional.mse_loss(student_y_hat_norm, teacher_y_hat_norm)

        loss = (
            loss_forecast
            + args.lambda_delta_distill * loss_delta_distill
            + args.lambda_pred_distill * loss_pred_distill
        )
        loss.backward()
        optimizer.step()

        batch_size = int(batch["x"].shape[0])
        totals["loss_sum"] += float(loss.detach().item()) * batch_size
        totals["forecast_loss_sum"] += float(loss_forecast.detach().item()) * batch_size
        totals["delta_distill_loss_sum"] += float(loss_delta_distill.detach().item()) * batch_size
        totals["pred_distill_loss_sum"] += float(loss_pred_distill.detach().item()) * batch_size
        totals["count"] += batch_size
    count = max(int(totals["count"]), 1)
    return {
        "loss": totals["loss_sum"] / count,
        "forecast_loss": totals["forecast_loss_sum"] / count,
        "delta_distill_loss": totals["delta_distill_loss_sum"] / count,
        "pred_distill_loss": totals["pred_distill_loss_sum"] / count,
    }


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed, deterministic=args.deterministic)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.overwrite)
    device = torch.device(args.device)

    include_probs = args.student_model == "legacy_multimodal_primitive_additive_soft"
    raw_train = TESS_Dataset(args.root, args.dataset, "train", include_probs=include_probs)
    raw_vali = TESS_Dataset(args.root, args.dataset, "vali", include_probs=include_probs)
    raw_test = TESS_Dataset(args.root, args.dataset, "test", include_probs=include_probs)
    scale_args = SimpleNamespace(scale=args.scale)
    train_ds, vali_ds, test_ds, scaler_info = maybe_scale_datasets(
        scale_args,
        raw_train,
        raw_vali,
        raw_test,
    )
    first = train_ds[0]
    seq_len = int(first["x"].shape[0])
    pred_len = int(first["y"].shape[0])

    teacher, teacher_config = load_teacher(
        Path(args.teacher_checkpoint),
        device,
        args.teacher_text_delta_scale,
    )
    student = build_student(args, seq_len, pred_len, device)
    copied_modules = initialize_student_from_teacher(
        student,
        teacher,
        init_primitive=args.init_student_primitive_from_teacher,
    )
    frozen_modules = freeze_student_numerical(student) if args.freeze_student_numerical else []

    trainable_params = [param for param in student.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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

    config = vars(args).copy()
    config.update(
        {
            "experiment_name": args.experiment_name,
            "model": args.student_model,
            "primitive_source": "text",
            "teacher_model": teacher_config.get("model", TEACHER_MODEL_NAME),
            "teacher_primitive_source": "gt",
            "student_primitive_source": "text",
            "seq_len": seq_len,
            "pred_len": pred_len,
            "dataset_sizes": {
                "train": len(train_ds),
                "vali": len(vali_ds),
                "test": len(test_ds),
            },
            "num_parameters": count_parameters(student),
            "teacher_num_parameters": count_parameters(teacher),
            "student_numerical_initialized_from_teacher": True,
            "student_primitive_initialized_from_teacher": args.init_student_primitive_from_teacher,
            "student_numerical_frozen": args.freeze_student_numerical,
            "copied_teacher_modules": copied_modules,
            "frozen_student_modules": frozen_modules,
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
        f"student={args.student_model} primitive_source=text "
        f"teacher={args.teacher_checkpoint} params={config['num_parameters']}"
    )
    if args.scale == "legacy_standard":
        print(
            "legacy_standard scaler "
            f"mean={config['scaler_mean']:.12g} std={config['scaler_std']:.12g}"
        )

    history = []
    best_vali_mse = float("inf")
    best_epoch = 0
    best_metrics: dict[str, dict[str, float]] | None = None
    for epoch in range(1, args.epochs + 1):
        train_losses = train_one_epoch(student, teacher, train_loader, optimizer, device, args)
        metrics = {
            split: evaluate_student(student, loader, device, args.max_eval_batches)
            for split, loader in eval_loaders.items()
        }
        row = {
            "epoch": epoch,
            "train_losses": train_losses,
            "train": metrics["train"],
            "vali": metrics["vali"],
            "test": metrics["test"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if metrics["vali"]["mse"] < best_vali_mse:
            best_vali_mse = metrics["vali"]["mse"]
            best_epoch = epoch
            best_metrics = metrics
            torch.save(
                checkpoint_payload(student, optimizer, config, epoch, best_vali_mse),
                output_dir / "best.pt",
            )
        torch.save(
            checkpoint_payload(student, optimizer, config, epoch, best_vali_mse),
            output_dir / "last.pt",
        )
        print(
            f"epoch={epoch:03d} train_total_loss={train_losses['loss']:.6f} "
            f"train_forecast_loss={train_losses['forecast_loss']:.6f} "
            f"train_delta_distill_loss={train_losses['delta_distill_loss']:.6f} "
            f"train_pred_distill_loss={train_losses['pred_distill_loss']:.6f} "
            f"vali_mse={metrics['vali']['mse']:.6f} vali_mae={metrics['vali']['mae']:.6f}"
        )

    last_metrics = {
        split: evaluate_student(student, loader, device, args.max_eval_batches)
        for split, loader in eval_loaders.items()
    }
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    student.load_state_dict(best_checkpoint["model_state"])
    best_reloaded = {
        split: evaluate_student(student, loader, device, args.max_eval_batches)
        for split, loader in eval_loaders.items()
    }
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
        "train_loss_decreased": history[-1]["train_losses"]["loss"] < history[0]["train_losses"]["loss"],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "experiment_name": args.experiment_name,
        "model": args.student_model,
        "primitive_source": "text",
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_primitive_source": "gt",
        "student_primitive_source": "text",
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
        "best_reloaded_test_mse": best_reloaded["test"]["mse"],
        "best_reloaded_test_mae": best_reloaded["test"]["mae"],
        "final_test_mse": last_metrics["test"]["mse"],
        "final_test_mae": last_metrics["test"]["mae"],
        "lambda_delta_distill": args.lambda_delta_distill,
        "lambda_pred_distill": args.lambda_pred_distill,
        "student_text_delta_scale": args.student_text_delta_scale,
        "teacher_text_delta_scale": args.teacher_text_delta_scale,
        "student_numerical_initialized_from_teacher": True,
        "student_primitive_initialized_from_teacher": args.init_student_primitive_from_teacher,
        "student_numerical_frozen": args.freeze_student_numerical,
        "train_loss_decreased": result["train_loss_decreased"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train additive text student from GT additive teacher.")
    parser.add_argument("--root", default="/home/lyc/workspace/TESS-RC2")
    parser.add_argument("--dataset", default="fnspid")
    parser.add_argument(
        "--teacher-checkpoint",
        "--teacher_checkpoint",
        default=(
            "/home/lyc/workspace/TESS-RC2/outputs/tess_basic/"
            "real_fnspid_legacy_standard/additive_gt_scale1/best.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--scale", choices=("raw", "legacy_standard"), default="legacy_standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default=(
            "/home/lyc/workspace/TESS-RC2/outputs/tess_basic/real_fnspid_legacy_standard/"
            "additive_text_distill_delta_lam0p1"
        ),
    )
    parser.add_argument("--experiment-name", "--experiment_name", default="additive_text_distill_delta_lam0p1")
    parser.add_argument("--student-model", "--student_model", default="legacy_multimodal_primitive_additive")
    parser.add_argument("--lambda-delta-distill", "--lambda_delta_distill", type=float, default=0.1)
    parser.add_argument("--lambda-pred-distill", "--lambda_pred_distill", type=float, default=0.0)
    parser.add_argument("--student-text-delta-scale", "--student_text_delta_scale", type=float, default=1.0)
    parser.add_argument("--teacher-text-delta-scale", "--teacher_text_delta_scale", type=float, default=1.0)
    parser.add_argument("--freeze-student-numerical", "--freeze_student_numerical", action="store_true")
    parser.add_argument(
        "--init-student-primitive-from-teacher",
        "--init_student_primitive_from_teacher",
        action="store_true",
    )
    parser.add_argument("--d-model", "--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--dropout2", type=float, default=0.3)
    parser.add_argument("--embedding-dropout", "--embedding_dropout", type=float, default=0.1)
    parser.add_argument("--primitive-emb-dim", "--primitive_emb_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--max-train-batches",
        "--max_train_batches",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for full training.",
    )
    parser.add_argument(
        "--max-eval-batches",
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional smoke-test limit; omit for full evaluation.",
    )
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
