#!/usr/bin/env python3
"""
Analyze cross-modal attention scaling and gradient-based saliency for
MultiModal_Baseline forecasts on the FNSPID dataset.

Per-sample metrics (attention weights, raw scores, gradient norms, MSE) are
exported to JSONL under results/rq1/<run_name>/, together with summary stats
and optional histograms (if matplotlib is available).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F

# Ensure project modules are discoverable when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "src"))

from model_trainer.utils.configurator import Config  # noqa: E402
from model_trainer.models.multimodal_baseline import MultiModal_Baseline  # noqa: E402
from model_trainer.common.dataloader import data_loader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export attention and saliency statistics for MultiModal_Baseline."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to model checkpoint (.pt). If omitted, random-initialized weights are used.",
    )
    parser.add_argument(
        "--dataset-alias",
        type=str,
        default="FNSPID/ver_camf",
        help="Dataset alias registered in configs (default: FNSPID/ver_camf).",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        help="Override dataset version (optional).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "vali", "test"],
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap on number of samples to process.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom subdirectory name under results/rq1/. Defaults to <split>_<alias_suffix>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory (results written inside this path).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., cpu, cuda:0). Defaults to config['device'].",
    )
    parser.add_argument(
        "--no-hist",
        action="store_true",
        help="Disable histogram rendering even if matplotlib is available.",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Load checkpoint with weights_only=True (PyTorch >=2.6).",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out_dir = args.output_dir
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
    else:
        alias_suffix = args.dataset_alias.replace("/", "_")
        default_name = f"{args.split}_{alias_suffix}"
        run_name = args.run_name or default_name
        out_dir = REPO_ROOT / "results" / "rq1" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device, weights_only: bool) -> None:
    if not checkpoint_path:
        return
    kwargs = {"map_location": device}
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([MultiModal_Baseline])
    if torch.__version__ >= "2.6.0":
        kwargs["weights_only"] = weights_only
    state = torch.load(str(checkpoint_path), **kwargs)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] Missing keys: {missing}; Unexpected keys: {unexpected}")


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def compute_gradient_norm(tensor: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    if tensor.grad is None:
        return torch.zeros(tensor.shape[0], device=tensor.device)
    flat = tensor.grad.view(tensor.shape[0], -1)
    if math.isinf(p):
        return flat.abs().max(dim=1).values
    if p == 1:
        return flat.abs().sum(dim=1)
    return flat.norm(p=p, dim=1)


def try_render_histogram(values: List[float], title: str, path: Path) -> None:
    if not values:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib not available ({exc}); skip {title}.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=min(50, max(10, len(values) // 5)), color="#1f77b4", alpha=0.85)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args)

    config_dict = {
        "dataset_alias": args.dataset_alias,
        "dataset_version": args.dataset_version,
        "auto_generate_embedding": False,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "batch_size": args.batch_size,
    }
    config = Config(config_dict, model="MultiModal_Baseline", dataset="FNSPID")
    device = torch.device(args.device or config["device"])

    model = MultiModal_Baseline(config).to(device)
    model.eval()
    load_checkpoint(model, args.checkpoint, device, weights_only=args.weights_only)

    train_loader, vali_loader, test_loader = data_loader(config)
    loader_map = {"train": train_loader, "vali": vali_loader, "test": test_loader}
    loader = loader_map[args.split]

    dataset_obj = loader.dataset
    if hasattr(dataset_obj, "valid_indices") and dataset_obj.valid_indices is not None:
        base_indices = list(dataset_obj.valid_indices)
    else:
        base_indices = list(range(len(dataset_obj)))
    sample_ids = [f"{args.split}-{idx:05d}" for idx in base_indices]

    metrics: List[Dict[str, float]] = []
    attn_values: List[float] = []
    attn_scores: List[float] = []
    grad_norms: List[float] = []
    mse_values: List[float] = []

    processed = 0
    criterion = torch.nn.MSELoss(reduction="none")

    current_index = 0
    for batch in loader:
        if args.max_samples is not None and processed >= args.max_samples:
            break

        batch = move_batch_to_device(batch, device)
        batch_x = batch["x"]
        batch_y = batch["y"]
        news_embed = batch.get("news_embed")
        if news_embed is None:
            continue

        cloned_embed = news_embed.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(batch_x, cloned_embed, flag="test")

        mse_tensor = criterion(outputs, batch_y).mean(dim=1)
        loss = mse_tensor.mean()
        loss.backward()

        attn = model.last_attention
        score = model.last_attention_scores
        grad_norm = compute_gradient_norm(cloned_embed, p=2.0)

        attn = attn.detach().cpu().flatten().tolist()
        score = score.detach().cpu().flatten().tolist() if isinstance(score, torch.Tensor) else [float("nan")] * len(attn)
        grad = grad_norm.detach().cpu().tolist()
        mse = mse_tensor.detach().cpu().tolist()
        preds_norm = outputs.detach().cpu()
        gts_norm = batch_y.detach().cpu()
        preds_raw = dataset_obj.inverse_transform(preds_norm.clone())
        gts_raw = dataset_obj.inverse_transform(gts_norm.clone())

        batch_size = len(attn)
        for i in range(batch_size):
            if args.max_samples is not None and processed >= args.max_samples:
                break
            if current_index >= len(sample_ids):
                break
            sid = sample_ids[current_index]
            metrics.append(
                {
                    "sample_id": sid,
                    "attention_weight": float(attn[i]),
                    "attention_score": float(score[i]),
                    "grad_norm_l2": float(grad[i]),
                    "mse": float(mse[i]),
                    "prediction_normalized": preds_norm[i].tolist(),
                    "ground_truth_normalized": gts_norm[i].tolist(),
                    "prediction_raw": preds_raw[i].tolist() if torch.is_tensor(preds_raw) else preds_raw[i],
                    "ground_truth_raw": gts_raw[i].tolist() if torch.is_tensor(gts_raw) else gts_raw[i],
                }
            )
            attn_values.append(float(attn[i]))
            attn_scores.append(float(score[i]))
            grad_norms.append(float(grad[i]))
            mse_values.append(float(mse[i]))
            processed += 1
            current_index += 1

    if not metrics:
        print("No samples processed; exiting.")
        return

    jsonl_path = output_dir / f"{args.split}_attention_metrics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in metrics:
            fp.write(json.dumps(row) + "\n")

    def summarize(values: Iterable[float]) -> Dict[str, float]:
        vals = [v for v in values if not math.isnan(v)]
        if not vals:
            return {"count": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan"), "std": float("nan")}
        count = len(vals)
        mean_val = sum(vals) / count
        variance = sum((v - mean_val) ** 2 for v in vals) / count if count > 1 else 0.0
        return {
            "count": count,
            "mean": float(mean_val),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "std": float(math.sqrt(variance)),
        }

    summary = {
        "dataset_alias": args.dataset_alias,
        "split": args.split,
        "samples": processed,
        "attention_weight": summarize(attn_values),
        "attention_score": summarize(attn_scores),
        "grad_norm_l2": summarize(grad_norms),
        "mse": summarize(mse_values),
    }

    summary_path = output_dir / f"{args.split}_attention_summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if not args.no_hist:
        try_render_histogram(attn_values, "Attention Weight Distribution", output_dir / f"{args.split}_attention_hist.png")
        try_render_histogram(grad_norms, "Gradient Norm Distribution", output_dir / f"{args.split}_grad_hist.png")

    print(f"Processed {processed} samples. Results saved to {output_dir}.")


if __name__ == "__main__":
    main()
