#!/usr/bin/env python3
"""
Evaluate a saved MMTSF checkpoint (.pt) and export per-sample predictions.

The script rebuilds the dataloaders from the accompanying config snapshot,
runs inference on the requested splits, and writes JSONL files that mirror
the structure produced during training (hist_data / ground_truth / prediction / news).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def add_src_to_path(root: Path) -> None:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MMTSF checkpoint and dump sample-level predictions.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved model.pt file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional path to config_snapshot.json. Defaults to <checkpoint_dir>/config_snapshot.json.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,vali,test",
        help="Comma-separated list of dataset splits to evaluate (choices: train, vali, test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write JSONL files. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Optional suffix appended to the output filenames before .jsonl.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (e.g. cpu, cuda:0). Default: cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override evaluation batch size. Defaults to value stored in config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open() as f:
        snapshot = json.load(f)
    if "config" not in snapshot:
        raise RuntimeError(f"{config_path} does not look like a config_snapshot.json")
    return snapshot["config"]


def normalise_paths(config: Dict, root: Path) -> None:
    # Force repository-local dataset roots so that relative paths from the snapshot work.
    dataset_name = config.get("dataset", "")
    if dataset_name:
        config["data_path"] = str(root / "dataset")
        config["dataset_root"] = str(root / "dataset" / dataset_name)
    config["auto_generate_embedding"] = bool(config.get("auto_generate_embedding", False))


def ensure_safe_globals(model_name: str) -> None:
    """Allow torch.load to unpickle custom layer classes safely."""
    import importlib

    safe_types: List[type] = []
    target_modules: List[str] = [
        "model_trainer.layers.AutoCorrelation",
        "model_trainer.layers.Autoformer_EncDec",
        "model_trainer.layers.Causal",
        "model_trainer.layers.Conv_Blocks",
        "model_trainer.layers.Embed",
        "model_trainer.layers.FourierCorrelation",
        "model_trainer.layers.MultiModal",
        "model_trainer.layers.MultiWaveletCorrelation",
        "model_trainer.layers.Pyraformer_EncDec",
        "model_trainer.layers.SelfAttention_Family",
        "model_trainer.layers.StandardNorm",
        "model_trainer.layers.Transformer_EncDec",
        "model_trainer.layers.diffusion",
        "model_trainer.layers.fedformer",
        "model_trainer.layers.mlp",
    ]

    # Model modules sometimes include additional helper classes.
    target_modules.append(f"model_trainer.models.{model_name.lower()}")

    for module_name in target_modules:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            safe_types.append(obj)

    if safe_types:
        torch.serialization.add_safe_globals(list(safe_types))


def load_model(config: Dict, checkpoint_path: Path, device: torch.device):
    from model_trainer.utils.utils import get_model

    model_name = config["model"]
    ModelClass = get_model(model_name)
    ensure_safe_globals(model_name)
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if isinstance(state, ModelClass):
        model = state
    else:
        model = ModelClass(config)
        if isinstance(state, dict):
            state_dict = state.get("model_state_dict") or state.get("state_dict") or state
        else:
            raise TypeError(f"Unsupported checkpoint payload type: {type(state)}")
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def prepare_dataloaders(config_obj):
    from model_trainer.common.dataloader import data_loader

    train_loader, vali_loader, test_loader = data_loader(config_obj)
    return {
        "train": train_loader,
        "vali": vali_loader,
        "test": test_loader,
    }


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")


def evaluate_split(trainer, loader, split: str):
    metrics, preds, trues, hist, news = trainer._evaluate_split(  # pylint: disable=protected-access
        loader,
        split,
        return_raw=True,
        include_details=True,
    )
    records = trainer._build_sample_records(  # pylint: disable=protected-access
        split,
        preds,
        trues,
        hist,
        news,
    )
    return metrics, records


def main() -> None:
    args = parse_args()
    root = repo_root()
    os.chdir(root)  # Ensure relative dataset paths resolve correctly.
    add_src_to_path(root)

    checkpoint_path = args.checkpoint.resolve()
    config_path = (args.config or checkpoint_path.parent / "config_snapshot.json").resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config snapshot not found at {config_path}")

    config_dict = load_config(config_path)
    normalise_paths(config_dict, root)
    config_dict["req_training"] = False
    config_dict["use_gpu"] = args.device.startswith("cuda")
    config_dict["device"] = args.device
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
        config_dict["eval_batch_size"] = args.batch_size

    from model_trainer.utils.configurator import Config
    config_obj = Config(config_dict, model=config_dict["model"], dataset=config_dict["dataset"])

    device = torch.device(args.device)
    model = load_model(config_obj.final_config_dict, checkpoint_path, device)

    from model_trainer.common.trainer import Trainer

    trainer = Trainer(model, config_obj)
    trainer.req_training = False
    trainer.model = model

    loaders = prepare_dataloaders(config_obj)
    output_dir = (args.output_dir or checkpoint_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_splits: Sequence[str] = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not requested_splits:
        raise ValueError("No dataset splits specified.")

    for split in requested_splits:
        if split not in loaders:
            raise ValueError(f"Unknown split '{split}'. Expected one of train, vali, test.")
        loader = loaders[split]
        if loader is None:
            raise RuntimeError(f"Split '{split}' is not available in the current configuration.")

        metrics, records = evaluate_split(trainer, loader, split)
        suffix = f"_{args.suffix}" if args.suffix else ""
        out_path = output_dir / f"{split}_samples{suffix}.jsonl"
        write_jsonl(out_path, records)

        if not args.quiet:
            metrics_str = ", ".join(f"{k}={float(v):.6f}" for k, v in metrics.items())
            print(f"[{split}] {metrics_str} -> {out_path}")


if __name__ == "__main__":
    main()
