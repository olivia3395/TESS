"""
Quick launcher for TimeXer on the 3-variable FNSPID (timx combined) dataset.

Example:
    PYTHONPATH=src python scripts/train_timexer_3var.py --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINER_ROOT = REPO_ROOT / "src" / "model_trainer"
# 确保可以直接导入短路径 layers.*（TimeXer 等模型使用）
for extra in [MODEL_TRAINER_ROOT]:
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

from model_trainer.utils.quick_start import quick_start


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimeXer on multi-var timx data")
    parser.add_argument(
        "--dataset-alias",
        default="FNSPID/ver_3var_timx_combined",
        help="Registered dataset alias",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument(
        "--patch-len",
        type=int,
        default=5,
        help="Patch length for TimeXer (should divide seq_len=15)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild combined dataset from ver_3var_timx even if files exist",
    )
    return parser.parse_args()


def _combine_sample(sample):
    hist = [x.strip() for x in sample["historical_data"].split(",")]
    shape = [x.strip() for x in sample["shape_code"].split(",")]
    news = [x.strip() for x in sample["news"].split(",")]
    if not (len(hist) == len(shape) == len(news)):
        raise ValueError("historical_data/shape_code/news length mismatch")

    combined = []
    for h, s, n in zip(hist, shape, news):
        combined.extend([h, s, n])
    return ", ".join(combined)


def prepare_combined_dataset(force_rebuild: bool = False):
    """Ensure ver_3var_timx_combined exists by interleaving the 3 variables."""
    src_root = Path("dataset/FNSPID/ver_3var_timx")
    tgt_root = Path("dataset/FNSPID/ver_3var_timx_combined")
    tgt_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "vali", "test"]:
        src_file = src_root / f"{split}.json"
        tgt_file = tgt_root / f"{split}.json"
        if tgt_file.exists() and not force_rebuild:
            continue

        with src_file.open() as f:
            data = json.load(f)

        combined_data = []
        for sample in data:
            combined_seq = _combine_sample(sample)
            combined_data.append(
                {
                    "historical_data": combined_seq,
                    "ground_truth": sample["ground_truth"],
                }
            )

        with tgt_file.open("w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

    # read one sample to infer seq_len/pred_len
    with (tgt_root / "train.json").open() as f:
        probe = json.load(f)[0]
    seq_len = len(probe["historical_data"].split(","))
    pred_len = len(probe["ground_truth"].split(","))
    return seq_len, pred_len


def main():
    args = parse_args()

    seq_len, pred_len = prepare_combined_dataset(force_rebuild=args.force_rebuild)

    if seq_len % args.patch_len != 0:
        raise ValueError(f"patch_len={args.patch_len} should divide seq_len={seq_len}")

    config_dict = {
        "dataset_alias": args.dataset_alias,
        "gpu_id": args.gpu,
        "distributed": False,
        # 仅使用数值序列输入，禁用文本/嵌入模态
        "use_multimodal": False,
        "use_text_news": False,
        "use_news_embedding": False,
        "use_llm_hidden": False,
        # TimeXer 序列形状
        "seq_len": seq_len,
        "pred_len": pred_len,
        "patch_len": args.patch_len,
        "batch_size": args.batch_size,
        "enc_in": 1,          # 输入变量数（当前为展平序列）
        "features": "M",      # 标记为多变量输入，避免缺省键问题
    }

    quick_start(
        model="TimeXer",
        dataset="FNSPID",
        config_dict=config_dict,
        save_model=True,
    )


if __name__ == "__main__":
    main()
