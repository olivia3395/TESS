"""
Quick launcher for TimeMixer on the 3-variable FNSPID (timx combined) dataset.

Example:
    PYTHONPATH=src python scripts/train_timemixer_3var.py --gpu 0
"""

import argparse
import json
import sys
from pathlib import Path

# Make `layers.*` imports in models work without modifying package structure
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINER_ROOT = REPO_ROOT / "src" / "model_trainer"
for extra in [MODEL_TRAINER_ROOT]:
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.append(extra_str)

from model_trainer.common.dataset import FnspidDataset
import model_trainer.common.dataloader as dl
import model_trainer.models.timemixer as timemixer
from model_trainer.utils.quick_start import quick_start


def parse_args():
    parser = argparse.ArgumentParser(description="Train TimeMixer on multi-var timx data")
    parser.add_argument(
        "--dataset-alias",
        default="FNSPID/ver_3var_timx_combined",
        help="Registered dataset alias",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--force-rebuild",
        default=False,
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
    src_root = REPO_ROOT / "dataset" / "FNSPID" / "ver_3var_timx_globalonly"
    tgt_root = REPO_ROOT / "dataset" / "FNSPID" / "ver_3var_timx_syn_conbined"
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
                    "historical_data": combined_seq,  # flat length 15, downstream reshape to [5,3]
                    "ground_truth": sample["ground_truth"],
                }
            )

        with tgt_file.open("w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)

    with (tgt_root / "train.json").open() as f:
        probe = json.load(f)[0]
    seq_len = len(probe["historical_data"].split(","))
    pred_len = len(probe["ground_truth"].split(","))
    return seq_len, pred_len


def main():
    args = parse_args()
    flat_seq_len=15
    seq_len=5
    flat_seq_len, pred_len = prepare_combined_dataset(force_rebuild=args.force_rebuild)
    if flat_seq_len % 3 != 0:
        raise ValueError(f"Flat sequence length {flat_seq_len} is not divisible by 3 channels")
    seq_len = flat_seq_len // 3

    class ReshapeFnspidDataset(FnspidDataset):
        """Wrap FnspidDataset to reshape flat 3-var seq into [L, C] before batching."""

        def __getitem__(self, idx):
            sample = super().__getitem__(idx)

            def _reshape(x_tensor):
                x_tensor = x_tensor.float()
                expected = self.config.get('seq_len', seq_len) * self.config.get('enc_in', 3)
                if x_tensor.numel() != expected:
                    raise ValueError(
                        f"x length {x_tensor.numel()} mismatch expected {expected}; "
                        "ensure combined data matches seq_len*enc_in"
                    )
                return x_tensor.view(self.config.get('seq_len', seq_len), self.config.get('enc_in', 3))

            if isinstance(sample, tuple):
                x = sample[0]
                reshaped = _reshape(x)
                return (reshaped, *sample[1:])
            if isinstance(sample, dict):
                sample = dict(sample)
                sample['x'] = _reshape(sample['x'])
                return sample
            raise TypeError(f"Unsupported sample type: {type(sample)}")

    # Swap dataset class for this run only
    dl.data_dict['FNSPID'] = ReshapeFnspidDataset

    # Monkey-patch TimeMixer forward to accept [B, L, C] without extra unsqueeze
    def _patched_forward(self, x_enc):
        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(-1)
        elif x_enc.dim() != 3:
            raise ValueError(f"TimeMixer expects [B, L] or [B, L, C], got {x_enc.shape}")
        dec_out = self.forecast(x_enc)
        if dec_out.dim() == 3:
            # TimeMixer returns [B, pred_len, C]; 取第一通道对齐标签
            dec_out = dec_out[..., 0]
        self.dec_out = dec_out[:, -self.pred_len:]
        return self.dec_out

    timemixer.TimeMixer.forward = _patched_forward

    config_dict = {
        "dataset_alias": args.dataset_alias,
        "gpu_id": args.gpu,
        "distributed": False,
        # 仅使用数值序列输入，禁用文本/嵌入模态
        "use_multimodal": False,
        "use_text_news": False,
        "use_news_embedding": False,
        "use_llm_hidden": False,
        # 形状配置
        "seq_len": seq_len,
        "pred_len": pred_len,
        "label_len": 0,
        "batch_size": args.batch_size,
        "enc_in": 3,       # 三通道输入
        "c_out": 1,
        # TimeMixer 额外必需键
        "channel_independence": False,  # 3 通道共享嵌入
        "down_sampling_window": 1,
        "down_sampling_layers": 3,
        "down_sampling_method": "avg",
        "use_norm": 1,
        "embed": "timeF",
        "freq": "h",
    }

    quick_start(
        model="TimeMixer",
        dataset="FNSPID",
        config_dict=config_dict,
        save_model=True,
    )


if __name__ == "__main__":
    main()
