"""Create learnable mock raw data and run the RC2 primitive cache pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primitive_inference_rc2.run_primitive_pipeline import run_pipeline


PRIMITIVES = ("distribution_shift", "volatility", "shape", "temporal_influence")
SPLITS = ("train", "vali", "test")


def make_record(idx: int, split: str, seq_len: int, pred_len: int, rng: random.Random) -> dict[str, object]:
    base = 5.0 + 0.03 * idx + {"train": 0.0, "vali": 0.5, "test": 1.0}[split]
    trend = 0.05 + 0.01 * ((idx % 5) - 2)
    seasonal = 0.04 * math.sin(idx * 0.3)
    x = [
        base + trend * t + seasonal * math.sin(t / 2.0) + rng.uniform(-0.01, 0.01)
        for t in range(seq_len)
    ]
    # Learnable future: mostly continuation from the observed trend plus tiny noise.
    last = x[-1]
    y = [
        last + trend * (t + 1) + 0.5 * seasonal * math.sin((seq_len + t) / 2.0)
        + rng.uniform(-0.01, 0.01)
        for t in range(pred_len)
    ]
    direction = "upward" if trend >= 0 else "downward"
    return {
        "news": f"MOCK company reports a stable {direction} operating trend for sample {idx}.",
        "historical_data": x,
        "ground_truth": y,
        "company_name": "MOCK",
    }


def write_split(root: Path, dataset: str, split: str, count: int, seq_len: int, pred_len: int, seed: int) -> None:
    rng = random.Random(seed)
    records = [make_record(idx, split, seq_len, pred_len, rng) for idx in range(count)]
    split_path = root / "dataset" / dataset / f"{split}.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    root = Path(args.root)
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    write_split(root, args.dataset, "train", args.n_train, args.seq_len, args.pred_len, args.seed)
    write_split(root, args.dataset, "vali", args.n_vali, args.seq_len, args.pred_len, args.seed + 1)
    write_split(root, args.dataset, "test", args.n_test, args.seq_len, args.pred_len, args.seed + 2)

    pipeline_result = run_pipeline(
        SimpleNamespace(
            root=str(root),
            dataset=args.dataset,
            primitives=list(PRIMITIVES),
            splits=list(SPLITS),
            backend="mock",
            prompt_source="fallback",
            num_samples=args.num_samples,
            limit=None,
            seed=args.seed,
            stages=["gt", "sampled", "gate", "grouped", "audit"],
            overwrite=True,
            temperature=0.7,
            top_p=0.95,
            max_tokens=64,
        )
    )
    return {"root": str(root), "manifest_path": pipeline_result["manifest_path"]}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create mock TESS data and primitive caches.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="fnspid")
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--n-vali", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--pred-len", type=int, default=6)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    result = run(build_arg_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
