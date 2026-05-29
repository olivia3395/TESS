#!/usr/bin/env python3
"""Utility to inspect a single sample from the FNSPID dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a sample from dataset/FNSPID/<version>/<split>.json."
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Dataset version folder, e.g. ver_camf or ver_turn1_final.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Sample position within the split (1-based).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "vali", "test"),
        default="train",
        help="Which split to inspect (default: train).",
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset/FNSPID",
        help="Root directory that contains the version folders.",
    )
    return parser.parse_args()


def load_samples(split_path: Path) -> list[dict]:
    if not split_path.exists():
        sys.exit(f"Missing split file: {split_path}")
    with split_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        sys.exit(f"Unexpected payload in {split_path}; expected a list.")
    return data


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    version_path = dataset_root / args.version
    if not version_path.exists():
        sys.exit(f"Dataset version not found: {version_path}")

    split_path = version_path / f"{args.split}.json"
    samples = load_samples(split_path)

    if args.index < 1:
        sys.exit("Index must be >= 1 because the dataset is 1-based in this tool.")
    zero_based = args.index - 1
    if zero_based >= len(samples):
        sys.exit(
            f"Index {args.index} is out of range; {split_path} only has {len(samples)} rows."
        )

    sample = samples[zero_based]
    print(f"Version: {args.version}")
    print(f"Split: {args.split} ({len(samples)} samples)")
    print(f"Sample #: {args.index}")
    print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
# python scripts/show_fnspid_sample.py --version ver_camf --split train --index 1,