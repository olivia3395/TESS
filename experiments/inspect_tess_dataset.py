"""Inspect raw TESS splits joined with grouped primitive cache records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primitive_inference_rc2.tess_dataset import DEFAULT_PRIMITIVE_ORDER, TESS_Dataset


def tensor_scalar(value: torch.Tensor | int) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def counter_to_json(counter: Counter[int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def inspect_split(root: str | Path, dataset_name: str, split: str) -> dict[str, Any]:
    dataset = TESS_Dataset(root=root, dataset=dataset_name, split=split)
    primitive_order = tuple(getattr(dataset, "primitive_order", DEFAULT_PRIMITIVE_ORDER))
    vocab_sizes = dict(dataset.primitive_vocab_sizes)
    text_missing = {primitive: 0 for primitive in primitive_order}
    gt_missing = {primitive: 0 for primitive in primitive_order}
    text_label_counts = {primitive: Counter() for primitive in primitive_order}
    gt_label_counts = {primitive: Counter() for primitive in primitive_order}
    gate_positive = {primitive: 0.0 for primitive in primitive_order}
    gate_total_positive = 0.0
    sample_ids = []
    duplicate_count = 0
    seen_sample_ids = set()
    index_mismatch_count = 0
    x_finite = True
    y_finite = True
    x_feature_dims = Counter()
    y_feature_dims = Counter()

    for row_idx in range(len(dataset)):
        item = dataset[row_idx]
        sample_id = tensor_scalar(item["sample_id"])
        sample_ids.append(sample_id)
        if sample_id in seen_sample_ids:
            duplicate_count += 1
        seen_sample_ids.add(sample_id)
        if sample_id < 0 or sample_id >= len(dataset.raw_records):
            index_mismatch_count += 1

        x = item["x"]
        y = item["y"]
        x_finite = x_finite and bool(torch.isfinite(x).all().item())
        y_finite = y_finite and bool(torch.isfinite(y).all().item())
        x_feature_dims[int(x.shape[-1])] += 1
        y_feature_dims[int(y.shape[-1])] += 1

        for primitive_idx, primitive in enumerate(primitive_order):
            text_id = int(item["text_primitive_ids"][primitive_idx].item())
            gt_id = int(item["gt_primitive_ids"][primitive_idx].item())
            if not bool(item["text_primitive_mask"][primitive_idx].item()):
                text_missing[primitive] += 1
                text_label_counts[primitive]["UNK"] += 1
            else:
                text_label_counts[primitive][str(text_id)] += 1
            if not bool(item["gt_primitive_mask"][primitive_idx].item()):
                gt_missing[primitive] += 1
                gt_label_counts[primitive]["UNK"] += 1
            else:
                gt_label_counts[primitive][str(gt_id)] += 1
            gate_value = float(item["gate_targets"][primitive_idx].item())
            gate_positive[primitive] += gate_value
            gate_total_positive += gate_value

    num_samples = len(dataset)
    text_missing_ratio = {
        primitive: text_missing[primitive] / num_samples if num_samples else 0.0
        for primitive in primitive_order
    }
    gt_missing_ratio = {
        primitive: gt_missing[primitive] / num_samples if num_samples else 0.0
        for primitive in primitive_order
    }
    gate_positive_rate = {
        primitive: gate_positive[primitive] / num_samples if num_samples else 0.0
        for primitive in primitive_order
    }
    return {
        "num_samples": num_samples,
        "seq_len_distribution": counter_to_json(dataset.seq_len_counts),
        "pred_len_distribution": counter_to_json(dataset.pred_len_counts),
        "x_finite": x_finite,
        "y_finite": y_finite,
        "single_variable": set(x_feature_dims) == {1} and set(y_feature_dims) == {1},
        "x_feature_dim_distribution": counter_to_json(x_feature_dims),
        "y_feature_dim_distribution": counter_to_json(y_feature_dims),
        "sample_id_min": min(sample_ids) if sample_ids else None,
        "sample_id_max": max(sample_ids) if sample_ids else None,
        "sample_id_duplicate_count": duplicate_count,
        "sample_id_missing_or_index_mismatch_count": index_mismatch_count,
        "text_primitive_missing_ratio": text_missing_ratio,
        "gt_primitive_missing_ratio": gt_missing_ratio,
        "text_primitive_label_distribution": {
            primitive: dict(text_label_counts[primitive]) for primitive in primitive_order
        },
        "gt_primitive_label_distribution": {
            primitive: dict(gt_label_counts[primitive]) for primitive in primitive_order
        },
        "gate_target_positive_rate": gate_positive_rate,
        "overall_gate_target_positive_rate": (
            gate_total_positive / (num_samples * len(primitive_order)) if num_samples else 0.0
        ),
        "primitive_order": list(primitive_order),
        "primitive_vocab_sizes_with_unk": vocab_sizes,
    }


def all_splits_are_5_in_5_out(split_reports: dict[str, dict[str, Any]]) -> bool:
    return all(
        report["seq_len_distribution"] == {"5": report["num_samples"]}
        and report["pred_len_distribution"] == {"5": report["num_samples"]}
        for report in split_reports.values()
    )


def print_report(report: dict[str, Any]) -> None:
    print(f"dataset={report['dataset']} root={report['root']}")
    print(f"primitive_order={report['primitive_order']}")
    print(f"primitive_vocab_sizes_with_unk={report['primitive_vocab_sizes_with_unk']}")
    print(f"all_splits_5_in_5_out={report['all_splits_5_in_5_out']}")
    print(f"all_splits_single_variable={report['all_splits_single_variable']}")
    if not report["all_splits_5_in_5_out"]:
        print("WARNING: one or more splits are not 5-in-5-out.")
    for split, split_report in report["splits"].items():
        print(f"\n[{split}]")
        print(f"N={split_report['num_samples']}")
        print(f"seq_len: {split_report['seq_len_distribution']}")
        print(f"pred_len: {split_report['pred_len_distribution']}")
        print(f"single_variable: {split_report['single_variable']}")
        print("text missing ratio:")
        for primitive, value in split_report["text_primitive_missing_ratio"].items():
            print(f"  {primitive}: {value:.6f}")
        print("gt missing ratio:")
        for primitive, value in split_report["gt_primitive_missing_ratio"].items():
            print(f"  {primitive}: {value:.6f}")
        print("gate positive rate:")
        for primitive, value in split_report["gate_target_positive_rate"].items():
            print(f"  {primitive}: {value:.6f}")
        print(f"overall gate positive rate: {split_report['overall_gate_target_positive_rate']:.6f}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    split_reports = {
        split: inspect_split(args.root, args.dataset, split)
        for split in ("train", "vali", "test")
    }
    first_split = next(iter(split_reports.values()))
    report = {
        "dataset": args.dataset,
        "root": str(Path(args.root)),
        "primitive_order": first_split["primitive_order"],
        "primitive_vocab_sizes_with_unk": first_split["primitive_vocab_sizes_with_unk"],
        "all_splits_5_in_5_out": all_splits_are_5_in_5_out(split_reports),
        "all_splits_single_variable": all(
            split_report["single_variable"] for split_report in split_reports.values()
        ),
        "splits": split_reports,
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect TESS_Dataset splits.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    report = run(build_arg_parser().parse_args())
    print_report(report)


if __name__ == "__main__":
    main()
