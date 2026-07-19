"""Build flat per-primitive gate caches.

Flat gate records align sampled predictions with GT labels by sample_id for one
primitive and split. gate_target is 1 only when the sampled pred_label equals
the GT primitive label; invalid or missing predictions become negative targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .cache_io import (
    gate_cache_path,
    gt_primitive_cache_path,
    report_path,
    sampled_inference_cache_path,
    write_json,
    write_json_object,
)
from .dataset_specs import VALID_SPLITS
from .primitive_specs import get_primitive_spec


SUPPORTED_PRIMITIVES = (
    "distribution_shift",
    "volatility",
    "shape",
    "temporal_influence",
)
PRESERVED_SAMPLED_FIELDS = (
    "margin",
    "self_consistency",
    "parse_rate",
    "valid_count",
    "num_samples",
    "sample_probs",
    "margin_type",
    "backend",
    "prompt_source",
    "prompt_template_path",
    "model",
)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required cache file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected record {idx} in {path} to be an object")
    return data


def build_gt_index(
    gt_records: list[dict[str, Any]],
    primitive: str,
    valid_labels: set[str],
) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for record in gt_records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, int):
            raise ValueError("GT record has missing or non-integer sample_id.")
        if sample_id in index:
            raise ValueError(f"Duplicate GT sample_id: {sample_id}")
        gt_label = record.get("gt_labels", {}).get(primitive)
        if gt_label not in valid_labels:
            raise ValueError(
                f"GT record sample_id={sample_id} has missing or invalid "
                f"{primitive} label: {gt_label!r}"
            )
        index[sample_id] = record
    return index


def build_gate_cache(args: argparse.Namespace) -> dict[str, Any]:
    if args.primitive not in SUPPORTED_PRIMITIVES:
        supported = ", ".join(SUPPORTED_PRIMITIVES)
        raise ValueError(f"Supported primitives: {supported}")
    if args.dataset != "fnspid":
        raise ValueError("Step 3C only supports --dataset fnspid")
    if args.split not in VALID_SPLITS:
        valid = ", ".join(VALID_SPLITS)
        raise ValueError(f"Unknown split '{args.split}'. Valid splits: {valid}")

    primitive_spec = get_primitive_spec(args.primitive)
    valid_labels = set(primitive_spec.labels)
    sampled_path = sampled_inference_cache_path(
        args.root, args.dataset, args.primitive, args.split
    )
    gt_path = gt_primitive_cache_path(args.root, args.dataset, args.split)
    output_path = gate_cache_path(args.root, args.dataset, args.primitive, args.split)
    report_output_path = report_path(
        args.root, f"gate_{args.primitive}_{args.dataset}_{args.split}.json"
    )

    sampled_records = load_records(sampled_path)
    gt_records = load_records(gt_path)
    gt_index = build_gt_index(gt_records, args.primitive, valid_labels)

    output_records: list[dict[str, Any]] = []
    missing_gt_count = 0
    invalid_pred_count = 0
    num_positive = 0

    for sampled_record in sampled_records:
        sample_id = sampled_record.get("sample_id")
        if not isinstance(sample_id, int):
            raise ValueError("Sampled inference record has missing or non-integer sample_id.")

        gt_record = gt_index.get(sample_id)
        if gt_record is None:
            missing_gt_count += 1
            raise ValueError(f"Missing GT record for sampled sample_id={sample_id}")

        pred_label = sampled_record.get("pred_label")
        gt_label = gt_record["gt_labels"][args.primitive]
        gt_label_id = primitive_spec.label_to_id[gt_label]

        if pred_label in valid_labels:
            label_id = primitive_spec.label_to_id[pred_label]
            gate_target = int(pred_label == gt_label)
        else:
            invalid_pred_count += 1
            label_id = None
            gate_target = 0

        num_positive += gate_target
        output_record = {
            "dataset": args.dataset,
            "split": args.split,
            "sample_id": sample_id,
            "primitive": args.primitive,
            "schema": "legacy_v1",
            "pred_label": pred_label,
            "gt_label": gt_label,
            "gate_target": gate_target,
            "label_id": label_id,
            "gt_label_id": gt_label_id,
        }
        for field in PRESERVED_SAMPLED_FIELDS:
            output_record[field] = sampled_record.get(field)
        output_records.append(output_record)

    write_json(output_path, output_records)

    num_gate_records = len(output_records)
    num_negative = num_gate_records - num_positive
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "primitive": args.primitive,
        "num_sampled_records": len(sampled_records),
        "num_gate_records": num_gate_records,
        "num_positive_gate_target": num_positive,
        "num_negative_gate_target": num_negative,
        "positive_rate": num_positive / num_gate_records if num_gate_records else 0.0,
        "missing_gt_count": missing_gt_count,
        "invalid_pred_count": invalid_pred_count,
        "input_sampled_path": str(sampled_path),
        "input_gt_path": str(gt_path),
        "output_gate_path": str(output_path),
    }
    write_json_object(report_output_path, report)
    return {"output_path": str(output_path), "report_path": str(report_output_path), "report": report}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build minimal gate cache.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--primitive", required=True)
    return parser


def main() -> None:
    result = build_gate_cache(build_arg_parser().parse_args())
    print(result["output_path"])
    print(result["report_path"])


if __name__ == "__main__":
    main()
