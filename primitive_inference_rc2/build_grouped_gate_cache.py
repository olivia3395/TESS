"""Build grouped per-sample gate caches and aggregate audit reports.

Flat gate caches are primitive-specific. The grouped cache combines all
primitive gate records for the same sample_id into one training-friendly record
after strict sample_id alignment checks within each split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .cache_io import (
    gate_cache_path,
    grouped_gate_cache_path,
    gt_primitive_metadata_path,
    report_path,
    write_json,
    write_json_object,
)
from .dataset_specs import VALID_SPLITS, get_dataset_spec


DEFAULT_PRIMITIVE_ORDER = (
    "distribution_shift",
    "volatility",
    "shape",
    "temporal_influence",
)
GROUPED_PRIMITIVE_FIELDS = (
    "pred_label",
    "gt_label",
    "gate_target",
    "label_id",
    "gt_label_id",
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
        raise FileNotFoundError(f"Required gate cache file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected record {idx} in {path} to be an object")
    return data


def records_by_sample_id(
    records: list[dict[str, Any]],
    primitive: str,
    split: str,
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, int):
            raise ValueError(f"{primitive}/{split} has missing or invalid sample_id.")
        if sample_id in indexed:
            raise ValueError(f"{primitive}/{split} has duplicate sample_id={sample_id}.")
        if record.get("primitive") != primitive:
            raise ValueError(
                f"{primitive}/{split} record sample_id={sample_id} has primitive "
                f"{record.get('primitive')!r}."
            )
        if record.get("split") != split:
            raise ValueError(
                f"{primitive}/{split} record sample_id={sample_id} has split "
                f"{record.get('split')!r}."
            )
        indexed[sample_id] = record
    return indexed


def validate_alignment(
    split: str,
    primitive_indexes: dict[str, dict[int, dict[str, Any]]],
) -> list[int]:
    reference_primitive = next(iter(primitive_indexes))
    reference_ids = set(primitive_indexes[reference_primitive])
    issues = []
    for primitive, index in primitive_indexes.items():
        ids = set(index)
        missing = sorted(reference_ids - ids)
        extra = sorted(ids - reference_ids)
        if missing:
            issues.append(f"{primitive} missing ids relative to {reference_primitive}: {missing}")
        if extra:
            issues.append(f"{primitive} has extra ids relative to {reference_primitive}: {extra}")
    if issues:
        raise ValueError(f"Gate sample_id alignment failed for split={split}: " + "; ".join(issues))
    return sorted(reference_ids)


def build_grouped_records(
    dataset: str,
    split: str,
    primitive_order: tuple[str, ...],
    primitive_indexes: dict[str, dict[int, dict[str, Any]]],
    sample_ids: list[int],
) -> list[dict[str, Any]]:
    grouped_records = []
    for sample_id in sample_ids:
        primitives: dict[str, dict[str, Any]] = {}
        gate_targets: dict[str, int] = {}
        label_ids: dict[str, int | None] = {}
        gt_label_ids: dict[str, int | None] = {}
        for primitive in primitive_order:
            flat_record = primitive_indexes[primitive][sample_id]
            gate_target = flat_record.get("gate_target")
            if gate_target not in (0, 1):
                raise ValueError(
                    f"{primitive}/{split} sample_id={sample_id} has invalid gate_target={gate_target!r}."
                )
            primitive_payload = {
                field: flat_record.get(field) for field in GROUPED_PRIMITIVE_FIELDS
            }
            primitives[primitive] = primitive_payload
            gate_targets[primitive] = gate_target
            label_ids[primitive] = flat_record.get("label_id")
            gt_label_ids[primitive] = flat_record.get("gt_label_id")

        grouped_records.append(
            {
                "dataset": dataset,
                "split": split,
                "sample_id": sample_id,
                "schema": "legacy_v1",
                "primitive_order": list(primitive_order),
                "primitives": primitives,
                "gate_targets": gate_targets,
                "label_ids": label_ids,
                "gt_label_ids": gt_label_ids,
            }
        )
    return grouped_records


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def audit_flat_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    positives = sum(1 for record in records if record.get("gate_target") == 1)
    negatives = sum(1 for record in records if record.get("gate_target") == 0)
    invalid_pred_count = sum(1 for record in records if record.get("label_id") is None)
    numeric_fields = {
        "parse_rate": [],
        "self_consistency": [],
        "margin": [],
    }
    for record in records:
        for field, values in numeric_fields.items():
            value = record.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
    return {
        "record_count": len(records),
        "num_positive_gate_target": positives,
        "num_negative_gate_target": negatives,
        "positive_rate": positives / len(records) if records else 0.0,
        "invalid_pred_count": invalid_pred_count,
        "missing_gt_count": 0,
        "parse_rate": summary(numeric_fields["parse_rate"]),
        "self_consistency": summary(numeric_fields["self_consistency"]),
        "margin": summary(numeric_fields["margin"]),
        "prompt_source_counts": dict(Counter(record.get("prompt_source") for record in records)),
        "backend_counts": dict(Counter(record.get("backend") for record in records)),
        "model_counts": dict(Counter(record.get("model") for record in records)),
    }


def gt_metadata_contains_all(
    root: str | Path,
    dataset: str,
    primitive_order: tuple[str, ...],
) -> bool:
    metadata_path = gt_primitive_metadata_path(root, dataset)
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    thresholds = metadata.get("thresholds", {})
    return all(primitive in thresholds for primitive in primitive_order)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_spec = get_dataset_spec(args.dataset, args.root)
    splits = tuple(args.splits)
    primitive_order = tuple(args.primitives)
    output_paths: dict[str, str] = {}
    grouped_counts: dict[str, int] = {}
    flat_counts: dict[str, dict[str, int]] = {}
    audit_by_split: dict[str, dict[str, Any]] = {}

    for split in splits:
        primitive_indexes = {}
        split_audit: dict[str, Any] = {}
        flat_counts[split] = {}
        for primitive in primitive_order:
            path = gate_cache_path(args.root, dataset_spec.name, primitive, split)
            records = load_records(path)
            flat_counts[split][primitive] = len(records)
            primitive_indexes[primitive] = records_by_sample_id(records, primitive, split)
            split_audit[primitive] = audit_flat_records(records)

        sample_ids = validate_alignment(split, primitive_indexes)
        grouped_records = build_grouped_records(
            dataset=dataset_spec.name,
            split=split,
            primitive_order=primitive_order,
            primitive_indexes=primitive_indexes,
            sample_ids=sample_ids,
        )
        output_path = grouped_gate_cache_path(args.root, dataset_spec.name, split)
        write_json(output_path, grouped_records)
        output_paths[split] = str(output_path)
        grouped_counts[split] = len(grouped_records)
        audit_by_split[split] = split_audit

    metadata_has_all = gt_metadata_contains_all(args.root, dataset_spec.name, primitive_order)
    audit_report = {
        "dataset": dataset_spec.name,
        "splits": list(splits),
        "primitives": list(primitive_order),
        "primitive_order": list(primitive_order),
        "grouped_gate_record_counts": grouped_counts,
        "flat_gate_record_counts": flat_counts,
        "flat_gate_audit": audit_by_split,
        "gt_metadata_contains_all_primitives": metadata_has_all,
    }
    audit_path = report_path(args.root, f"primitive_cache_audit_{dataset_spec.name}.json")
    write_json_object(audit_path, audit_report)
    return {
        "output_paths": output_paths,
        "audit_path": str(audit_path),
        "audit": audit_report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build grouped gate cache and audit report.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--splits", nargs="+", default=list(VALID_SPLITS))
    parser.add_argument("--primitives", nargs="+", default=list(DEFAULT_PRIMITIVE_ORDER))
    return parser


def main() -> None:
    result = run(build_arg_parser().parse_args())
    print(json.dumps(result["output_paths"], ensure_ascii=False, indent=2))
    print(result["audit_path"])


if __name__ == "__main__":
    main()
