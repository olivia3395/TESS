"""Build or update split-level GT primitive caches.

The cache is one record per raw sample with nested gt_labels and gt_scores.
Running a single primitive updates that primitive in existing records while
preserving previously generated primitive labels and threshold metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .cache_io import (
    gt_primitive_cache_path,
    gt_primitive_metadata_path,
    report_path,
    write_json,
    write_json_object,
)
from .dataset_specs import VALID_SPLITS, get_dataset_spec
from .gt_labelers import (
    DistributionShiftLabeler,
    ShapeLabeler,
    TemporalInfluenceLabeler,
    VALID_DISTRIBUTION_SHIFT_LABELS,
    VALID_SHAPE_LABELS,
    VALID_TEMPORAL_INFLUENCE_LABELS,
    VALID_VOLATILITY_LABELS,
    VolatilityLabeler,
)


SUPPORTED_PRIMITIVES = (
    "distribution_shift",
    "volatility",
    "shape",
    "temporal_influence",
)
LABELERS = {
    "distribution_shift": DistributionShiftLabeler,
    "volatility": VolatilityLabeler,
    "shape": ShapeLabeler,
    "temporal_influence": TemporalInfluenceLabeler,
}
VALID_LABELS = {
    "distribution_shift": VALID_DISTRIBUTION_SHIFT_LABELS,
    "volatility": VALID_VOLATILITY_LABELS,
    "shape": VALID_SHAPE_LABELS,
    "temporal_influence": VALID_TEMPORAL_INFLUENCE_LABELS,
}


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected record {idx} in {path} to be an object")
    return data


def load_existing_cache(path: Path) -> tuple[list[dict[str, Any]] | None, bool]:
    if not path.exists():
        return None, False
    return load_records(path), True


def build_split_cache(
    dataset: str,
    split: str,
    records: list[dict[str, Any]],
    primitive: str,
    labeler: (
        DistributionShiftLabeler
        | VolatilityLabeler
        | ShapeLabeler
        | TemporalInfluenceLabeler
    ),
    thresholds: dict[str, Any],
    existing_records: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    if existing_records is not None and len(existing_records) != len(records):
        raise ValueError(
            f"Existing GT cache for {dataset}/{split} has {len(existing_records)} "
            f"records but raw split has {len(records)} records."
        )

    output_records = []
    label_counts: Counter[str] = Counter()
    valid_labels = set(VALID_LABELS[primitive])
    for sample_id, record in enumerate(records):
        label, scores = labeler.compute(record, thresholds)
        if label not in valid_labels:
            raise ValueError(f"Unexpected {primitive} label: {label}")
        label_counts[label] += 1
        if existing_records is None:
            output_record = {
                "dataset": dataset,
                "split": split,
                "sample_id": sample_id,
                "schema": "legacy_v1",
            }
        else:
            output_record = dict(existing_records[sample_id])
            if output_record.get("sample_id") != sample_id:
                raise ValueError(
                    f"Existing GT cache sample_id mismatch in {split}: "
                    f"expected {sample_id}, got {output_record.get('sample_id')}"
                )
            output_record.setdefault("dataset", dataset)
            output_record.setdefault("split", split)
            output_record.setdefault("schema", "legacy_v1")

        output_record.setdefault("gt_labels", {})
        output_record.setdefault("gt_scores", {})
        output_record.setdefault("threshold_ref", {})
        output_record["gt_labels"][primitive] = label
        output_record["gt_scores"][primitive] = scores
        output_record["threshold_ref"][primitive] = {
            "path": f"data_cache/gt_primitive/{dataset}/metadata.json",
            "fit_split": thresholds["fit_split"],
            "method": thresholds["method"],
        }
        output_records.append(output_record)
    return output_records, label_counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.primitive not in SUPPORTED_PRIMITIVES:
        supported = ", ".join(SUPPORTED_PRIMITIVES)
        raise ValueError(f"Supported primitives: {supported}")

    dataset_spec = get_dataset_spec(args.dataset, args.root)
    split_records = {
        split: load_records(dataset_spec.split_path(split)) for split in VALID_SPLITS
    }

    labeler = LABELERS[args.primitive]()
    thresholds = labeler.fit(split_records["train"])

    metadata_path = gt_primitive_metadata_path(args.root, dataset_spec.name)
    if metadata_path.exists():
        threshold_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        threshold_payload.setdefault("dataset", dataset_spec.name)
        threshold_payload.setdefault("schema", "legacy_v1")
        threshold_payload.setdefault("thresholds", {})
    else:
        threshold_payload = {
            "dataset": dataset_spec.name,
            "schema": "legacy_v1",
            "thresholds": {},
        }
    threshold_payload["thresholds"][args.primitive] = thresholds
    threshold_payload["notes"] = (
        "Thresholds are fitted from train split only and applied to train/vali/test."
    )
    write_json_object(metadata_path, threshold_payload)

    report: dict[str, Any] = {
        "dataset": dataset_spec.name,
        "primitive": args.primitive,
        "schema": "legacy_v1",
        "threshold_source": "train",
        "thresholds": thresholds,
        "cache_update_mode_by_split": {},
        "splits": {},
    }

    output_paths = {}
    for split, records in split_records.items():
        output_path = gt_primitive_cache_path(args.root, dataset_spec.name, split)
        existing_records, existed = load_existing_cache(output_path)
        output_records, label_counts = build_split_cache(
            dataset=dataset_spec.name,
            split=split,
            records=records,
            primitive=args.primitive,
            labeler=labeler,
            thresholds=thresholds,
            existing_records=existing_records,
        )
        write_json(output_path, output_records)
        output_paths[split] = str(output_path)
        report["cache_update_mode_by_split"][split] = (
            "updated_existing" if existed else "created_new"
        )
        report["splits"][split] = {
            "num_records": len(output_records),
            "label_counts": {
                label: label_counts.get(label, 0)
                for label in VALID_LABELS[args.primitive]
            },
        }

    report_output_path = report_path(
        args.root, f"gt_{args.primitive}_{dataset_spec.name}.json"
    )
    write_json_object(report_output_path, report)
    return {
        "metadata_path": str(metadata_path),
        "report_path": str(report_output_path),
        "output_paths": output_paths,
        "report": report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build GT primitive cache.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--primitive", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run(args)
    print(json.dumps(result["output_paths"], ensure_ascii=False, indent=2))
    print(result["metadata_path"])
    print(result["report_path"])


if __name__ == "__main__":
    main()
