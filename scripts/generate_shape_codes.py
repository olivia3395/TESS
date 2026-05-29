#!/usr/bin/env python3
"""Annotate FNSPID samples with a shape_code derived from ground truth moves.

The script reads JSON splits (train/vali/test) from a source directory,
classifies the ground_truth trajectory into one of five buckets, and writes a
new dataset directory that mirrors the inputs while appending a shape_code
field to every record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence


SHAPE_CODE_INCREASING = 1
SHAPE_CODE_DECREASING = 2
SHAPE_CODE_RISE_FALL = 3
SHAPE_CODE_FALL_RISE = 4
SHAPE_CODE_OSCILLATING = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify ground truth sequences into five movement shapes and add "
            "a shape_code field."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("MMTSF_LIB/dataset/FNSPID/ver_camf"),
        help="Directory containing the input JSON splits (train/vali/test).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("MMTSF_LIB/dataset/FNSPID/ver_synchronized_shape"),
        help="Output directory for the augmented JSON splits.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Minimum delta required to treat a step as increasing/decreasing.",
    )
    return parser.parse_args()


def parse_series(series: str) -> List[float]:
    """Convert a comma-separated numeric string into a float list."""
    parts = [part.strip() for part in series.split(",")]
    return [float(part) for part in parts if part]


def sign_from_diff(diff: float, tolerance: float) -> int:
    """Return 1 for upward move, -1 for downward move, or 0 within tolerance."""
    if diff > tolerance:
        return 1
    if diff < -tolerance:
        return -1
    return 0


def compress_signs(signs: Iterable[int]) -> List[int]:
    """Drop zeros and consecutive duplicates to highlight turning points."""
    compressed: List[int] = []
    for sign in signs:
        if sign == 0:
            continue
        if not compressed or sign != compressed[-1]:
            compressed.append(sign)
    return compressed


def classify_shape(values: Sequence[float], tolerance: float) -> int:
    """Assign a shape_code based on monotonicity and a single turning point."""
    if len(values) < 2:
        return SHAPE_CODE_OSCILLATING

    diffs = [values[idx + 1] - values[idx] for idx in range(len(values) - 1)]
    signs = compress_signs(sign_from_diff(diff, tolerance) for diff in diffs)

    if not signs:
        return SHAPE_CODE_OSCILLATING
    if len(signs) == 1:
        return (
            SHAPE_CODE_INCREASING
            if signs[0] > 0
            else SHAPE_CODE_DECREASING
        )
    if len(signs) == 2:
        first, second = signs
        if first == 1 and second == -1:
            return SHAPE_CODE_RISE_FALL
        if first == -1 and second == 1:
            return SHAPE_CODE_FALL_RISE
    return SHAPE_CODE_OSCILLATING


def annotate_file(
    source_path: Path, destination_path: Path, tolerance: float
) -> None:
    with source_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    annotated = []
    for record in payload:
        if "ground_truth" not in record:
            raise KeyError(
                f"Missing 'ground_truth' in record from {source_path}"
            )
        values = parse_series(record["ground_truth"])
        shape_code = classify_shape(values, tolerance)
        updated_record = dict(record)
        updated_record["shape_code"] = shape_code
        annotated.append(updated_record)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as file_obj:
        json.dump(annotated, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def main() -> None:
    args = parse_args()
    source_dir = args.source
    destination_dir = args.destination

    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found under {source_dir}")

    for json_file in json_files:
        target_path = destination_dir / json_file.name
        annotate_file(json_file, target_path, args.tolerance)
        print(f"Wrote {target_path}")


if __name__ == "__main__":
    main()
