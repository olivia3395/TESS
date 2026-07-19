"""Path and JSON helpers for primitive cache artifacts.

All cache writers use stable human-readable JSON so intermediate GT, sampled,
gate, grouped-gate, report, and manifest files can be inspected directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def sampled_inference_cache_path(
    root: str | Path,
    dataset: str,
    primitive: str,
    split: str,
) -> Path:
    return (
        Path(root)
        / "data_cache"
        / "sampled_inference"
        / dataset
        / primitive
        / f"{split}.json"
    )


def gt_primitive_cache_path(
    root: str | Path,
    dataset: str,
    split: str,
) -> Path:
    return Path(root) / "data_cache" / "gt_primitive" / dataset / f"{split}.json"


def gt_primitive_metadata_path(root: str | Path, dataset: str) -> Path:
    return Path(root) / "data_cache" / "gt_primitive" / dataset / "metadata.json"


def gate_cache_path(
    root: str | Path,
    dataset: str,
    primitive: str,
    split: str,
) -> Path:
    return Path(root) / "data_cache" / "gate" / dataset / primitive / f"{split}.json"


def grouped_gate_cache_path(root: str | Path, dataset: str, split: str) -> Path:
    return Path(root) / "data_cache" / "gate_grouped" / dataset / f"{split}.json"


def report_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "data_cache" / "reports" / filename


def write_json(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_json_object(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
