#!/usr/bin/env python3
"""Collect MSE/MAE/RMSE metrics for specified ver_gen runs into a single file."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

METRIC_KEYS = ("MSE", "MAE", "RMSE")


@dataclass(frozen=True)
class RunSpec:
    key: str
    description: str
    run_dir: Path


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        key="gen1",
        description="Five-step analysis + strength + direction",
        run_dir=Path("output/FNSPID/ver_gen1/MultiModal_Baseline/Nov-09-2025-15-48-57"),
    ),
    RunSpec(
        key="gen2",
        description="Five-step strength + direction",
        run_dir=Path("output/FNSPID/ver_gen2/MultiModal_Baseline/Nov-09-2025-15-48-56"),
    ),
    RunSpec(
        key="gen9",
        description="Five-step direction (01 encoded)",
        run_dir=Path("output/FNSPID/ver_gen9/MultiModal_Baseline/Nov-09-2025-16-46-04"),
    ),
    RunSpec(
        key="gen4",
        description="Weekly global direction",
        run_dir=Path("output/FNSPID/ver_gen4/MultiModal_Baseline/Nov-09-2025-15-48-56"),
    ),
    RunSpec(
        key="gen5",
        description="GT vs historical mean direction (5 steps)",
        run_dir=Path("output/FNSPID/ver_gen5/MultiModal_Baseline/Nov-09-2025-15-48-56"),
    ),
    RunSpec(
        key="gen6",
        description="GT vs previous step direction (5 steps)",
        run_dir=Path("output/FNSPID/ver_gen6/MultiModal_Baseline/Nov-09-2025-15-48-56"),
    ),
    RunSpec(
        key="gen8",
        description="GT vs historical mean strength + direction",
        run_dir=Path("output/FNSPID/ver_gen8/MultiModal_Baseline/Nov-09-2025-16-00-41"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect mse/mae/rmse from the manifest.json files of selected runs."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/selected_gen_metrics.json"),
        help="Where to store the collected metrics (default: analysis/selected_gen_metrics.json).",
    )
    return parser.parse_args()


def read_metrics(manifest_path: Path) -> Dict[str, float | None]:
    with manifest_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    metrics = payload.get("best_metrics") or payload.get("best_test_metrics") or {}
    return {key: float(metrics[key]) if key in metrics else None for key in METRIC_KEYS}


def collect_metrics(specs: Iterable[RunSpec]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for spec in specs:
        manifest = spec.run_dir / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Manifest not found for {spec.key}: {manifest}")
        metrics = read_metrics(manifest)
        record: Dict[str, object] = {
            "key": spec.key,
            "description": spec.description,
            "run_dir": str(spec.run_dir),
            "manifest": str(manifest),
        }
        record.update(metrics)
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    records = collect_metrics(RUN_SPECS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2, ensure_ascii=False)
    print(f"Saved metrics for {len(records)} runs to {args.output}")


if __name__ == "__main__":
    main()

