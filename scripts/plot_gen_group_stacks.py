#!/usr/bin/env python
"""Plot stacked bars for curated ver_gen groupings with natural-language labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SegmentConfig:
    version: str
    label: str


@dataclass(frozen=True)
class GroupConfig:
    label: str
    segments: Sequence[SegmentConfig]


GROUPS: Sequence[GroupConfig] = (
    GroupConfig(
        label="LLM Analysis",
        segments=(
            SegmentConfig("ver_gen2", "Five-Day Trend Strength"),
            SegmentConfig("ver_gen9", "Five-Day Trend"),
        ),
    ),
    GroupConfig(
        label="gt Compare synthetic data against historical averages",
        segments=(
            SegmentConfig("ver_gen8", "Five-Day Trend Strength"),
            SegmentConfig("ver_gen5", "Five-Day Trend"),
            
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw stacked bars comparing curated ver_gen groupings."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/FNSPID"),
        help="Root directory that stores per-version outputs (default: output/FNSPID).",
    )
    parser.add_argument(
        "--model",
        default="MultiModal_Baseline",
        help="Model sub-folder to look up (default: MultiModal_Baseline).",
    )
    parser.add_argument(
        "--metric",
        default="MSE",
        help="Metric key to extract from manifest.json (default: MSE).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/FNSPID/mse_gen_group_stacks.png"),
        help="File to save the figure (default: output/FNSPID/mse_gen_group_stacks.png).",
    )
    return parser.parse_args()


def locate_latest_manifest(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    manifests = sorted(
        (p for p in model_dir.glob("*/manifest.json") if p.is_file()),
        key=lambda item: item.stat().st_mtime,
    )
    return manifests[-1] if manifests else None


def read_metric(manifest_path: Path, metric_key: str) -> float:
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("best_metrics") or payload.get("best_test_metrics") or {}
    value = metrics.get(metric_key, math.nan)
    return float(value)


def collect_group_metrics(
    base_dir: Path, model: str, metric_key: str
) -> Tuple[List[str], List[List[float]], Dict[str, Path]]:
    x_labels: List[str] = []
    grouped_values: List[List[float]] = []
    manifest_map: Dict[str, Path] = {}

    for group in GROUPS:
        x_labels.append(group.label)
        segment_values: List[float] = []

        for seg in group.segments:
            manifest = locate_latest_manifest(base_dir / seg.version / model)
            if manifest is None:
                value = math.nan
            else:
                value = read_metric(manifest, metric_key)
                manifest_map[f"{seg.version}:{model}"] = manifest
            segment_values.append(value)

        grouped_values.append(segment_values)

    return x_labels, grouped_values, manifest_map


def plot_stacked_bars(
    x_labels: Sequence[str],
    grouped_values: Sequence[Sequence[float]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x_positions = list(range(len(x_labels)))

    color_cycle = plt.get_cmap("tab20").colors
    color_map: OrderedDict[str, Tuple[float, float, float]] = OrderedDict()
    seen_labels: set[str] = set()

    for group_idx, (label, values) in enumerate(zip(x_labels, grouped_values)):
        bottom = 0.0
        for seg_idx, value in enumerate(values):
            if math.isnan(value):
                continue

            seg_label = GROUPS[group_idx].segments[seg_idx].label
            if seg_label not in color_map:
                color_map[seg_label] = color_cycle[len(color_map) % len(color_cycle)]
            color = color_map[seg_label]

            plot_label = seg_label if seg_label not in seen_labels else None
            bars = ax.bar(  # Keep handle for future inspection if needed
                x_positions[group_idx],
                value,
                bottom=bottom,
                color=color,
                width=0.55,
                label=plot_label,
                edgecolor="white",
                linewidth=0.7,
            )
            seen_labels.add(seg_label)
            bottom += value

            ax.bar_label(
                bars,
                labels=[f"{value:.4f}"],
                label_type="center",
                fontsize=8,
                color="white",
            )

    ax.set_xticks(x_positions, x_labels, rotation=0)
    ax.set_ylabel("Best MSE")
    ax.set_title("Multi-granularity trend representation")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    print(f"Stacked bar figure saved to {output_path}")


def main() -> None:
    args = parse_args()
    x_labels, grouped_values, manifest_map = collect_group_metrics(
        args.base_dir, args.model, args.metric
    )
    plot_stacked_bars(x_labels, grouped_values, args.output)

    for key, manifest in manifest_map.items():
        print(f"{key} -> {manifest}")


if __name__ == "__main__":
    main()


#  python scripts/plot_gen_group_stacks.py 
