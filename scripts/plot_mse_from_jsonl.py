#!/usr/bin/env python
"""Plot per-sample MSE curves for ver_gen1-ver_gen4 test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

DEFAULT_VERSIONS = ["ver_gen1", "ver_gen2", "ver_gen3", "ver_gen4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-sample MSE curves from test_scores.jsonl files."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("output/FNSPID"),
        help="Root directory that contains ver_gen* folders (default: output/FNSPID).",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        default=DEFAULT_VERSIONS,
        help="Ordered list of version folders to include (default: ver_gen1-ver_gen4).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="MultiModal_Baseline",
        help="Model sub-folder used within each version. Leave empty to search all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/FNSPID/mse_ver_gen1-4.png"),
        help="Path of the PNG file to write (default: output/FNSPID/mse_ver_gen1-4.png).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving the figure.",
    )
    return parser.parse_args()


def resolve_score_file(root: Path, version: str, model_name: str | None) -> Path:
    """Return the latest test_scores.jsonl path for the requested version."""
    version_dir = root / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Missing version directory: {version_dir}")

    candidates: List[Path] = []
    if model_name:
        model_dir = version_dir / model_name
        if model_dir.exists():
            candidates = sorted(model_dir.rglob("test_scores.jsonl"), 
                   key=lambda p: p.stat().st_mtime)

    # if not candidates:
    #     candidates = sorted(version_dir.rglob("test_scores.jsonl"))

    if not candidates:
        raise FileNotFoundError(f"No test_scores.jsonl found under {version_dir}")

    return candidates[-1]


def load_mse_series(score_file: Path) -> List[float]:
    """Load the MSE values from a JSONL file."""
    values: List[float] = []
    with score_file.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "mse" not in record:
                raise KeyError(f"`mse` missing in {score_file} line {line_num}")
            values.append(float(record["mse"]))

    if not values:
        raise ValueError(f"{score_file} does not contain any records.")

    return values


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_series(series: Dict[str, List[float]], output_path: Path, show_plot: bool) -> None:
    """Create and save the line chart."""
    plt.figure(figsize=(10, 6))
    for version, values in series.items():
        x_axis = range(1, len(values) + 1)
        plt.plot(x_axis, values, marker="o", markersize=2, linewidth=1.2, label=version)

    plt.xlabel("Sample Index")
    plt.ylabel("MSE")
    plt.title("ver_gen1-ver_gen4 Test MSE Curves")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    plt.tight_layout()

    ensure_parent_dir(output_path)
    plt.savefig(output_path, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def format_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()

    series: Dict[str, List[float]] = {}
    source_paths: Dict[str, Path] = {}
    for version in args.versions:
        score_file = resolve_score_file(args.root, version, args.model_name or None)
        series[version] = load_mse_series(score_file)
        source_paths[version] = score_file

    plot_series(series, args.output, args.show)

    print(f"Saved plot to {format_relative(args.output)}")
    for version in args.versions:
        print(
            f"{version}: {len(series[version])} samples from {format_relative(source_paths[version])}"
        )


if __name__ == "__main__":
    main()
