"""Summarize additive primitive FNSPID experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = (
    "additive_text_scale0",
    "additive_text_scale1",
    "additive_text_soft_scale1",
    "additive_gt_scale1",
    "additive_text_distill_delta_lam0p1",
    "additive_text_soft_distill_delta_lam0p1",
    "additive_text_gate_scale1",
    "additive_gt_gate_scale1",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def first_available(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def format_value(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def row_for_experiment(out_root: Path, experiment_name: str) -> dict[str, Any]:
    output_dir = out_root / experiment_name
    summary = read_json(output_dir / "summary.json")
    metrics = read_json(output_dir / "metrics.json")

    return {
        "experiment_name": experiment_name,
        "output_dir": str(output_dir),
        "best_epoch": first_available(
            summary.get("best_epoch"),
            metrics.get("best_epoch"),
        ),
        "test_mse": first_available(
            summary.get("best_reloaded_test_mse"),
            nested_get(metrics, ("best_reloaded", "test", "mse")),
            nested_get(metrics, ("best_recorded", "test", "mse")),
            summary.get("final_test_mse"),
            nested_get(metrics, ("last", "test", "mse")),
        ),
        "test_mae": first_available(
            summary.get("best_reloaded_test_mae"),
            nested_get(metrics, ("best_reloaded", "test", "mae")),
            nested_get(metrics, ("best_recorded", "test", "mae")),
            summary.get("final_test_mae"),
            nested_get(metrics, ("last", "test", "mae")),
        ),
        "vali_mse": first_available(
            summary.get("best_vali_mse"),
            nested_get(metrics, ("best_reloaded", "vali", "mse")),
            nested_get(metrics, ("best_recorded", "vali", "mse")),
        ),
        "train_mse": first_available(
            nested_get(metrics, ("best_reloaded", "train", "mse")),
            nested_get(metrics, ("best_recorded", "train", "mse")),
            nested_get(metrics, ("last", "train", "mse")),
        ),
        "mean_dynamic_scale": first_available(
            summary.get("best_reloaded_test_mean_dynamic_scale"),
            nested_get(metrics, ("best_reloaded", "test", "mean_dynamic_scale")),
            nested_get(metrics, ("best_recorded", "test", "mean_dynamic_scale")),
            summary.get("final_test_mean_dynamic_scale"),
            nested_get(metrics, ("last", "test", "mean_dynamic_scale")),
        ),
        "mean_gate_weight": first_available(
            summary.get("best_reloaded_test_mean_gate_weight"),
            nested_get(metrics, ("best_reloaded", "test", "mean_gate_weight")),
            nested_get(metrics, ("best_recorded", "test", "mean_gate_weight")),
            summary.get("final_test_mean_gate_weight"),
            nested_get(metrics, ("last", "test", "mean_gate_weight")),
        ),
        "lambda_delta_distill": first_available(
            summary.get("lambda_delta_distill"),
            nested_get(metrics, ("config", "lambda_delta_distill")),
        ),
        "lambda_pred_distill": first_available(
            summary.get("lambda_pred_distill"),
            nested_get(metrics, ("config", "lambda_pred_distill")),
        ),
        "student_numerical_initialized_from_teacher": first_available(
            summary.get("student_numerical_initialized_from_teacher"),
            nested_get(metrics, ("config", "student_numerical_initialized_from_teacher")),
        ),
        "student_numerical_frozen": first_available(
            summary.get("student_numerical_frozen"),
            nested_get(metrics, ("config", "student_numerical_frozen")),
        ),
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = (
        "experiment_name",
        "output_dir",
        "best_epoch",
        "test_mse",
        "test_mae",
        "vali_mse",
        "train_mse",
        "lambda_delta_distill",
        "lambda_pred_distill",
        "student_numerical_initialized_from_teacher",
        "student_numerical_frozen",
        "mean_dynamic_scale",
        "mean_gate_weight",
    )
    rendered = [
        {column: format_value(row.get(column)) for column in columns}
        for row in rows
    ]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rendered:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize additive primitive experiment outputs.")
    parser.add_argument(
        "--out-root",
        default="/home/lyc/workspace/TESS-RC2/outputs/tess_basic/real_fnspid_legacy_standard",
        help="Directory containing additive primitive experiment output folders.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_root = Path(args.out_root)
    print_table([row_for_experiment(out_root, name) for name in EXPERIMENTS])


if __name__ == "__main__":
    main()
