"""Compare legacy TimesNet and RC2 TimesNet numeric runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _latest(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def find_legacy_snapshot(root: Path) -> Path | None:
    return _latest(list(root.rglob("config_snapshot.json")))


def find_legacy_manifest(root: Path) -> Path | None:
    return _latest(list(root.rglob("manifest.json")))


def legacy_row(root: Path) -> dict[str, Any]:
    snapshot_path = find_legacy_snapshot(root)
    manifest_path = find_legacy_manifest(root)
    snapshot = _read_json(snapshot_path)
    manifest = _read_json(manifest_path)
    config = snapshot.get("config", {})
    best_metrics = snapshot.get("best_test_metrics", {}) or {}
    best_vali_metrics = (
        manifest.get("epoch_metrics", {})
        .get("best", {})
        .get("vali_metrics", {})
    )
    return {
        "run_name": "legacy_timesnet",
        "code_path": "legacy",
        "model": "TimesNet",
        "metric_scale": "legacy_train_history_standardized",
        "best_vali_mse": snapshot.get("best_valid_score"),
        "best_vali_mae": best_vali_metrics.get("MAE"),
        "best_reloaded_test_mse": best_metrics.get("MSE"),
        "best_reloaded_test_mae": best_metrics.get("MAE"),
        "final_test_mse": None,
        "final_test_mae": None,
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "lr": config.get("learning_rate"),
        "seed": config.get("seed"),
        "notes": f"snapshot={snapshot_path}; manifest={manifest_path}; legacy trainer stores best state in memory and exports best split metrics when sample export is enabled",
    }


def rc2_row(path: Path) -> dict[str, Any]:
    summary = _read_json(path / "summary.json")
    metrics = _read_json(path / "metrics.json")
    config = summary.get("config") or metrics.get("config", {})
    best_test = summary.get("best_reloaded", metrics.get("best_reloaded", {})).get("test", {})
    final_test = summary.get("last", metrics.get("last", {})).get("test", {})
    best_vali = summary.get("best_reloaded", metrics.get("best_reloaded", {})).get("vali", {})
    return {
        "run_name": "rc2_legacy_timesnet",
        "code_path": "rc2",
        "model": config.get("model", "legacy_timesnet"),
        "metric_scale": summary.get("metric_scale") or metrics.get("metric_scale"),
        "best_vali_mse": summary.get("best_vali_mse"),
        "best_vali_mae": best_vali.get("mae"),
        "best_reloaded_test_mse": summary.get("best_reloaded_test_mse"),
        "best_reloaded_test_mae": best_test.get("mae"),
        "final_test_mse": summary.get("final_test_mse"),
        "final_test_mae": final_test.get("mae"),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "lr": config.get("lr"),
        "seed": config.get("seed"),
        "notes": f"scaler_mean={summary.get('scaler_mean')}; scaler_std={summary.get('scaler_std')}; best_reload={summary.get('best_reload_check')}",
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value).replace("\n", " ")


def print_table(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run_name",
        "code_path",
        "model",
        "metric_scale",
        "best_vali_mse",
        "best_vali_mae",
        "best_reloaded_test_mse",
        "best_reloaded_test_mae",
        "final_test_mse",
        "final_test_mae",
        "epochs",
        "batch_size",
        "lr",
        "seed",
        "notes",
    ]
    print(",".join(columns))
    for row in rows:
        print(",".join(fmt(row.get(column)) for column in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare legacy and RC2 TimesNet runs.")
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--rc2-output-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    rows = []
    if args.legacy_root:
        rows.append(legacy_row(args.legacy_root))
    if args.rc2_output_dir:
        rows.append(rc2_row(args.rc2_output_dir))

    if args.output_json and not args.print_only:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_table(rows)


if __name__ == "__main__":
    main()
