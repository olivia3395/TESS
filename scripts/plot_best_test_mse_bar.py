#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
读取三个 config_snapshot.json 的 best_test_mse，画柱状图并保存为 PNG。

模型：
- UniModal_Baseline / FNSPID / ver_camf / best_epoch_Oct-20-2025-18-24-46
- MultiModal_Baseline / FNSPID / ver_camf / best
- MultiModal_Baseline / FNSPID / ver_global_shape_volatility_natural / best
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 兼容无图形界面环境
import matplotlib.pyplot as plt


def main() -> None:
    base = Path(__file__).resolve().parents[1]  # 指向 MMTSF_LIB 根目录

    paths = {
        "UniModal_ver_camf": base
        / "saved"
        / "UniModal_Baseline"
        / "FNSPID"
        / "ver_camf"
        / "best_epoch_Oct-20-2025-18-24-46"
        / "config_snapshot.json",
        "MultiModal_ver_camf": base
        / "saved"
        / "MultiModal_Baseline"
        / "FNSPID"
        / "ver_camf"
        / "best"
        / "config_snapshot.json",
        "MultiModal_global_shape": base
        / "saved"
        / "MultiModal_Baseline"
        / "FNSPID"
        / "ver_global_shape_volatility_natural"
        / "best"
        / "config_snapshot.json",
    }

    labels = []
    values = []

    for label, path in paths.items():
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        mse = cfg.get("best_test_mse")
        if mse is None:
            mse = cfg.get("best_test_metrics", {}).get("MSE")
        # labels.append(label)
        values.append(mse)

    labels = ["UniModal", "Original_text", "Ours"]
    print("Labels:", labels)
    print("MSEs  :", values)

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green"])
    plt.ylabel("Average MSE")
    plt.title("non-stationary shape MSE Comparison(FNSPID)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    out_dir = base / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "best_test_mse_bar.png"
    plt.savefig(out_path, dpi=200)
    print("Saved figure to", out_path)


if __name__ == "__main__":
    main()


