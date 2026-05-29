#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 FNSPID/ver_camf 的 historical_data 生成 shape 标签：
- Rise: 一直增（整体单调上升）
- Fall: 一直减（整体单调下降）
- Peak: 先增后减（只有一个拐点）
- Recover: 先减后增（只有一个拐点）
- Oscillate: 其它情况（振荡/多拐点）

会读取 dataset/FNSPID/<source-version>/{train,vali,test}.json
写入  dataset/FNSPID/<target-version>/{train,vali,test}.json，并为每条样本新增字段 shape。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

import numpy as np

DEFAULT_SPLITS = ("train", "vali", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为 FNSPID/ver_camf 数据集中的 historical_data 打 shape 标签，"
            "新增字段 shape，并写入新的版本目录。"
        )
    )
    parser.add_argument(
        "--dataset",
        default="Bitcoin",
        help="数据集文件夹名（默认：FNSPID）",
    )
    parser.add_argument(
        "--source-version",
        default="ver_camf",
        help="源版本文件夹名（默认：ver_camf）",
    )
    parser.add_argument(
        "--target-version",
        default="ver_camf_shape_gt",
        help="目标版本文件夹名（默认：ver_camf_shape）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="要处理的划分名（默认：train vali test）",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    """从 path 读取 JSON list."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} in {path} is {type(item).__name__}, expected dict")
    return cast(List[Dict[str, Any]], data)


def write_json(data: List[Dict[str, Any]], path: Path) -> None:
    """写回 JSON 文件，末尾加换行方便 diff。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n")


def parse_historical_data(value: Any) -> np.ndarray:
    """
    ver_camf 中 historical_data 是形如 'a, b, c, d, e' 的字符串，这里解析成 float 数组。
    """
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=float)
    if isinstance(value, (list, tuple)):
        return np.asarray([float(x) for x in value], dtype=float)
    raise TypeError(f"Unsupported historical_data type: {type(value)}")


def classify_shape(values: np.ndarray, eps: float = 1e-6) -> str:
    """
    根据时间序列形状打标签：
    - Rise: 全程无明显下降，且存在上升
    - Fall: 全程无明显上升，且存在下降
    - Peak: 先升后降，只有一个拐点
    - Recover: 先降后升，只有一个拐点
    - Oscillate: 其它情况（多拐点 / 来回震荡 / 近似平）
    """
    if values.ndim != 1 or values.size < 2:
        return "Oscillate"

    diffs = np.diff(values)
    pos = diffs > eps
    neg = diffs < -eps

    # 单调上升 / 单调下降
    if pos.any() and not neg.any():
        return "Rise"
    if neg.any() and not pos.any():
        return "Fall"

    n = values.size
    # 尝试寻找唯一拐点
    for k in range(1, n - 1):
        left = diffs[:k]
        right = diffs[k:]

        # Peak: 左侧非下降（>=0），右侧非上升（<=0），且左右都存在"明显"上升/下降
        if (
            np.all(left >= -eps)
            and np.all(right <= eps)
            and (left > eps).any()
            and (right < -eps).any()
        ):
            return "Peak"

        # Recover: 左侧非上升（<=0），右侧非下降（>=0）
        if (
            np.all(left <= eps)
            and np.all(right >= -eps)
            and (left < -eps).any()
            and (right > eps).any()
        ):
            return "Recover"

    # 多次反转或形状较复杂
    return "Oscillate"


def add_shape_field(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """对每条样本解析 historical_data，新增 shape 字段。"""
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if "ground_truth" not in rec:
            raise KeyError(f"Record {idx} missing 'historical_data'")
        try:
            hist = parse_historical_data(rec["ground_truth"])
        except Exception as exc:  # 保留现场信息
            raise ValueError(f"Failed to parse historical_data for record {idx}: {exc}") from exc

        shape = classify_shape(hist)
        new_rec = dict(rec)
        new_rec["shape"] = shape
        out.append(new_rec)
    return out


def main() -> None:
    args = parse_args()
    # scripts/ 的上一级是 MMTSF_LIB 根目录
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "dataset" / args.dataset / args.source_version
    target_root = project_root / "dataset" / args.dataset / args.target_version

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    for split in args.splits:
        source_path = source_root / f"{split}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing split file: {source_path}")

        records = load_records(source_path)
        labeled = add_shape_field(records)

        target_path = target_root / source_path.name
        write_json(labeled, target_path)
        print(f"{split}: wrote {len(labeled)} records -> {target_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()


