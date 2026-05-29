"""为 FNSPID/ver_camf 数据集添加 global_volatility 标签。

波动率计算方式：
1. 计算相邻时间步差值的绝对值：|x[i+1] - x[i]|
2. 除以五个时间步数据的均值
3. 除以4取平均（因为有4个差值）

使用train集的20和80分位点作为阈值，对所有数据集划分Low/Medium/High。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

import numpy as np

DEFAULT_SPLITS = ("train", "vali", "test")


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
    """解析 historical_data 字符串为 float 数组。"""
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return np.asarray([float(p) for p in parts], dtype=float)
    if isinstance(value, (list, tuple)):
        return np.asarray([float(x) for x in value], dtype=float)
    raise TypeError(f"Unsupported historical_data type: {type(value)}")


def compute_volatility(values: np.ndarray) -> float:
    """
    计算波动率：
    1. 后一个时间步减前一个时间步，取绝对值
    2. 除以五个时间步数据的均值
    3. 除以4取平均
    
    公式：volatility = mean(|diff|) / mean(values) / 4
    """
    if values.size < 2:
        return 0.0
    
    # 计算相邻差值绝对值
    diffs = np.abs(np.diff(values))
    
    # 计算均值
    mean_value = np.mean(values)
    
    if mean_value == 0:
        return 0.0
    
    # 波动率 = 平均差值 / 均值 / 4
    volatility = np.mean(diffs) / mean_value / 4.0
    
    return float(volatility)


def add_volatility_field(
    records: Iterable[Dict[str, Any]], 
    p20: float, 
    p80: float
) -> List[Dict[str, Any]]:
    """为每条记录计算波动率并添加 global_volatility 标签。"""
    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if "historical_data" not in rec:
            raise KeyError(f"Record {idx} missing 'historical_data'")
        try:
            hist = parse_historical_data(rec["historical_data"])
        except Exception as exc:
            raise ValueError(f"Failed to parse historical_data for record {idx}: {exc}") from exc

        vol = compute_volatility(hist)
        
        # 根据分位点划分
        if vol < p20:
            label = "Low"
        elif vol < p80:
            label = "Medium"
        else:
            label = "High"
        
        new_rec = dict(rec)
        new_rec["global_volatility"] = label
        out.append(new_rec)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "为 FNSPID/ver_camf 数据集中的 historical_data 计算 global_volatility 标签，"
            "使用train集的20和80分位点作为阈值。"
        )
    )
    parser.add_argument(
        "--dataset",
        default="FNSPID",
        help="数据集文件夹名（默认：FNSPID）",
    )
    parser.add_argument(
        "--source-version",
        default="ver_camf",
        help="源版本文件夹名（默认：ver_camf）",
    )
    parser.add_argument(
        "--target-version",
        default="ver_camf_global_volatility",
        help="目标版本文件夹名（默认：ver_camf_global_volatility）",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="要处理的划分名（默认：train vali test）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]  # 指向 MMTSF_LIB
    source_root = project_root / "dataset" / args.dataset / args.source_version
    target_root = project_root / "dataset" / args.dataset / args.target_version

    if not source_root.is_dir():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    # 首先从train集计算分位点
    train_path = source_root / "train.json"
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    
    train_records = load_records(train_path)
    train_volatilities = []
    for idx, rec in enumerate(train_records):
        try:
            hist = parse_historical_data(rec["historical_data"])
            vol = compute_volatility(hist)
            train_volatilities.append(vol)
        except Exception as exc:
            raise ValueError(f"Failed to compute volatility for train record {idx}: {exc}") from exc
    
    p20 = np.percentile(train_volatilities, 20)
    p80 = np.percentile(train_volatilities, 80)
    
    print(f"Train set volatility statistics:")
    print(f"  Count: {len(train_volatilities)}")
    print(f"  Min: {np.min(train_volatilities):.6f}")
    print(f"  Max: {np.max(train_volatilities):.6f}")
    print(f"  Mean: {np.mean(train_volatilities):.6f}")
    print(f"  20th percentile: {p20:.6f}")
    print(f"  80th percentile: {p80:.6f}")
    print()

    # 处理所有划分
    for split in args.splits:
        source_path = source_root / f"{split}.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing split file: {source_path}")

        records = load_records(source_path)
        labeled = add_volatility_field(records, p20, p80)

        target_path = target_root / source_path.name
        write_json(labeled, target_path)
        print(f"{split}: wrote {len(labeled)} records -> {target_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()














