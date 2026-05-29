"""Generate ver_gen5 dataset derived from ver_camf with encoded news sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import List, Optional, Sequence, Tuple


def parse_series(series_str: str) -> List[float]:
    """Convert comma-separated string into list of floats."""
    parts = [part.strip() for part in series_str.split(",")]
    return [float(part) for part in parts if part]

def _mean_and_scale(
    hist_values: Sequence[float],
    gt_values: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Return mean and a safe standard deviation (fallback if variance is tiny)."""
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean/scale.")

    mean_val = sum(hist_values) / len(hist_values)
    variance = sum((value - mean_val) ** 2 for value in hist_values) / len(hist_values)
    sigma = variance ** 0.5
    if sigma < 1e-6:
        fallback_values = list(hist_values)
        if gt_values:
            fallback_values.extend(gt_values)
        max_diff = max((abs(value - mean_val) for value in fallback_values), default=0.0)
        sigma = max(max_diff, 1.0)
    return mean_val, sigma

def encode_sequence3(hist_values: List[float], news:str) -> List[int]:
    """Encode ground truth values relative to the mean of historical values."""
    trends = re.findall(r'\{([^}]+)\}', news)

    
    # 转换为编码
    encoding_map = {
        'Rising': 1,
        'Falling': -1,
        'Stable': 0
    }
    
    encoded = [encoding_map.get(trend.strip(), 0) for trend in trends]
    return encoded

def encode_sequence2(hist_values: List[float], gt_values: List[float]) -> List[int]:
    """Encode ground truth values relative to the mean of historical values."""
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean.")
    last_val = hist_values[-1]
    encoded = []
    for value in gt_values:
        diff = abs(value - last_val)
        if diff < 1:
            encoded.append(0)
        elif value < last_val:
            encoded.append(-1)
        else:
            encoded.append(1)
        last_val = value
    return encoded

def encode_sequence(hist_values: List[float], gt_values: List[float]) -> List[int]:
    """Encode ground truth values relative to the mean of historical values."""
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean.")
    mean_val = sum(hist_values) / len(hist_values)
    encoded = []
    for value in gt_values:
        diff = abs(value - mean_val)
        if diff < 1:
            encoded.append(0)
        elif value < mean_val:
            encoded.append(-1)
        else:
            encoded.append(1)
    return encoded

def encode_sequence5(news:str) -> List[int]:
    """机械地直接取gen2中的趋势词."""
    trends = re.findall(r'\{([^}]+)\}', news)

    encoding_map = {
        'Rising': 1,
        'Falling': -1,
        'Stable': 0
    }
    encoded=[encoding_map.get(trend.strip().split(",")[-1].strip(),0) for trend in trends]
    return encoded
def encode_sequence6(hist_values: List[float], gt_values: List[float]) -> List[int]:
    """
    Encode trend based on comparison between historical mean and ground truth mean.
    
    - 1: Rising (ground truth mean > historical mean)
    - -1: Falling (ground truth mean < historical mean)
    - 0: Stable (means are approximately equal)
    """
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean.")
    if not gt_values:
        raise ValueError("ground_truth data is empty, cannot compute mean.")
    
    hist_mean = sum(hist_values) / len(hist_values)
    gt_mean = sum(gt_values) / len(gt_values)
    
    diff = abs(gt_mean - hist_mean)
    # if diff < 1:  # 阈值可调整
    #     return [0]  # Stable
    if gt_mean >= hist_mean:
        return [1]  # Rising
    else:
        return [-1]  # Falling

def encode_sequence4(hist_values: List[float], gt_values: List[float]) -> List[str]:
    """Encode direction vs. mean along with five-level strength using string codes."""
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean.")

    mean_val = sum(hist_values) / len(hist_values)
    variance = sum((value - mean_val) ** 2 for value in hist_values) / len(hist_values)
    sigma = variance ** 0.5
    if sigma < 1e-6:
        max_diff = max((abs(value - mean_val) for value in hist_values), default=0.0)
        sigma = max(max_diff, 1.0)

    strength_thresholds = (0.25, 0.75, 1.25, 1.75)
    encoded: List[str] = []
    for value in gt_values:
        diff = value - mean_val
        abs_diff = abs(diff)
        if abs_diff < 1:
            direction = 0
        else:
            direction = 1 if diff > 0 else -1

        z_score = abs_diff / (sigma + 1e-6)
        if z_score < strength_thresholds[0]:
            strength = 1
        elif z_score < strength_thresholds[1]:
            strength = 2
        elif z_score < strength_thresholds[2]:
            strength = 3
        elif z_score < strength_thresholds[3]:
            strength = 4
        else:
            strength = 5

        encoded.append(f"{direction}{strength}")
    return encoded

def encode_sequence7(
    hist_values: List[float],
    gt_values: List[float],
    strength_thresholds: Sequence[float],
) -> List[str]:
    """
    Encode direction (only 1 or -1) and strength using per-series normalized differences.

    Strength thresholds are expected to contain four ascending boundary values so that the
    resulting five buckets distribute training samples as evenly as possible. Thresholds are
    computed on absolute z-scores from the training split and applied consistently to all splits.
    """
    if not hist_values:
        raise ValueError("historical_data is empty, cannot compute mean.")
    if len(strength_thresholds) != 4:
        raise ValueError("strength_thresholds must contain four boundary values.")

    mean_val, sigma = _mean_and_scale(hist_values, gt_values)
    encoded: List[str] = []
    for value in gt_values:
        diff = value - mean_val
        abs_diff = abs(diff)
        direction = 1 if diff >= 0 else -1  # Treat tiny/equal diffs as rising.

        z_score = abs_diff / sigma
        strength = len(strength_thresholds) + 1
        for idx, threshold in enumerate(strength_thresholds):
            if z_score <= threshold:
                strength = idx + 1
                break

        encoded.append(f"{direction}{strength}")
    return encoded

def _interpolated_quantile(sorted_values: List[float], quantile: float) -> float:
    """Return the interpolated quantile value for a sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    clamped_quantile = max(0.0, min(1.0, quantile))
    position = clamped_quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(len(sorted_values) - 1, lower_index + 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction

def compute_strength_thresholds(train_records: List[dict]) -> Tuple[float, float, float, float]:
    """Compute four thresholds that split the training absolute diffs into ~equal-sized groups."""
    abs_diffs: List[float] = []
    for record in train_records:
        historical = parse_series(record["historical_data"])
        if not historical:
            raise ValueError("historical_data is empty in training record, cannot compute thresholds.")
        gt_values = parse_series(record["ground_truth"])
        mean_val, sigma = _mean_and_scale(historical, gt_values)
        abs_diffs.extend(abs((value - mean_val) / sigma) for value in gt_values)

    if not abs_diffs:
        return (0.0, 0.0, 0.0, 0.0)

    abs_diffs.sort()
    quantiles = (0.2, 0.4, 0.6, 0.8)
    thresholds = tuple(_interpolated_quantile(abs_diffs, q) for q in quantiles)
    return thresholds

def encode_sequence10(news: str) -> List[str]:
    """
    Drop the final digit from every two-digit encoded news token.

    Example: ``-14, 13`` becomes ``-1, 1``.
    """
    codes = [code.strip() for code in news.split(",")]
    new_codes: List[str] = []
    for code in codes:
        stripped = code.strip()
        if not stripped:
            continue

        sign = ""
        digits = stripped
        if stripped.startswith(("+", "-")):
            sign = stripped[0]
            digits = stripped[1:]

        if len(digits) == 2 and digits.isdigit():
            stripped = f"{sign}{digits[0]}"
        new_codes.append(stripped)
    return new_codes

def process_split(
    split: str,
    src_dir: Path,
    dst_dir: Path,
    args,
    strength_thresholds: Optional[Sequence[float]] = None,
) -> None:
    """Process a single split JSON file."""
    src_path = src_dir / f"{split}.json"
    dst_path = dst_dir / f"{split}.json"
    with src_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    new_records = []
    for idx, record in enumerate(records):
        historical = parse_series(record["historical_data"])
        ground_truth = parse_series(record["ground_truth"])
        news=record["news"]
        if(args.generate == 5):
            encoded = encode_sequence(historical, ground_truth)
        elif(args.generate == 6):
            encoded = encode_sequence2(historical, ground_truth)
        elif(args.generate == 7):
            encoded = encode_sequence3(historical, news)
        elif(args.generate == 8):
            if strength_thresholds is None:
                raise ValueError(
                    "Strength thresholds must be precomputed from the training split for encode_sequence7."
                )
            encoded = encode_sequence7(historical, ground_truth, strength_thresholds)
        elif(args.generate == 9):
            encoded = encode_sequence5(news)
        elif(args.generate == 10):
            encoded = encode_sequence10(news)
        else:
            encoded = encode_sequence6(historical, ground_truth)
        updated = dict(record)
        updated["news"] = ", ".join(str(code) for code in encoded)
        new_records.append(updated)

    dst_dir.mkdir(parents=True, exist_ok=True)
    with dst_path.open("w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a derived dataset where the news field is replaced by "
            "five-step encoded sequences computed from historical data means."
        )
    )
    parser.add_argument(
        "--source",
        default="dataset/Bitcoin/ver_synchronized",
        help="Path to the source dataset directory containing train/vali/test JSON files.",
    )
    parser.add_argument(
        "--dest",
        default="dataset/Bitcoin/ver_synchronized_trendonly",
        help="Destination directory for the generated dataset.",
    )
    parser.add_argument(
        "--generate",
        default=10,

    )
    args = parser.parse_args()
    
    src_dir = Path(args.source)
    dst_dir = Path(args.dest)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

    strength_thresholds: Optional[Tuple[float, float, float, float]] = None
    if args.generate == 8:
        train_path = src_dir / "train.json"
        if not train_path.exists():
            raise FileNotFoundError(f"Training split {train_path} does not exist for threshold computation.")
        with train_path.open("r", encoding="utf-8") as f:
            train_records = json.load(f)
        strength_thresholds = compute_strength_thresholds(train_records)

    for split in ("train", "vali", "test"):
        process_split(split, src_dir, dst_dir,args, strength_thresholds)

    print(f"Generated dataset saved to {dst_dir}")


if __name__ == "__main__":
    main()

# python src/generate_qwen_embedding/generate_ver_gen5_dataset.py --dest dataset/FNSPID/ver_gen6
