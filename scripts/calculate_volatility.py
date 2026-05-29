#!/usr/bin/env python
"""Calculate volatility (amplitude) for ground truth data and classify into high/medium/low categories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from collections import Counter


def calculate_volatility(gt_values: List[float]) -> float:
    """
    Calculate volatility for a sequence.
    
    Formula: 
    - From the second time step, calculate |gt[i] - gt[i-1]| / mean(gt) for each step
    - Sum all these values and divide by (number_of_steps - 1) to get average
    
    Args:
        gt_values: List of ground truth values
        
    Returns:
        Volatility value (float)
    """
    if len(gt_values) < 2:
        return 0.0
    
    gt_array = np.array(gt_values, dtype=float)
    mean_gt = np.mean(gt_array)
    
    if mean_gt == 0:
        return 0.0
    
    # Calculate absolute differences between consecutive steps
    diffs = np.abs(np.diff(gt_array))
    
    # Normalize by mean and calculate average
    normalized_diffs = diffs / mean_gt
    volatility = np.mean(normalized_diffs)
    
    return float(volatility)


def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_volatility_statistics(train_data: List[Dict[str, Any]]) -> tuple[float, float]:
    """
    Calculate volatility statistics from training set to determine thresholds.
    
    Returns:
        (low_threshold, high_threshold) - 33rd and 66th percentiles
    """
    volatilities = []
    
    for record in train_data:
        ground_truth = record.get("ground_truth", [])
        if not ground_truth:
            continue
        
        # Handle string format (comma-separated values)
        if isinstance(ground_truth, str):
            try:
                gt_values = [float(x.strip()) for x in ground_truth.split(',') if x.strip()]
            except (ValueError, TypeError):
                continue
        elif isinstance(ground_truth, list):
            try:
                gt_values = [float(x) for x in ground_truth]
            except (ValueError, TypeError):
                continue
        else:
            continue
        
        if len(gt_values) < 2:
            continue
        
        volatility = calculate_volatility(gt_values)
        volatilities.append(volatility)
    
    if len(volatilities) == 0:
        # Default thresholds if no valid data
        return 0.01, 0.05
    
    # Calculate percentiles - using 33rd and 66th percentiles
    low_threshold = np.percentile(volatilities, 33)  # 33rd percentile
    high_threshold = np.percentile(volatilities, 66)  # 66th percentile
    
    print(f"\nVolatility Statistics from Training Set:")
    print(f"  Total samples: {len(volatilities)}")
    print(f"  Min: {np.min(volatilities):.6f}")
    print(f"  Max: {np.max(volatilities):.6f}")
    print(f"  Mean: {np.mean(volatilities):.6f}")
    print(f"  Median: {np.median(volatilities):.6f}")
    print(f"  33rd percentile (low threshold): {low_threshold:.6f}")
    print(f"  66th percentile (high threshold): {high_threshold:.6f}")
    
    return float(low_threshold), float(high_threshold)


def classify_volatility(volatility: float, low_threshold: float, high_threshold: float) -> str:
    """
    Classify volatility into high/medium/low categories.
    
    Args:
        volatility: Volatility value
        low_threshold: 33rd percentile threshold
        high_threshold: 66th percentile threshold
        
    Returns:
        "Low", "Medium", or "High" (capitalized)
    """
    if volatility <= low_threshold:
        return "Low"
    elif volatility >= high_threshold:
        return "High"
    else:
        return "Medium"


def process_dataset(
    input_file: Path, 
    output_file: Path, 
    low_threshold: float, 
    high_threshold: float
) -> None:
    """Process dataset and add volatility labels."""
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    print(f"Processing {len(data)} samples...")
    
    volatility_counts = Counter()
    volatility_values = []
    
    for i, record in enumerate(data):
        # Create a copy to preserve original fields
        new_record = {k: v for k, v in record.items()}
        ground_truth = record.get("ground_truth", [])
        
        if not ground_truth:
            if (i + 1) % 1000 == 0:
                print(f"Warning: Sample {i} has no ground_truth")
            new_record["global_volatility"] = "Medium"  # Default
            data[i] = new_record
            continue
        
        # Handle string format (comma-separated values)
        if isinstance(ground_truth, str):
            try:
                gt_values = [float(x.strip()) for x in ground_truth.split(',') if x.strip()]
            except (ValueError, TypeError):
                if (i + 1) % 1000 == 0:
                    print(f"Warning: Sample {i} has invalid ground_truth string format")
                new_record["global_volatility"] = "Medium"  # Default
                data[i] = new_record
                continue
        elif isinstance(ground_truth, list):
            try:
                gt_values = [float(x) for x in ground_truth]
            except (ValueError, TypeError):
                if (i + 1) % 1000 == 0:
                    print(f"Warning: Sample {i} has invalid ground_truth list format")
                new_record["global_volatility"] = "Medium"  # Default
                data[i] = new_record
                continue
        else:
            if (i + 1) % 1000 == 0:
                print(f"Warning: Sample {i} has unexpected ground_truth type: {type(ground_truth)}")
            new_record["global_volatility"] = "Medium"  # Default
            data[i] = new_record
            continue
        
        # Calculate volatility
        volatility = calculate_volatility(gt_values)
        volatility_values.append(volatility)
        
        # Classify volatility
        volatility_label = classify_volatility(volatility, low_threshold, high_threshold)
        new_record["global_volatility"] = volatility_label
        data[i] = new_record
        volatility_counts[volatility_label] += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Volatility Classification Statistics")
    print("=" * 80)
    total = len(data)
    for label in ["Low", "Medium", "High"]:
        count = volatility_counts[label]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{label:12s}: {count:5d} ({percentage:5.2f}%)")
    
    if volatility_values:
        print(f"\nVolatility Value Statistics:")
        print(f"  Mean: {np.mean(volatility_values):.6f}")
        print(f"  Median: {np.median(volatility_values):.6f}")
        print(f"  Min: {np.min(volatility_values):.6f}")
        print(f"  Max: {np.max(volatility_values):.6f}")
    print("=" * 80)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    # Input directory
    input_dir = project_root / "dataset" / "Electricity" / "ver_camf"
    
    # Output directory for new dataset version
    output_dir = project_root / "dataset" / "Electricity" / "ver_synchronized_volatility"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Calculate thresholds from training set
    train_input = input_dir / "train.json"
    if not train_input.exists():
        print(f"Error: Training file not found: {train_input}")
        return
    
    print("=" * 80)
    print("Step 1: Calculating volatility thresholds from training set")
    print("=" * 80)
    train_data = load_data(train_input)
    low_threshold, high_threshold = calculate_volatility_statistics(train_data)
    
    # Step 2: Process all three datasets
    print("\n" + "=" * 80)
    print("Step 2: Processing datasets with calculated thresholds")
    print("=" * 80)
    
    # Process test set
    test_input = input_dir / "test.json"
    test_output = output_dir / "test.json"
    
    if test_input.exists():
        print("\nProcessing test set...")
        process_dataset(test_input, test_output, low_threshold, high_threshold)
    else:
        print(f"Warning: {test_input} not found")
    
    # Process train set
    train_output = output_dir / "train.json"
    if train_input.exists():
        print("\nProcessing train set...")
        process_dataset(train_input, train_output, low_threshold, high_threshold)
    
    # Process vali set
    vali_input = input_dir / "vali.json"
    vali_output = output_dir / "vali.json"
    
    if vali_input.exists():
        print("\nProcessing vali set...")
        process_dataset(vali_input, vali_output, low_threshold, high_threshold)
    else:
        print(f"Warning: {vali_input} not found")
    
    print("\n" + "=" * 80)
    print("All datasets processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

