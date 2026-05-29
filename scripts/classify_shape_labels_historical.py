#!/usr/bin/env python
"""Classify historical data into 5 shape types: Rise, Fall, Peak, Recover, Oscillate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter


def load_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_first_derivative(values: np.ndarray) -> np.ndarray:
    """Calculate first derivative (differences between consecutive points)."""
    return np.diff(values)


def find_significant_changes(diff: np.ndarray, threshold_ratio: float = 0.3) -> List[int]:
    """Find positions where derivative changes sign significantly."""
    if len(diff) < 2:
        return []
    
    # Normalize differences to handle different scales
    abs_diff = np.abs(diff)
    threshold = np.max(abs_diff) * threshold_ratio
    
    sign_changes = []
    for i in range(1, len(diff)):
        # Check if sign changes and change is significant
        if diff[i-1] * diff[i] < 0 and (abs(diff[i-1]) > threshold or abs(diff[i]) > threshold):
            sign_changes.append(i)
    
    return sign_changes


def classify_trend_segments(values: np.ndarray, num_segments: int = 4) -> Tuple[str, float]:
    """Classify trend by dividing into segments and analyzing each."""
    n = len(values)
    segment_size = n // num_segments
    
    # Calculate average slope for each segment
    segment_slopes = []
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < num_segments - 1 else n
        segment = values[start:end]
        if len(segment) > 1:
            slope = (segment[-1] - segment[0]) / max(len(segment), 1)
            segment_slopes.append(slope)
    
    if len(segment_slopes) < 2:
        return "Unknown", 0.0
    
    # Normalize slopes by the range of values
    value_range = np.max(values) - np.min(values)
    if value_range > 0:
        normalized_slopes = [s / value_range for s in segment_slopes]
    else:
        normalized_slopes = segment_slopes
    
    # Classify based on segment slopes (more lenient threshold)
    first_half_avg = np.mean(normalized_slopes[:len(normalized_slopes)//2])
    second_half_avg = np.mean(normalized_slopes[len(normalized_slopes)//2:])
    
    # Use a more lenient threshold (lower value means easier to classify as Peak/Recover)
    # Also check if there's a clear pattern even with small threshold
    threshold = max(np.std(normalized_slopes) * 0.2, 0.002) if np.std(normalized_slopes) > 0 else 0.002
    
    # Peak: first half increases, second half decreases
    # Be more lenient - allow if the pattern is clear even if threshold is small
    if first_half_avg > threshold and second_half_avg < -threshold:
        return "Peak", abs(first_half_avg) + abs(second_half_avg)
    # Also check if first half is clearly positive and second half is clearly negative
    elif first_half_avg > 0 and second_half_avg < 0 and abs(first_half_avg) + abs(second_half_avg) > 0.01:
        return "Peak", abs(first_half_avg) + abs(second_half_avg)
    # Recover: first half decreases, second half increases
    elif first_half_avg < -threshold and second_half_avg > threshold:
        return "Recover", abs(first_half_avg) + abs(second_half_avg)
    # Also check if first half is clearly negative and second half is clearly positive
    elif first_half_avg < 0 and second_half_avg > 0 and abs(first_half_avg) + abs(second_half_avg) > 0.01:
        return "Recover", abs(first_half_avg) + abs(second_half_avg)
    # Rise: overall positive trend
    elif np.mean(normalized_slopes) > threshold:
        return "Rise", abs(np.mean(normalized_slopes))
    # Fall: overall negative trend
    elif np.mean(normalized_slopes) < -threshold:
        return "Fall", abs(np.mean(normalized_slopes))
    else:
        return "Oscillate", np.std(normalized_slopes)


def detect_oscillation(values: np.ndarray, min_oscillations: int = 5) -> bool:
    """Detect if the sequence oscillates (has multiple direction changes)."""
    diff = calculate_first_derivative(values)
    sign_changes = find_significant_changes(diff, threshold_ratio=0.15)
    sign_changes = find_significant_changes(diff, threshold_ratio=0.15)
    
    # Only classify as oscillate if there are many sign changes (more strict)
    # This allows Peak/Recover patterns with some fluctuations
    if len(sign_changes) >= min_oscillations:
        return True
    
    # Check for strong alternating pattern (very regular oscillation)
    if len(sign_changes) >= 4:
        # Check if changes are relatively evenly distributed
        intervals = np.diff([0] + sign_changes + [len(diff)])
        if len(intervals) >= 3:
            # Check if intervals are relatively uniform (oscillation pattern)
            interval_std = np.std(intervals)
            interval_mean = np.mean(intervals)
            if interval_mean > 0 and interval_std / interval_mean < 0.4:
                return True
    
    return False


def classify_shape(values: List[float]) -> Tuple[int, str]:
    """
    Classify shape into 5 types:
    1. Rise - consistently increasing
    2. Fall - consistently decreasing
    3. Peak - increase then decrease (main pattern)
    4. Recover - decrease then increase (main pattern)
    5. Oscillate - oscillating pattern
    
    Returns: (label_id, label_name)
    """
    values_array = np.array(values)
    
    if len(values_array) < 3:
        return 5, "Oscillate"  # Default for very short sequences
    
    # First check for oscillation (highest priority)
    if detect_oscillation(values_array):
        return 5, "Oscillate"
    
    # Calculate overall trend
    overall_slope = (values_array[-1] - values_array[0]) / len(values_array)
    overall_change = values_array[-1] - values_array[0]
    
    # Calculate first derivative
    diff = calculate_first_derivative(values_array)
    
    # Find significant sign changes
    sign_changes = find_significant_changes(diff, threshold_ratio=0.25)
    
    # Classify using segment analysis
    trend_type, confidence = classify_trend_segments(values_array, num_segments=4)
    
    # Additional check: try to identify Peak/Recover patterns more aggressively
    # by looking at the maximum/minimum point position
    max_idx = np.argmax(values_array)
    min_idx = np.argmin(values_array)
    n = len(values_array)
    
    # If maximum is in the first 2/3 and minimum is after it, or vice versa, it might be Peak/Recover
    if max_idx < 2 * n // 3 and min_idx > max_idx:
        # Check if it's a peak pattern: increase to max, then decrease
        if max_idx > n // 4:  # Max should not be too early
            first_part_slope = (values_array[max_idx] - values_array[0]) / max(max_idx, 1)
            second_part_slope = (values_array[-1] - values_array[max_idx]) / max(n - max_idx, 1)
            value_range = np.max(values_array) - np.min(values_array)
            if value_range > 0:
                first_norm = first_part_slope / value_range
                second_norm = second_part_slope / value_range
                if first_norm > 0.003 and second_norm < -0.003:
                    trend_type = "Peak"
                    confidence = abs(first_norm) + abs(second_norm)
    
    if min_idx < 2 * n // 3 and max_idx > min_idx:
        # Check if it's a recover pattern: decrease to min, then increase
        if min_idx > n // 4:  # Min should not be too early
            first_part_slope = (values_array[min_idx] - values_array[0]) / max(min_idx, 1)
            second_part_slope = (values_array[-1] - values_array[min_idx]) / max(n - min_idx, 1)
            value_range = np.max(values_array) - np.min(values_array)
            if value_range > 0:
                first_norm = first_part_slope / value_range
                second_norm = second_part_slope / value_range
                if first_norm < -0.003 and second_norm > 0.003:
                    trend_type = "Recover"
                    confidence = abs(first_norm) + abs(second_norm)
    
    # Refine classification (more lenient for Peak/Recover)
    # Use multiple split points to find the best Peak/Recover pattern
    n = len(values_array)
    value_range = np.max(values_array) - np.min(values_array)
    if value_range == 0:
        return 5, "Oscillate"
    
    # Initialize variables for best scores
    best_peak_score = 0
    best_recover_score = 0
    best_peak_split = n // 2
    best_recover_split = n // 2
    
    # Try multiple split points
    split_points = list(range(n // 4, 3 * n // 4, max(1, n // 12)))
    
    for split in split_points:
        if split < 2 or split >= n - 2:
            continue
        
        first_half_slope = (values_array[split] - values_array[0]) / max(split, 1)
        second_half_slope = (values_array[-1] - values_array[split]) / max(n - split, 1)
        
        # Normalize by value range
        first_norm = first_half_slope / value_range
        second_norm = second_half_slope / value_range
        
        # Score for Peak (increase then decrease)
        if first_norm > 0 and second_norm < 0:
            peak_score = abs(first_norm) + abs(second_norm)
            if peak_score > best_peak_score:
                best_peak_score = peak_score
                best_peak_split = split
        
        # Score for Recover (decrease then increase)
        if first_norm < 0 and second_norm > 0:
            recover_score = abs(first_norm) + abs(second_norm)
            if recover_score > best_recover_score:
                best_recover_score = recover_score
                best_recover_split = split
    
    # Use a very lenient threshold (lowered to increase Peak/Recover detection)
    threshold = 0.002
    
    # Check for clear monotonic trends first (Rise/Fall) before Peak/Recover
    # If the sequence is clearly monotonic (high ratio), prioritize Rise/Fall
    # Lower threshold to 0.45 (45% of steps in same direction) - very lenient
    positive_ratio = np.sum(diff > 0) / len(diff) if len(diff) > 0 else 0
    negative_ratio = np.sum(diff < 0) / len(diff) if len(diff) > 0 else 0
    
    if positive_ratio >= 0.45 and overall_slope > 0:
        # Only classify as Peak if Peak pattern is very strong
        if best_peak_score < threshold * 2.0:  # More lenient - allow weak peak patterns
            return 1, "Rise"
    if negative_ratio >= 0.45 and overall_slope < 0:
        # Only classify as Recover if Recover pattern is very strong
        if best_recover_score < threshold * 2.0:  # More lenient - allow weak recover patterns
            return 2, "Fall"
    
    # Check Peak/Recover patterns
    peak_detected = False
    recover_detected = False
    
    if best_peak_score > threshold or trend_type == "Peak":
        # Verify it's actually peak-like
        split = best_peak_split if best_peak_score > threshold else n // 2
        first_half_slope = (values_array[split] - values_array[0]) / max(split, 1)
        second_half_slope = (values_array[-1] - values_array[split]) / max(n - split, 1)
        first_norm = first_half_slope / value_range
        second_norm = second_half_slope / value_range
        
        # Very lenient check: just need opposite signs with some magnitude
        if first_norm > threshold * 0.2 and second_norm < -threshold * 0.2:
            # Only classify as Oscillate if there are MANY sign changes (very strict)
            if len(sign_changes) >= 7:
                return 5, "Oscillate"
            peak_detected = True
    
    if best_recover_score > threshold or trend_type == "Recover":
        # Verify it's actually recover-like
        split = best_recover_split if best_recover_score > threshold else n // 2
        first_half_slope = (values_array[split] - values_array[0]) / max(split, 1)
        second_half_slope = (values_array[-1] - values_array[split]) / max(n - split, 1)
        first_norm = first_half_slope / value_range
        second_norm = second_half_slope / value_range
        
        # Very lenient check: just need opposite signs with some magnitude
        if first_norm < -threshold * 0.2 and second_norm > threshold * 0.2:
            # Only classify as Oscillate if there are MANY sign changes (very strict)
            if len(sign_changes) >= 7:
                return 5, "Oscillate"
            recover_detected = True
    
    # Return Peak/Recover if detected
    if peak_detected:
        return 3, "Peak"
    if recover_detected:
        return 4, "Recover"
    
    # If Peak/Recover not detected, check for Rise/Fall with more lenient criteria
    # Lower threshold to 0.42 (42% of steps in same direction) - very lenient
    if positive_ratio >= 0.42 and overall_slope > 0:
        return 1, "Rise"
    if negative_ratio >= 0.42 and overall_slope < 0:
        return 2, "Fall"
    
    # Fallback to original logic
    if trend_type == "Peak":
        peak_found = False
        for split in split_points:
            if split < 2 or split >= n - 2:
                continue
            first_half_slope = (values_array[split] - values_array[0]) / max(split, 1)
            second_half_slope = (values_array[-1] - values_array[split]) / max(n - split, 1)
            
            # Normalize by value range
            value_range = np.max(values_array) - np.min(values_array)
            if value_range > 0:
                first_half_slope_norm = first_half_slope / value_range
                second_half_slope_norm = second_half_slope / value_range
                threshold = 0.005  # Very lenient threshold
            else:
                first_half_slope_norm = first_half_slope
                second_half_slope_norm = second_half_slope
                threshold = 0.0001
            
            if first_half_slope_norm > threshold and second_half_slope_norm < -threshold:
                peak_found = True
                break
        
        if peak_found:
            return 3, "Peak"
        else:
            # Might be oscillate or other pattern
            if len(sign_changes) >= 6:  # More strict for oscillation
                return 5, "Oscillate"
            elif overall_slope > 0:
                # Check if it's clearly rising
                positive_ratio = np.sum(diff > 0) / len(diff) if len(diff) > 0 else 0
                if positive_ratio >= 0.42:
                    return 1, "Rise"
                return 5, "Oscillate"
            else:
                # Check if it's clearly falling
                negative_ratio = np.sum(diff < 0) / len(diff) if len(diff) > 0 else 0
                if negative_ratio >= 0.42:
                    return 2, "Fall"
                return 5, "Oscillate"
    
    elif trend_type == "Recover":
        # Verify it's actually recover-like: first part decreases, second part increases
        # Use multiple split points to be more lenient
        n = len(values_array)
        split_points = [n // 3, n // 2, 2 * n // 3]
        
        recover_found = False
        for split in split_points:
            if split < 2 or split >= n - 2:
                continue
            first_half_slope = (values_array[split] - values_array[0]) / max(split, 1)
            second_half_slope = (values_array[-1] - values_array[split]) / max(n - split, 1)
            
            # Normalize by value range
            value_range = np.max(values_array) - np.min(values_array)
            if value_range > 0:
                first_half_slope_norm = first_half_slope / value_range
                second_half_slope_norm = second_half_slope / value_range
                threshold = 0.005  # Very lenient threshold
            else:
                first_half_slope_norm = first_half_slope
                second_half_slope_norm = second_half_slope
                threshold = 0.0001
            
            if first_half_slope_norm < -threshold and second_half_slope_norm > threshold:
                recover_found = True
                break
        
        if recover_found:
            return 4, "Recover"
        else:
            # Might be oscillate or other pattern
            if len(sign_changes) >= 6:  # More strict for oscillation
                return 5, "Oscillate"
            elif overall_slope > 0:
                # Check if it's clearly rising
                positive_ratio = np.sum(diff > 0) / len(diff) if len(diff) > 0 else 0
                if positive_ratio >= 0.42:
                    return 1, "Rise"
                return 5, "Oscillate"
            else:
                # Check if it's clearly falling
                negative_ratio = np.sum(diff < 0) / len(diff) if len(diff) > 0 else 0
                if negative_ratio >= 0.42:
                    return 2, "Fall"
                return 5, "Oscillate"
    
    elif trend_type == "Rise":
        # Check if it's truly monotonic or has some fluctuations
        positive_ratio = np.sum(diff > 0) / len(diff) if len(diff) > 0 else 0
        # More lenient check: at least 42% of steps are increasing
        if positive_ratio >= 0.42 and overall_slope > 0:
            # Make sure it's not a strong Peak pattern
            if best_peak_score < threshold * 1.2:  # More lenient - allow weak peak patterns
                return 1, "Rise"
        # If it has strong peak-like pattern, prefer Peak
        if best_peak_score > threshold * 2.0:
            return 3, "Peak"
        # Fallback: if mostly increasing, classify as Rise (very lenient)
        if positive_ratio >= 0.40 and overall_slope > 0:
            return 1, "Rise"
        return 5, "Oscillate"
    
    elif trend_type == "Fall":
        # Check if it's truly monotonic or has some fluctuations
        negative_ratio = np.sum(diff < 0) / len(diff) if len(diff) > 0 else 0
        # More lenient check: at least 42% of steps are decreasing
        if negative_ratio >= 0.42 and overall_slope < 0:
            # Make sure it's not a strong Recover pattern
            if best_recover_score < threshold * 1.2:  # More lenient - allow weak recover patterns
                return 2, "Fall"
        # If it has strong recover-like pattern, prefer Recover
        if best_recover_score > threshold * 2.0:
            return 4, "Recover"
        # Fallback: if mostly decreasing, classify as Fall (very lenient)
        if negative_ratio >= 0.40 and overall_slope < 0:
            return 2, "Fall"
        return 5, "Oscillate"
    
    else:  # Oscillate or uncertain
        # Last chance to check for Peak/Recover before defaulting to Oscillate
        if best_peak_score > threshold * 0.5:
            return 3, "Peak"
        if best_recover_score > threshold * 0.5:
            return 4, "Recover"
        # Also check for Rise/Fall as last resort
        if positive_ratio >= 0.42 and overall_slope > 0:
            return 1, "Rise"
        if negative_ratio >= 0.42 and overall_slope < 0:
            return 2, "Fall"
        return 5, "Oscillate"


def process_dataset(input_file: Path, output_file: Path) -> None:
    """Process dataset and add shape labels based on historical data."""
    print(f"Loading data from {input_file}...")
    data = load_data(input_file)
    
    print(f"Processing {len(data)} samples...")
    
    shape_counts = Counter()
    
    for i, record in enumerate(data):
        # Create a copy to preserve original fields
        new_record = {k: v for k, v in record.items()}
        historical_data = record.get("historical_data", [])  # Changed from "hist_data" to "historical_data"
        if not historical_data:
            if (i + 1) % 1000 == 0:
                print(f"Warning: Sample {i} has no historical_data")
            new_record["shape"] = "Oscillate"
            data[i] = new_record
            continue
        
        # Handle string format (comma-separated values)
        if isinstance(historical_data, str):
            try:
                hist_values = [float(x.strip()) for x in historical_data.split(',') if x.strip()]
            except (ValueError, TypeError):
                if (i + 1) % 1000 == 0:
                    print(f"Warning: Sample {i} has invalid historical_data string format")
                new_record["shape"] = "Oscillate"
                data[i] = new_record
                continue
        elif isinstance(historical_data, list):
            # Convert to float if needed
            try:
                hist_values = [float(x) for x in historical_data]
            except (ValueError, TypeError):
                if (i + 1) % 1000 == 0:
                    print(f"Warning: Sample {i} has invalid historical_data list format")
                new_record["shape"] = "Oscillate"
                data[i] = new_record
                continue
        else:
            if (i + 1) % 1000 == 0:
                print(f"Warning: Sample {i} has unexpected historical_data type: {type(historical_data)}")
            new_record["shape"] = "Oscillate"
            data[i] = new_record
            continue
        
        # Classify shape
        label_id, label_name = classify_shape(hist_values)  # Using historical_data instead of ground_truth
        # Only add 'shape' field, keep original fields unchanged
        new_record["shape"] = label_name
        data[i] = new_record
        shape_counts[label_name] += 1
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...")
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Shape Classification Statistics (based on historical data)")
    print("=" * 80)
    total = len(data)
    for shape_name in ["Rise", "Fall", "Peak", "Recover", "Oscillate"]:
        count = shape_counts[shape_name]
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{shape_name:12s}: {count:5d} ({percentage:5.2f}%)")
    print("=" * 80)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    
    # Output directory for new dataset version (using historical data)
    output_dir = project_root / "dataset" / "Electricity" / "ver_synchronized_shape_historical"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process test set
    test_input = project_root / "dataset" / "Electricity" / "ver_camf" / "test.json"
    test_output = output_dir / "test.json"
    
    if test_input.exists():
        print("Processing test set using historical data...")
        process_dataset(test_input, test_output)
    else:
        print(f"Error: {test_input} not found")
    
    # Process train set if exists
    train_input = project_root / "dataset" / "Electricity" / "ver_camf" / "train.json"
    train_output = output_dir / "train.json"
    
    if train_input.exists():
        print("\nProcessing train set using historical data...")
        process_dataset(train_input, train_output)
    
    # Process vali set if exists
    vali_input = project_root / "dataset" / "Electricity" / "ver_camf" / "vali.json"
    vali_output = output_dir / "vali.json"
    
    if vali_input.exists():
        print("\nProcessing vali set using historical data...")
        process_dataset(vali_input, vali_output)


if __name__ == "__main__":
    main()