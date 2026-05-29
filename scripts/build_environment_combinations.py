"""
Build Environment dataset combinations similar to Bitcoin dataset structure.
Generates combinations of ver_global, ver_shape, ver_temporal_shape_shape, ver_volatility
in three formats: base, natural, and structured.
"""

import json
import os
import re
from pathlib import Path


def extract_value(text, pattern):
    """Extract value from text using pattern."""
    if not text:
        return ""
    match = re.search(pattern, text)
    return match.group(1) if match else text.strip()


def format_base(val1, val2, label1, label2):
    """Base format: comma-separated values."""
    return f"{val1},{val2}"


def format_natural(val1, val2, label1, label2):
    """Natural language format."""
    # Map labels to natural language descriptions
    label_map = {
        "global trend": "global trend",
        "shape": "shape",
        "temporal influence shape": "temporal influence shape",
        "global volatility": "global volatility"
    }
    desc1 = label_map.get(label1, label1)
    desc2 = label_map.get(label2, label2)
    return f"The {desc1} is {val1} and the {desc2} is {val2}"


def format_structured(val1, val2, label1, label2):
    """Structured format."""
    return f"{label1}: {val1}, {label2}: {val2}"


def load_dataset(base_dir, dataset_name, split):
    """Load a dataset split."""
    file_path = os.path.join(base_dir, dataset_name, f"{split}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_field_value(item, dataset_name):
    """Extract the relevant value from dataset item."""
    if dataset_name == "ver_global":
        return item.get("global_trend", "")
    elif dataset_name == "ver_shape":
        news = item.get("news", "")
        # Extract value from "Shape:Peak" format
        match = re.search(r"Shape:\s*(.+)", news)
        return match.group(1).strip() if match else ""
    elif dataset_name == "ver_temporal_shape":
        news = item.get("news", "")
        # Extract value from "Temporal Influence Shape: Delayed" format
        match = re.search(r"Temporal Influence Shape:\s*(.+)", news)
        return match.group(1).strip() if match else ""
    elif dataset_name == "ver_volatility":
        news = item.get("news", "")
        # Extract value from "Global volatility: Medium" format
        match = re.search(r"Global volatility:\s*(.+)", news)
        return match.group(1).strip() if match else ""
    else:
        return ""


def get_label_name(dataset_name):
    """Get the label name for a dataset."""
    label_map = {
        "ver_global": "global trend",
        "ver_shape": "shape",
        "ver_temporal_shape": "temporal influence shape",
        "ver_volatility": "global volatility"
    }
    return label_map.get(dataset_name, dataset_name)


def create_combination(base_dir, output_dir, dataset1_name, dataset2_name, format_func, format_name):
    """Create a combination dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    label1 = get_label_name(dataset1_name)
    label2 = get_label_name(dataset2_name)
    
    for split in ["train", "vali", "test"]:
        try:
            data1 = load_dataset(base_dir, dataset1_name, split)
            data2 = load_dataset(base_dir, dataset2_name, split)
            
            if len(data1) != len(data2):
                print(f"Warning: {split} length mismatch: {len(data1)} vs {len(data2)}")
                continue
            
            merged = []
            for item1, item2 in zip(data1, data2):
                val1 = extract_field_value(item1, dataset1_name)
                val2 = extract_field_value(item2, dataset2_name)
                
                if not val1 or not val2:
                    continue
                
                # Create combined news field
                combined_news = format_func(val1, val2, label1, label2)
                
                # Create new item based on first dataset
                new_item = {
                    "historical_data": item1.get("historical_data", ""),
                    "ground_truth": item1.get("ground_truth", ""),
                    "news": combined_news
                }
                merged.append(new_item)
            
            output_file = output_path / f"{split}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            
            print(f"[{format_name}] {split}: {len(merged)} samples -> {output_file}")
            
        except Exception as e:
            print(f"Error processing {split} for {dataset1_name}+{dataset2_name} ({format_name}): {e}")


def get_output_name(dataset1, dataset2):
    """Generate output dataset name following Bitcoin naming convention."""
    # Remove "ver_" prefix and combine
    name1 = dataset1.replace("ver_", "")
    name2 = dataset2.replace("ver_", "")
    
    # Special handling for naming conventions (following Bitcoin order)
    # Order priority: global > shape > temporal > volatility
    if name1 == "global":
        if name2 == "shape":
            return "ver_global_shape"
        elif name2 == "temporal_shape":
            return "ver_global_temporal_shape"
        elif name2 == "volatility":
            return "ver_global_volatility"
    elif name1 == "shape":
        if name2 == "temporal_shape":
            return "ver_shape_temporal_shape"
        elif name2 == "volatility":
            # Bitcoin uses ver_volatility_shape, so reverse order
            return "ver_volatility_shape"
    elif name1 == "temporal_shape":
        if name2 == "volatility":
            # Bitcoin uses ver_volatility_temporal, so reverse order
            return "ver_volatility_temporal_shape"
    elif name1 == "volatility":
        # If volatility is first, keep it first
        if name2 == "shape":
            return "ver_volatility_shape"
        elif name2 == "temporal_shape":
            return "ver_volatility_temporal_shape"
    
    # Fallback: combine names
    return f"ver_{name1}_{name2}"


def main():
    base_dir = "MMTSF_LIB/dataset/Environment"
    output_base = "MMTSF_LIB/dataset/Environment"
    
    # Define combinations: (dataset1, dataset2)
    # Note: Order matters for naming - volatility should come first when combined with shape/temporal
    combinations = [
        ("ver_global", "ver_shape"),
        ("ver_global", "ver_temporal_shape"),
        ("ver_global", "ver_volatility"),
        ("ver_shape", "ver_temporal_shape"),
        ("ver_volatility", "ver_shape"),  # Reversed to match Bitcoin naming
        ("ver_volatility", "ver_temporal_shape"),  # Reversed to match Bitcoin naming
    ]
    
    # Define formats: (format_func, format_suffix)
    formats = [
        (format_base, ""),
        (format_natural, "_natural"),
        (format_structured, "_structured"),
    ]
    
    print("=" * 80)
    print("Building Environment dataset combinations")
    print("=" * 80)
    print()
    
    total = 0
    for dataset1, dataset2 in combinations:
        base_output_name = get_output_name(dataset1, dataset2)
        for format_func, format_suffix in formats:
            output_name = f"{base_output_name}{format_suffix}"
            output_dir = os.path.join(output_base, output_name)
            
            print(f"Creating: {output_name}")
            create_combination(
                base_dir,
                output_dir,
                dataset1,
                dataset2,
                format_func,
                format_suffix or "base"
            )
            print()
            total += 1
    
    print("=" * 80)
    print(f"Completed! Generated {total} dataset combinations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
