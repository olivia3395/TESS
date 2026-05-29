#!/usr/bin/env python
"""Capitalize the first letter of global_volatility field values."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


def capitalize_volatility_field(data: List[Dict[str, Any]]) -> None:
    """Capitalize the first letter of global_volatility field values."""
    mapping = {
        "low": "Low",
        "medium": "Medium",
        "high": "High"
    }
    
    count = 0
    for record in data:
        if "global_volatility" in record:
            old_value = record["global_volatility"]
            if old_value in mapping:
                record["global_volatility"] = mapping[old_value]
                count += 1
    
    return count


def process_file(file_path: Path) -> None:
    """Process a single JSON file."""
    print(f"Processing {file_path.name}...")
    
    # Load data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} samples")
    
    # Capitalize volatility field
    count = capitalize_volatility_field(data)
    print(f"  Updated {count} records")
    
    # Save back
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved to {file_path}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "dataset" / "Electricity" / "ver_synchronized_volatility"
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    # Process all JSON files in the directory
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return
    
    print("=" * 80)
    print("Capitalizing global_volatility field values")
    print("=" * 80)
    
    for json_file in sorted(json_files):
        process_file(json_file)
        print()
    
    print("=" * 80)
    print("All files processed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

