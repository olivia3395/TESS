#!/usr/bin/env python3
"""Normalize trend annotations inside dataset JSON files.

The script scans every JSON file under a given directory and replaces any
occurrence of the value ``0`` inside ``step_trends`` or ``global_trend`` fields
with ``-1``. The original files are updated in-place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace all zero values with -1 inside step_trends and "
            "global_trend fields."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/FNSPID/ver_gen_new_total"),
        help="Directory that contains the target JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and report pending updates without modifying them.",
    )
    return parser.parse_args()


def replace_zero(value: Any) -> Tuple[Any, int]:
    """Recursively replace numeric zeros with -1."""
    if isinstance(value, bool):
        return value, 0
    if isinstance(value, int):
        return (-1, 1) if value == 0 else (value, 0)
    if isinstance(value, list):
        updated_list = []
        replacements = 0
        for item in value:
            new_item, delta = replace_zero(item)
            updated_list.append(new_item)
            replacements += delta
        return updated_list, replacements
    if isinstance(value, dict):
        updated_dict = {}
        replacements = 0
        for key, item in value.items():
            new_item, delta = replace_zero(item)
            updated_dict[key] = new_item
            replacements += delta
        return updated_dict, replacements
    return value, 0


def replace_trend_fields(payload: Any) -> int:
    """Walk the JSON payload and update trend-related fields."""
    replacements = 0
    if isinstance(payload, dict):
        for key in ("step_trends", "global_trend"):
            if key in payload:
                payload[key], delta = replace_zero(payload[key])
                replacements += delta
        for value in payload.values():
            if isinstance(value, (dict, list)):
                replacements += replace_trend_fields(value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, (dict, list)):
                replacements += replace_trend_fields(item)
    return replacements


def process_file(path: Path, dry_run: bool) -> int:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    replacements = replace_trend_fields(data)
    if replacements and not dry_run:
        with path.open("w", encoding="utf-8") as file_obj:
            json.dump(data, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
    return replacements


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise SystemExit(f"Directory not found: {root}")

    total_files = 0
    total_replacements = 0
    for json_path in sorted(root.rglob("*.json")):
        total_files += 1
        replacements = process_file(json_path, args.dry_run)
        total_replacements += replacements
        if replacements:
            action = "would update" if args.dry_run else "updated"
            print(f"{action} {json_path} ({replacements} replacements)")

    print(
        f"Processed {total_files} file(s); "
        f"applied {total_replacements} replacement(s)."
    )


if __name__ == "__main__":
    main()
