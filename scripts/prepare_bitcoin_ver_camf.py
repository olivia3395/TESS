"""Generate Bitcoin/ver_camf dataset by stripping prompt fields from ver_base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, cast

DEFAULT_SPLITS = ("train", "vali", "test")


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array of dict records from path."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} in {path} is {type(item).__name__}, expected dict")
    return cast(List[Dict[str, Any]], data)


def drop_prompt(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return new records with the prompt field removed, preserving other keys."""
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        cleaned.append({key: value for key, value in record.items() if key != "prompt"})
    return cleaned


def write_json(data: List[Dict[str, Any]], path: Path) -> None:
    """Write data to path with a trailing newline for git-friendly diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Bitcoin dataset version without prompt fields."
    )
    parser.add_argument(
        "--dataset",
        default="Electricity",
        help="Dataset folder name under MMTSF_LIB/dataset (default: Bitcoin).",
    )
    parser.add_argument(
        "--source-version",
        default="ver_base",
        help="Source version folder to read from (default: ver_base).",
    )
    parser.add_argument(
        "--target-version",
        default="ver_camf",
        help="Target version folder to write to (default: ver_camf).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset split names to process (default: train vali test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        cleaned = drop_prompt(records)

        target_path = target_root / source_path.name
        write_json(cleaned, target_path)

        print(f"{split}: wrote {len(cleaned)} records -> {target_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
