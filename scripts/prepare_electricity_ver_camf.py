"""Generate ver_camf splits for Electricity and Environment datasets."""
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


def strip_prompt_and_rename_meta(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Electricity: drop prompt and rename meta_domain to news."""
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        transformed: Dict[str, Any] = {}
        for key, value in record.items():
            if key == "prompt":
                continue
            transformed["news" if key == "meta_domain" else key] = value
        cleaned.append(transformed)
    return cleaned


def trim_news_prefix(news: Any) -> Any:
    """If news is a string, drop any content through 'Notably' and tidy casing."""
    if not isinstance(news, str):
        return news
    marker = "Notably"
    idx = news.find(marker)
    if idx == -1:
        return news
    trimmed = news[idx + len(marker) :].lstrip(" ,")
    if trimmed:
        trimmed = trimmed[0].upper() + trimmed[1:]
    return trimmed


def select_environment_fields(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Environment: keep historical_data/ground_truth, add news from filter_events."""
    required_keys = ("ground_truth", "historical_data", "filter_events")
    cleaned: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        missing = [key for key in required_keys if key not in record]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Record {idx} missing required keys: {missing_str}")
        cleaned.append(
            {
                "historical_data": record["historical_data"],
                "ground_truth": record["ground_truth"],
                "news": trim_news_prefix(record["filter_events"]),
            }
        )
    return cleaned


def transform_records(dataset: str, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dataset_lower = dataset.lower()
    if dataset_lower == "electricity":
        return strip_prompt_and_rename_meta(records)
    if dataset_lower == "environment":
        return select_environment_fields(records)
    raise ValueError(f"Unsupported dataset for transformation: {dataset}")


def write_json(data: List[Dict[str, Any]], path: Path) -> None:
    """Write data to path with a trailing newline for git-friendly diffs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(serialized + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create ver_camf splits. Electricity: drop prompt and rename meta_domain to news. "
            "Environment: keep historical/gt/meta_data and rename filter_events to news."
        )
    )
    parser.add_argument(
        "--dataset",
        default="Electricity",
        help="Dataset folder name under MMTSF_LIB/dataset (default: Electricity).",
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
        cleaned = transform_records(args.dataset, records)

        target_path = target_root / source_path.name
        write_json(cleaned, target_path)

        print(f"{split}: wrote {len(cleaned)} records -> {target_path.relative_to(project_root)}")


if __name__ == "__main__":
    main()
