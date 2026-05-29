#!/usr/bin/env python3
"""Compare per-sample MSE values across multiple JSONL reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ScoreEntry:
    mse: float
    line_number: int


@dataclass(frozen=True)
class WinningCase:
    sample_id: str
    target_mse: float
    best_other_mse: float
    line_number: int

    @property
    def margin(self) -> float:
        return self.best_other_mse - self.target_mse


def load_scores(path: Path) -> Dict[str, ScoreEntry]:
    """Load a JSONL file into a mapping of sample_id -> ScoreEntry."""
    scores: Dict[str, ScoreEntry] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc

            sample_id = record.get("sample_id")
            if sample_id is None:
                raise ValueError(f"{path}:{line_number} missing sample_id")
            if "mse" not in record:
                raise ValueError(f"{path}:{line_number} missing mse field")

            scores[sample_id] = ScoreEntry(mse=float(record["mse"]), line_number=line_number)
    return scores


def compare_scores(
    target: Dict[str, ScoreEntry], others: List[Dict[str, ScoreEntry]]
) -> List[WinningCase]:
    """Return winners where target mse is strictly lower than all other sets."""
    winners: List[WinningCase] = []
    for sample_id, target_entry in target.items():
        try:
            other_entries = [dataset[sample_id] for dataset in others]
        except KeyError:
            # Skip if a dataset lacks this sample_id; comparisons would be unfair.
            continue

        if all(target_entry.mse < entry.mse for entry in other_entries):
            best_other_mse = min(entry.mse for entry in other_entries)
            winners.append(
                WinningCase(
                    sample_id=sample_id,
                    target_mse=target_entry.mse,
                    best_other_mse=best_other_mse,
                    line_number=target_entry.line_number,
                )
            )
    return winners


def infer_dataset_test_file(target_path: Path) -> Optional[Path]:
    """Infer dataset test file from the target output path."""
    parts = target_path.resolve().parts
    output_idx: Optional[int] = None
    for idx, token in enumerate(parts):
        if token == "output":
            output_idx = idx
            break
    if output_idx is None:
        return None
    if len(parts) <= output_idx + 2:
        return None

    dataset_root = parts[output_idx + 1]
    alias = parts[output_idx + 2]

    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "dataset" / dataset_root / alias

    for filename in ("test.json", "test.jsonl"):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def load_dataset_news(path: Path) -> Dict[str, Optional[str]]:
    """Load dataset file and return sample_id -> news mapping."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError
        records = data
    except (json.JSONDecodeError, ValueError):
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
                if isinstance(record, dict):
                    records.append(record)

    news_map: Dict[str, Optional[str]] = {}
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        sample_id = record.get("sample_id") or f"test_{idx}"
        news_map[sample_id] = record.get("news")
    return news_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find sample_ids whose mse in the target file is lower than all other files."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="JSONL file providing the candidate (baseline) scores.",
    )
    parser.add_argument(
        "others",
        nargs="+",
        type=Path,
        help="JSONL files to compare against.",
    )
    parser.add_argument(
        "--dataset-test-file",
        type=Path,
        default=None,
        help="Path to the dataset test JSON/JSONL file for retrieving raw news (default: inferred from target path).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of best-improved cases to surface (default: 4).",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be a positive integer")

    target_path = args.target.resolve()
    target_scores = load_scores(target_path)
    other_scores = [load_scores(path.resolve()) for path in args.others]

    winners = compare_scores(target_scores, other_scores)
    if args.dataset_test_file:
        news_file = args.dataset_test_file.resolve()
        if not news_file.exists():
            raise FileNotFoundError(f"{news_file} does not exist")
    else:
        news_file = infer_dataset_test_file(target_path)
    news_map = load_dataset_news(news_file) if news_file else {}

    top_cases = sorted(winners, key=lambda case: case.margin, reverse=True)[: args.top_k]
    top_cases_payload = [
        {
            "sample_id": case.sample_id,
            "line_number": case.line_number,
            "target_mse": case.target_mse,
            "best_other_mse": case.best_other_mse,
            "margin": case.margin,
            "news": news_map.get(case.sample_id),
        }
        for case in top_cases
    ]

    payload = {
        "count": len(winners),
        "sample_ids": [case.sample_id for case in winners],
        "top_cases": top_cases_payload,
    }
    if news_file:
        payload["news_source"] = str(news_file)

    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
