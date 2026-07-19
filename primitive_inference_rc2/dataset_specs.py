"""Dataset layout definitions for the clean TESS-RC2 repository.

The refactor reads raw records from dataset/<dataset>/<split>.json and avoids
the legacy ver_primitive layout except as read-only reference material.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_SPLITS = ("train", "vali", "test")
VALID_DATASETS = ("fnspid", "bitcoin", "electricity", "environment")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_dir: Path

    def split_path(self, split: str) -> Path:
        if split not in VALID_SPLITS:
            valid = ", ".join(VALID_SPLITS)
            raise ValueError(f"Unknown split '{split}'. Valid splits: {valid}")
        return self.dataset_dir / f"{split}.json"


def get_dataset_spec(name: str, root: str | Path = ".") -> DatasetSpec:
    if name not in VALID_DATASETS:
        valid = ", ".join(VALID_DATASETS)
        raise ValueError(f"Unknown dataset '{name}'. Valid datasets: {valid}")
    root_path = Path(root)
    return DatasetSpec(name=name, dataset_dir=root_path / "dataset" / name)
