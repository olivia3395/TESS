"""Runtime dataset adapter for TESS no-gate and gated experiments.

TESS_Dataset joins raw forecasting samples with grouped primitive gate cache
records in memory. The returned item separates model inputs
(`x`, `text_primitive_ids`) from auxiliary target/evaluation fields
(`y`, `gt_primitive_ids`, `gate_targets`) so no-gate training can ignore gate
targets while gated experiments can consume them explicitly.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .cache_io import grouped_gate_cache_path
from .dataset_specs import get_dataset_spec
from .gt_labelers import parse_numeric_sequence
from .primitive_specs import get_primitive_spec


DEFAULT_PRIMITIVE_ORDER = (
    "distribution_shift",
    "volatility",
    "shape",
    "temporal_influence",
)


class TESS_Dataset(Dataset):
    """In-memory join of raw sequence records and grouped primitive cache."""

    def __init__(
        self,
        root: str | Path,
        dataset: str,
        split: str,
        primitive_order: Sequence[str] | None = None,
        include_aux: bool = True,
        include_probs: bool = False,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.include_aux = include_aux
        self.include_probs = include_probs
        self.primitive_order = tuple(primitive_order or DEFAULT_PRIMITIVE_ORDER)
        self.primitive_label_counts = {
            primitive: len(get_primitive_spec(primitive).labels)
            for primitive in self.primitive_order
        }
        self.primitive_vocab_sizes = {
            primitive: label_count + 1
            for primitive, label_count in self.primitive_label_counts.items()
        }
        self.primitive_unk_ids = dict(self.primitive_label_counts)

        dataset_spec = get_dataset_spec(dataset, self.root)
        raw_path = dataset_spec.split_path(split)
        grouped_path = grouped_gate_cache_path(self.root, dataset, split)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw split file does not exist: {raw_path}")
        if not grouped_path.exists():
            raise FileNotFoundError(f"Grouped gate cache does not exist: {grouped_path}")

        self.raw_records = self._load_json_list(raw_path)
        self.grouped_records = self._load_json_list(grouped_path)
        self.samples = self._validate_and_build_samples()
        self.seq_len_counts = Counter(len(sample["x_values"]) for sample in self.samples)
        self.pred_len_counts = Counter(len(sample["y_values"]) for sample in self.samples)
        self.seq_lens = dict(self.seq_len_counts)
        self.pred_lens = dict(self.pred_len_counts)

    @staticmethod
    def _load_json_list(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON list in {path}")
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Expected record {idx} in {path} to be an object")
        return data

    def _validate_and_build_samples(self) -> list[dict[str, Any]]:
        samples = []
        grouped_primitive_order: tuple[str, ...] | None = None
        for row_idx, grouped_record in enumerate(self.grouped_records):
            sample_id = grouped_record.get("sample_id")
            if not isinstance(sample_id, int):
                raise ValueError(f"Grouped record {row_idx} has non-integer sample_id.")
            if sample_id < 0 or sample_id >= len(self.raw_records):
                raise ValueError(
                    f"Grouped record {row_idx} sample_id={sample_id} cannot index "
                    f"{len(self.raw_records)} raw records."
                )
            if grouped_record.get("dataset") != self.dataset:
                raise ValueError(f"Grouped record {row_idx} has wrong dataset.")
            if grouped_record.get("split") != self.split:
                raise ValueError(f"Grouped record {row_idx} has wrong split.")

            record_order = tuple(grouped_record.get("primitive_order", ()))
            if grouped_primitive_order is None:
                grouped_primitive_order = record_order
            elif record_order != grouped_primitive_order:
                raise ValueError("Grouped records do not share one primitive_order.")
            missing = [primitive for primitive in self.primitive_order if primitive not in record_order]
            if missing:
                raise ValueError(f"Grouped record {row_idx} missing primitives: {missing}")

            raw_record = self.raw_records[sample_id]
            x_values = self._parse_sequence(raw_record.get("historical_data"), "historical_data", sample_id)
            y_values = self._parse_sequence(raw_record.get("ground_truth"), "ground_truth", sample_id)

            text_ids, text_mask = self._ids_and_mask(
                grouped_record.get("label_ids", {}), "label_ids", row_idx
            )
            gt_ids, gt_mask = self._ids_and_mask(
                grouped_record.get("gt_label_ids", {}), "gt_label_ids", row_idx
            )
            gate_targets = self._gate_targets(grouped_record.get("gate_targets", {}), row_idx)
            text_margins = self._text_margins(grouped_record.get("primitives", {}))
            text_probs, text_prob_mask = self._text_probs(grouped_record.get("primitives", {}))
            samples.append(
                {
                    "sample_id": sample_id,
                    "x_values": x_values,
                    "y_values": y_values,
                    "text_ids": text_ids,
                    "text_mask": text_mask,
                    "text_margins": text_margins,
                    "text_probs": text_probs,
                    "text_prob_mask": text_prob_mask,
                    "gt_ids": gt_ids,
                    "gt_mask": gt_mask,
                    "gate_targets": gate_targets,
                }
            )
        return samples

    @staticmethod
    def _parse_sequence(value: Any, field: str, sample_id: int) -> list[float]:
        values = parse_numeric_sequence(value)
        if not values:
            raise ValueError(f"sample_id={sample_id} has empty {field}.")
        return values

    def _ids_and_mask(
        self,
        id_map: dict[str, Any],
        field: str,
        row_idx: int,
    ) -> tuple[list[int], list[bool]]:
        ids = []
        mask = []
        if not isinstance(id_map, dict):
            raise ValueError(f"Grouped record {row_idx} has invalid {field}.")
        for primitive in self.primitive_order:
            raw_id = id_map.get(primitive)
            unk_id = self.primitive_unk_ids[primitive]
            if raw_id is None:
                ids.append(unk_id)
                mask.append(False)
                continue
            if not isinstance(raw_id, int) or raw_id < 0 or raw_id >= unk_id:
                raise ValueError(
                    f"Grouped record {row_idx} has invalid {field}.{primitive}: {raw_id!r}"
                )
            ids.append(raw_id)
            mask.append(True)
        return ids, mask

    def _gate_targets(self, gate_map: dict[str, Any], row_idx: int) -> list[float]:
        if not isinstance(gate_map, dict):
            raise ValueError(f"Grouped record {row_idx} has invalid gate_targets.")
        targets = []
        for primitive in self.primitive_order:
            target = gate_map.get(primitive)
            if target not in (0, 1):
                raise ValueError(
                    f"Grouped record {row_idx} has invalid gate_targets.{primitive}: {target!r}"
                )
            targets.append(float(target))
        return targets

    def _text_margins(self, primitive_map: dict[str, Any]) -> list[float]:
        margins = []
        if not isinstance(primitive_map, dict):
            return [0.0 for _ in self.primitive_order]
        for primitive in self.primitive_order:
            primitive_record = primitive_map.get(primitive, {})
            margin = primitive_record.get("margin") if isinstance(primitive_record, dict) else None
            if isinstance(margin, bool) or margin is None:
                margins.append(0.0)
                continue
            try:
                margin_value = float(margin)
            except (TypeError, ValueError):
                margin_value = 0.0
            margins.append(margin_value if torch.isfinite(torch.tensor(margin_value)) else 0.0)
        return margins

    def _text_probs(self, primitive_map: dict[str, Any]) -> tuple[list[list[float]], list[list[bool]]]:
        max_vocab_size = max(self.primitive_vocab_sizes.values())
        probs_by_primitive = []
        mask_by_primitive = []
        if not isinstance(primitive_map, dict):
            return (
                [[0.0 for _ in range(max_vocab_size)] for _ in self.primitive_order],
                [[False for _ in range(max_vocab_size)] for _ in self.primitive_order],
            )
        for primitive in self.primitive_order:
            spec = get_primitive_spec(primitive)
            vocab_size = self.primitive_vocab_sizes[primitive]
            probs = [0.0 for _ in range(max_vocab_size)]
            prob_mask = [idx < vocab_size for idx in range(max_vocab_size)]
            sample_probs = {}
            primitive_record = primitive_map.get(primitive, {})
            if isinstance(primitive_record, dict) and isinstance(primitive_record.get("sample_probs"), dict):
                sample_probs = primitive_record["sample_probs"]
            total = 0.0
            for label, raw_prob in sample_probs.items():
                if label not in spec.label_to_id:
                    continue
                try:
                    prob = float(raw_prob)
                except (TypeError, ValueError):
                    continue
                if not torch.isfinite(torch.tensor(prob)) or prob < 0.0:
                    continue
                label_id = spec.label_to_id[label]
                probs[label_id] = prob
                total += prob
            if total > 0.0:
                probs = [prob / total for prob in probs]
            probs_by_primitive.append(probs)
            mask_by_primitive.append(prob_mask)
        return probs_by_primitive, mask_by_primitive

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample = self.samples[index]
        item: dict[str, torch.Tensor | int] = {
            "x": torch.tensor(sample["x_values"], dtype=torch.float32).unsqueeze(-1),
            "y": torch.tensor(sample["y_values"], dtype=torch.float32).unsqueeze(-1),
            "text_primitive_ids": torch.tensor(sample["text_ids"], dtype=torch.long),
            "text_primitive_mask": torch.tensor(sample["text_mask"], dtype=torch.bool),
            "text_primitive_margins": torch.tensor(sample["text_margins"], dtype=torch.float32),
            "sample_id": torch.tensor(sample["sample_id"], dtype=torch.long),
        }
        if self.include_aux:
            item.update(
                {
                    "gt_primitive_ids": torch.tensor(sample["gt_ids"], dtype=torch.long),
                    "gt_primitive_mask": torch.tensor(sample["gt_mask"], dtype=torch.bool),
                    "gate_targets": torch.tensor(sample["gate_targets"], dtype=torch.float32),
                }
            )
        if self.include_probs:
            item.update(
                {
                    "text_primitive_probs": torch.tensor(sample["text_probs"], dtype=torch.float32),
                    "text_primitive_prob_mask": torch.tensor(sample["text_prob_mask"], dtype=torch.bool),
                }
            )
        return item
