"""Legacy primitive schemas and lightweight output parsing.

PrimitiveSpec preserves the label order used later for label_id/gt_label_id.
The parser intentionally returns the first valid label found in model text so
mock and OpenAI-compatible completions share the same downstream path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    labels: tuple[str, ...]
    legacy_output_field: str

    @property
    def label_to_id(self) -> dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}


PRIMITIVE_SPECS: dict[str, PrimitiveSpec] = {
    "distribution_shift": PrimitiveSpec(
        name="distribution_shift",
        labels=("STRONG-RISE", "MILD-RISE", "STABLE", "MILD-DROP", "STRONG-DROP"),
        legacy_output_field="distribution_shift",
    ),
    "volatility": PrimitiveSpec(
        name="volatility",
        labels=("High", "Medium", "Low"),
        legacy_output_field="global_volatility",
    ),
    "shape": PrimitiveSpec(
        name="shape",
        labels=("Rise", "Fall", "Peak", "Recover", "Oscillate"),
        legacy_output_field="shape",
    ),
    "temporal_influence": PrimitiveSpec(
        name="temporal_influence",
        labels=("Immediate", "Delayed", "Sustained"),
        legacy_output_field="Temporal Influence Shape",
    ),
}


def get_primitive_spec(name: str) -> PrimitiveSpec:
    try:
        return PRIMITIVE_SPECS[name]
    except KeyError as exc:
        valid = ", ".join(PRIMITIVE_SPECS)
        raise ValueError(f"Unknown primitive '{name}'. Valid primitives: {valid}") from exc


def parse_label(text: str, spec: PrimitiveSpec) -> str | None:
    if not text:
        return None

    matches: list[tuple[int, int, str]] = []
    for order, label in enumerate(spec.labels):
        pattern = re.compile(rf"(?<![\w-]){re.escape(label)}(?![\w-])", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            matches.append((match.start(), order, label))

    if not matches:
        return None
    return min(matches)[2]
