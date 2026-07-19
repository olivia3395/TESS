"""Prompt construction for primitive inference.

The builder prefers legacy prompt templates when available, then falls back to
a deterministic built-in prompt. Template formatting supports the placeholders
used by legacy FNSPID prompts without modifying the legacy directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .primitive_specs import PrimitiveSpec


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    prompt_source: str
    prompt_template_path: str | None


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def find_legacy_template_path(
    root: str | Path,
    dataset: str,
    primitive: str,
) -> Path | None:
    path = (
        Path(root)
        / "legacy"
        / "primitive_inference"
        / "prompt_templates"
        / dataset
        / f"{primitive}.txt"
    )
    return path if path.exists() else None


def build_fallback_prompt(primitive_spec: PrimitiveSpec, news: str) -> str:
    labels = ", ".join(primitive_spec.labels)
    return (
        "You are a time-series primitive classifier.\n\n"
        f"Primitive: {primitive_spec.name}\n"
        f"Valid labels: {labels}\n\n"
        f"News:\n{news}\n\n"
        "Return exactly one valid label from the list."
    )


def build_prompt(
    root: Path,
    dataset: str,
    primitive_spec: PrimitiveSpec,
    record: dict[str, Any],
    prompt_source: str = "auto",
) -> PromptBuildResult:
    news = str(record.get("news", ""))
    if prompt_source == "fallback":
        return PromptBuildResult(
            prompt=build_fallback_prompt(primitive_spec, news),
            prompt_source="fallback",
            prompt_template_path=None,
        )
    if prompt_source != "auto":
        raise ValueError("--prompt-source must be 'auto' or 'fallback'")

    template_path = find_legacy_template_path(root, dataset, primitive_spec.name)
    if template_path is None:
        return PromptBuildResult(
            prompt=build_fallback_prompt(primitive_spec, news),
            prompt_source="fallback",
            prompt_template_path=None,
        )

    template = template_path.read_text(encoding="utf-8").strip()
    prompt = _format_legacy_template(template, record, primitive_spec)
    return PromptBuildResult(
        prompt=prompt,
        prompt_source="legacy_template",
        prompt_template_path=str(template_path),
    )


def _format_legacy_template(
    template: str,
    record: dict[str, Any],
    primitive_spec: PrimitiveSpec,
) -> str:
    news = str(record.get("news", ""))
    historical_data = str(record.get("historical_data", ""))
    company_name = str(
        record.get("company_name")
        or record.get("company")
        or _extract_company_name_hint(news)
        or ""
    )
    mapping = _SafeFormatDict(
        historical_data=historical_data,
        news=news,
        text=news,
        input=news,
        company=company_name,
        company_name=company_name,
        mean_value=_compute_mean_value(historical_data),
        labels=", ".join(primitive_spec.labels),
        primitive=primitive_spec.name,
    )
    try:
        prompt = template.format_map(mapping)
    except (KeyError, IndexError, ValueError):
        prompt = template

    return (
        f"{prompt}\n\n"
        "Context for this record:\n"
        f"Primitive: {primitive_spec.name}\n"
        f"Valid labels: {', '.join(primitive_spec.labels)}\n"
        f"News:\n{news}\n"
        "Return exactly one valid label from the list."
    )


def _compute_mean_value(historical_data: str) -> str:
    values = [float(match) for match in re.findall(r"-?\d+(?:\.\d+)?", historical_data)]
    if not values:
        return ""
    return f"{sum(values) / len(values):.4f}"


def _extract_company_name_hint(news: str) -> str:
    match = re.search(r"\b([A-Z][A-Za-z0-9&.,'-]*(?:\s+[A-Z][A-Za-z0-9&.,'-]*){0,3})\b", news)
    return match.group(1) if match else ""
