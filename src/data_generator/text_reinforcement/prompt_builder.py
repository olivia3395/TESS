from __future__ import annotations

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    base_system: str
    base_user: str
    variants: Dict[str, str]
    default_max_tokens: int

    def render(self, sample: Dict[str, Any], variant: str) -> Dict[str, str]:
        if variant not in self.variants:
            raise KeyError(f"Unknown variant '{variant}'")
        user_prompt = self.base_user.format(
            historical_data=sample.get("historical_data", ""),
            original_news=sample.get("news", ""),
        )
        suffix = self.variants[variant].strip()
        if suffix:
            user_prompt = f"{user_prompt.rstrip()}\n{suffix}"
        return {
            "system_prompt": self.base_system,
            "user_prompt": user_prompt,
        }


@functools.lru_cache(maxsize=4)
def load_prompt_template(config_path: str | Path) -> PromptTemplate:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found: {config_path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        base_system = raw["base_system"].strip()
        base_user = raw["base_user"].strip()
        variants_raw = raw.get("variants", {})
        variants = {}
        for name, value in variants_raw.items():
            if isinstance(value, dict) and 'user_suffix' in value:
                variants[name] = value['user_suffix']
            else:
                variants[name] = value
        default_max_tokens = int(raw.get("default_max_tokens", 320))
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Invalid prompt configuration: %s", json.dumps(raw, ensure_ascii=False)[:500])
        raise ValueError("Malformed prompt configuration") from exc
    return PromptTemplate(
        base_system=base_system,
        base_user=base_user,
        variants=variants,
        default_max_tokens=default_max_tokens,
    )
