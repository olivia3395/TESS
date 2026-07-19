"""LLM client abstractions for sampled primitive inference.

MockLLMClient is deterministic and dependency-free for local cache validation.
OpenAICompatibleClient is a thin optional wrapper for vLLM/OpenAI-compatible
chat completions and imports the openai package only when that backend is used.
"""

from __future__ import annotations

import hashlib
import random
import sys
from typing import Sequence

from .primitive_specs import PrimitiveSpec


class BaseLLMClient:
    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def __init__(self, spec: PrimitiveSpec, seed: int = 0) -> None:
        self.spec = spec
        self.seed = seed
        self._counter = 0

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        prompt_text = "\n".join(message.get("content", "") for message in messages)
        digest = hashlib.sha256(
            f"{self.seed}:{self._counter}:{prompt_text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        self._counter += 1
        label = rng.choice(self.spec.labels)
        return f"Predicted primitive label: {label}."


class OpenAICompatibleClient(BaseLLMClient):
    """OpenAI-compatible chat client for vLLM or similar servers.

    For Qwen3 on vLLM, thinking/reasoning can be disabled per request
    by passing chat_template_kwargs.enable_thinking=False through extra_body.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        thinking_mode: str = "off",
        top_k: int | None = 20,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for --backend openai-compatible. "
                "Install it with `pip install openai`, or use --backend mock."
            ) from exc

        if thinking_mode not in {"off", "on", "auto"}:
            raise ValueError(
                f"Unsupported thinking_mode={thinking_mode!r}. "
                "Expected one of: off, on, auto."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.thinking_mode = thinking_mode
        self.top_k = top_k

    @property
    def enable_thinking(self) -> bool | None:
        if self.thinking_mode == "off":
            return False
        if self.thinking_mode == "on":
            return True
        return None

    def _build_extra_body(self) -> dict | None:
        extra_body = {}

        if self.thinking_mode == "off":
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif self.thinking_mode == "on":
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}

        if self.top_k is not None:
            extra_body["top_k"] = self.top_k

        return extra_body or None

    def generate(
        self,
        messages,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        extra_body = self._build_extra_body()

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                "OpenAI-compatible generation failed. "
                "Check --base-url, --api-key, --model, and whether the vLLM server is running."
            ) from exc

        return response.choices[0].message.content or ""


VLLMOpenAIClient = OpenAICompatibleClient
