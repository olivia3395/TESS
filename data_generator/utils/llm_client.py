"""utils/llm_client.py
============================================================
统一 OpenAI‑compatible SDK 封装，供 LeadAgent / SubAgent 调用。
核心改动：简化调用，只需传入 **System Prompt** 与 **User Prompt**。

功能要点
--------
- ✅ `achat()` / `chat()`：各 1 行代码即可发起请求
- ✅ 同步 / 异步接口
- ✅ 指数退避自动重试
- ✅ 可选流式响应
- ✅ Token 估算与成本预估
- ✅ 全局并发控制（asyncio.Semaphore）
- ✅ 可自定义 `base_url` 对接 FastChat / DashScope / Ollama 等网关


使用示例
--------
```python
from llm_client import LLMClient
import asyncio

client = LLMClient(model="gpt-3.5-turbo-0125", api_key="sk‑...")

async def main():
    res = await client.achat(
        user_prompt="Summarise quick sort in 50 words.",
        system_prompt="You are a concise algorithm tutor.",
        max_tokens=64,
    )
    print(res.content, res.cost_estimate)

asyncio.run(main())
```
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
try:
    from openai import AsyncOpenAI  # type: ignore
    from openai import AsyncAzureOpenAI  # type: ignore
except ImportError:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore
    AsyncAzureOpenAI = None  # type: ignore
try:  # pragma: no cover
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore

# ---------------------------------------------------------------------------
# 配置 & 常量
# ---------------------------------------------------------------------------

_MODEL_PRICING = {
    "gpt-3.5-turbo-0125": 0.0005,
    "gpt-4o-mini": 0.005,
    "gpt-4o": 0.01,
}

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost: float

# ---------------------------------------------------------------------------
# 客户端实现
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI ChatCompletion 封装（兼容 OpenAI‑compatible 网关）。"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        *,
        max_retries: int = 1,
        backoff_base: float = 1.5,
        semaphore: Optional[asyncio.Semaphore] = None,
        default_system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        stream: bool = False,
        metrics_callback: Optional[Callable[[Dict[str, float]], None]] = None,
        **default_params: Any,
    ) -> None:
        api_key = api_key or "9tH07lBQBzlFhT96StmDA4Zd8fByxPJ6QT9UqHiLC9G3o2c9IJuPJQQJ99BIACHYHv6XJ3w3AAABACOG2jiA"
        if not api_key:
            raise ValueError("[LLMClient] API key not provided and environment variable not set.")

        azure_endpoint = base_url or "https://liyu-intelligentagent-gpt4o.openai.azure.com/"
        azure_api_version = "2025-01-01-preview"

        self.model = model
        self.sem = semaphore
        self.default_params = default_params
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.default_system_prompt = default_system_prompt
        self.stream = stream
        self.metrics_callback = metrics_callback

        if azure_endpoint:
            if AsyncAzureOpenAI is None:
                raise ImportError("AsyncAzureOpenAI not available - install openai>=1.0.0")
            self.client = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint.rstrip('/'),
            )
        else:
            if AsyncOpenAI is None:
                raise ImportError("AsyncOpenAI not available - install openai>=1.0.0 or provide Azure endpoint")
            base_url_env = os.getenv("OPENAI_BASE_URL")
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url_env)
        # 可选 tiktoken 编码器
        self._enc = None
        if tiktoken is not None:
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except Exception:
                pass

    # -------------------------- 对外接口 --------------------------- #

    async def achat(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        **params: Any,
    ) -> LLMResult:
        """**推荐的异步接口**。只需给出 system + user 文本。"""
        messages = [
            {"role": "system", "content": system_prompt or self.default_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if self.sem:
            async with self.sem:
                return await self._run(messages, max_tokens, params)
        return await self._run(messages, max_tokens, params)

    def chat(
        self,
        *,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        **params: Any,
    ) -> LLMResult:  # pragma: no cover sync helper
        return asyncio.run(self.achat(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            **params,
        ))

    # -------------------------- 核心执行 --------------------------- #

    async def _run(self, messages, max_tokens, override_params: Dict[str, Any]):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": self.stream,
            **self.default_params,
            **override_params,
        }
        attempt = 0
        while True:
            try:
                if self.stream:
                    parts = []
                    async for chunk in await self.client.chat.completions.create(**payload):
                        parts.append(chunk)
                    resp = self._merge_stream(parts)
                else:
                    resp = await self.client.chat.completions.create(**payload)
                return self._build_result(resp)
            except:
                if attempt >= self.max_retries:
                    raise
                await asyncio.sleep(self.backoff_base ** attempt)
                attempt += 1

    # -------------------------- 私有辅助 --------------------------- #

    def _merge_stream(self, chunks):
        content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        return {
            "choices": [{"message": {"content": content}}],
            "usage": chunks[-1].get("usage") if chunks else None,
        }

    def _build_result(self, resp):
        content = resp.choices[0].message.content.strip()
        usage = getattr(resp, "usage", None)
        if usage is not None:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
        else:
            prompt_tokens = self._approx_tokens(content) // 2
            completion_tokens = self._approx_tokens(content)

        cost = self._calc_cost(prompt_tokens + completion_tokens)

        result = LLMResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost
        )
        if callable(getattr(self, 'metrics_callback', None)):
            try:
                self.metrics_callback({
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': prompt_tokens + completion_tokens,
                    'cost': cost,
                })
            except Exception:  # noqa: BLE001
                pass
        return result
# Baseline Metric COmp

    def _approx_tokens(self, text: str) -> int:
        if self._enc:
            return len(self._enc.encode(text))
        return len(text.split())

    def _calc_cost(self, tokens: int) -> float:
        price = _MODEL_PRICING.get(self.model, 0.002)
        return tokens / 1000 * price

__all__ = ["LLMClient", "LLMResult"]
