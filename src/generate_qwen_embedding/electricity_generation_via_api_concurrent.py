#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Electricity load generation via remote API with concurrent requests.
Keeps the prompt/extraction/retry logic from electricity_generation_via_api.py.
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import requests
except ImportError as exc:  # pragma: no cover - runtime guard
        raise SystemExit("The `requests` package is required. Install with `pip install requests`.") from exc


STEP_LINE_PATTERN = re.compile(
    r"(?:\d+\s*:)?\s*Step\s*(\d*)\s*:.*?(?:[sS]trength)[:\s]*([A-Za-z]+).*?(?:[tT]rend)[:\s]*([A-Za-z]+)",
    re.IGNORECASE,
)

GLOBAL_TREND_PATTERN = re.compile(r"Global\s*trend\s*[:\s]*\s*(Rising|Falling)", re.IGNORECASE)

STRENGTH_ENCODING = {
    "emerging": 1,
    "moderate": 2,
    "significant": 3,
    "prominent": 4,
    "dominant": 5,
}

TREND_ENCODING = {
    "rising": 1,
    "falling": -1,
}


def load_split(data_path: str, split: str) -> list:
    path = Path(data_path) / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_structured_trend_fields(news_text: str, num_steps: int) -> Tuple[List[int], List[int], int]:
    step1_pattern = re.compile(r"(Step\s*1[:\s]|Step1[:\s])", re.IGNORECASE)
    step1_match = step1_pattern.search(news_text)
    if step1_match:
        news_text = news_text[step1_match.start() :]

    strengths = [0] * num_steps
    trends = [0] * num_steps

    matches = []
    for match in STEP_LINE_PATTERN.finditer(news_text):
        step_num_str = match.group(1).strip()
        raw_strength = match.group(2).strip().lower()
        raw_trend = match.group(3).strip().lower()
        step_idx = int(step_num_str) - 1 if step_num_str else 0
        matches.append((step_idx, raw_strength, raw_trend))

    if not matches:
        loose_pattern = re.compile(
            r"(?:\d+\s*[:\.]\s*)?[Aa]nalysis:.*?[sS]trength[:\s]*([A-Za-z]+).*?[tT]rend[:\s]*([A-Za-z]+)",
            re.IGNORECASE,
        )
        for i, match in enumerate(loose_pattern.finditer(news_text)):
            raw_strength = match.group(1).strip().lower()
            raw_trend = match.group(2).strip().lower()
            step_idx = i if i < num_steps else num_steps - 1
            matches.append((step_idx, raw_strength, raw_trend))

    if not matches:
        lines = news_text.split("\n")
        step_count = 0
        i = 0
        while i < len(lines) and step_count < num_steps:
            line = lines[i].strip()
            if ("analysis" in line.lower() or re.search(r"Step\d*:", line, re.IGNORECASE)) and ":" in line:
                strength = None
                trend = None
                for j in range(1, 4):
                    if i + j < len(lines):
                        next_line = lines[i + j].strip()
                        if "strength:" in next_line.lower() and not strength:
                            strength_match = re.search(r"[sS]trength[:\s]*([A-Za-z]+)", next_line, re.IGNORECASE)
                            if strength_match:
                                strength = strength_match.group(1).strip().lower()
                        elif "trend:" in next_line.lower() and not trend:
                            trend_match = re.search(r"[tT]rend[:\s]*([A-Za-z]+)", next_line, re.IGNORECASE)
                            if trend_match:
                                trend = trend_match.group(1).strip().lower()
                if strength and trend:
                    step_idx = step_count
                    matches.append((step_idx, strength, trend))
                    step_count += 1
            i += 1

    for step_idx, raw_strength, raw_trend in matches:
        if 0 <= step_idx < num_steps:
            strengths[step_idx] = STRENGTH_ENCODING.get(raw_strength, 0)
            trends[step_idx] = TREND_ENCODING.get(raw_trend, -1)

    global_match = GLOBAL_TREND_PATTERN.search(news_text)
    if global_match:
        global_trend = TREND_ENCODING.get(global_match.group(1).strip().lower(), -1)
    else:
        global_trend = 0

    return strengths, trends, global_trend


def is_extraction_successful(strengths: List[int], trends: List[int]) -> bool:
    return all(strength != 0 and trend != 0 for strength, trend in zip(strengths, trends))


def clean_generated_text(text: str) -> str:
    if "<think>" in text:
        if "</think>" in text:
            after_think_end = text.split("</think>", 1)
            text = after_think_end[1] if len(after_think_end) > 1 else after_think_end[0]
        else:
            text = text.split("<think>", 1)[0]

    step1_match = re.search(r"(?:\d+\s*:)?\s*Step\s*1\s*:", text, re.IGNORECASE)
    if step1_match:
        text = text[step1_match.start() :]
    return text.strip()


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[str, Dict[str, Any]]:
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"API error {response.status_code}: {response.text[:200]}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("API response missing choices field")

    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    if not content:
        raise RuntimeError("API response missing content field")

    return str(content), data.get("usage") or {}


def generate_text_with_retry(
    prompt: str,
    args: argparse.Namespace,
    num_steps: int,
    sample_idx: int,
) -> Tuple[str, List[int], List[int], int, Dict[str, float]]:
    retry = 0
    total_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    last_text = ""

    while retry < args.max_retries:
        try:
            raw_text, usage = call_chat_api(
                args.api_base,
                args.api_key,
                args.model,
                prompt,
                args.max_tokens,
                args.temperature,
                args.request_timeout,
            )
            total_usage["prompt_tokens"] += float(usage.get("prompt_tokens", 0))
            total_usage["completion_tokens"] += float(usage.get("completion_tokens", 0))
            total_usage["total_tokens"] += float(usage.get("total_tokens", 0))
        except Exception as exc:  # pragma: no cover - network failures
            retry += 1
            sleep = min(args.retry_backoff ** retry, args.max_retry_sleep)
            print(f"[Sample {sample_idx}] API call failed (attempt {retry}/{args.max_retries}): {exc}")
            time.sleep(sleep)
            continue

        cleaned = clean_generated_text(raw_text)
        strengths, trends, global_trend = extract_structured_trend_fields(cleaned, num_steps=num_steps)

        if is_extraction_successful(strengths, trends):
            return cleaned, strengths, trends, global_trend, total_usage

        last_text = cleaned
        retry += 1
        sleep = min(args.retry_backoff ** retry, args.max_retry_sleep)
        print(f"[Sample {sample_idx}] Extraction failed, retrying ({retry}/{args.max_retries})...")
        time.sleep(sleep)

    print(f"[Sample {sample_idx}] Failed after {args.max_retries} retries, keeping last attempt")
    if not last_text:
        last_text = ""
    strengths, trends, global_trend = extract_structured_trend_fields(last_text, num_steps=num_steps)
    return last_text, strengths, trends, global_trend, total_usage


def build_prompt(item: dict, template: str, few_shots: str) -> Tuple[str, float]:
    historical_data = item.get("historical_data", "")
    news = item.get("news") or item.get("prompt", "")
    values = [float(x.strip()) for x in historical_data.split(",") if x.strip()]
    mean_value = sum(values) / len(values) if values else 0.0
    prompt = template.format(
        historical_data=historical_data,
        news=news,
        few_shots=few_shots,
        mean_value=mean_value,
    )
    return prompt, mean_value


def load_checkpoint(checkpoint_file: Path, total_len: int):
    results: List[Any] = [None] * total_len
    usage_totals = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    if not checkpoint_file.exists():
        return results, usage_totals

    with checkpoint_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = payload.get("idx")
            item = payload.get("item")
            usage = payload.get("usage", {})
            if idx is None or not (0 <= idx < total_len):
                continue
            results[idx] = item
            usage_totals["prompt_tokens"] += float(usage.get("prompt_tokens", 0))
            usage_totals["completion_tokens"] += float(usage.get("completion_tokens", 0))
            usage_totals["total_tokens"] += float(usage.get("total_tokens", 0))
    return results, usage_totals


def append_checkpoint(checkpoint_file: Path, idx: int, item: dict, usage: Dict[str, float], lock: asyncio.Lock):
    async def _write():
        async with lock:
            with checkpoint_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"idx": idx, "item": item, "usage": usage}, ensure_ascii=False))
                f.write("\n")
    return _write()


async def process_split_async(args: argparse.Namespace, split_name: str):
    data = load_split(args.data_path, split_name)
    if args.test_limit is not None:
        data = data[: args.test_limit]

    prompts: List[str] = []
    for item in data:
        prompt, _ = build_prompt(item, args.prompt_template, args.few_shots)
        prompts.append(prompt)
    total_len = len(data)

    checkpoint_dir = Path(args.checkpoint_dir or args.output_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_dir / f"{split_name}_checkpoint.jsonl"
    results, aggregate_usage = load_checkpoint(checkpoint_file, total_len)

    pending_indices = [i for i, item in enumerate(results) if item is None]
    if not pending_indices:
        print(f"{split_name}: all samples already in checkpoint, writing final output...")

    semaphore = asyncio.Semaphore(args.concurrency)
    results_list: List[Dict[str, Any]] = results  # alias for clarity
    ckpt_lock = asyncio.Lock()

    async def handle(idx: int, item: dict, prompt: str):
        loop = asyncio.get_running_loop()
        async with semaphore:
            text, strengths, trends, global_trend, usage = await loop.run_in_executor(
                None, generate_text_with_retry, prompt, args, args.num_steps, idx
            )
        new_item = deepcopy(item)
        new_item["news"] = text
        new_item["step_strengths"] = strengths
        new_item["step_trends"] = trends
        new_item["global_trend"] = global_trend
        results_list[idx] = new_item
        aggregate_usage["prompt_tokens"] += usage["prompt_tokens"]
        aggregate_usage["completion_tokens"] += usage["completion_tokens"]
        aggregate_usage["total_tokens"] += usage["total_tokens"]
        await append_checkpoint(checkpoint_file, idx, new_item, usage, ckpt_lock)
        return idx

    tasks = [
        asyncio.create_task(handle(idx, data[idx], prompts[idx])) for idx in pending_indices
    ]
    if tasks:
        await asyncio.gather(*tasks)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{split_name}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)

    print(
        f"Saved generated {split_name} data to {output_file}. "
        f"Usage prompt/completion/total: "
        f"{aggregate_usage['prompt_tokens']:.0f}/"
        f"{aggregate_usage['completion_tokens']:.0f}/"
        f"{aggregate_usage['total_tokens']:.0f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate electricity dataset trends via remote API with concurrent requests"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_camf",
        help="Path to Electricity dataset split folder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="/home/llh/MMTSF/MMTSF_LIB/dataset/Electricity/ver_generated_withfewshots_api",
        help="Output path for generated texts",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="https://api2.aigcbest.top/v1",
        help="API base url (OpenAI-compatible), e.g., https://api2.aigcbest.top/v1",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default='sk-qb5mKgNnjOuAPJAiQWfp9svOXUe6GIps0hWj0KZFZRmASDXQ',
        help="API key (defaults to env AIGCBEST_API_KEY / OPENAI_API_KEY)",
    )
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name to request")
    parser.add_argument("--splits", nargs="+", default=["train", "vali", "test"], help="Dataset splits to process")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens for completion")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-steps", type=int, default=48, help="Number of step outputs expected from the model")
    parser.add_argument("--max-retries", type=int, default=5, help="Maximum retries per sample")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Exponential backoff base for retries")
    parser.add_argument("--max-retry-sleep", type=float, default=15.0, help="Upper bound on retry sleep seconds")
    parser.add_argument("--request-timeout", type=int, default=120, help="HTTP request timeout in seconds")
    parser.add_argument("--test-limit", type=int, default=None, help="Limit number of samples for quick tests")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to store per-split checkpoint jsonl files (default: output-path)",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "You are a professional electricity demand analyst. Your task is to analyze electricity load movements and related contextual news to determine trend direction and strength for each of the 48 time steps, then conclude the global trend.\n"
            "\n"
            "Instructions:\n"
            "1. The field historical_data contains exactly 48 comma-separated past electricity load values for a single region or market (e.g., NSW). Index these values in order as Step1 (the earliest in this window) through Step48 (the latest).\n"
            "2. For each of the 48 steps (Step1–Step48), you will analyze that step's load value from the sequence {{{historical_data}}} compared to the historical mean value {mean_value}.\n"
            "3. Determine the trend direction strictly by the numeric rule: Rising if the step load value > {mean_value}, Falling if the step load value < {mean_value}.\n"
            "4. Determine the trend strength based on the magnitude of the difference between the step load value and {mean_value}, choosing ONLY from: Emerging, Moderate, Significant, Prominent, Dominant.\n"
            "5. Determine the Global trend by comparing the predicted future mean electricity load with the historical mean value {mean_value}. If the predicted future mean is greater than {mean_value}, the Global trend is Rising; if it is lower, the Global trend is Falling.\n"
            "6. Use the information in News (e.g., weather, economic activity, policy changes, and other meta-domain factors) only to support your brief analyses in natural language. However, the labels trend and strength for each step must always follow the numeric rules above and must not be changed based on your own assumptions.\n"
            "\n"
            "You MUST output in the following EXACT format with no extra text:\n"
            "Step1: Analysis:<brief analysis for step 1>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "Step2: Analysis:<brief analysis for step 2>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "...\n"
            "Step48: Analysis:<brief analysis for step 48>, strength:<one of Emerging/Moderate/Significant/Prominent/Dominant>, trend:<Rising or Falling>\n"
            "Global trend:<Rising or Falling>\n"
            "\n"
            "\n"
            "Historical data: {{{historical_data}}}\n"
            "Mean value: {mean_value}\n"
            "News: {news}\n"
            "Few_shots: {few_shots}\n"
            "In the few_shots examples, the field trends uses -1 to represent Falling and 1 to represent Rising, and the field step_strengths uses integers 1-5 to represent, from weakest to strongest, the five strength labels: Emerging (1), Moderate (2), Significant (3), Prominent (4), Dominant (5).\n"
            "Provide your analysis in the exact format specified above:"
        ),
        help="Prompt template for electricity demand generation over 48 steps",
    )
    parser.add_argument(
        "--few-shots",
        type=str,
        default="src/generate_qwen_embedding/few_shot_samples_elec.txt",
        help="Path to few-shot examples file",
    )
    return parser.parse_args()


async def async_main():
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Please provide --api-key or set AIGCBEST_API_KEY/OPENAI_API_KEY.")

    with open(args.few_shots, "r", encoding="utf-8") as f:
        args.few_shots = f.read()

    for split_name in args.splits:
        print(f"Processing {split_name} split via API with concurrency={args.concurrency}...")
        await process_split_async(args, split_name)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
