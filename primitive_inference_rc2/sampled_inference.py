"""Sampled primitive inference CLI.

Each output record stores repeated sampled labels, smoothed probabilities, a
self-consistency score, and a log-probability margin. With the mock backend this
simulates the confidence features later consumed by gate-cache builders without
requiring a live LLM server.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .cache_io import sampled_inference_cache_path, write_json
from .dataset_specs import get_dataset_spec
from .llm_clients import MockLLMClient, OpenAICompatibleClient
from .primitive_specs import PrimitiveSpec, get_primitive_spec, parse_label
from .prompting import build_prompt


DEFAULT_ALPHA = 1e-3


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset split file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Expected record {idx} in {path} to be an object")
    return data


def build_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You classify time-series news into one primitive label.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


def summarize_samples(
    parsed_labels: list[str | None],
    spec: PrimitiveSpec,
    alpha: float,
) -> dict[str, Any]:
    """Convert parsed sampled labels into confidence-style cache fields."""
    counts = {label: 0 for label in spec.labels}
    for label in parsed_labels:
        if label is not None:
            counts[label] += 1

    valid_count = sum(counts.values())
    num_samples = len(parsed_labels)
    denom = valid_count + alpha * len(spec.labels)
    if denom > 0:
        probs = {label: (counts[label] + alpha) / denom for label in spec.labels}
    else:
        uniform = 1.0 / len(spec.labels)
        probs = {label: uniform for label in spec.labels}

    if valid_count == 0:
        return {
            "pred_label": None,
            "sample_counts": counts,
            "sample_probs": probs,
            "margin": 0.0,
            "self_consistency": 0.0,
            "parse_rate": 0.0,
            "valid_count": 0,
            "num_samples": num_samples,
        }

    ranked_counts = sorted(
        counts.items(),
        key=lambda item: (-item[1], spec.label_to_id[item[0]]),
    )
    pred_label, max_count = ranked_counts[0]
    ranked_probs = sorted(probs.values(), reverse=True)
    margin = math.log(ranked_probs[0]) - math.log(ranked_probs[1])

    return {
        "pred_label": pred_label,
        "sample_counts": counts,
        "sample_probs": probs,
        "margin": margin,
        "self_consistency": max_count / valid_count,
        "parse_rate": valid_count / num_samples if num_samples else 0.0,
        "valid_count": valid_count,
        "num_samples": num_samples,
    }


def run(args: argparse.Namespace) -> Path:
    dataset_spec = get_dataset_spec(args.dataset, args.root)
    primitive_spec = get_primitive_spec(args.primitive)
    input_path = dataset_spec.split_path(args.split)
    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    model_name = args.model
    if args.backend == "mock":
        client = MockLLMClient(primitive_spec, seed=args.seed)
        model_name = model_name or "mock"
    elif args.backend == "openai-compatible":
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            thinking_mode=args.thinking_mode,
            top_k=args.top_k,
        )
    else:
        raise ValueError("--backend must be 'mock' or 'openai-compatible'")

    output_records: list[dict[str, Any]] = []

    for sample_id, record in enumerate(records):
        news = str(record.get("news", ""))
        prompt_result = build_prompt(
            root=Path(args.root),
            dataset=dataset_spec.name,
            primitive_spec=primitive_spec,
            record=record,
            prompt_source=args.prompt_source,
        )
        messages = build_messages(prompt_result.prompt)
        parsed_labels: list[str | None] = []
        for _ in range(args.num_samples):
            generated = client.generate(
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            parsed_labels.append(parse_label(generated, primitive_spec))

        summary = summarize_samples(parsed_labels, primitive_spec, args.alpha)
        output_records.append(
            {
                "dataset": dataset_spec.name,
                "split": args.split,
                "sample_id": sample_id,
                "primitive": primitive_spec.name,
                "schema": "legacy_v1",
                "raw_news": news,
                **summary,
                "margin_type": "sampled_self_consistency_margin",
                "backend": args.backend,
                "prompt_source": prompt_result.prompt_source,
                "prompt_template_path": prompt_result.prompt_template_path,
                "model": model_name,
            }
        )

    output_path = sampled_inference_cache_path(
        args.root,
        dataset_spec.name,
        primitive_spec.name,
        args.split,
    )
    write_json(output_path, output_records)
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampled primitive inference.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--primitive", required=True)
    parser.add_argument("--backend", choices=("mock", "openai-compatible"), default="mock")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default=None)
    parser.add_argument("--prompt-source", choices=("auto", "fallback"), default="auto")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--thinking-mode",
        choices=["off", "on", "auto"],
        default="off",
        help=(
            "Thinking mode for OpenAI-compatible backends. "
            "For Qwen3 on vLLM, 'off' sends chat_template_kwargs.enable_thinking=False."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Optional vLLM top_k sampling parameter for OpenAI-compatible backend.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    output_path = run(args)
    print(output_path)


if __name__ == "__main__":
    main()
