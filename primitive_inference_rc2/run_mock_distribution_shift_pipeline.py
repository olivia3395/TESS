"""Small legacy wrapper for the distribution_shift mock pipeline.

The unified run_primitive_pipeline module supersedes this convenience script,
but keeping it runnable preserves earlier smoke-test workflows.
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from .build_gate_cache import build_gate_cache
from .cache_io import report_path, write_json_object
from .dataset_specs import VALID_SPLITS
from .sampled_inference import DEFAULT_ALPHA, run as run_sampled_inference


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    if args.dataset != "fnspid":
        raise ValueError("Step 3C only supports --dataset fnspid")

    split_results: dict[str, dict[str, object]] = {}
    for split in VALID_SPLITS:
        sampled_args = SimpleNamespace(
            root=args.root,
            dataset=args.dataset,
            split=split,
            primitive="distribution_shift",
            backend="mock",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            model=None,
            prompt_source="auto",
            num_samples=args.num_samples,
            limit=args.limit,
            seed=args.seed,
            alpha=DEFAULT_ALPHA,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        sampled_path = run_sampled_inference(sampled_args)

        gate_args = SimpleNamespace(
            root=args.root,
            dataset=args.dataset,
            split=split,
            primitive="distribution_shift",
        )
        gate_result = build_gate_cache(gate_args)

        split_results[split] = {
            "sampled_path": str(sampled_path),
            "gate_path": gate_result["output_path"],
            "report_path": gate_result["report_path"],
            "gate_summary": gate_result["report"],
        }

    aggregate_report = {
        "dataset": args.dataset,
        "primitive": "distribution_shift",
        "backend": "mock",
        "splits": {
            split: split_result["gate_summary"] for split, split_result in split_results.items()
        },
    }
    aggregate_report_path = report_path(
        args.root, "gate_distribution_shift_fnspid_all_splits.json"
    )
    write_json_object(aggregate_report_path, aggregate_report)
    return {
        "splits": split_results,
        "aggregate_report_path": str(aggregate_report_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mock distribution_shift sampled inference and gate cache for all splits."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default="fnspid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    result = run_pipeline(args)
    for split, split_result in result["splits"].items():
        print(split)
        print(split_result["sampled_path"])
        print(split_result["gate_path"])
        print(split_result["report_path"])
    print(result["aggregate_report_path"])


if __name__ == "__main__":
    main()
