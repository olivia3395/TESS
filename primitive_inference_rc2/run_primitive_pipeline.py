"""Unified primitive cache pipeline runner.

The runner orchestrates existing GT, sampled, flat gate, grouped gate, and audit
modules without shelling out. It records stage statuses and generated paths in a
manifest so mock or real-backend runs are reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .build_gate_cache import build_gate_cache
from .build_grouped_gate_cache import DEFAULT_PRIMITIVE_ORDER, run as run_grouped_gate
from .build_gt_cache import run as run_gt_cache
from .cache_io import (
    gate_cache_path,
    grouped_gate_cache_path,
    gt_primitive_cache_path,
    gt_primitive_metadata_path,
    report_path,
    sampled_inference_cache_path,
    write_json_object,
)
from .dataset_specs import VALID_SPLITS, get_dataset_spec
from .sampled_inference import DEFAULT_ALPHA, run as run_sampled_inference


DEFAULT_STAGES = ("gt", "sampled", "gate", "grouped", "audit")
VALID_STAGES = set(DEFAULT_STAGES)


def metadata_contains_primitive(root: str | Path, dataset: str, primitive: str) -> bool:
    metadata_path = gt_primitive_metadata_path(root, dataset)
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return primitive in metadata.get("thresholds", {})


def gt_outputs_exist(root: str | Path, dataset: str, primitive: str) -> bool:
    report = report_path(root, f"gt_{primitive}_{dataset}.json")
    return (
        all(gt_primitive_cache_path(root, dataset, split).exists() for split in VALID_SPLITS)
        and metadata_contains_primitive(root, dataset, primitive)
        and report.exists()
    )


def sampled_output_exists(
    root: str | Path,
    dataset: str,
    primitive: str,
    split: str,
) -> bool:
    return sampled_inference_cache_path(root, dataset, primitive, split).exists()


def gate_outputs_exist(
    root: str | Path,
    dataset: str,
    primitive: str,
    split: str,
) -> bool:
    return gate_cache_path(root, dataset, primitive, split).exists() and report_path(
        root, f"gate_{primitive}_{dataset}_{split}.json"
    ).exists()


def grouped_outputs_exist(root: str | Path, dataset: str, splits: tuple[str, ...]) -> bool:
    return all(grouped_gate_cache_path(root, dataset, split).exists() for split in splits)


def audit_output_exists(root: str | Path, dataset: str) -> bool:
    return report_path(root, f"primitive_cache_audit_{dataset}.json").exists()


def append_status(
    statuses: list[dict[str, Any]],
    stage: str,
    status: str,
    paths: list[str],
    detail: dict[str, Any] | None = None,
) -> None:
    entry: dict[str, Any] = {"stage": stage, "status": status, "paths": paths}
    if detail:
        entry["detail"] = detail
    statuses.append(entry)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    unknown_stages = [stage for stage in args.stages if stage not in VALID_STAGES]
    if unknown_stages:
        valid = ", ".join(DEFAULT_STAGES)
        raise ValueError(f"Unknown stages: {unknown_stages}. Valid stages: {valid}")
    if args.backend != "mock":
        raise ValueError("This runner stage only supports --backend mock.")

    dataset_spec = get_dataset_spec(args.dataset, args.root)
    splits = tuple(args.splits)
    primitives = tuple(args.primitives)
    stages = tuple(args.stages)
    start_time = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    statuses: list[dict[str, Any]] = []
    created_or_updated_paths: list[str] = []
    skipped: list[dict[str, Any]] = []

    if "gt" in stages:
        for primitive in primitives:
            expected_paths = [
                str(gt_primitive_cache_path(args.root, dataset_spec.name, split))
                for split in VALID_SPLITS
            ]
            expected_paths.extend(
                [
                    str(gt_primitive_metadata_path(args.root, dataset_spec.name)),
                    str(report_path(args.root, f"gt_{primitive}_{dataset_spec.name}.json")),
                ]
            )
            if not args.overwrite and gt_outputs_exist(args.root, dataset_spec.name, primitive):
                skipped.append({"stage": "gt", "primitive": primitive, "reason": "existing_outputs"})
                append_status(statuses, "gt", "skipped_existing", expected_paths, {"primitive": primitive})
                continue
            result = run_gt_cache(
                SimpleNamespace(root=args.root, dataset=dataset_spec.name, primitive=primitive)
            )
            paths = list(result["output_paths"].values()) + [
                result["metadata_path"],
                result["report_path"],
            ]
            created_or_updated_paths.extend(paths)
            append_status(statuses, "gt", "completed", paths, {"primitive": primitive})

    if "sampled" in stages:
        for primitive in primitives:
            for split in splits:
                output_path = sampled_inference_cache_path(
                    args.root, dataset_spec.name, primitive, split
                )
                if not args.overwrite and output_path.exists():
                    skipped.append(
                        {
                            "stage": "sampled",
                            "primitive": primitive,
                            "split": split,
                            "reason": "existing_output",
                        }
                    )
                    append_status(
                        statuses,
                        "sampled",
                        "skipped_existing",
                        [str(output_path)],
                        {"primitive": primitive, "split": split},
                    )
                    continue
                path = run_sampled_inference(
                    SimpleNamespace(
                        root=args.root,
                        dataset=dataset_spec.name,
                        split=split,
                        primitive=primitive,
                        backend=args.backend,
                        base_url="http://localhost:8000/v1",
                        api_key="EMPTY",
                        model=None,
                        prompt_source=args.prompt_source,
                        num_samples=args.num_samples,
                        limit=args.limit,
                        seed=args.seed,
                        alpha=DEFAULT_ALPHA,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                    )
                )
                created_or_updated_paths.append(str(path))
                append_status(
                    statuses,
                    "sampled",
                    "completed",
                    [str(path)],
                    {"primitive": primitive, "split": split},
                )

    if "gate" in stages:
        for primitive in primitives:
            for split in splits:
                output_path = gate_cache_path(args.root, dataset_spec.name, primitive, split)
                gate_report_path = report_path(
                    args.root, f"gate_{primitive}_{dataset_spec.name}_{split}.json"
                )
                if not args.overwrite and gate_outputs_exist(
                    args.root, dataset_spec.name, primitive, split
                ):
                    skipped.append(
                        {
                            "stage": "gate",
                            "primitive": primitive,
                            "split": split,
                            "reason": "existing_outputs",
                        }
                    )
                    append_status(
                        statuses,
                        "gate",
                        "skipped_existing",
                        [str(output_path), str(gate_report_path)],
                        {"primitive": primitive, "split": split},
                    )
                    continue
                result = build_gate_cache(
                    SimpleNamespace(
                        root=args.root,
                        dataset=dataset_spec.name,
                        split=split,
                        primitive=primitive,
                    )
                )
                paths = [result["output_path"], result["report_path"]]
                created_or_updated_paths.extend(paths)
                append_status(
                    statuses,
                    "gate",
                    "completed",
                    paths,
                    {"primitive": primitive, "split": split},
                )

    if "grouped" in stages or "audit" in stages:
        expected_grouped = [
            str(grouped_gate_cache_path(args.root, dataset_spec.name, split))
            for split in splits
        ]
        expected_audit = str(report_path(args.root, f"primitive_cache_audit_{dataset_spec.name}.json"))
        have_grouped = grouped_outputs_exist(args.root, dataset_spec.name, splits)
        have_audit = audit_output_exists(args.root, dataset_spec.name)
        should_run = args.overwrite or ("grouped" in stages and not have_grouped) or (
            "audit" in stages and not have_audit
        )
        if should_run:
            result = run_grouped_gate(
                SimpleNamespace(
                    root=args.root,
                    dataset=dataset_spec.name,
                    splits=list(splits),
                    primitives=list(primitives),
                )
            )
            paths = list(result["output_paths"].values()) + [result["audit_path"]]
            created_or_updated_paths.extend(paths)
            append_status(statuses, "grouped_audit", "completed", paths)
        else:
            skipped.append(
                {
                    "stage": "grouped_audit",
                    "reason": "existing_outputs",
                }
            )
            append_status(
                statuses,
                "grouped_audit",
                "skipped_existing",
                expected_grouped + [expected_audit],
            )

    end_time = datetime.now(timezone.utc)
    manifest = {
        "dataset": dataset_spec.name,
        "splits": list(splits),
        "primitives": list(primitives),
        "stages": list(stages),
        "backend": args.backend,
        "prompt_source": args.prompt_source,
        "num_samples": args.num_samples,
        "limit": args.limit,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": time.perf_counter() - start_perf,
        "created_or_updated_paths": created_or_updated_paths,
        "stage_statuses": statuses,
        "skipped_stages_or_files": skipped,
        "environment_note": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_executable": os.environ.get("_", ""),
        },
    }
    manifest_path = report_path(
        args.root, f"primitive_pipeline_manifest_{dataset_spec.name}.json"
    )
    write_json_object(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the primitive cache pipeline.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--primitives", nargs="+", default=list(DEFAULT_PRIMITIVE_ORDER))
    parser.add_argument("--splits", nargs="+", default=list(VALID_SPLITS))
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--prompt-source", choices=("auto", "fallback"), default="auto")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--thinking-mode", choices=["off", "on", "auto"], default="off")
    parser.add_argument("--top-k", type=int, default=20)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    result = run_pipeline(args)
    print(result["manifest_path"])


if __name__ == "__main__":
    main()
