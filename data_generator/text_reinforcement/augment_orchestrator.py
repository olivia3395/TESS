from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from ..utils.llm_client import LLMClient, LLMResult
from .async_executor import AsyncBatchExecutor, ExecFailure, ExecResult
from .prompt_builder import PromptTemplate, load_prompt_template

logger = logging.getLogger(__name__)


@dataclass
class PromptJob:
    sample_index: int
    sample_id: str
    split: str
    variant: str
    payload: Dict[str, Any]
    system_prompt: str
    user_prompt: str
    max_tokens: int


@dataclass
class AugmentMetrics:
    samples: int
    duration_s: float
    avg_latency_ms: float
    p95_latency_ms: float
    retries: int
    max_concurrency: int
    skipped_samples: int
    failed_samples: int


class AugmentOrchestrator:
    def __init__(
        self,
        *,
        dataset_root: Path,
        base_version: str,
        turn_id: int,
        prompt_config: Path,
        executor: AsyncBatchExecutor,
        llm_client: LLMClient,
        batch_size: int = 32,
        max_tokens: int = 320,
        resume_failures: bool = True,
        progress_log_interval: int = 1000,
    ) -> None:
        self.dataset_root = dataset_root
        self.base_version = base_version
        self.turn_id = turn_id
        self.executor = executor
        self.llm_client = llm_client
        self.batch_size = batch_size
        self.prompt_template: PromptTemplate = load_prompt_template(prompt_config)
        self.max_tokens = max_tokens or self.prompt_template.default_max_tokens
        self.resume_failures = resume_failures
        self.progress_log_interval = max(0, progress_log_interval)

    async def run(self, variants: Sequence[str]) -> Dict[str, AugmentMetrics]:
        metrics: Dict[str, AugmentMetrics] = {}
        for variant in variants:
            metrics[variant] = await self._run_variant(variant)
        return metrics

    async def _run_variant(self, variant: str) -> AugmentMetrics:
        start = time.perf_counter()
        variant_name = f"ver_turn{self.turn_id}_{variant}"
        out_dir = self.dataset_root / variant_name
        out_dir.mkdir(parents=True, exist_ok=True)
        failures_path = out_dir / "augmentation_failures.jsonl"

        failures: Dict[str, Dict[str, Any]] = {}
        if self.resume_failures:
            failures = self._load_failure_log(failures_path)

        latencies: List[float] = []
        retries = 0
        total_samples = 0
        skipped_samples = 0

        for split in ("train", "vali", "test"):
            source_path = self.dataset_root / self.base_version / f"{split}.json"
            if not source_path.exists():
                raise FileNotFoundError(f"Missing source split: {source_path}")
            with source_path.open("r", encoding="utf-8") as f:
                records = json.load(f)

            jobs_all = self._build_jobs(records, split=split, variant=variant, base_index=0)
            total_split = len(jobs_all)

            out_path = out_dir / f"{split}.json"
            existing_records = self._load_existing_records(out_path) if self.resume_failures else {}
            completed_ids = {sid for sid, record in existing_records.items() if self._is_record_complete(record)}
            skipped_samples += len(completed_ids)

            pending_failures = failures.get(split, {})
            for sample_id in list(pending_failures.keys()):
                if sample_id in completed_ids:
                    pending_failures.pop(sample_id, None)

            jobs_to_process: List[PromptJob] = []
            processed = len(completed_ids)
            for job in jobs_all:
                if job.sample_id in pending_failures or job.sample_id not in completed_ids:
                    jobs_to_process.append(job)

            split_success_map: Dict[str, Dict[str, Any]] = {}

            for chunk_start in range(0, len(jobs_to_process), self.batch_size):
                chunk = jobs_to_process[chunk_start : chunk_start + self.batch_size]
                if not chunk:
                    continue
                results, failures_batch = await self.executor.map(
                    chunk,
                    self._call_llm,
                    is_rate_limit_error=_is_rate_limit_error,
                    collect_failures=True,
                )
                for exec_result in results:
                    latencies.append(exec_result.latency_s)
                    retries += max(exec_result.attempts - 1, 0)
                    record = self._build_augmented_sample(exec_result.job, exec_result.result)
                    split_success_map[exec_result.job.sample_id] = record
                    if exec_result.job.split in failures:
                        failures[exec_result.job.split].pop(exec_result.job.sample_id, None)
                    processed += 1
                for failure in failures_batch:
                    failure_entry = self._make_failure_entry(failure)
                    failures.setdefault(failure.job.split, {})[failure.job.sample_id] = failure_entry
                if self.progress_log_interval and (
                    processed >= total_split or processed % self.progress_log_interval == 0
                ):
                    logger.info(
                        "Augment progress | variant=%s split=%s processed=%d/%d concurrency=%d",
                        variant,
                        split,
                        processed,
                        total_split,
                        self.executor.target_concurrency,
                    )

            final_records: List[Dict[str, Any]] = []
            for job in jobs_all:
                if job.sample_id in split_success_map:
                    final_records.append(split_success_map[job.sample_id])
                elif job.sample_id in existing_records:
                    final_records.append(existing_records[job.sample_id])
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final_records, f, ensure_ascii=False, indent=2)
            total_samples += len(final_records)

        failed_samples = sum(len(entries) for entries in failures.values())
        if failures or failures_path.exists():
            self._write_failure_log(failures_path, failures)

        duration = time.perf_counter() - start
        avg_latency = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
        p95_latency = _percentile(latencies, 95) * 1000 if latencies else 0.0
        return AugmentMetrics(
            samples=total_samples,
            duration_s=duration,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            retries=retries,
            max_concurrency=self.executor.target_concurrency,
            skipped_samples=skipped_samples,
            failed_samples=failed_samples,
        )

    def _load_existing_records(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.warning("AugmentOrchestrator: failed to parse existing records from %s", path)
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for item in data:
            sample_id = item.get("source_sample_id") or item.get("sample_id")
            if sample_id:
                records[sample_id] = item
        return records

    def _load_failure_log(self, path: Path) -> Dict[str, Dict[str, Any]]:
        failures: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return failures
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                split = entry.get("split")
                sample_id = entry.get("sample_id")
                if not split or not sample_id:
                    continue
                failures.setdefault(split, {})[sample_id] = entry
        return failures

    def _write_failure_log(self, path: Path, failures: Dict[str, Dict[str, Any]]) -> None:
        total = sum(len(entries) for entries in failures.values())
        if total == 0:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for split, entries in failures.items():
                for sample_id, entry in entries.items():
                    payload = dict(entry)
                    payload.setdefault("split", split)
                    payload.setdefault("sample_id", sample_id)
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _make_failure_entry(self, failure: ExecFailure[PromptJob]) -> Dict[str, Any]:
        return {
            "split": failure.job.split,
            "sample_id": failure.job.sample_id,
            "variant": failure.job.variant,
            "attempts": failure.attempts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": repr(failure.error)[:500],
        }

    def _is_record_complete(self, record: Dict[str, Any]) -> bool:
        news = record.get("news")
        return bool(news and str(news).strip())

    def _build_jobs(
        self,
        records: Sequence[Dict[str, Any]],
        *,
        split: str,
        variant: str,
        base_index: int = 0,
    ) -> List[PromptJob]:
        jobs: List[PromptJob] = []
        for offset, record in enumerate(records):
            prompt = self.prompt_template.render(record, variant)
            sample_index = base_index + offset
            job = PromptJob(
                sample_index=sample_index,
                sample_id=record.get("sample_id") or f"{split}-{sample_index:05d}",
                split=split,
                variant=variant,
                payload=record,
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
                max_tokens=self.max_tokens,
            )
            jobs.append(job)
        return jobs

    async def _call_llm(self, job: PromptJob) -> LLMResult:
        return await self.llm_client.achat(
            system_prompt=job.system_prompt,
            user_prompt=job.user_prompt,
            max_tokens=job.max_tokens,
        )

    def _build_augmented_sample(self, job: PromptJob, llm_result: LLMResult) -> Dict[str, Any]:
        enriched = dict(job.payload)
        enriched["news"] = llm_result.content
        enriched["augment_variant"] = job.variant
        enriched["turn_id"] = self.turn_id
        enriched["source_sample_id"] = job.sample_id
        return enriched


def _is_rate_limit_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status in (429, 503):
        return True
    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message or "retry later" in message


def _percentile(data: Iterable[float], percent: float) -> float:
    ordered = sorted(data)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * percent / 100.0
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    d0 = ordered[f] * (c - k)
    d1 = ordered[c] * (k - f)
    return d0 + d1
