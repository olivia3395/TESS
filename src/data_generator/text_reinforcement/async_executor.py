from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Awaitable, Callable, Deque, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ExecResult(Generic[T, R]):
    job: T
    result: R
    latency_s: float
    attempts: int


@dataclass
class ExecFailure(Generic[T]):
    job: T
    error: BaseException
    attempts: int


class AsyncBatchExecutor:
    """Adaptive concurrency executor for LLM calls."""

    def __init__(
        self,
        *,
        initial_concurrency: int = 120,
        max_concurrency: int = 500,
        min_concurrency: int = 64,
        latency_increase_threshold: float = 1.0,
        latency_decrease_threshold: float = 2.0,
        perf_window: int = 50,
        max_retries: int = 3,
        retry_backoff: float = 0.75,
        retry_multiplier: float = 1.7,
    ) -> None:
        if min_concurrency <= 0 or max_concurrency < min_concurrency:
            raise ValueError("Invalid concurrency bounds")
        self._target_concurrency = max(min(initial_concurrency, max_concurrency), min_concurrency)
        self._max_concurrency = max_concurrency
        self._min_concurrency = min_concurrency
        self._latency_increase_threshold = latency_increase_threshold
        self._latency_decrease_threshold = latency_decrease_threshold
        self._perf_window = perf_window
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._retry_multiplier = retry_multiplier
        self._latencies: Deque[float] = deque(maxlen=perf_window)
        self._rate_limit_events = 0
        self._max_observed_concurrency = self._target_concurrency

    @property
    def target_concurrency(self) -> int:
        return self._target_concurrency

    @property
    def rate_limit_events(self) -> int:
        return self._rate_limit_events

    @property
    def max_observed_concurrency(self) -> int:
        return self._max_observed_concurrency

    async def map(
        self,
        jobs: Sequence[T],
        worker: Callable[[T], Awaitable[R]],
        *,
        is_rate_limit_error: Optional[Callable[[BaseException], bool]] = None,
        collect_failures: bool = False,
    ) -> Union[List[ExecResult[T, R]], Tuple[List[ExecResult[T, R]], List[ExecFailure[T]]]]:
        if not jobs:
            return []
        concurrency = min(self._target_concurrency, len(jobs))
        semaphore = asyncio.Semaphore(concurrency)
        results: List[ExecResult[T, R]] = []
        failures: List[ExecFailure[T]] = []
        has_rate_limit = False

        async def run_job(job: T) -> Optional[ExecResult[T, R]]:
            nonlocal has_rate_limit
            attempt = 0
            backoff = self._retry_backoff
            while True:
                await semaphore.acquire()
                start = time.perf_counter()
                try:
                    attempt += 1
                    value = await worker(job)
                    latency = time.perf_counter() - start
                    self._latencies.append(latency)
                    return ExecResult(job=job, result=value, latency_s=latency, attempts=attempt)
                except Exception as exc:  # noqa: BLE001
                    is_rate_limited = bool(is_rate_limit_error and is_rate_limit_error(exc))
                    if is_rate_limited:
                        has_rate_limit = True
                        self._rate_limit_events += 1
                        if attempt <= self._max_retries:
                            await asyncio.sleep(backoff)
                            backoff *= self._retry_multiplier
                            continue
                    if collect_failures:
                        failures.append(ExecFailure(job=job, error=exc, attempts=attempt))
                        return None
                    raise
                finally:
                    semaphore.release()

        tasks = [asyncio.create_task(run_job(job)) for job in jobs]
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                if result is not None:
                    results.append(result)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(Exception):
                        await task

        self._adjust_concurrency(has_rate_limit)
        if collect_failures:
            return results, failures
        return results

    def _adjust_concurrency(self, had_rate_limit: bool) -> None:
        if not self._latencies:
            return
        p95_latency = self._percentile(self._latencies, 95)
        if had_rate_limit or p95_latency > self._latency_decrease_threshold:
            new_target = max(int(self._target_concurrency * 0.5), self._min_concurrency)
            if new_target < self._target_concurrency:
                logger.info(
                    "AsyncBatchExecutor: reducing concurrency from %s to %s (p95=%.3fs, rate_limit=%s)",
                    self._target_concurrency,
                    new_target,
                    p95_latency,
                    had_rate_limit,
                )
                self._target_concurrency = new_target
        elif p95_latency < self._latency_increase_threshold:
            new_target = min(int(self._target_concurrency * 1.2), self._max_concurrency)
            if new_target > self._target_concurrency:
                logger.info(
                    "AsyncBatchExecutor: increasing concurrency from %s to %s (p95=%.3fs)",
                    self._target_concurrency,
                    new_target,
                    p95_latency,
                )
                self._target_concurrency = new_target
                self._max_observed_concurrency = max(self._max_observed_concurrency, self._target_concurrency)

    @staticmethod
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
