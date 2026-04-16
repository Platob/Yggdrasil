"""
Bounded thread-pool for large / infinite job streams.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from typing import Any, Deque, Iterable, Iterator, Optional, Set

from .job import Job
from .job_result import JobResult

__all__ = ["JobPoolExecutor"]

LOGGER = logging.getLogger(__name__)


class JobPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor helper for large / infinite :class:`Job` streams.

    Keeps at most *max_in_flight* futures submitted at any time so the caller
    is never overwhelmed when the job source is unbounded.

    Args:
        max_workers:     Thread-pool size (``None`` → ``os.cpu_count()``).
        max_in_flight:   Max in-flight futures (``None`` → ``max_workers * 2``).
        job_name_prefix: Thread-name prefix forwarded to ``ThreadPoolExecutor``.

    Yields (via :meth:`as_completed`):
        :class:`JobResult` in completion order (``ordered=False``) or
        submission order (``ordered=True``).
    """

    def __init__(
        self,
        max_workers: int | None = None,
        max_in_flight: int | None = None,
        job_name_prefix: str = "",
    ) -> None:
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=job_name_prefix,
        )
        self.job_name_prefix = job_name_prefix
        self.max_in_flight: int = max_in_flight if max_in_flight else (self._max_workers * 2)

    # ------------------------------------------------------------------
    # Submission helpers
    # ------------------------------------------------------------------

    def submit_job(self, job: Job) -> Future:
        """Submit a single :class:`Job` and return its :class:`Future`."""
        return self.submit(job.run)

    def _try_submit_next(self, it: Iterator[Job]) -> Optional[Future]:
        try:
            job = next(it)
        except StopIteration:
            return None
        return self.submit_job(job)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        max_workers: int | None = None,
    ) -> "JobPoolExecutor":
        if isinstance(obj, cls):
            return cls  # type: ignore[return-value]
        return cls(max_workers=max_workers or os.cpu_count())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_workers(self) -> int:
        return self._max_workers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cancel_all(fs: Iterable[Future]) -> None:
        """Best-effort cancellation of pending (not-yet-started) futures."""
        for f in fs:
            if not f.done():
                f.cancel()

    def _unpack(
        self,
        fut: Future,
        raise_error: bool,
        pending: Iterable[Future],
    ) -> JobResult:
        exc = fut.exception()
        if exc is not None:
            if raise_error:
                self._cancel_all(pending)
                raise exc
            LOGGER.exception(exc)
            return JobResult(None, exc)
        return JobResult(fut.result(), None)

    # ------------------------------------------------------------------
    # Public streaming API
    # ------------------------------------------------------------------

    def as_completed(
        self,
        jobs: Iterable[Job],
        *,
        ordered: bool = False,
        max_in_flight: int | None = None,
        cancel_on_exit: bool = False,
        shutdown_on_exit: bool = False,
        shutdown_wait: bool = False,
        raise_error: bool = True,
    ) -> Iterator[JobResult]:
        """Consume a (possibly infinite) :class:`Job` iterable and yield :class:`JobResult` objects.

        Args:
            jobs:            Source of :class:`Job` objects (may be infinite).
            ordered:         ``False`` (default) → completion order.
                             ``True``            → strict submission order.
            max_in_flight:   Override the instance-level window for this call.
            cancel_on_exit:  Cancel pending futures on generator close / error.
            shutdown_on_exit: Shut down the pool on generator close / error.
            shutdown_wait:   Wait for running threads when shutting down.
            raise_error:     Re-raise job exceptions at the yield site.

        Yields:
            :class:`JobResult` — one per submitted job.
        """
        window = max_in_flight or self.max_in_flight
        it = iter(jobs)

        if ordered:
            inflight: Deque[Future] = deque()
            try:
                for _ in range(window):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    inflight.append(fut)

                while inflight:
                    head = inflight[0]
                    if not head.done():
                        wait({head}, return_when=FIRST_COMPLETED)

                    job_result = self._unpack(
                        inflight.popleft(),
                        raise_error=raise_error,
                        pending=inflight,
                    )
                    if job_result.ok:
                        yield job_result
                    elif raise_error:
                        raise job_result.exception  # type: ignore[misc]

                    nxt = self._try_submit_next(it)
                    if nxt is not None:
                        inflight.append(nxt)

            finally:
                if cancel_on_exit:
                    self._cancel_all(inflight)
                if shutdown_on_exit:
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)

        else:
            pending: Set[Future] = set()
            try:
                for _ in range(window):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    pending.add(fut)

                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for fut in done:
                        pending.discard(fut)

                        job_result = self._unpack(
                            fut,
                            raise_error=raise_error,
                            pending=pending,
                        )
                        if job_result.ok:
                            yield job_result
                        elif raise_error:
                            raise job_result.exception  # type: ignore[misc]

                        nxt = self._try_submit_next(it)
                        if nxt is not None:
                            pending.add(nxt)

            finally:
                if cancel_on_exit:
                    self._cancel_all(pending)
                if shutdown_on_exit:
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)

