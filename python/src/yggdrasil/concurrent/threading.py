"""
infinite_threadpool.py

A tiny helper around ThreadPoolExecutor for infinite / huge job streams with a
bounded in-flight window, supporting both completion-order and submission-order yields.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Generator, Iterable, Iterator, Optional, Set, Tuple

__all__ = ["Job", "JobThreadPoolExecutor"]


@dataclass(frozen=True, slots=True)
class Job:
    """Immutable bundle describing a unit of work."""

    @classmethod
    def make(cls, func: Callable[..., Any], *args: Any, **kwargs: Any) -> "Job":
        return cls(func=func, args=args, kwargs=kwargs)

    func: Callable[..., Any]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def run(self) -> Any:
        return self.func(*self.args, **self.kwargs)


class JobThreadPoolExecutor(ThreadPoolExecutor):
    """
    ThreadPoolExecutor helper for infinite / huge job streams with a bounded in-flight window.

    - ordered=False: yield futures as completed (completion order)
    - ordered=True: yield futures strictly in submission order (can block behind slow jobs)

    max_in_flight controls the maximum number of concurrently in-flight futures the helper
    will keep submitted at any time. (Alias of max_buffer; max_buffer kept for back-compat.)
    """

    def submit_job(self, job: Job) -> Future:
        return self.submit(job.run)

    def _try_submit_next(self, it: Iterator[Job]) -> Optional[Future]:
        try:
            job = next(it)
        except StopIteration:
            return None
        return self.submit_job(job)

    @staticmethod
    def _cancel_all(fs: Iterable[Future]) -> None:
        # Best effort: cancels queued/not-started futures.
        # Futures already running won't be stopped by cancel().
        for f in fs:
            if not f.done():
                f.cancel()

    def as_completed(
        self,
        job_generator: Iterable[Job],
        *,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
        cancel_on_exit: bool = False,
        shutdown_on_exit: bool = False,
        shutdown_wait: bool = False,
    ) -> Generator[Future, None, None]:
        """
        Consume a (possibly infinite) Job iterable and yield Futures.

        Cleanup behavior:
          - If the caller stops iteration early (break/return), or an exception happens,
            we cancel any remaining queued futures (best-effort).
          - Optionally, also shutdown the executor (Py 3.9+: cancel_futures=True).

        Args:
            job_generator: Source of Job objects (can be infinite).
            ordered:
                - False: yield futures as completed (completion order).
                - True: yield futures in submission order (may block behind slow jobs).
            max_in_flight: Max number of in-flight jobs at once (must be > 0).
            cancel_on_exit: Cancel remaining futures when generator exits early.
            shutdown_on_exit: Also call shutdown(cancel_futures=True) on exit.
            shutdown_wait: If shutdown_on_exit, whether to wait for worker threads.
        """
        if not max_in_flight:
            max_in_flight = self._max_workers * 2

        it = iter(job_generator)

        if ordered:
            inflight: Deque[Future] = deque()

            try:
                # Prime in-flight window
                for _ in range(max_in_flight):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    inflight.append(fut)

                # Drain + refill, strictly preserving submission order
                while inflight:
                    head = inflight[0]
                    if not head.done():
                        wait({head}, return_when=FIRST_COMPLETED)

                    yield inflight.popleft()

                    fut = self._try_submit_next(it)
                    if fut is not None:
                        inflight.append(fut)

            finally:
                if cancel_on_exit:
                    self._cancel_all(inflight)
                if shutdown_on_exit:
                    # Py 3.9+: cancels futures still in the queue (not running yet)
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)

        else:
            pending: Set[Future] = set()

            try:
                # Prime in-flight window
                for _ in range(max_in_flight):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    pending.add(fut)

                # Drain + refill, yielding as tasks complete
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)

                    for fut in done:
                        pending.discard(fut)
                        yield fut

                        new_fut = self._try_submit_next(it)
                        if new_fut is not None:
                            pending.add(new_fut)

            finally:
                if cancel_on_exit:
                    self._cancel_all(pending)
                if shutdown_on_exit:
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)
