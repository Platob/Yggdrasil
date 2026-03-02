"""
infinite_threadpool.py

A tiny helper around ThreadPoolExecutor for infinite / huge job streams with a
bounded in-flight window, supporting both completion-order and submission-order yields.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from threading import Thread
from typing import Any, Callable, Deque, Dict, Generator, Iterable, Iterator, Optional, Set, Tuple

__all__ = ["Job", "JobPoolExecutor"]


LOGGER = logging.getLogger(__name__)


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

    def fire_and_forget(self):
        t = Thread(
            target=self.func,
            args=self.args or (),
            kwargs=self.kwargs or {}
        )

        t.start()


class JobPoolExecutor(ThreadPoolExecutor):
    """
    ThreadPoolExecutor helper for infinite / huge job streams with a bounded in-flight window.

    - ordered=False: yield futures as completed (completion order)
    - ordered=True: yield futures strictly in submission order (can block behind slow jobs)

    max_in_flight controls the maximum number of concurrently in-flight futures the helper
    will keep submitted at any time. (Alias of max_buffer; max_buffer kept for back-compat.)
    """

    def __init__(
        self,
        max_workers: int = None,
        max_in_flight: int = None,
        job_name_prefix: str = '',
    ):
        super().__init__(
            max_workers=max_workers,
            thread_name_prefix=job_name_prefix,
        )

        self.job_name_prefix = job_name_prefix
        self.max_in_flight = max_in_flight if max_in_flight else max_workers * 2

    def submit_job(self, job: Job) -> Future:
        return self.submit(job.run)

    def _try_submit_next(self, it: Iterator[Job]) -> Optional[Future]:
        try:
            job = next(it)
        except StopIteration:
            return None
        return self.submit_job(job)

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        max_workers: Optional[int] = None,
    ):
        if isinstance(obj, cls):
            return cls

        return cls(max_workers=max_workers or os.cpu_count())

    @staticmethod
    def _cancel_all(fs: Iterable[Future]) -> None:
        # Best effort: cancels queued/not-started futures.
        # Futures already running won't be stopped by cancel().
        for f in fs:
            if not f.done():
                f.cancel()

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def _unpack(
        self,
        fut: Future,
        raise_error: bool,
        pending: list
    ) -> Tuple[Future, Any, Optional[BaseException]]:
        exc = fut.exception()
        if exc is not None:
            if raise_error:
                self._cancel_all(pending)
                raise exc

            LOGGER.exception(exc)

            return fut, None, exc
        return fut, fut.result(), None

    def as_completed(
        self,
        job_generator: Iterable[Job],
        *,
        ordered: bool = False,
        max_in_flight: Optional[int] = None,
        cancel_on_exit: bool = False,
        shutdown_on_exit: bool = False,
        shutdown_wait: bool = False,
        raise_error: bool = True,
    ) -> Generator[Tuple[Future, Any, Optional[BaseException]], None, None]:
        """
        Consume a (possibly infinite) Job iterable and yield (future, result, exc) triples.

        Each yield is a 3-tuple:
            future  – the completed Future object
            result  – future.result() on success, None on error
            exc     – the exception on failure, None on success

        If raise_error=True, exceptions are re-raised at the yield site instead of
        being returned as the third element.
        """
        max_in_flight = max_in_flight or self.max_in_flight
        it = iter(job_generator)

        if ordered:
            inflight: Deque[Future] = deque()

            try:
                for _ in range(max_in_flight):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    inflight.append(fut)

                while inflight:
                    head = inflight[0]
                    if not head.done():
                        wait({head}, return_when=FIRST_COMPLETED)

                    f, result, exc = self._unpack(
                        inflight.popleft(),
                        raise_error=raise_error,
                        pending=inflight,
                    )

                    if exc is None:
                        yield result

                    fut = self._try_submit_next(it)
                    if fut is not None:
                        inflight.append(fut)

            finally:
                if cancel_on_exit:
                    self._cancel_all(inflight)
                if shutdown_on_exit:
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)

        else:
            pending: Set[Future] = set()

            try:
                for _ in range(max_in_flight):
                    fut = self._try_submit_next(it)
                    if fut is None:
                        break
                    pending.add(fut)

                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)

                    for fut in done:
                        pending.discard(fut)

                        f, result, exc = self._unpack(
                            fut,
                            raise_error=raise_error,
                            pending=pending,
                        )

                        if exc is None:
                            yield result

                        new_fut = self._try_submit_next(it)
                        if new_fut is not None:
                            pending.add(new_fut)

            finally:
                if cancel_on_exit:
                    self._cancel_all(pending)
                if shutdown_on_exit:
                    self.shutdown(wait=shutdown_wait, cancel_futures=True)
