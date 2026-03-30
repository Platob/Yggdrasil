"""
AsyncJob — abstract awaitable handle for an async operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from yggdrasil.dataclasses.waiting import WaitingConfigArg

from ..job_result import JobResult

__all__ = ["AsyncJob"]

T = TypeVar("T")


class AsyncJob(ABC, Generic[T]):
    """Abstract handle for an async operation that can be awaited.

    Implementors must provide :meth:`wait`, :attr:`is_done`, and :meth:`result`.
    """

    @abstractmethod
    def wait(
        self,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
    ) -> Optional[T]:
        """Block until the job completes and return its result.

        Args:
            wait:        ``None`` / ``True``  → block indefinitely.
                         ``False``            → non-blocking poll; returns ``None``
                         if not yet finished.
                         ``WaitingConfig`` or a numeric number of seconds → timed wait.
            raise_error: Re-raise the job's exception when ``True``; return
                         ``None`` when ``False``.

        Returns:
            The job's return value, or ``None`` on failure / timeout / non-blocking
            miss (when ``raise_error=False``).

        Raises:
            TimeoutError: If a timed wait elapses before the job finishes.
            BaseException: Any exception raised by the job (when ``raise_error=True``).
        """

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """``True`` when the job has finished (success or failure)."""

    @abstractmethod
    def result(self) -> Optional[JobResult[T]]:
        """Return the :class:`~yggdrasil.concurrent.job_result.JobResult` if done, else ``None``."""

