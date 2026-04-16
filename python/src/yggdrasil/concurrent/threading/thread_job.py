"""
ThreadJob — daemon-thread-backed implementation of AsyncJob.
"""

from __future__ import annotations

import logging
import threading as _threading
from dataclasses import dataclass, field
from threading import Thread
from typing import Optional, TypeVar

from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg

from ..job import Job
from ..job_result import JobResult
from .base import AsyncJob

__all__ = ["ThreadJob"]

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class ThreadJob(AsyncJob[T]):
    """A :class:`~yggdrasil.concurrent.job.Job` running in a daemon
    :class:`~threading.Thread`, started immediately on construction.

    Use :meth:`wait` to block and retrieve the result::

        handle = Job.make(fetch_data, url).fire_and_forget()

        # … do other work …

        data = handle.wait()           # block until thread finishes
        data = handle.wait(wait=5.0)   # block at most 5 seconds
        data = handle.wait(wait=False) # non-blocking poll
    """

    job: Job[T]

    # Internal mutable state — excluded from the public constructor
    _result: Optional[T] = field(default=None, init=False, repr=False)
    _exception: Optional[BaseException] = field(default=None, init=False, repr=False)
    _done: _threading.Event = field(
        default_factory=_threading.Event, init=False, repr=False,
    )
    _thread: Optional[Thread] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        func_name = getattr(self.job.func, "__qualname__", repr(self.job.func))
        t = Thread(
            target=self._run,
            name=f"ThreadJob-{func_name}",
            daemon=True,
        )
        self._thread = t
        t.start()
        LOGGER.debug("ThreadJob started: %s", func_name)

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def _run(self) -> None:
        func_name = getattr(self.job.func, "__qualname__", repr(self.job.func))
        try:
            self._result = self.job.run()
            LOGGER.debug("ThreadJob finished ok: %s", func_name)
        except BaseException as exc:  # noqa: BLE001
            self._exception = exc
            LOGGER.debug("ThreadJob raised %r: %s", exc, func_name)
        finally:
            self._done.set()

    # ------------------------------------------------------------------
    # AsyncJob implementation
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        """``True`` when the thread has finished (success or failure)."""
        return self._done.is_set()

    def result(self) -> Optional[JobResult[T]]:
        """Return the :class:`~yggdrasil.concurrent.job_result.JobResult` if done, else ``None``."""
        if not self.is_done:
            return None
        return JobResult(result=self._result, exception=self._exception)

    def wait(
        self,
        wait: WaitingConfigArg = None,
        raise_error: bool = True,
    ) -> Optional[T]:
        """Block until the thread finishes.

        Args:
            wait:        ``None`` / ``True``  → block indefinitely.
                         ``False``            → non-blocking poll.
                         ``WaitingConfig`` or numeric seconds → timed wait.
            raise_error: Re-raise the thread's exception when ``True``.

        Returns:
            The job return value, or ``None`` on failure / timeout / non-blocking
            miss (when ``raise_error=False``).

        Raises:
            TimeoutError: When a timed wait elapses before the thread finishes.
            BaseException: The exception from the job (when ``raise_error=True``).
        """
        # --- Non-blocking poll ---
        if wait is False:
            if not self._done.is_set():
                return None

        # --- Block indefinitely ---
        elif wait is None or wait is True:
            self._done.wait(timeout=None)

        # --- Timed wait via WaitingConfig ---
        else:
            config = WaitingConfig.check_arg(wait)
            timeout: float | None = config.timeout if config.timeout else None
            completed = self._done.wait(timeout=timeout)
            if not completed:
                func_name = getattr(self.job.func, "__qualname__", repr(self.job.func))
                LOGGER.warning(
                    "ThreadJob timed out after %.1fs: %s", timeout, func_name,
                )
                if raise_error:
                    raise TimeoutError(
                        f"ThreadJob({func_name!r}) did not complete within {timeout}s"
                    )
                return None

        # --- Propagate exception or return result ---
        if self._exception is not None:
            if raise_error:
                raise self._exception
            return None

        return self._result

