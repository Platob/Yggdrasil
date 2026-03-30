"""
Job — immutable unit of work.

Thread-specific classes (AsyncJob, ThreadJob) live in
:mod:`yggdrasil.concurrent.threading`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from yggdrasil.concurrent.threading import ThreadJob

__all__ = ["Job"]

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Job(Generic[T]):
    """Immutable, callable bundle describing a unit of work.

    Create with :meth:`make` rather than the constructor directly::

        job = Job.make(my_func, arg1, arg2, key=val)
        result = job.run()                # synchronous
        handle = job.fire_and_forget()    # background thread, returns ThreadJob
        handle.wait()                     # block until done
    """

    func: Callable[..., T]
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        merged_args = self.args + args
        merged_kwargs = {**self.kwargs, **kwargs}
        return self.func(*merged_args, **merged_kwargs)

    @classmethod
    def make(
        cls,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> "Job[T]":
        """Build a :class:`Job` from a callable and its arguments."""
        return cls(func=func, args=args, kwargs=kwargs)

    def run(self) -> T:
        """Execute the job synchronously in the calling thread."""
        return self.func(*self.args, **self.kwargs)

    def thread(self) -> "ThreadJob[T]":
        """Start this job in a daemon thread and return an awaitable :class:`~yggdrasil.concurrent.threading.ThreadJob`.

        The thread is started immediately.  Call :meth:`~yggdrasil.concurrent.threading.ThreadJob.wait`
        to block until completion and retrieve the result.
        """
        # Lazy import avoids the job → threading → job circular dependency.
        from yggdrasil.concurrent.threading import ThreadJob  # noqa: PLC0415
        return ThreadJob(job=self)

    def fire_and_forget(self) -> "ThreadJob[T]":
        """Alias for :meth:`thread`.

        Returns the :class:`~yggdrasil.concurrent.threading.ThreadJob` handle so callers
        can optionally await the result — or simply ignore it for true fire-and-forget.
        """
        return self.thread()
