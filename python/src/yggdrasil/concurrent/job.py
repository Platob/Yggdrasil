"""
Job — minimal callable bundle.

Thread-specific classes (AsyncJob, ThreadJob) live in
:mod:`yggdrasil.concurrent.threading`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from yggdrasil.concurrent.threading import ThreadJob

__all__ = ["Job"]

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


# Shared empty sentinels — Job stores these in the no-arg case so
# ``Job.make(func)`` never allocates a fresh tuple / dict. Neither
# is mutated by Job; callers reading ``job.kwargs`` only ever iterate
# or merge-from it (never mutate).
_EMPTY_ARGS: Tuple[Any, ...] = ()
_EMPTY_KWARGS: Dict[str, Any] = {}


class Job(Generic[T]):
    """Minimal callable bundle: a function plus its bound args / kwargs.

    Deliberately not a :func:`dataclass`. ``Job`` is on the hot path
    for every fan-out (``JobPoolExecutor.as_completed``), every
    fire-and-forget thread spawn, and every retry layer — the saved
    dataclass __init__ / equality / repr overhead matters per job. A
    plain slotted class with a hand-written ``__init__`` is roughly
    half the per-construction cost.

    Build via :meth:`make` (collects ``*args / **kwargs`` like a
    normal call site) or pass them explicitly to the constructor.

    Usage::

        job = Job.make(my_func, arg1, arg2, key=val)
        result = job.run()                # synchronous
        handle = job.fire_and_forget()    # background thread, returns ThreadJob
        handle.wait()                     # block until done
    """

    __slots__ = ("func", "args", "kwargs")

    func: Callable[..., T]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def __init__(
        self,
        func: Callable[..., T],
        args: Tuple[Any, ...] = _EMPTY_ARGS,
        kwargs: Dict[str, Any] = _EMPTY_KWARGS,
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"Job({getattr(self.func, '__qualname__', self.func)!r})"

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        # Hot path: ``job()`` with no extra args / kwargs — common
        # under ``ThreadPoolExecutor.submit(job)`` and direct call
        # sites. Skip the merge allocations.
        if not args and not kwargs:
            return self.func(*self.args, **self.kwargs)
        merged_args = self.args + args if args else self.args
        merged_kwargs = {**self.kwargs, **kwargs} if kwargs else self.kwargs
        return self.func(*merged_args, **merged_kwargs)

    @classmethod
    def make(
        cls,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> "Job[T]":
        """Build a :class:`Job` from a callable and its arguments."""
        # Reuse the shared empty sentinels for the no-arg case so we
        # never allocate a per-job tuple / dict. Python's ``*args /
        # **kwargs`` collection already gave us the args tuple / kwargs
        # dict for free; nothing to copy.
        return cls(
            func,
            args if args else _EMPTY_ARGS,
            kwargs if kwargs else _EMPTY_KWARGS,
        )

    def run(self) -> T:
        """Execute the job synchronously in the calling thread."""
        # Branch the most common shapes out of the call: ``run()`` on
        # ``Job.make(func)`` is hit by every job submitted to
        # :class:`JobPoolExecutor`, so the saved unpack cost compounds
        # across large fan-out streams.
        kwargs = self.kwargs
        if not kwargs:
            args = self.args
            if not args:
                return self.func()
            return self.func(*args)
        return self.func(*self.args, **kwargs)

    def thread(self) -> "ThreadJob[T]":
        """Start this job in a daemon thread and return a :class:`~yggdrasil.concurrent.threading.ThreadJob` handle.

        The thread is started immediately. Call
        :meth:`~yggdrasil.concurrent.threading.ThreadJob.wait` to block
        until completion and retrieve the result.
        """
        # Lazy import avoids the job → threading → job circular dependency.
        from yggdrasil.concurrent.threading import ThreadJob  # noqa: PLC0415
        return ThreadJob(job=self)

    def fire_and_forget(self) -> "ThreadJob[T]":
        """Alias for :meth:`thread` — kept for the more readable call site."""
        return self.thread()
