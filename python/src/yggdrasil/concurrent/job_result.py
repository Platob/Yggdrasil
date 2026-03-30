"""
Immutable result wrapper for an async job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

__all__ = ["JobResult"]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class JobResult(Generic[T]):
    """Immutable outcome of a :class:`~yggdrasil.concurrent.job.Job` execution.

    Either holds a *result* (success) or an *exception* (failure).
    Both fields are always present; exactly one is ``None`` in the normal case.
    """

    result: Optional[T]
    exception: Optional[BaseException]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """``True`` when the job succeeded (no exception captured)."""
        return self.exception is None

    def get(self) -> T:
        """Return the result or re-raise the captured exception.

        Raises:
            BaseException: The exception that was captured during job execution.
        """
        if self.exception is not None:
            raise self.exception
        return self.result  # type: ignore[return-value]

    def __bool__(self) -> bool:
        return self.ok

