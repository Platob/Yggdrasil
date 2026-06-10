"""FX-rate exceptions — one ``BackendError`` family under ``YGGException``.

A :class:`BackendError` means a single upstream (Frankfurter, Fawaz,
ER-API, …) couldn't answer; the orchestrator catches it and walks to the
next backend. :class:`AllBackendsFailed` is raised only when the whole
chain is exhausted, carrying the per-backend failures for diagnosis.
"""
from __future__ import annotations

from yggdrasil.exceptions.base import YGGException

__all__ = ["FxRateError", "BackendError", "AllBackendsFailed", "NoBackendsError"]


class FxRateError(YGGException):
    """Base for every fxrate error."""


class BackendError(FxRateError):
    """A single FX backend failed to return a usable response."""


class NoBackendsError(FxRateError):
    """No backend was configured to answer the request."""


class AllBackendsFailed(FxRateError):
    """Every backend in the chain failed; carries the per-backend reasons."""

    def __init__(self, failures: dict[str, Exception]) -> None:
        self.failures = failures
        detail = "; ".join(f"{name}: {exc}" for name, exc in failures.items())
        super().__init__(f"All FX backends failed ({detail}).")
