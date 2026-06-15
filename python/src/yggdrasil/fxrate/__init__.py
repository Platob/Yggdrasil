"""FX rate orchestration — multi-backend, polars-native, geography-enriched.

Pull spot and time-series exchange rates from public APIs (Frankfurter,
Fawaz) through a pluggable :class:`~yggdrasil.fxrate.backends.Backend`
layer. Results come back as polars DataFrames. Everything rides
:class:`yggdrasil.http_.HTTPSession`, so the caller gets the project-level
retry / cache semantics for free.

Quick start::

    from yggdrasil.fxrate import FxRate
    fx = FxRate()
    df = fx.fetch(pairs=[("EUR", "USD"), ("EUR", "GBP")],
                  start="2024-01-01", end="2024-01-31")

The orchestrator tries each backend in order and falls through to the next
on :class:`BackendError`, so a single upstream outage degrades gracefully.
"""
from __future__ import annotations

from .backends import Backend, BackendError
from .session import FxQuote, FxRate

__all__ = ["FxRate", "FxQuote", "Backend", "BackendError"]
