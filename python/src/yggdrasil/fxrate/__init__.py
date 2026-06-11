"""FX rate orchestration.

Fetch foreign-exchange rates across a fallback chain of providers and
get back a long polars frame. Hand :class:`FxRate` a list of
:class:`~yggdrasil.fxrate.backends.Backend` instances, then ask for a
timeseries or the latest snapshot::

    from yggdrasil.fxrate import FxRate
    fx = FxRate(backends=[primary, secondary])
    df = fx.fetch([("EUR", "USD"), ("EUR", "GBP")], "2024-01-01", "2024-01-30")

Inputs are loose on purpose — currencies as ISO codes / aliases /
:class:`Currency`, moments as ISO strings / epochs / ``datetime`` — and
normalised on the way in. A backend that can't answer raises
:class:`BackendError`, which rolls the request over to the next provider.
"""
from __future__ import annotations

from .backends import Backend
from .session import BackendError, FxQuote, FxRate

__all__ = ["FxRate", "FxQuote", "BackendError", "Backend"]
