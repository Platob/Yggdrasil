"""yggdrasil.fxrate — schema-aware FX rate fetching with backend fallback.

Public surface::

    from yggdrasil.fxrate import FxRate, FxQuote

    fx = FxRate()                                   # ECB → ER-API chain
    df = fx.fetch(pairs=[("EUR", "USD")],
                  start="2024-01-01", end="2024-01-30")   # polars frame
    spot = fx.latest(pairs=["EUR/USD", "USD/JPY"])

:class:`FxRate` orchestrates: coerce inputs → group pairs by source →
walk the :class:`~yggdrasil.fxrate.backends.Backend` chain (falling back on
:class:`BackendError`) → assemble a long polars frame → optionally enrich
with geography. Each pair is a :class:`FxQuote` on the wire.
"""
from __future__ import annotations

from .backends import Backend, ExchangeRateBackend, FrankfurterBackend
from .exceptions import AllBackendsFailed, BackendError, FxRateError, NoBackendsError
from .quote import FxQuote
from .session import FxRate, FxRateClient

__all__ = [
    "FxRate",
    "FxRateClient",
    "FxQuote",
    "Backend",
    "FrankfurterBackend",
    "ExchangeRateBackend",
    "FxRateError",
    "BackendError",
    "NoBackendsError",
    "AllBackendsFailed",
]
