"""Trading-oriented FX rate orchestration.

:class:`FxRate` fans currency-pair requests across a fallback chain of
free FX providers (:class:`Frankfurter`, :class:`Fawaz`, :class:`ErApi`),
grouping pairs by source currency, walking to the next backend on
:class:`BackendError`, and assembling the result into a long-format
polars frame — optionally enriched with country geography.

Typical use::

    from yggdrasil.fxrate import FxRate
    from yggdrasil.fxrate.backends import Frankfurter, Fawaz, ErApi

    fx = FxRate(backends=[Frankfurter(), Fawaz(), ErApi()])
    df = fx.fetch(
        pairs=[("EUR", "USD"), ("EUR", "GBP")],
        start="2024-01-01", end="2024-01-30",
    )
    spot = fx.latest(pairs=[("EUR", "USD")])
"""
from __future__ import annotations

from .backends import Backend, BackendError, ErApi, Fawaz, Frankfurter, FxQuote
from .session import FxRate

__all__ = [
    "FxRate",
    "FxQuote",
    "Backend",
    "BackendError",
    "Frankfurter",
    "Fawaz",
    "ErApi",
]
