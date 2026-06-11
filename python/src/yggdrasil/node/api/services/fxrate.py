"""FX rate service for the node API.

Wraps FxRate with the FrankfurterBackend; offline-safe (returns empty
if the network is down or the backend raises BackendError).
"""
from __future__ import annotations

import datetime as dt
import logging

_LOG = logging.getLogger("yggdrasil.node.fxrate")

# Supported pairs as (source, target) tuples — the common crosses
COMMON_PAIRS = [
    ("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY"),
    ("USD", "EUR"), ("USD", "GBP"), ("USD", "JPY"),
    ("GBP", "USD"), ("GBP", "EUR"),
]


class FxRateService:
    def __init__(self) -> None:
        try:
            from yggdrasil.fxrate import FxRate
            from yggdrasil.fxrate.frankfurter import FrankfurterBackend
            self._fx = FxRate(backends=[FrankfurterBackend()])
        except Exception:
            self._fx = None

    def latest(self, pairs: list[tuple[str, str]] | None = None) -> list[dict]:
        if self._fx is None:
            return []
        if pairs is None:
            pairs = COMMON_PAIRS
        try:
            quotes = self._fx.latest(pairs)
            return [
                {
                    "source": q.source, "target": q.target,
                    "value": q.value, "date": q.from_timestamp.strftime("%Y-%m-%d"),
                    "sampling": q.sampling,
                }
                for q in quotes
            ]
        except Exception as exc:
            _LOG.debug("FX latest failed: %s", exc)
            return []

    def timeseries(
        self,
        source: str,
        target: str,
        start: str,
        end: str,
    ) -> list[dict]:
        if self._fx is None:
            return []
        try:
            df = self._fx.fetch([(source, target)], start, end)
            if df is None or df.is_empty():
                return []
            return [
                {
                    "source": row["source"], "target": row["target"],
                    "value": row["value"],
                    "date": row["from_timestamp"].strftime("%Y-%m-%d"),
                }
                for row in df.iter_rows(named=True)
            ]
        except Exception as exc:
            _LOG.debug("FX timeseries failed: %s", exc)
            return []
