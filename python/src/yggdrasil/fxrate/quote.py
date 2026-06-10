"""The atom of FX data — a single :class:`FxQuote` (one rate, one window).

A quote is ``source → target`` over a half-open ``[from_timestamp,
to_timestamp)`` window at a given ``sampling`` (``"1d"``, ``"latest"``,
…) carrying the rate ``value`` (units of *target* per 1 unit of
*source*). Quotes are frozen, hashable, and picklable — long-frame
assembly fans a list of them straight into polars.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FxQuote:
    source: str
    target: str
    from_timestamp: dt.datetime
    to_timestamp: dt.datetime
    sampling: str
    value: float

    def as_row(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "from_timestamp": self.from_timestamp,
            "to_timestamp": self.to_timestamp,
            "sampling": self.sampling,
            "value": self.value,
        }
