"""Trading signal generation — stateless, pure-function momentum signals.

Strategy: z-score of the latest price against a rolling window.
  z > +1.5  → SELL  (price elevated vs recent history)
  z < -1.5  → BUY   (price depressed vs recent history)
  otherwise → HOLD
"""
from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Signal, SignalKind

ZSCORE_BUY = -1.5
ZSCORE_SELL = 1.5

# Polars is faster for large windows; pure-Python is fine for < 200 points.
_POLARS_THRESHOLD = 200


def _zscore(values: list[float]) -> float:
    """Z-score of the last value in *values* against the full window.

    Uses polars vectorization for large windows (≥ _POLARS_THRESHOLD) to
    avoid O(n) Python loops on month-long hourly series.
    """
    n = len(values)
    if n < 3:
        return 0.0
    if n >= _POLARS_THRESHOLD:
        try:
            import polars as pl
            s = pl.Series(values, dtype=pl.Float64)
            mean = s.mean()
            std = s.std()
            if std and std > 0:
                return float((values[-1] - mean) / std)
            return 0.0
        except ImportError:
            pass
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 0.0
    return (values[-1] - mean) / std if std > 0 else 0.0


def compute_signals(
    prices: list[dict],
    zone: str,
    series: str,
) -> list["Signal"]:
    """Derive BUY/SELL/HOLD signals from a flat price list.

    *prices* is the output of :func:`~yggdrasil.bot.market.fetch_prices`.
    Returns one :class:`~yggdrasil.bot.models.Signal` per zone/series group.
    """
    from .models import Signal

    if not prices:
        return []

    values = [row["value"] for row in prices if row.get("value") is not None]
    if not values:
        return []

    z = _zscore(values)
    kind: "SignalKind" = "SELL" if z > ZSCORE_SELL else "BUY" if z < ZSCORE_BUY else "HOLD"
    mean = sum(values) / len(values)
    latest_price = values[-1]
    latest_ts = prices[-1]["timestamp"]
    if isinstance(latest_ts, str):
        latest_ts = dt.datetime.fromisoformat(latest_ts)

    reason = (
        f"Latest {series} price {latest_price:.2f} is "
        f"{abs(z):.2f}σ {'above' if z > 0 else 'below'} 7-day mean {mean:.2f}."
    )

    return [Signal(
        zone=zone,
        series=series,
        kind=kind,
        price=latest_price,
        mean=round(mean, 4),
        zscore=round(z, 4),
        ts=latest_ts,
        reason=reason,
    )]
