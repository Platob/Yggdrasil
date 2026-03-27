from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen

from yggdrasil.data.enums.currency import Currency

__all__ = ["FxRate", "fetch_fx_rates"]


@dataclass(frozen=True, slots=True)
class FxRate:
    source: Currency
    target: Currency
    value: float


def _to_utc_datetime(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _fetch_daily_rates(source: Currency, target: Currency, start: dt.date, end: dt.date) -> dict[dt.date, float]:
    endpoint = f"https://api.frankfurter.app/{start.isoformat()}..{end.isoformat()}"
    qs = urlencode({"from": source.code, "to": target.code})
    with urlopen(f"{endpoint}?{qs}") as resp:  # nosec: B310 - trusted static host + encoded query
        payload = json.loads(resp.read().decode("utf-8"))

    rates = payload.get("rates", {})
    result: dict[dt.date, float] = {}
    for day, values in rates.items():
        raw = values.get(target.code)
        if raw is None:
            continue
        result[dt.date.fromisoformat(day)] = float(raw)
    if not result:
        raise ValueError(
            f"No FX rates returned for {source.code}/{target.code} between {start} and {end}"
        )
    return result


def fetch_fx_rates(
    source: Currency | str,
    target: Currency | str,
    start: dt.datetime,
    end: dt.datetime,
    *,
    sampling: int = 3600,
    as_polars: bool = True,
):
    src = Currency.parse(source)
    tgt = Currency.parse(target)

    start_utc = _to_utc_datetime(start)
    end_utc = _to_utc_datetime(end)

    if end_utc < start_utc:
        raise ValueError("end must be >= start")
    if sampling <= 0:
        raise ValueError("sampling must be > 0 seconds")

    daily = _fetch_daily_rates(src, tgt, start_utc.date(), end_utc.date())
    day_keys = sorted(daily)

    rows: list[tuple[dt.datetime, FxRate]] = []
    cursor = start_utc
    delta = dt.timedelta(seconds=int(sampling))

    while cursor <= end_utc:
        d = cursor.date()
        candidates = [day for day in day_keys if day <= d]
        use_day = candidates[-1] if candidates else day_keys[0]
        rate = FxRate(source=src, target=tgt, value=daily[use_day])
        rows.append((cursor, rate))
        cursor += delta

    if as_polars:
        from yggdrasil.polars.lib import polars as pl

        return pl.DataFrame(
            {
                "at": [at for at, _ in rows],
                "source": [r.source.code for _, r in rows],
                "target": [r.target.code for _, r in rows],
                "value": [r.value for _, r in rows],
            }
        )

    return [r for _, r in rows]
