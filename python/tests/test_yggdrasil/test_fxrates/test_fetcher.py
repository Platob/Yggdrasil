from __future__ import annotations

import datetime as dt
import json
from unittest.mock import patch

from yggdrasil.fxrates.fetcher import FxRate, fetch_fx_rates


class _Resp:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._payload).encode("utf-8")


def test_fetch_fx_rates_polars_default() -> None:
    payload = {
        "rates": {
            "2026-01-01": {"EUR": 0.9},
            "2026-01-02": {"EUR": 0.91},
        }
    }
    with patch("yggdrasil.fxrates.fetcher.urlopen", return_value=_Resp(payload)):
        df = fetch_fx_rates(
            "USD",
            "EUR",
            dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.timezone.utc),
            dt.datetime(2026, 1, 1, 2, 0, tzinfo=dt.timezone.utc),
            sampling=3600,
        )

    assert df.shape[0] == 3
    assert df["source"].to_list() == ["USD", "USD", "USD"]
    assert df["target"].to_list() == ["EUR", "EUR", "EUR"]
    assert df["value"].to_list() == [0.9, 0.9, 0.9]


def test_fetch_fx_rates_list_mode() -> None:
    payload = {"rates": {"2026-01-01": {"CHF": 0.8}}}
    with patch("yggdrasil.fxrates.fetcher.urlopen", return_value=_Resp(payload)):
        rows = fetch_fx_rates(
            "USD",
            "CHF",
            dt.datetime(2026, 1, 1, 0, 0),
            dt.datetime(2026, 1, 1, 2, 0),
            sampling=3600,
            as_polars=False,
        )

    assert all(isinstance(r, FxRate) for r in rows)
    assert [r.value for r in rows] == [0.8, 0.8, 0.8]
