from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.enums.currency import Currency
from yggdrasil.fxrate import BackendError, FxQuote, FxRate
from yggdrasil.fxrate.backends import Backend
from yggdrasil.fxrate.exceptions import AllBackendsFailed
from yggdrasil.fxrate.session import (
    _coerce_currency,
    _coerce_datetime,
    _coerce_pair,
    _group_pairs_by_source,
)


class _Stub(Backend):
    name = "stub"
    base_url = "stub://"
    default_sampling = "1d"

    def __init__(self, quotes=None, raise_with=None):
        self._quotes = quotes or []
        self._raise = raise_with

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        if self._raise:
            raise self._raise
        return self._quotes

    def fetch_latest(self, session, *, source, targets, at):
        if self._raise:
            raise self._raise
        return self._quotes


def _q(source="EUR", target="USD"):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    return FxQuote(source, target, now, now + dt.timedelta(days=1), "1d", 1.1)


# -- coercion ---------------------------------------------------------------

def test_coerce_currency_alias():
    assert _coerce_currency("$").code == "USD"
    assert _coerce_currency(Currency.EUR) is Currency.EUR


def test_coerce_pair_forms():
    assert _coerce_pair(("EUR", "USD")) == (Currency.EUR, Currency.USD)
    assert _coerce_pair("EUR/USD") == (Currency.EUR, Currency.USD)
    assert _coerce_pair("EURUSD") == (Currency.EUR, Currency.USD)


def test_coerce_datetime_forms():
    assert _coerce_datetime("2024-01-01").year == 2024
    assert _coerce_datetime(1704067200).tzinfo is not None
    assert _coerce_datetime(dt.date(2024, 1, 1)).day == 1


def test_group_pairs_by_source():
    grouped = _group_pairs_by_source([
        (Currency.EUR, Currency.USD),
        (Currency.EUR, Currency.GBP),
        (Currency.USD, Currency.JPY),
    ])
    assert grouped == {"EUR": ["USD", "GBP"], "USD": ["JPY"]}


# -- orchestration ----------------------------------------------------------

def test_fetch_assembles_frame():
    fx = FxRate(backends=(_Stub([_q()]),))
    df = fx.fetch(pairs=["EUR/USD"], start="2024-01-01", end="2024-01-01")
    assert df.columns[:2] == ["source", "target"]
    assert df.height == 1


def test_fetch_lazy_returns_lazyframe():
    import polars as pl

    fx = FxRate(backends=(_Stub([_q()]),))
    out = fx.fetch(pairs=["EUR/USD"], start="2024-01-01", end="2024-01-01", lazy=True)
    assert isinstance(out, pl.LazyFrame)


def test_fallback_walks_to_next_backend():
    fx = FxRate(backends=(_Stub(raise_with=BackendError("down")), _Stub([_q()])))
    df = fx.fetch(pairs=["EUR/USD"], start="2024-01-01", end="2024-01-01")
    assert df.height == 1


def test_all_backends_failing_raises():
    fx = FxRate(backends=(_Stub(raise_with=BackendError("a")), _Stub(raise_with=BackendError("b"))))
    with pytest.raises(AllBackendsFailed):
        fx.fetch(pairs=["EUR/USD"], start="2024-01-01", end="2024-01-01")
