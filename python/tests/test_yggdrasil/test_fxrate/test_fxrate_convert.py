"""Tests for :meth:`FxRate.convert`.

The scalar convert helper is what curated callers reach for when
they want "give me one amount in EUR" without going through the
DataFrame helpers. Exercises the short-circuit on same-currency,
the snapshot path (``at=None``), and the historical path
(``at=<datetime>``) — all against a stub backend, no network.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.enums.currency import Currency
from yggdrasil.fxrate import BackendError, FxRate

from ._helpers import StubFxBackend, make_quote


class TestConvertSameCurrency:

    def test_short_circuits_without_backend_call(self) -> None:
        # No quotes registered — would raise via _fetch_group_with_fallback
        # if it tried; the same-currency short-circuit means no call fires.
        b = StubFxBackend(quotes=())
        fx = FxRate(backends=(b,))
        assert fx.convert(100.0, "USD", "USD") == 100.0
        assert fx.convert(0.0, Currency.EUR, Currency.EUR) == 0.0
        assert b.calls == []


class TestConvertLatest:

    def test_basic_conversion(self) -> None:
        b = StubFxBackend(quotes=(make_quote("EUR", "USD", value=1.10),))
        fx = FxRate(backends=(b,))
        assert fx.convert(100.0, "EUR", "USD") == pytest.approx(110.0)
        # latest path — not timeseries
        assert b.calls[0]["kind"] == "latest"

    def test_alias_resolution(self) -> None:
        b = StubFxBackend(quotes=(make_quote("USD", "EUR", value=0.92),))
        fx = FxRate(backends=(b,))
        # Aliases parse through Currency.from_ — '$' → USD, '€' → EUR.
        assert fx.convert(100.0, "$", "€") == pytest.approx(92.0)

    def test_currency_instances(self) -> None:
        b = StubFxBackend(quotes=(make_quote("EUR", "JPY", value=160.0),))
        fx = FxRate(backends=(b,))
        assert fx.convert(100.0, Currency.EUR, Currency.JPY) == pytest.approx(16000.0)


class TestConvertHistorical:

    def test_historical_at_datetime(self) -> None:
        b = StubFxBackend(quotes=(
            make_quote("EUR", "USD", date="2024-06-15", value=1.08),
        ))
        fx = FxRate(backends=(b,))
        at = dt.datetime(2024, 6, 15, tzinfo=dt.timezone.utc)
        assert fx.convert(100.0, "EUR", "USD", at=at) == pytest.approx(108.0)
        assert b.calls[0]["kind"] == "timeseries"

    def test_historical_picks_closest_not_after(self) -> None:
        b = StubFxBackend(quotes=(
            make_quote("EUR", "USD", date="2024-06-13", value=1.07),
            make_quote("EUR", "USD", date="2024-06-14", value=1.08),
            make_quote("EUR", "USD", date="2024-06-15", value=1.09),
        ))
        fx = FxRate(backends=(b,))
        # Sunday request — the stub doesn't have Sunday data, but the
        # fetch returned three business-day rows surrounding the
        # weekend. ``_pick_historical_rate`` picks the closest-but-not-
        # after timestamp.
        at = dt.datetime(2024, 6, 14, 12, 0, tzinfo=dt.timezone.utc)
        # 2024-06-14 row carries 1.08; the next row (2024-06-15) is
        # AFTER our timestamp, so we settle on 1.08.
        assert fx.convert(100.0, "EUR", "USD", at=at) == pytest.approx(108.0)


class TestConvertFailure:

    def test_no_rate_raises_backenderror(self) -> None:
        # Stub returns zero rows for the pair — every backend fails.
        b = StubFxBackend(quotes=())
        fx = FxRate(backends=(b,))
        with pytest.raises(BackendError, match="EUR->USD"):
            fx.convert(100.0, "EUR", "USD")
