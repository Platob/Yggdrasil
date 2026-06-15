"""Tests for the FX-rate path — yggdrasil.fxrate + FxRateSkill.

Covers the per-call coercion hot path, the two real backends' request/parse
shaping (HTTP mocked — no network), the multi-backend fallback walk, the
polars frame assembly, and the Loki FxRateSkill end-to-end.
"""
from __future__ import annotations

import datetime as dt
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.enums.currency import Currency
from yggdrasil.fxrate import BackendError, FxQuote, FxRate
from yggdrasil.fxrate.backends import Backend, FawazBackend, FrankfurterBackend
from yggdrasil.fxrate.session import (
    _coerce_currency,
    _coerce_datetime,
    _coerce_pair,
    _group_pairs_by_source,
)

_UTC = dt.timezone.utc


def _quote(source: str, target: str, value: float = 1.1) -> FxQuote:
    return FxQuote(
        source=source,
        target=target,
        from_timestamp=dt.datetime(2024, 1, 1, tzinfo=_UTC),
        to_timestamp=dt.datetime(2024, 1, 2, tzinfo=_UTC),
        sampling="1d",
        value=value,
    )


class _StubBackend(Backend):
    name = "stub"
    base_url = "stub://"

    def __init__(self, quotes=None, raise_with=None):
        self._quotes = quotes or []
        self._raise = raise_with

    def fetch_timeseries(self, session, *, source, targets, start, end, sampling):
        if self._raise is not None:
            raise self._raise
        return self._quotes

    def fetch_latest(self, session, *, source, targets, at):
        if self._raise is not None:
            raise self._raise
        return self._quotes


class TestCoercion(unittest.TestCase):
    def test_currency_iso_alias_identity(self):
        self.assertEqual(_coerce_currency("EUR"), Currency.EUR)
        self.assertEqual(_coerce_currency("$"), Currency.USD)
        self.assertIs(_coerce_currency(Currency.EUR), Currency.EUR)

    def test_pair(self):
        self.assertEqual(_coerce_pair(("eur", "usd")), (Currency.EUR, Currency.USD))

    def test_datetime_date_only_string(self):
        self.assertEqual(
            _coerce_datetime("2024-01-01"),
            dt.datetime(2024, 1, 1, tzinfo=_UTC),
        )

    def test_datetime_iso_with_tz(self):
        self.assertEqual(
            _coerce_datetime("2024-01-01T10:00:00+00:00"),
            dt.datetime(2024, 1, 1, 10, tzinfo=_UTC),
        )

    def test_datetime_z_suffix(self):
        self.assertEqual(
            _coerce_datetime("2024-01-01T10:00:00Z"),
            dt.datetime(2024, 1, 1, 10, tzinfo=_UTC),
        )

    def test_datetime_naive_date_and_epoch(self):
        self.assertEqual(
            _coerce_datetime(dt.date(2024, 1, 1)),
            dt.datetime(2024, 1, 1, tzinfo=_UTC),
        )
        self.assertEqual(
            _coerce_datetime(1704067200), dt.datetime(2024, 1, 1, tzinfo=_UTC)
        )

    def test_datetime_naive_datetime_gets_utc(self):
        self.assertEqual(
            _coerce_datetime(dt.datetime(2024, 1, 1, 5)),
            dt.datetime(2024, 1, 1, 5, tzinfo=_UTC),
        )

    def test_datetime_bad_string_raises_value_error(self):
        with self.assertRaises(ValueError):
            _coerce_datetime("not-a-date-at-all")

    def test_datetime_bad_type_raises_type_error(self):
        with self.assertRaises(TypeError):
            _coerce_datetime(object())

    def test_group_pairs_by_source(self):
        pairs = [
            (Currency.EUR, Currency.USD),
            (Currency.EUR, Currency.GBP),
            (Currency.USD, Currency.JPY),
        ]
        self.assertEqual(
            _group_pairs_by_source(pairs),
            {"EUR": ["USD", "GBP"], "USD": ["JPY"]},
        )


class TestFrankfurterBackend(unittest.TestCase):
    def test_timeseries_request_and_parse(self):
        resp = MagicMock()
        resp.json.return_value = {
            "base": "EUR",
            "rates": {
                "2024-01-01": {"USD": 1.10, "GBP": 0.87},
                "2024-01-02": {"USD": 1.11, "GBP": 0.88},
            },
        }
        sess = MagicMock()
        sess.get.return_value = resp
        quotes = FrankfurterBackend().fetch_timeseries(
            sess, source="EUR", targets=["USD", "GBP"],
            start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2), sampling="1d",
        )
        self.assertEqual(len(quotes), 4)
        url = sess.get.call_args.args[0]
        self.assertIn("2024-01-01..2024-01-02", url)
        self.assertIn("from=EUR", url)
        first = quotes[0]
        self.assertEqual((first.source, first.target), ("EUR", "USD"))
        self.assertEqual(first.from_timestamp, dt.datetime(2024, 1, 1, tzinfo=_UTC))

    def test_failure_wraps_backend_error(self):
        sess = MagicMock()
        sess.get.side_effect = RuntimeError("boom")
        with self.assertRaises(BackendError):
            FrankfurterBackend().fetch_timeseries(
                sess, source="EUR", targets=["USD"],
                start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 1), sampling="1d",
            )


class TestFawazBackend(unittest.TestCase):
    def test_latest_lowercases_and_filters_targets(self):
        resp = MagicMock()
        resp.json.return_value = {"date": "2024-01-01", "eur": {"usd": 1.1, "gbp": 0.87, "jpy": 160.0}}
        sess = MagicMock()
        sess.get.return_value = resp
        quotes = FawazBackend().fetch_latest(
            sess, source="EUR", targets=["USD", "GBP"], at=dt.datetime(2024, 1, 1, tzinfo=_UTC),
        )
        self.assertEqual({q.target for q in quotes}, {"USD", "GBP"})  # JPY filtered out
        url = sess.get.call_args.args[0]
        self.assertIn("/eur.json", url)


class TestOrchestration(unittest.TestCase):
    def test_fetch_assembles_long_frame(self):
        fx = FxRate(backends=[_StubBackend([_quote("EUR", "USD"), _quote("EUR", "GBP")])])
        df = fx.fetch(pairs=[("EUR", "USD"), ("EUR", "GBP")], start="2024-01-01", end="2024-01-31")
        self.assertEqual(df.height, 2)
        self.assertEqual(
            df.columns,
            ["source", "target", "from_timestamp", "to_timestamp", "sampling", "value"],
        )

    def test_lazy_returns_lazyframe(self):
        import polars as pl

        fx = FxRate(backends=[_StubBackend([_quote("EUR", "USD")])])
        out = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-31", lazy=True)
        self.assertIsInstance(out, pl.LazyFrame)

    def test_empty_quotes_typed_frame(self):
        import polars as pl

        fx = FxRate(backends=[_StubBackend([])])
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-31")
        self.assertEqual(df.height, 0)
        self.assertEqual(df.schema["value"], pl.Float64)

    def test_fallback_on_backend_error(self):
        primary = _StubBackend(raise_with=BackendError("down"))
        secondary = _StubBackend([_quote("EUR", "USD")])
        fx = FxRate(backends=[primary, secondary])
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        self.assertEqual(df.height, 1)

    def test_all_backends_fail_raises(self):
        fx = FxRate(backends=[_StubBackend(raise_with=BackendError("a"))])
        with self.assertRaises(BackendError):
            fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")

    def test_latest(self):
        fx = FxRate(backends=[_StubBackend([_quote("EUR", "USD")])])
        self.assertEqual(fx.latest(pairs=[("EUR", "USD")]).height, 1)

    def test_default_chain_is_frankfurter_then_fawaz(self):
        fx = FxRate()
        self.assertEqual([b.name for b in fx._backends], ["frankfurter", "fawaz"])


class TestFxRateSkill(unittest.TestCase):
    def test_registered(self):
        from yggdrasil.loki.skill import REGISTRY

        self.assertIn("fx_rate", REGISTRY)

    def test_skill_runs_with_string_pairs(self):
        from yggdrasil.loki.skills import FxRateSkill

        fx = FxRate(backends=[_StubBackend([_quote("EUR", "USD")])])
        with patch("yggdrasil.fxrate.FxRate", return_value=fx):
            res = FxRateSkill().run(
                MagicMock(), pairs=["EUR/USD"], start="2024-01-01", end="2024-01-31",
            )
        self.assertTrue(res["available"])
        self.assertEqual(res["mode"], "timeseries")
        self.assertEqual(res["pairs"], [("EUR", "USD")])
        self.assertEqual(res["rows"], 1)

    def test_skill_latest_when_no_window(self):
        from yggdrasil.loki.skills import FxRateSkill

        fx = FxRate(backends=[_StubBackend([_quote("EUR", "USD")])])
        with patch("yggdrasil.fxrate.FxRate", return_value=fx):
            res = FxRateSkill().run(MagicMock(), pairs=[["EUR", "USD"]])
        self.assertEqual(res["mode"], "latest")

    def test_skill_bad_pair_string_raises(self):
        from yggdrasil.loki.skills import FxRateSkill

        with self.assertRaises(ValueError):
            FxRateSkill().run(MagicMock(), pairs=["EURUSD"], start="2024-01-01", end="2024-01-02")


if __name__ == "__main__":
    unittest.main()
