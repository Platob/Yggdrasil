"""Unit tests for :class:`yggdrasil.fxrate.FxRate`.

These exercise input coercion, backend orchestration / fallback,
frame assembly, and geography enrichment with a stub backend —
no network. Live HTTP tests live in
``test_fxrate_integration.py``.
"""
from __future__ import annotations

import datetime as dt

import pytest

from yggdrasil.enums.currency import Currency
from yggdrasil.fxrate import (
    FX_FRAME_COLUMNS,
    FX_FRAME_GEO_COLUMNS,
    BackendError,
    FxRate,
)

from ._helpers import StubFxBackend, make_quote


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


class TestInputCoercion:

    def test_string_pairs_normalise_to_uppercase(self) -> None:
        b = StubFxBackend(quotes=(
            make_quote("EUR", "USD", value=1.10),
        ))
        fx = FxRate(backends=(b,))
        df = fx.fetch(pairs=[("eur", "usd")], start="2024-01-01", end="2024-01-01")
        assert df["source"].to_list() == ["EUR"]
        assert df["target"].to_list() == ["USD"]
        assert b.calls[0]["source"] == "EUR"
        assert b.calls[0]["targets"] == ("USD",)

    def test_currency_alias_resolves_via_parse(self) -> None:
        b = StubFxBackend(quotes=(make_quote("USD", "EUR", value=0.92),))
        fx = FxRate(backends=(b,))
        df = fx.fetch(pairs=[("$", "€")], start="2024-01-01", end="2024-01-01")
        assert df["source"].to_list() == ["USD"]
        assert df["target"].to_list() == ["EUR"]

    def test_currency_instance_accepted(self) -> None:
        b = StubFxBackend(quotes=(make_quote("USD", "JPY", value=150.0),))
        fx = FxRate(backends=(b,))
        df = fx.fetch(
            pairs=[(Currency.USD, Currency.JPY)],
            start="2024-01-01", end="2024-01-01",
        )
        assert df["source"].to_list() == ["USD"]
        assert df["target"].to_list() == ["JPY"]

    def test_iso_string_start_end(self) -> None:
        b = StubFxBackend(quotes=(make_quote("EUR", "USD", value=1.1),))
        fx = FxRate(backends=(b,))
        fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-05")
        call = b.calls[0]
        assert call["start"] == dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert call["end"] == dt.datetime(2024, 1, 5, tzinfo=dt.timezone.utc)

    def test_naive_datetime_treated_as_utc(self) -> None:
        b = StubFxBackend(quotes=(make_quote("EUR", "USD", value=1.1),))
        fx = FxRate(backends=(b,))
        fx.fetch(
            pairs=[("EUR", "USD")],
            start=dt.datetime(2024, 3, 1, 12, 0, 0),
            end=dt.datetime(2024, 3, 1, 14, 0, 0),
        )
        call = b.calls[0]
        assert call["start"].tzinfo == dt.timezone.utc
        assert call["end"].tzinfo == dt.timezone.utc

    def test_epoch_seconds_accepted(self) -> None:
        b = StubFxBackend(quotes=(make_quote("EUR", "USD", value=1.1),))
        fx = FxRate(backends=(b,))
        # 2024-01-01T00:00:00Z
        fx.fetch(pairs=[("EUR", "USD")], start=1704067200, end=1704153600)
        call = b.calls[0]
        assert call["start"] == dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        assert call["end"] == dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)

    def test_empty_pairs_raises(self) -> None:
        fx = FxRate(backends=(StubFxBackend(),))
        with pytest.raises(ValueError, match="at least one"):
            fx.fetch(pairs=[], start="2024-01-01", end="2024-01-02")

    def test_invalid_pair_shape_raises(self) -> None:
        fx = FxRate(backends=(StubFxBackend(),))
        with pytest.raises(ValueError, match="length 2"):
            fx.fetch(pairs=[("EUR",)], start="2024-01-01", end="2024-01-02")  # type: ignore[list-item]

    def test_identical_source_target_raises(self) -> None:
        fx = FxRate(backends=(StubFxBackend(),))
        with pytest.raises(ValueError, match="identical"):
            fx.fetch(pairs=[("EUR", "EUR")], start="2024-01-01", end="2024-01-02")

    def test_start_after_end_raises(self) -> None:
        fx = FxRate(backends=(StubFxBackend(),))
        with pytest.raises(ValueError, match="after end"):
            fx.fetch(
                pairs=[("EUR", "USD")],
                start="2024-01-10", end="2024-01-01",
            )


# ---------------------------------------------------------------------------
# Backend fallback orchestration
# ---------------------------------------------------------------------------


class TestBackendFallback:

    def test_first_backend_used_when_it_returns_data(self) -> None:
        primary = StubFxBackend(
            name="primary",
            quotes=(make_quote("EUR", "USD", value=1.10),),
        )
        secondary = StubFxBackend(name="secondary", quotes=())
        fx = FxRate(backends=(primary, secondary))
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        assert df["value"].to_list() == [1.10]
        assert len(primary.calls) == 1
        # Secondary not consulted — primary already returned data.
        assert len(secondary.calls) == 0

    def test_falls_through_on_backend_error(self) -> None:
        primary = StubFxBackend(
            name="primary",
            raise_with=BackendError("simulated outage"),
        )
        secondary = StubFxBackend(
            name="secondary",
            quotes=(make_quote("EUR", "USD", value=1.11),),
        )
        fx = FxRate(backends=(primary, secondary))
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        assert df["value"].to_list() == [1.11]
        assert len(primary.calls) == 1
        assert len(secondary.calls) == 1

    def test_falls_through_on_unexpected_exception(self) -> None:
        # Non-BackendError still triggers fallback (any backend that
        # crashes shouldn't sink the whole fetch).
        primary = StubFxBackend(
            name="primary",
            raise_with=RuntimeError("kaboom"),
        )
        secondary = StubFxBackend(
            name="secondary",
            quotes=(make_quote("EUR", "USD", value=1.12),),
        )
        fx = FxRate(backends=(primary, secondary))
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        assert df["value"].to_list() == [1.12]

    def test_empty_return_triggers_fallback(self) -> None:
        primary = StubFxBackend(name="primary", quotes=())
        secondary = StubFxBackend(
            name="secondary",
            quotes=(make_quote("EUR", "USD", value=1.13),),
        )
        fx = FxRate(backends=(primary, secondary))
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        assert df["value"].to_list() == [1.13]
        assert len(primary.calls) == 1
        assert len(secondary.calls) == 1

    def test_all_backends_fail_raises_backend_error(self) -> None:
        b1 = StubFxBackend(name="one", raise_with=BackendError("oops 1"))
        b2 = StubFxBackend(name="two", raise_with=BackendError("oops 2"))
        fx = FxRate(backends=(b1, b2))
        with pytest.raises(BackendError, match="All FX backends failed"):
            fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")

    def test_no_backends_configured_raises(self) -> None:
        fx = FxRate(backends=())
        with pytest.raises(BackendError, match="no configured backends"):
            fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")

    def test_grouping_one_call_per_source(self) -> None:
        # Three pairs but only two distinct sources — backend hit twice.
        primary = StubFxBackend(
            quotes=(make_quote("EUR", "USD", value=1.1),),
        )
        fx = FxRate(backends=(primary,))
        fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP"), ("USD", "JPY")],
            start="2024-01-01", end="2024-01-01",
        )
        sources = [c["source"] for c in primary.calls]
        assert sorted(sources) == ["EUR", "USD"]
        # EUR call carried both targets.
        eur_call = next(c for c in primary.calls if c["source"] == "EUR")
        assert sorted(eur_call["targets"]) == ["GBP", "USD"]

    def test_dedup_target_per_source(self) -> None:
        primary = StubFxBackend(quotes=())
        fx = FxRate(backends=(primary,))
        fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "USD"), ("EUR", "GBP")],
            start="2024-01-01", end="2024-01-01",
        )
        eur_call = primary.calls[0]
        # USD must only appear once even though the caller listed it twice.
        assert eur_call["targets"].count("USD") == 1


# ---------------------------------------------------------------------------
# Frame schema + ordering
# ---------------------------------------------------------------------------


class TestFrameSchema:

    def _two_day_frame(self):
        primary = StubFxBackend(quotes=(
            make_quote("EUR", "USD", date="2024-01-02", value=1.10),
            make_quote("EUR", "USD", date="2024-01-01", value=1.11),
            make_quote("EUR", "GBP", date="2024-01-01", value=0.86),
        ))
        fx = FxRate(backends=(primary,))
        return fx.fetch(
            pairs=[("EUR", "USD"), ("EUR", "GBP")],
            start="2024-01-01", end="2024-01-02",
        )

    def test_columns_match_contract(self) -> None:
        df = self._two_day_frame()
        assert tuple(df.columns) == FX_FRAME_COLUMNS

    def test_timestamps_are_utc_microseconds(self) -> None:
        import polars as pl
        df = self._two_day_frame()
        assert df.schema["from_timestamp"] == pl.Datetime("us", time_zone="UTC")
        assert df.schema["to_timestamp"] == pl.Datetime("us", time_zone="UTC")

    def test_value_is_float64(self) -> None:
        import polars as pl
        df = self._two_day_frame()
        assert df.schema["value"] == pl.Float64

    def test_sorted_by_from_source_target(self) -> None:
        df = self._two_day_frame()
        rows = list(zip(
            df["from_timestamp"].to_list(),
            df["source"].to_list(),
            df["target"].to_list(),
        ))
        assert rows == sorted(rows)

    def test_sampling_passes_through(self) -> None:
        primary = StubFxBackend(quotes=(
            make_quote("EUR", "USD", value=1.1, sampling="1h"),
        ))
        fx = FxRate(backends=(primary,))
        df = fx.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
            sampling="1h",
        )
        assert df["sampling"].to_list() == ["1h"]
        assert primary.calls[0]["sampling"] == "1h"


# ---------------------------------------------------------------------------
# Geography enrichment
# ---------------------------------------------------------------------------


class TestGeoEnrichment:

    def test_geo_columns_added_only_when_requested(self) -> None:
        primary = StubFxBackend(quotes=(make_quote("EUR", "USD", value=1.1),))
        fx = FxRate(backends=(primary,))
        plain = fx.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
        )
        assert "source_lat" not in plain.columns
        with_geo = fx.fetch(
            pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01",
            geo=True,
        )
        for col in FX_FRAME_GEO_COLUMNS:
            assert col in with_geo.columns

    def test_geo_unknown_currency_yields_none(self) -> None:
        # A currency we know the GeoZone catalog can't resolve — XYZ
        # isn't a real ISO 4217 code, so both sides should fall through
        # to null geo. Currency() accepts any 3-letter alpha-3 token.
        primary = StubFxBackend(quotes=(make_quote("XYZ", "ABC", value=1.0),))
        fx = FxRate(backends=(primary,))
        df = fx.fetch(
            pairs=[("XYZ", "ABC")], start="2024-01-01", end="2024-01-01",
            geo=True,
        )
        # The columns exist but the values are null — geography
        # enrichment never breaks the fetch.
        assert df["source_country_iso"].to_list() == [None]
        assert df["source_lat"].to_list() == [None]
        assert df["target_country_iso"].to_list() == [None]


# ---------------------------------------------------------------------------
# Singleton + transient backends
# ---------------------------------------------------------------------------


class TestSingleton:

    def test_same_no_base_url_collapses_to_one_instance(self) -> None:
        a = FxRate()
        b = FxRate()
        assert a is b

    def test_backends_override_on_second_construction(self) -> None:
        FxRate()  # warm
        b = StubFxBackend(quotes=(make_quote("EUR", "USD", value=2.0),))
        fx = FxRate(backends=(b,))
        df = fx.fetch(pairs=[("EUR", "USD")], start="2024-01-01", end="2024-01-01")
        assert df["value"].to_list() == [2.0]
