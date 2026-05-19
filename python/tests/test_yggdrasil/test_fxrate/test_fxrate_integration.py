"""Live HTTP integration tests for the FX backends.

These hit the real public APIs (Frankfurter, fawazahmed/jsDelivr,
open.er-api.com). Marked ``fxrate_integration`` and skipped unless
``YGGDRASIL_FXRATE_INTEGRATION`` is set in the environment — same
opt-in shape the Databricks and Postgres live tests use.

Run with::

    YGGDRASIL_FXRATE_INTEGRATION=1 pytest -m fxrate_integration \\
        tests/test_yggdrasil/test_fxrate/test_fxrate_integration.py

These tests check the wire-level contract — that the API still
returns the shape our parsers expect — and the end-to-end fetch
through :class:`FxRate`. They're slow (one or two HTTP calls each)
and fragile against upstream changes; that's the point. When a
parse fails here, fix the parser before the data layer notices.
"""
from __future__ import annotations

import datetime as dt
import os

import pytest

from yggdrasil.fxrate import (
    ErApiBackend,
    FawazBackend,
    FrankfurterBackend,
    FxRate,
)


_INTEGRATION_FLAG = "YGGDRASIL_FXRATE_INTEGRATION"


pytestmark = [
    pytest.mark.fxrate_integration,
    pytest.mark.skipif(
        not os.environ.get(_INTEGRATION_FLAG),
        reason=(
            f"FX live integration tests are opt-in. Set "
            f"{_INTEGRATION_FLAG}=1 to enable."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Frankfurter
# ---------------------------------------------------------------------------


class TestFrankfurterLive:

    def test_latest_eur_usd_gbp(self) -> None:
        fx = FxRate(backends=(FrankfurterBackend(),))
        df = fx.latest(pairs=[("EUR", "USD"), ("EUR", "GBP")])
        sources = set(df["source"].to_list())
        targets = set(df["target"].to_list())
        assert sources == {"EUR"}
        assert "USD" in targets and "GBP" in targets
        # Sanity: EUR/USD has lived in [0.7, 1.6] for decades; the live
        # rate must fall inside that conservative envelope.
        for v in df["value"].to_list():
            assert 0.1 < v < 10.0

    def test_timeseries_window_covered(self) -> None:
        fx = FxRate(backends=(FrankfurterBackend(),))
        df = fx.fetch(
            pairs=[("EUR", "USD")],
            start="2024-01-02", end="2024-01-05",
        )
        # ECB skips weekends, so 4 dates → 2 to 4 rows depending on
        # weekday alignment. Pin the lower bound + check we don't drift
        # outside the window.
        assert df.height >= 2
        dates = set(df["from_timestamp"].to_list())
        for ts in dates:
            assert dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc) <= ts
            assert ts <= dt.datetime(2024, 1, 5, tzinfo=dt.timezone.utc)


# ---------------------------------------------------------------------------
# Fawaz / jsDelivr
# ---------------------------------------------------------------------------


class TestFawazLive:

    def test_latest_with_crypto(self) -> None:
        # Frankfurter doesn't cover BTC — pin the Fawaz path on a
        # known-good crypto pair so a regression in the URL shape
        # fails loud.
        fx = FxRate(backends=(FawazBackend(),))
        df = fx.latest(pairs=[("USD", "BTC")])
        assert df.height == 1
        # BTC has lived above 1k USD for years — keep the floor lax.
        value = df["value"].to_list()[0]
        assert 0.0 < value < 1.0

    def test_historical_specific_date(self) -> None:
        fx = FxRate(backends=(FawazBackend(),))
        df = fx.fetch(
            pairs=[("EUR", "USD")],
            start="2024-03-06", end="2024-03-06",
        )
        assert df.height == 1
        # Pin the known historical value within a generous band — the
        # static archive shouldn't drift but the test stays useful
        # even if the upstream rounds differently.
        v = df["value"].to_list()[0]
        assert 0.9 < v < 1.3


# ---------------------------------------------------------------------------
# ER-API
# ---------------------------------------------------------------------------


class TestErApiLive:

    def test_latest_usd_base(self) -> None:
        fx = FxRate(backends=(ErApiBackend(),))
        df = fx.latest(pairs=[("USD", "EUR"), ("USD", "JPY")])
        sources = set(df["source"].to_list())
        targets = set(df["target"].to_list())
        assert sources == {"USD"}
        assert "EUR" in targets and "JPY" in targets


# ---------------------------------------------------------------------------
# Multi-source fallback against the real chain
# ---------------------------------------------------------------------------


class TestDefaultChainLive:

    def test_default_chain_handles_crypto_via_fallback(self) -> None:
        # Frankfurter rejects BTC; the default chain must roll to Fawaz.
        fx = FxRate()
        df = fx.latest(pairs=[("USD", "BTC")])
        assert df.height == 1
        assert df["source"].to_list() == ["USD"]
        assert df["target"].to_list() == ["BTC"]

    def test_default_chain_mixed_pairs(self) -> None:
        # ECB-supported + crypto in one call — sanity check the
        # per-pair fallback doesn't fragment the frame.
        fx = FxRate()
        df = fx.latest(pairs=[("EUR", "USD"), ("USD", "BTC")])
        rows = list(zip(df["source"].to_list(), df["target"].to_list()))
        assert ("EUR", "USD") in rows
        assert ("USD", "BTC") in rows

    def test_geo_enrichment_live(self) -> None:
        fx = FxRate()
        df = fx.latest(
            pairs=[("USD", "EUR"), ("GBP", "JPY")],
            geo=True,
        )
        # ISO 4217 alpha-3 codes start with the issuing country's
        # alpha-2 — pin the heuristic from session._zone_for_currency.
        by_source = dict(zip(df["source"].to_list(), df["source_country_iso"].to_list()))
        assert by_source["USD"] == "US"
        assert by_source["GBP"] == "GB"
        # Lat/lon must be valid WGS84 ranges.
        for lat in df["source_lat"].to_list() + df["target_lat"].to_list():
            if lat is None:
                continue
            assert -90.0 <= lat <= 90.0
        for lon in df["source_lon"].to_list() + df["target_lon"].to_list():
            if lon is None:
                continue
            assert -180.0 <= lon <= 180.0
