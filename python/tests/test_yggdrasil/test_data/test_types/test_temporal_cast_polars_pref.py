"""Tests for the polars-first tz helper in ``temporal``.

The helper exists so timezone-aware arrow casts do not touch pyarrow's
Arrow C++ tz database (which fails on Windows without IANA tzdata set up).
Polars ships its own tz database via chrono-tz, so routing tz transitions
through polars sidesteps the Windows failure entirely while staying
equivalent to the pyarrow semantics.
"""
from __future__ import annotations

from unittest.mock import patch

import pyarrow as pa
import pytest

from yggdrasil.data.types import temporal as tc


pl = pytest.importorskip("polars")


def _naive_ts_array(values, unit="us"):
    return pa.array(values, type=pa.timestamp(unit))


def _aware_ts_array(values, tz, unit="us"):
    return pa.array(values, type=pa.timestamp(unit, tz))


# ---------------------------------------------------------------------------
# Polars-first path is actually taken
# ---------------------------------------------------------------------------

def test_polars_path_does_not_touch_pyarrow_assume_timezone():
    """With polars available, pc.assume_timezone must not be called."""
    arr = _naive_ts_array([0, 1_000_000])
    with patch.object(tc.pc, "assume_timezone") as pa_assume:
        out = tc.retimestamp_prefer_polars(arr, unit="us", tz="Europe/Paris")
    assert out.type == pa.timestamp("us", "Europe/Paris")
    pa_assume.assert_not_called()


def test_polars_path_does_not_touch_pyarrow_cast_for_tz_change():
    aware = _aware_ts_array([0, 1_000_000], tz="UTC")
    with patch.object(tc.pc, "cast") as pa_cast:
        out = tc.retimestamp_prefer_polars(aware, unit="us", tz="Europe/Paris")
    assert out.type == pa.timestamp("us", "Europe/Paris")
    pa_cast.assert_not_called()


def test_fallback_when_polars_unavailable():
    """When _polars_or_none returns None, pyarrow path is used."""
    arr = _naive_ts_array([0])
    with patch.object(tc, "_polars_or_none", return_value=None), \
         patch.object(tc.pc, "assume_timezone", wraps=tc.pc.assume_timezone) as pa_assume:
        out = tc.retimestamp_prefer_polars(arr, unit="us", tz="UTC")
    assert out.type == pa.timestamp("us", "UTC")
    pa_assume.assert_called_once()


def test_fallback_for_seconds_unit():
    """Polars has no 's' time unit, so seconds path must fall back."""
    arr = _naive_ts_array([0], unit="s")
    with patch.object(tc.pc, "assume_timezone", wraps=tc.pc.assume_timezone) as pa_assume:
        out = tc.retimestamp_prefer_polars(arr, unit="s", tz="UTC")
    assert out.type == pa.timestamp("s", "UTC")
    pa_assume.assert_called_once()


def test_no_tz_either_side_uses_pyarrow_cast_only():
    """No tz on source or target — no tz database involved at all."""
    arr = _naive_ts_array([0], unit="ns")
    out = tc.retimestamp_prefer_polars(arr, unit="us", tz=None)
    assert out.type == pa.timestamp("us")


# ---------------------------------------------------------------------------
# Semantic equivalence: polars path matches pyarrow semantics
# ---------------------------------------------------------------------------

def test_naive_to_tz_aware_unsafe_preserves_wallclock():
    """unsafe_tz=True: 2023-01-02 03:04 as Paris stays 03:04 Paris."""
    arr = pa.array(
        [pa.scalar(0, type=pa.timestamp("us"))],  # 1970-01-01 00:00:00 naive
        type=pa.timestamp("us"),
    )
    out = tc.retimestamp_prefer_polars(arr, unit="us", tz="Europe/Paris", unsafe_tz=True)
    assert out.type == pa.timestamp("us", "Europe/Paris")
    # The wall-clock reading in the target zone is the same as the naive input.
    as_polars = pl.from_arrow(out)
    dt_in_paris = as_polars.to_list()[0]
    assert dt_in_paris.hour == 0 and dt_in_paris.minute == 0


def test_naive_to_tz_aware_safe_treats_source_as_utc():
    """unsafe_tz=False: naive is assumed UTC and shifted to target tz."""
    arr = _naive_ts_array([0])
    out = tc.retimestamp_prefer_polars(arr, unit="us", tz="Europe/Paris", unsafe_tz=False)
    assert out.type == pa.timestamp("us", "Europe/Paris")
    as_polars = pl.from_arrow(out)
    dt_paris = as_polars.to_list()[0]
    # 1970-01-01 UTC is 01:00 Paris (CET, no DST in January).
    assert dt_paris.hour == 1


def test_tz_aware_to_tz_aware_same_instant():
    aware = _aware_ts_array([0], tz="UTC")
    out = tc.retimestamp_prefer_polars(aware, unit="us", tz="America/New_York")
    assert out.type == pa.timestamp("us", "America/New_York")
    as_polars = pl.from_arrow(out)
    dt_ny = as_polars.to_list()[0]
    # 1970-01-01 00:00 UTC = 1969-12-31 19:00 New York (EST).
    assert dt_ny.year == 1969 and dt_ny.hour == 19


def test_tz_aware_to_naive_matches_pyarrow():
    """tz-aware → naive preserves the underlying UTC-epoch integer in both
    pyarrow and polars. This test pins that equivalence so the polars path
    can't silently drift from pyarrow's cast semantics.
    """
    import pyarrow.compute as pc

    aware = _aware_ts_array([0, 42_000_000], tz="Europe/Paris")
    polars_out = tc.retimestamp_prefer_polars(aware, unit="us", tz=None)
    pyarrow_out = pc.cast(aware, pa.timestamp("us"))
    assert polars_out.type == pa.timestamp("us")
    assert polars_out.to_pylist() == pyarrow_out.to_pylist()


def test_unit_conversion_carries_through():
    arr = _naive_ts_array([0], unit="ns")
    out = tc.retimestamp_prefer_polars(arr, unit="ms", tz="UTC")
    assert out.type == pa.timestamp("ms", "UTC")


# ---------------------------------------------------------------------------
# Call-site integration
# ---------------------------------------------------------------------------

def test_arrow_cast_to_timestamp_uses_polars_helper():
    """arrow_cast_to_timestamp must route through the polars-first helper."""
    arr = _naive_ts_array([0])
    with patch.object(tc, "retimestamp_prefer_polars", wraps=tc.retimestamp_prefer_polars) as spy:
        out = tc.arrow_cast_to_timestamp(arr, unit="us", tz="Europe/Paris", unsafe_tz=True)
    spy.assert_called_once()
    assert out.type == pa.timestamp("us", "Europe/Paris")


def test_arrow_str_to_timestamp_uses_polars_helper():
    """arrow_str_to_timestamp must route the tz tail through the helper."""
    arr = pa.array(["2023-01-02T03:04:05"])
    with patch.object(tc, "retimestamp_prefer_polars", wraps=tc.retimestamp_prefer_polars) as spy:
        out = tc.arrow_str_to_timestamp(arr, unit="us", tz="Europe/Paris", unsafe_tz=True)
    spy.assert_called_once()
    assert out.type == pa.timestamp("us", "Europe/Paris")
