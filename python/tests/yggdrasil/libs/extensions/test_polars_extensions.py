from __future__ import annotations

from datetime import datetime, timedelta
import zoneinfo

import polars as pl
import pytest

# Import module so monkeypatching (setattr on pl.DataFrame) happens.
import yggdrasil.libs.extensions.polars_extensions as ext  # noqa: F401


def _dt(y, m, d, hh=0, mm=0, ss=0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=zoneinfo.ZoneInfo("UTC"))


def test_join_coalesced_prefers_left_and_falls_back_to_right():
    left = pl.DataFrame(
        {
            "k": [1, 2, 3],
            "v": [10, None, 30],
            "only_left": ["a", "b", "c"],
        }
    )
    right = pl.DataFrame(
        {
            "k": [1, 2, 3],
            "v": [100, 200, 300],
            "only_right": ["x", "y", "z"],
        }
    )

    out = pl.DataFrame.join_coalesced(left, right, on="k", how="left")

    assert out.columns == ["k", "v", "only_left", "only_right"]
    assert out.to_dict(as_series=False) == {
        "k": [1, 2, 3],
        "v": [10, 200, 30],
        "only_left": ["a", "b", "c"],
        "only_right": ["x", "y", "z"],
    }


def test_resample_upsample_inserts_missing_timestamps_and_forward_fills():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "ts": [base, base + timedelta(minutes=2), base + timedelta(minutes=4)],
            "price": [1.0, 2.0, 3.0],
        }
    )

    out = df.resample(
        time_col="ts",
        every="1m",
        group_by="symbol",
        fill="forward",
    ).sort("ts")

    assert out.height == 5
    assert out["ts"].to_list() == [base + timedelta(minutes=i) for i in range(5)]
    assert out["price"].to_list() == [1.0, 1.0, 2.0, 2.0, 3.0]


def test_resample_upsample_group_by_separates_series():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "symbol": ["A", "A", "B", "B"],
            "ts": [
                base,
                base + timedelta(minutes=2),
                base + timedelta(minutes=1),
                base + timedelta(minutes=3),
            ],
            "price": [10.0, 20.0, 100.0, 300.0],
        }
    )

    out = df.resample(
        time_col="ts",
        every="1m",
        group_by="symbol",
        fill="forward",
    )

    a = out.filter(pl.col("symbol") == "A").sort("ts")
    assert a["ts"].to_list() == [base + timedelta(minutes=i) for i in range(3)]
    assert a["price"].to_list() == [10.0, 10.0, 20.0]

    b = out.filter(pl.col("symbol") == "B").sort("ts")
    assert b["ts"].to_list() == [base + timedelta(minutes=i) for i in range(1, 4)]
    assert b["price"].to_list() == [100.0, 100.0, 300.0]


def test_resample_downsample_group_by_dynamic_with_dict_aggs():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "symbol": ["A"] * 6,
            "ts": [base + timedelta(minutes=i) for i in range(6)],
            "qty": [1, 2, 3, 4, 5, 6],
            "px": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
        }
    )

    out = df.resample(
        time_col="ts",
        every="3m",
        group_by="symbol",
        agg={"qty": "sum", "px": "last"},
        label="left",
        closed="left",
    ).sort("ts")

    assert out["ts"].to_list() == [base + timedelta(minutes=0), base + timedelta(minutes=3)]
    assert out["qty"].to_list() == [1 + 2 + 3, 4 + 5 + 6]
    assert out["px"].to_list() == [12.0, 22.0]


def test_resample_downsample_with_expr_aggs_list():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "symbol": ["A"] * 4,
            "ts": [base + timedelta(minutes=i) for i in range(4)],
            "x": [1, 2, 3, 4],
        }
    )

    out = df.resample(
        time_col="ts",
        every="2m",
        group_by="symbol",
        agg=[
            pl.col("x").mean().alias("x_mean"),
            pl.col("x").max().alias("x_max"),
        ],
    ).sort("ts")

    assert out["ts"].to_list() == [base + timedelta(minutes=0), base + timedelta(minutes=2)]
    assert out["x_mean"].to_list() == [1.5, 3.5]
    assert out["x_max"].to_list() == [2, 4]


def test_resample_invalid_fill_raises():
    df = pl.DataFrame({"ts": [_dt(2025, 1, 1, 0, 0, 0)], "v": [1]})
    with pytest.raises(ValueError):
        df.resample(every="1m", time_col="ts", fill="lolnope")  # type: ignore[arg-type]


# -------------------------
# New tests: time_col inference
# -------------------------

def test_resample_infers_first_datetime_column_when_time_col_missing():
    base = _dt(2025, 1, 1, 0, 0, 0)

    # ts_a should be inferred (first Datetime in schema order)
    df = pl.DataFrame(
        {
            "id": [1, 1],
            "ts_a": [base, base + timedelta(minutes=2)],
            "ts_b": [base + timedelta(hours=1), base + timedelta(hours=1, minutes=2)],
            "v": [10, 20],
        }
    )

    out = df.resample(
        every="1m",
        group_by="id",
        fill="forward",
        # time_col omitted on purpose
    )

    g = out.filter(pl.col("id") == 1).sort("ts_a")
    assert g.height == 3
    assert g["ts_a"].to_list() == [base + timedelta(minutes=i) for i in range(3)]
    assert g["v"].to_list() == [10, 10, 20]


def test_resample_infer_raises_when_no_datetime_columns_exist():
    # Date-only should NOT be accepted if you require "Datetime only"
    df = pl.DataFrame(
        {
            "d": [datetime(2025, 1, 1).date(), datetime(2025, 1, 2).date()],
            "x": [1, 2],
        }
    )
    with pytest.raises(ValueError):
        df.resample(every="1d", fill="forward")  # time_col omitted


def test_resample_infer_prefers_datetime_not_date_even_if_date_is_first():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            # Date column comes first in schema
            "d": [base.date(), base.date()],
            # Datetime column exists and should be inferred
            "ts": [base, base + timedelta(minutes=2)],
            "v": [1, 2],
        }
    )

    out = df.resample(every="1m", fill="forward")  # time_col omitted
    out = out.sort("ts")

    assert out["ts"].to_list() == [base + timedelta(minutes=i) for i in range(3)]
    assert out["v"].to_list() == [1, 1, 2]


def test_resample_explicit_time_col_overrides_inference():
    base = _dt(2025, 1, 1, 0, 0, 0)
    df = pl.DataFrame(
        {
            "ts_a": [base, base + timedelta(minutes=2)],
            "ts_b": [base + timedelta(hours=1), base + timedelta(hours=1, minutes=2)],
            "v": [10, 20],
        }
    )

    out = df.resample(time_col="ts_b", every="1m", fill="forward").sort("ts_b")

    assert out["ts_b"].to_list() == [
        base + timedelta(hours=1, minutes=i) for i in range(3)
    ]
    assert out["v"].to_list() == [10, 10, 20]
