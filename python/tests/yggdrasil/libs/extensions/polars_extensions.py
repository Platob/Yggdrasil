# test_time_resample.py

import datetime as dt
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from yggdrasil.libs.extensions.polars_extensions import time_resample  # <-- change this


def _dt(year, month, day, hour=0, minute=0, tzinfo=None):
    return dt.datetime(year, month, day, hour, minute, tzinfo=tzinfo)


def test_time_resample_berlin_dst_join_and_fill_debug():
    berlin_tz = ZoneInfo("Europe/Berlin")

    # Original UTC anchors
    utc_0 = dt.datetime(2025, 3, 30, 0, 0, tzinfo=dt.timezone.utc)
    utc_1 = dt.datetime(2025, 3, 30, 1, 0, tzinfo=dt.timezone.utc)
    utc_2 = dt.datetime(2025, 3, 30, 2, 0, tzinfo=dt.timezone.utc)

    df = pl.DataFrame(
        {
            "ts": pl.Series(
                "ts",
                [utc_0, utc_1, utc_2],
                dtype=pl.Datetime("us", "Etc/UTC"),
            ),
            "value": [100, None, 200],
        }
    )

    # Run resample: join + forward fill in Berlin tz
    res = time_resample(
        df,
        time_column="ts",
        every="1h",
        time_unit="us",
        time_zone="Europe/Berlin",
        maintain_order=True,
        fill_strategy="forward",
    )

    # Lists for debugging
    orig_ts_list = df["ts"].to_list()
    orig_val_list = df["value"].to_list()

    ts_list = res["ts"].to_list()
    val_list = res["value"].to_list()

    # Expected Berlin axis you specified
    berlin_list = [
        dt.datetime(2025, 3, 30, 1, 0, tzinfo=berlin_tz),
        dt.datetime(2025, 3, 30, 2, 0, tzinfo=berlin_tz),
        dt.datetime(2025, 3, 30, 3, 0, tzinfo=berlin_tz),
        dt.datetime(2025, 3, 30, 4, 0, tzinfo=berlin_tz),
    ]

    # Expected values:
    # - 01:00 Berlin ← from utc_0 (100)
    # - 02:00 Berlin ← new from axis, gets filled forward from 100
    # - 03:00 Berlin ← from utc_1 (which mapped here), but original is None → forward fill continues
    # - 04:00 Berlin ← from utc_2 (200)
    expected_values = [100, 100, 100, 200]

    # Basic dtype / tz sanity
    ts_dtype = res.schema["ts"]
    assert isinstance(ts_dtype, pl.Datetime)
    assert ts_dtype.time_zone == "Europe/Berlin"
    assert ts_dtype.time_unit == "us"

    # Assert exact axis + values
    assert ts_list == berlin_list
    assert val_list == expected_values