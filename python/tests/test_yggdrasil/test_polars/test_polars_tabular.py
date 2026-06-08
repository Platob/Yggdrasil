"""Tests for :class:`yggdrasil.polars.tabular.PolarsTabular`."""

from __future__ import annotations

import polars as pl
import pyarrow as pa

from yggdrasil.polars.tabular import PolarsTabular


def test_holds_native_frame_and_schema() -> None:
    t = PolarsTabular(pl.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
    assert not t.is_empty()
    assert bool(t) is True
    assert t.collect_schema().names == ["id", "name"]


def test_read_polars_is_native() -> None:
    df = pl.DataFrame({"id": [1, 2]})
    t = PolarsTabular(df)
    out = t.read_polars_frame()
    assert isinstance(out, pl.DataFrame)
    assert out.to_dicts() == [{"id": 1}, {"id": 2}]


def test_cross_engine_reads() -> None:
    t = PolarsTabular(pl.DataFrame({"id": [1, 2], "v": [1.5, 2.5]}))
    assert t.read_arrow_table().to_pylist() == [{"id": 1, "v": 1.5}, {"id": 2, "v": 2.5}]
    assert t.read_pandas_frame().to_dict("records") == [{"id": 1, "v": 1.5}, {"id": 2, "v": 2.5}]


def test_empty_holder() -> None:
    t = PolarsTabular()
    assert t.is_empty()
    assert bool(t) is False
    assert t.read_arrow_table().num_rows == 0


def test_write_overwrite_and_append() -> None:
    t = PolarsTabular(pl.DataFrame({"a": [1]}))
    t.write_table(pl.DataFrame({"a": [2]}))  # default mode = overwrite
    assert t.read_arrow_table().to_pylist() == [{"a": 2}]
    from yggdrasil.enums import Mode
    t.write_table(pl.DataFrame({"a": [3]}), mode=Mode.APPEND)
    assert t.read_arrow_table()["a"].to_pylist() == [2, 3]


def test_write_arrow_into_polars_holder() -> None:
    t = PolarsTabular()
    t.write_table(pa.table({"a": [1, 2, 3]}))
    assert isinstance(t.frame, pl.DataFrame)
    assert t.read_arrow_table().num_rows == 3


def test_filter_through_tabular_api() -> None:
    t = PolarsTabular(pl.DataFrame({"id": [1, 2, 3], "region": ["US", "EU", "US"]}))
    out = t.filter("region = 'US'").read_arrow_table()
    assert out.num_rows == 2
