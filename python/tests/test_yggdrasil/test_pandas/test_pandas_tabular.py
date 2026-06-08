"""Tests for :class:`yggdrasil.pandas.tabular.PandasTabular`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.pandas.tabular import PandasTabular

pd = pytest.importorskip("pandas")


def test_holds_native_frame_and_schema() -> None:
    t = PandasTabular(pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
    assert not t.is_empty()
    assert t.collect_schema().names == ["id", "name"]


def test_read_pandas_is_native() -> None:
    t = PandasTabular(pd.DataFrame({"id": [1, 2]}))
    out = t.read_pandas_frame()
    assert isinstance(out, pd.DataFrame)
    assert out.to_dict("records") == [{"id": 1}, {"id": 2}]


def test_cross_engine_reads() -> None:
    t = PandasTabular(pd.DataFrame({"id": [1, 2], "v": [1.5, 2.5]}))
    assert t.read_arrow_table().to_pylist() == [{"id": 1, "v": 1.5}, {"id": 2, "v": 2.5}]
    import polars as pl
    assert isinstance(t.read_polars_frame(), pl.DataFrame)


def test_empty_holder() -> None:
    t = PandasTabular()
    assert t.is_empty()
    assert t.read_arrow_table().num_rows == 0


def test_write_overwrite_and_append() -> None:
    from yggdrasil.enums import Mode

    t = PandasTabular(pd.DataFrame({"a": [1]}))
    t.write_table(pd.DataFrame({"a": [2]}))  # default = overwrite
    assert t.read_arrow_table().to_pylist() == [{"a": 2}]
    t.write_table(pd.DataFrame({"a": [3]}), mode=Mode.APPEND)
    assert t.read_arrow_table()["a"].to_pylist() == [2, 3]


def test_write_arrow_into_pandas_holder() -> None:
    t = PandasTabular()
    t.write_table(pa.table({"a": [1, 2, 3]}))
    assert isinstance(t.frame, pd.DataFrame)
    assert t.read_arrow_table().num_rows == 3
