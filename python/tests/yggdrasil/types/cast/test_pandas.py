# tests/types/test_pandas_cast.py
import math

import pytest

pa = pytest.importorskip("pyarrow")
pandas = pytest.importorskip("pandas")

from yggdrasil.types.cast.registry import convert
from yggdrasil.types.cast.pandas import (
    cast_pandas_series,
    cast_pandas_dataframe,
    arrow_array_to_pandas_series,
    arrow_table_to_pandas_dataframe,
    record_batch_reader_to_pandas_dataframe,
    pandas_series_to_arrow_array,
    pandas_dataframe_to_arrow_table,
    pandas_dataframe_to_record_batch_reader,
)


def test_cast_pandas_series_via_function_identity_int():
    s = pandas.Series([1, 2, 3], name="col")
    out = cast_pandas_series(s, options=None)

    assert isinstance(out, pandas.Series)
    assert out.name == s.name
    assert out.index.equals(s.index)
    assert out.tolist() == [1, 2, 3]


def test_cast_pandas_dataframe_via_function_identity_basic():
    df = pandas.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = cast_pandas_dataframe(df, options=None)

    assert isinstance(out, pandas.DataFrame)
    # same columns and values, index preserved
    pandas.testing.assert_frame_equal(out, df)


def test_convert_pandas_series_to_pandas_series_registered():
    s = pandas.Series([10, 20, 30], name="n")

    out = convert(s, pandas.Series)

    assert isinstance(out, pandas.Series)
    assert out.tolist() == [10, 20, 30]
    assert out.name == s.name
    assert out.index.equals(s.index)


def test_convert_pandas_series_to_arrow_array_registered():
    s = pandas.Series([1, 2, None], name="n")

    arr = convert(s, pa.Array)

    assert isinstance(arr, pa.Array)
    assert arr.to_pylist() == [1, 2, None]


def test_convert_arrow_array_to_pandas_series_registered():
    arr = pa.array([1, 2, 3])

    s = convert(arr, pandas.Series)

    assert isinstance(s, pandas.Series)
    assert s.tolist() == [1, 2, 3]


def test_convert_pandas_dataframe_to_arrow_table_registered():
    df = pandas.DataFrame({"a": [1, 2], "b": ["u", "v"]})

    table = convert(df, pa.Table)

    assert isinstance(table, pa.Table)
    assert table.column("a").to_pylist() == [1, 2]
    assert table.column("b").to_pylist() == ["u", "v"]


def test_convert_arrow_table_to_pandas_dataframe_registered():
    df = pandas.DataFrame({"a": [1, 2], "b": ["u", "v"]})
    table = pa.Table.from_pandas(df, preserve_index=False)

    out = convert(table, pandas.DataFrame)

    assert isinstance(out, pandas.DataFrame)
    # Arrow may tweak dtypes but values/columns should match
    pandas.testing.assert_frame_equal(out.reset_index(drop=True),
                                      df.reset_index(drop=True))


def test_convert_pandas_dataframe_to_record_batch_reader_registered():
    df = pandas.DataFrame({"x": [1, 2, 3]})

    reader = convert(df, pa.RecordBatchReader)

    assert isinstance(reader, pa.RecordBatchReader)

    # Re-materialize and compare roundtrip
    batches = list(reader)
    table = pa.Table.from_batches(batches)
    df_roundtrip = table.to_pandas()

    pandas.testing.assert_frame_equal(
        df_roundtrip.reset_index(drop=True),
        df.reset_index(drop=True),
    )


def test_convert_record_batch_reader_to_pandas_dataframe_registered():
    df = pandas.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    out = convert(reader, pandas.DataFrame)

    assert isinstance(out, pandas.DataFrame)
    pandas.testing.assert_frame_equal(
        out.reset_index(drop=True),
        df.reset_index(drop=True),
    )



def test_arrow_array_to_pandas_series_direct_helper():
    arr = pa.array([1.5, 2.5, None])

    s = arrow_array_to_pandas_series(arr, cast_options=None)

    assert isinstance(s, pandas.Series)
    assert len(s) == 3
    assert s.iloc[0] == 1.5
    assert s.iloc[1] == 2.5
    assert math.isnan(s.iloc[2])


def test_arrow_table_to_pandas_dataframe_direct_helper():
    df = pandas.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    table = pa.Table.from_pandas(df, preserve_index=False)

    out = arrow_table_to_pandas_dataframe(table, cast_options=None)

    assert isinstance(out, pandas.DataFrame)
    pandas.testing.assert_frame_equal(
        out.reset_index(drop=True),
        df.reset_index(drop=True),
    )


def test_record_batch_reader_to_pandas_dataframe_direct_helper():
    df = pandas.DataFrame({"a": [1, 2, 3]})
    table = pa.Table.from_pandas(df, preserve_index=False)
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())

    out = record_batch_reader_to_pandas_dataframe(reader, cast_options=None)

    assert isinstance(out, pandas.DataFrame)
    pandas.testing.assert_frame_equal(
        out.reset_index(drop=True),
        df.reset_index(drop=True),
    )
