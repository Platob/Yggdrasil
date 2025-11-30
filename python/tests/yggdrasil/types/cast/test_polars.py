# test_polars_cast.py

import pyarrow as pa
import pytest

polars = pytest.importorskip("polars")
import polars as pl  # noqa: F401

from yggdrasil.types.cast.polars import (
    cast_polars_series,
    cast_polars_dataframe,
    polars_series_to_arrow_array,
    polars_dataframe_to_arrow_table,
    arrow_array_to_polars_series,
    arrow_table_to_polars_dataframe,
    polars_dataframe_to_record_batch_reader,
    record_batch_reader_to_polars_dataframe,
)
from yggdrasil.types.cast.arrow import ArrowCastOptions
from yggdrasil.types import convert


# ---------------------------------------------------------------------------
# Series casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_series_simple_numeric_cast():
    s = polars.Series("a", [1, 2, 3])

    target_field = pa.field("a", pa.float64(), nullable=True)
    opts = ArrowCastOptions(target_field=target_field)

    casted = cast_polars_series(s, opts)

    assert isinstance(casted, polars.Series)
    assert casted.name == "a"
    # dtype will be Polars Float64-ish
    assert "float" in str(casted.dtype).lower()
    assert casted.to_list() == [1.0, 2.0, 3.0]


def test_cast_polars_series_fill_non_nullable_defaults():
    s = polars.Series("a", [1, None, 3])

    target_field = pa.field("a", pa.int64(), nullable=False)
    opts = ArrowCastOptions(target_field=target_field)

    casted = cast_polars_series(s, opts)

    # null should be replaced with default_arrow_python_value(int64) -> 0
    assert casted.to_list() == [1, 0, 3]
    # ensure integer-ish dtype
    assert "int" in str(casted.dtype).lower()


def test_cast_polars_series_schema_target_uses_first_field():
    s = polars.Series("a", [1, 2, 3])

    schema = pa.schema(
        [pa.field("a", pa.string(), nullable=True)]
    )
    opts = ArrowCastOptions(target_field=schema)

    casted = cast_polars_series(s, opts)

    values = casted.to_list()

    # Implementation currently returns a struct series with a single field "a"
    # represented as dicts: {"a": "1"}, {"a": "2"}, ...
    if values and isinstance(values[0], dict):
        assert all(set(v.keys()) == {"a"} for v in values)
        extracted = [v["a"] for v in values]
    else:
        # Fallback: if Polars ever returns a flat series instead
        extracted = [str(v) for v in values]

    assert extracted == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# DataFrame casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_dataframe_basic_schema_cast():
    df = polars.DataFrame({"A": [1, 2], "B": ["x", "y"]})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=True),
        ]
    )
    opts = ArrowCastOptions(
        target_field=target_schema,
        strict_match_names=False,  # allow case-insensitive matching
    )

    casted = cast_polars_dataframe(df, opts)

    assert isinstance(casted, polars.DataFrame)
    assert casted.columns == ["a", "b"]
    assert casted["a"].to_list() == [1, 2]
    assert casted["b"].to_list() == ["x", "y"]


def test_cast_polars_dataframe_missing_column_add_missing_false_raises():
    df = polars.DataFrame({"a": [1, 2]})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=True),
        ]
    )

    opts = ArrowCastOptions(
        target_field=target_schema,
        add_missing_columns=False,
        strict_match_names=True,
    )

    with pytest.raises(pa.ArrowInvalid, match="Missing column b"):
        cast_polars_dataframe(df, opts)


def test_cast_polars_dataframe_add_missing_column_with_defaults():
    df = polars.DataFrame({"a": [1, 2]})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.string(), nullable=False),
        ]
    )

    opts = ArrowCastOptions(
        target_field=target_schema,
        add_missing_columns=True,
        strict_match_names=True,
    )

    casted = cast_polars_dataframe(df, opts)

    assert casted.columns == ["a", "b"]
    assert casted["a"].to_list() == [1, 2]
    # default_arrow_python_value(string) -> ""
    assert casted["b"].to_list() == ["", ""]


def test_cast_polars_dataframe_allow_add_columns_true_keeps_extras():
    df = polars.DataFrame({"a": [1], "extra": [42]})

    target_schema = pa.schema(
        [pa.field("a", pa.int32(), nullable=True)],
    )

    opts = ArrowCastOptions(
        target_field=target_schema,
        allow_add_columns=True,
        strict_match_names=True,
    )

    casted = cast_polars_dataframe(df, opts)

    assert casted.columns == ["a", "extra"]
    assert casted["a"].to_list() == [1]
    assert casted["extra"].to_list() == [42]


def test_cast_polars_dataframe_allow_add_columns_false_drops_extras():
    df = polars.DataFrame({"a": [1], "extra": [42]})

    target_schema = pa.schema(
        [pa.field("a", pa.int32(), nullable=True)],
    )

    opts = ArrowCastOptions(
        target_field=target_schema,
        allow_add_columns=False,
        strict_match_names=True,
    )

    casted = cast_polars_dataframe(df, opts)

    assert casted.columns == ["a"]
    assert casted["a"].to_list() == [1]


# ---------------------------------------------------------------------------
# Polars <-> Arrow conversions (direct helper tests)
# ---------------------------------------------------------------------------

def test_polars_series_to_arrow_array_and_back():
    s = polars.Series("a", [1, 2, 3])

    arr = polars_series_to_arrow_array(s)
    assert isinstance(arr, pa.Array)
    assert arr.to_pylist() == [1, 2, 3]

    s2 = arrow_array_to_polars_series(arr)
    assert isinstance(s2, polars.Series)
    assert s2.to_list() == [1, 2, 3]


def test_polars_dataframe_to_arrow_table_and_back():
    df = polars.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    table = polars_dataframe_to_arrow_table(df)
    assert isinstance(table, pa.Table)
    assert table.column("a").to_pylist() == [1, 2]

    df2 = arrow_table_to_polars_dataframe(table)
    assert isinstance(df2, polars.DataFrame)
    assert df2.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_polars_dataframe_to_arrow_table_with_cast_options():
    df = polars.DataFrame({"a": [1, 2]})

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions(target_field=target_schema)

    table = polars_dataframe_to_arrow_table(df, opts)
    assert table.schema.field("a").type == pa.int64()
    assert table.column("a").to_pylist() == [1, 2]


# ---------------------------------------------------------------------------
# RecordBatchReader <-> Polars DataFrame
# ---------------------------------------------------------------------------

def _make_record_batch_reader_from_table(table: pa.Table) -> pa.RecordBatchReader:
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


def test_polars_dataframe_to_record_batch_reader_roundtrip():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    rbr = polars_dataframe_to_record_batch_reader(df)
    assert isinstance(rbr, pa.RecordBatchReader)

    # Convert back to Polars
    df2 = record_batch_reader_to_polars_dataframe(rbr)

    assert isinstance(df2, polars.DataFrame)
    assert df2.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_record_batch_reader_to_polars_dataframe_with_arrow_cast():
    table = pa.table({"A": [1, 2, 3]})
    rbr = _make_record_batch_reader_from_table(table)

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions(target_field=target_schema, strict_match_names=False)

    df = record_batch_reader_to_polars_dataframe(rbr, opts)

    assert df.columns == ["a"]
    assert df["a"].to_list() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Integration tests using convert(...)
# ---------------------------------------------------------------------------

def test_convert_polars_series_to_arrow_array_and_back():
    s = polars.Series("a", [1, 2, 3])

    arr = convert(s, pa.Array)
    assert isinstance(arr, pa.Array)
    assert arr.to_pylist() == [1, 2, 3]

    s2 = convert(arr, polars.Series)
    assert isinstance(s2, polars.Series)
    assert s2.to_list() == [1, 2, 3]


def test_convert_polars_dataframe_to_arrow_table_and_back():
    df = polars.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    table = convert(df, pa.Table)
    assert isinstance(table, pa.Table)
    assert table.column("a").to_pylist() == [1, 2]

    df2 = convert(table, polars.DataFrame)
    assert isinstance(df2, polars.DataFrame)
    assert df2.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_convert_polars_dataframe_to_record_batch_reader_and_back():
    df = polars.DataFrame({"a": [10, 20, 30]})

    rbr = convert(df, pa.RecordBatchReader)
    assert isinstance(rbr, pa.RecordBatchReader)

    df2 = convert(rbr, polars.DataFrame)
    assert isinstance(df2, polars.DataFrame)
    assert df2.to_dict(as_series=False) == df.to_dict(as_series=False)


def test_cast_polars_dataframe_with_arrow_schema_cast_direct():
    """
    Directly test schema-based casting using cast_polars_dataframe, since convert(...)
    may treat same-type conversions as identity and ignore cast_options.
    """
    df = polars.DataFrame({"A": [1, 2, 3]})

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )

    opts = ArrowCastOptions(target_field=target_schema, strict_match_names=False)
    df_cast = cast_polars_dataframe(df, opts)

    assert df_cast.columns == ["a"]
    assert df_cast["a"].to_list() == [1, 2, 3]


def test_convert_arrow_record_batch_reader_to_polars_with_cast():
    table = pa.table({"A": [1, None, 3]})
    rbr = _make_record_batch_reader_from_table(table)

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions(target_field=target_schema, strict_match_names=False)

    df = convert(rbr, polars.DataFrame, options=opts)

    assert df.columns == ["a"]
    # null should be filled with 0 by Arrow-side cast before Polars conversion
    assert df["a"].to_list() == [1, 0, 3]
