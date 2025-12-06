# test_polars_cast.py

import pyarrow as pa
import pytest

polars = pytest.importorskip("polars")
import polars as pl  # noqa: F401

from yggdrasil.types.cast.polars_cast import (
    cast_polars_array,
    cast_polars_dataframe,
    polars_series_to_arrow_array,
    polars_dataframe_to_arrow_table,
    arrow_array_to_polars_series,
    arrow_table_to_polars_dataframe,
    polars_dataframe_to_record_batch_reader,
    record_batch_reader_to_polars_dataframe,
)
from yggdrasil.types.cast.arrow_cast import CastOptions, arrow_schema_to_field
from yggdrasil.types import convert


# ---------------------------------------------------------------------------
# Series casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_series_struct_child_defaults_and_missing_added():
    series = polars.Series("payload", [{"value": 1}, {"value": None}])

    target_field = pa.field(
        "payload",
        pa.struct(
            [
                pa.field("value", pa.int64(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        ),
        nullable=True,
    )

    opts = CastOptions.safe_init(target_field=target_field, add_missing_columns=True)
    casted = cast_polars_array(series, opts)
    evaluated = (
        polars.DataFrame({"payload": series})
        .with_columns(casted)
        .get_column("payload")
    )

    assert isinstance(evaluated, polars.Series)
    assert evaluated.dtype.fields[0].name == "value"
    assert evaluated.to_list() == [
        {"value": 1, "label": ""},
        {"value": 0, "label": ""},
    ]


def test_cast_polars_series_list_of_structs_preserves_null_lists():
    series = polars.Series("items", [[{"count": 1}], [{"count": None}], None])

    target_field = pa.field(
        "items",
        pa.list_(
            pa.field(
                "item",
                pa.struct([pa.field("count", pa.int64(), nullable=False)]),
                nullable=True,
            )
        ),
        nullable=True,
    )

    casted = cast_polars_array(series, CastOptions.safe_init(target_field=target_field))

    assert isinstance(casted, polars.Series)
    assert casted.dtype.inner == polars.Struct([polars.Field("count", polars.Int64)])
    assert casted.to_list() == [
        [{"count": 1}],
        [{"count": 0}],
        None,
    ]


# ---------------------------------------------------------------------------
# DataFrame casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_dataframe_nested_schema_and_defaults():
    df = polars.DataFrame(
        {
            "Meta": [{"id": 1}, {"id": 2}],
            "payload": [{"score": "7"}, {"score": None}],
            "extra": [True, False],
        }
    )

    target_schema = pa.schema(
        [
            pa.field(
                "meta",
                pa.struct(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("tag", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "payload",
                pa.struct([pa.field("score", pa.int64(), nullable=False)]),
                nullable=False,
            ),
            pa.field("extra", pa.bool_(), nullable=True),
        ]
    )

    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema),
        add_missing_columns=True,
        allow_add_columns=False,
        strict_match_names=False,
    )

    casted = cast_polars_dataframe(df, opts)

    assert casted.columns == ["meta", "payload", "extra"]
    assert casted["meta"].to_list() == [
        {"id": 1, "tag": ""},
        {"id": 2, "tag": ""},
    ]
    assert casted["payload"].to_list() == [
        {"score": 7},
        {"score": 0},
    ]
    assert casted["extra"].to_list() == [True, False]


def test_cast_polars_dataframe_preserves_extras_when_allowed():
    df = polars.DataFrame({"a": [1], "keep": [9]})

    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(pa.schema([pa.field("a", pa.int64(), nullable=False)])),
        allow_add_columns=True,
    )

    casted = cast_polars_dataframe(df, opts)

    assert casted.columns == ["a", "keep"]
    assert casted["a"].to_list() == [1]
    assert casted["keep"].to_list() == [9]


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


def test_polars_dataframe_to_arrow_table_with_cast_options():
    df = polars.DataFrame({"vals": [[1, 2], [None, 3]]})

    target_schema = pa.schema(
        [
            pa.field(
                "vals",
                pa.list_(pa.field("item", pa.int64(), nullable=False)),
                nullable=False,
            )
        ]
    )

    opts = CastOptions.safe_init(target_field=arrow_schema_to_field(target_schema))
    table = polars_dataframe_to_arrow_table(df, opts)

    assert isinstance(table, pa.Table)
    assert table.schema == target_schema
    assert table.column("vals").to_pylist() == [[1, 2], [0, 3]]


def test_arrow_table_to_polars_dataframe_round_trip_structs():
    table = pa.table(
        {
            "data": pa.array(
                [
                    {"count": 1, "label": "x"},
                    {"count": None, "label": "y"},
                ],
                type=pa.struct(
                    [
                        pa.field("count", pa.int64(), nullable=False),
                        pa.field("label", pa.string(), nullable=False),
                    ]
                ),
            )
        }
    )

    df = arrow_table_to_polars_dataframe(
        table,
        CastOptions.safe_init(target_field=arrow_schema_to_field(table.schema)),
    )
    assert isinstance(df, polars.DataFrame)
    assert df["data"].to_list() == [
        {"count": 1, "label": "x"},
        {"count": None, "label": "y"},
    ]


# ---------------------------------------------------------------------------
# RecordBatchReader <-> Polars DataFrame
# ---------------------------------------------------------------------------

def _make_record_batch_reader_from_table(table: pa.Table) -> pa.RecordBatchReader:
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


def test_record_batch_reader_to_polars_dataframe_with_cast():
    table = pa.table({"COUNT": [1, None, 3]})
    rbr = _make_record_batch_reader_from_table(table)

    target_schema = pa.schema([pa.field("count", pa.int64(), nullable=False)])
    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema), strict_match_names=False
    )

    df = record_batch_reader_to_polars_dataframe(rbr, opts)

    assert df.columns == ["count"]
    assert df["count"].to_list() == [1, 0, 3]


def test_polars_dataframe_to_record_batch_reader_roundtrip():
    df = polars.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    rbr = polars_dataframe_to_record_batch_reader(df)
    assert isinstance(rbr, pa.RecordBatchReader)

    df2 = record_batch_reader_to_polars_dataframe(rbr)
    assert df2.to_dict(as_series=False) == df.to_dict(as_series=False)


# ---------------------------------------------------------------------------
# Integration tests using convert(...)
# ---------------------------------------------------------------------------

def test_convert_polars_dataframe_to_arrow_table_with_schema_hint():
    df = polars.DataFrame({"amount": [1, None]})
    target_schema = pa.schema([pa.field("amount", pa.int64(), nullable=False)])

    table = convert(
        df,
        pa.Table,
        options=CastOptions.safe_init(target_field=arrow_schema_to_field(target_schema)),
    )

    assert isinstance(table, pa.Table)
    assert table.schema == target_schema
    assert table.column("amount").to_pylist() == [1, 0]


def test_convert_arrow_record_batch_reader_to_polars_with_cast():
    table = pa.table({"A": [1, None, 3]})
    rbr = _make_record_batch_reader_from_table(table)

    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema), strict_match_names=False
    )

    df = convert(rbr, polars.DataFrame, options=opts)

    assert df.columns == ["a"]
    assert df["a"].to_list() == [1, 0, 3]
