# test_polars_cast.py
import datetime
import zoneinfo

import pyarrow as pa
import pytest

polars = pytest.importorskip("polars")
import polars as pl  # noqa: F401

from yggdrasil.polars.cast import (
    cast_polars_array,
    cast_polars_dataframe,
    polars_dataframe_to_arrow_table,
    arrow_table_to_polars_dataframe, arrow_type_to_polars_type, polars_type_to_arrow_type, arrow_field_to_polars_field,
    polars_field_to_arrow_field,
)
from yggdrasil.types.cast.cast_options import CastOptions
from yggdrasil.types.cast.arrow_cast import arrow_schema_to_field
from yggdrasil.types import convert


# ---------------------------------------------------------------------------
# Series casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_series_struct_child_defaults_and_missing_added():
    series = pl.Series("payload", [{"value": 1}, {"value": None}])

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

    opts = CastOptions.safe_init(target_field=target_field)
    casted = cast_polars_array(series, opts)
    evaluated = (
        pl.DataFrame({"payload": series})
        .with_columns(casted)
        .get_column("payload")
    )

    assert isinstance(evaluated, pl.Series)
    assert evaluated.dtype.fields[0].name == "value"
    assert evaluated.to_list() == [
        {"value": 1, "label": ""},
        {"value": 0, "label": ""},
    ]


def test_cast_polars_series_list_of_structs_preserves_null_lists():
    series = pl.Series("items", [[{"count": 1, "timestamp": "2025-12-10 10:00:00"}], [{"count": None}], None])

    target_field = pa.field(
        "items",
        pa.list_(
            pa.field(
                "item",
                pa.struct([
                    pa.field("count", pa.int64(), nullable=False),
                    pa.field("timestamp", pa.timestamp("us", "UTC"), nullable=False)
                ]),
                nullable=False,
            )
        ),
        nullable=False,
    )

    casted = cast_polars_array(series, CastOptions.safe_init(target_field=target_field))

    assert isinstance(casted, pl.Series)
    assert casted.dtype.inner == pl.Struct([
        pl.Field("count", pl.Int64),
        pl.Field("timestamp", pl.Datetime("us", "UTC"))
    ])
    assert casted.to_list() == [
        [{"count": 1, 'timestamp': datetime.datetime(2025, 12, 10, 10, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))}],
        [{"count": 0, 'timestamp': datetime.datetime(1970, 1, 1, 0, 0, tzinfo=zoneinfo.ZoneInfo(key='UTC'))}],
        [],
    ]


# ---------------------------------------------------------------------------
# DataFrame casting tests
# ---------------------------------------------------------------------------

def test_cast_polars_dataframe_nested_schema_and_defaults():
    df = pl.DataFrame(
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
    df = pl.DataFrame({"a": [1], "keep": [9]})

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

def test_polars_dataframe_to_arrow_table_with_cast_options():
    df = pl.DataFrame({"vals": [[1, 2], [None, 3]]})

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
    assert isinstance(df, pl.DataFrame)
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


# ---------------------------------------------------------------------------
# Integration tests using convert(...)
# ---------------------------------------------------------------------------

def test_convert_polars_dataframe_to_arrow_table_with_schema_hint():
    df = pl.DataFrame({"amount": [1, None]})
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

    df = convert(rbr, pl.DataFrame, options=opts)

    assert df.columns == ["a"]
    assert df["a"].to_list() == [1, 0, 3]



def test_arrow_to_polars_primitive_bool_int_string():
    assert arrow_type_to_polars_type(pa.bool_()) == polars.Boolean()

    assert arrow_type_to_polars_type(pa.int64()) == polars.Int64()
    assert arrow_type_to_polars_type(pa.uint32()) == polars.UInt32()

    # Arrow string/large_string -> Polars Utf8
    assert arrow_type_to_polars_type(pa.string()) == polars.Utf8()
    assert arrow_type_to_polars_type(pa.large_string()) == polars.Utf8()


def test_arrow_to_polars_list_and_struct():
    inner = pa.int64()
    list_type = pa.list_(inner)
    pl_list = arrow_type_to_polars_type(list_type)
    assert isinstance(pl_list, polars.List)
    assert pl_list.inner == polars.Int64()

    struct_type = pa.struct(
        [
            pa.field("a", pa.int32()),
            pa.field("b", pa.string()),
        ]
    )
    pl_struct = arrow_type_to_polars_type(struct_type)
    # Struct dtype
    assert isinstance(pl_struct, polars.Struct)
    # Fields should map correctly
    field_names = [f.name for f in pl_struct.fields]
    assert field_names == ["a", "b"]
    dtypes = [f.dtype for f in pl_struct.fields]
    assert dtypes[0] == polars.Int32()
    assert dtypes[1] == polars.Utf8()


def test_arrow_to_polars_timestamp_and_duration():
    ts = pa.timestamp("us", tz="UTC")
    pl_ts = arrow_type_to_polars_type(ts)
    assert isinstance(pl_ts, polars.Datetime)
    assert pl_ts.time_unit == "us"
    assert pl_ts.time_zone == "UTC"

    dur = pa.duration("ms")
    pl_dur = arrow_type_to_polars_type(dur)
    assert isinstance(pl_dur, polars.Duration)
    assert pl_dur.time_unit == "ms"


def test_arrow_to_polars_map_type_is_list_of_struct():
    # Arrow map(key_type, item_type)
    m = pa.map_(pa.string(), pa.int64())
    pl_dtype = arrow_type_to_polars_type(m)

    # Represented as List(Struct(key, value))
    assert isinstance(pl_dtype, polars.List)
    inner = pl_dtype.inner
    assert isinstance(inner, polars.Struct)

    field_names = [f.name for f in inner.fields]
    assert field_names == ["key", "value"]
    key_field, value_field = inner.fields
    assert key_field.dtype == polars.Utf8()
    assert value_field.dtype == polars.Int64()


def test_arrow_field_to_polars_field_and_back_roundtrip():
    arrow_field = pa.field("x", pa.int64(), nullable=False)

    pl_field = arrow_field_to_polars_field(arrow_field)
    # Depending on Polars version, this is either pl.Field or (name, dtype)
    if hasattr(polars, "Field") and isinstance(pl_field, polars.Field):
        assert pl_field.name == "x"
        assert pl_field.dtype == polars.Int64()
        pl_repr = pl_field
    else:
        name, dtype = pl_field
        assert name == "x"
        assert dtype == polars.Int64()
        pl_repr = (name, dtype)

    arrow_field2 = polars_field_to_arrow_field(pl_repr)
    assert isinstance(arrow_field2, pa.Field)
    assert arrow_field2.name == "x"
    # We don't assert nullable equality strictly, since implementation
    # defaults to nullable=True
    assert arrow_field2.type == pa.int64()


def test_polars_to_arrow_primitive_roundtrip():
    for pl_type, arrow_type in [
        (polars.Boolean, pa.bool_()),
        (polars.Int64, pa.int64()),
        (polars.UInt32, pa.uint32()),
        (polars.Float64, pa.float64()),
        (polars.Utf8, pa.large_string()),
        (polars.Date, pa.date32()),
    ]:
        # class
        assert polars_type_to_arrow_type(pl_type) == arrow_type
        # instance
        assert polars_type_to_arrow_type(pl_type()) == arrow_type


def test_polars_to_arrow_list_and_struct():
    # List<Int64>
    pl_list = polars.List(polars.Int64)
    arrow_list = polars_type_to_arrow_type(pl_list)
    assert pa.types.is_list(arrow_list)
    assert arrow_list.value_type == pa.int64()

    # Struct with fields
    Field = getattr(polars, "Field", None)
    if Field is not None:
        pl_struct = polars.Struct(
            [Field("a", polars.Int32), Field("b", polars.Utf8)]
        )
    else:
        pl_struct = polars.Struct({"a": polars.Int32, "b": polars.Utf8})

    arrow_struct = polars_type_to_arrow_type(pl_struct)
    assert pa.types.is_struct(arrow_struct)
    assert [f.name for f in arrow_struct] == ["a", "b"]
    assert arrow_struct.field("a").type == pa.int32()
    assert arrow_struct.field("b").type == pa.large_string()


def test_polars_to_arrow_datetime_and_duration():
    pl_dt = polars.Datetime(time_unit="ms", time_zone="UTC")
    arrow_dt = polars_type_to_arrow_type(pl_dt)
    assert pa.types.is_timestamp(arrow_dt)
    assert arrow_dt.unit == "ms"
    assert arrow_dt.tz == "UTC"

    pl_dur = polars.Duration(time_unit="us")
    arrow_dur = polars_type_to_arrow_type(pl_dur)
    assert pa.types.is_duration(arrow_dur)
    assert arrow_dur.unit == "us"


def test_arrow_to_polars_unsupported_raises():
    dec = pa.decimal128(10, 2)
    with pytest.raises(TypeError):
        arrow_type_to_polars_type(dec)


def test_polars_to_arrow_unsupported_raises():
    class FakeDtype:
        pass

    with pytest.raises(TypeError):
        polars_type_to_arrow_type(FakeDtype())
