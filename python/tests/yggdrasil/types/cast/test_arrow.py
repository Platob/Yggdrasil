from dataclasses import dataclass
import datetime as dt

# test_arrow_cast.py
import pyarrow as pa
import pytest

from yggdrasil.types import convert
from yggdrasil.types.cast.arrow_cast import (
    ArrowCastOptions,
    cast_arrow_array,
    cast_arrow_table,
    cast_arrow_batch,
    cast_arrow_record_batch_reader,
    default_arrow_array,
    table_to_record_batch,
    record_batch_to_table,
    table_to_record_batch_reader,
    record_batch_reader_to_table,
    record_batch_to_record_batch_reader,
    record_batch_reader_to_record_batch,
    DEFAULT_CAST_OPTIONS,
    pylist_to_arrow_table,
    pylist_to_record_batch,
)


# ---------------------------------------------------------------------------
# ArrowCastOptions tests
# ---------------------------------------------------------------------------

def test_arrow_cast_options_check_arg_none_returns_default():
    opts = ArrowCastOptions.check_arg(None)
    # Should not allocate a new object
    assert opts == DEFAULT_CAST_OPTIONS


def test_arrow_cast_options_check_arg_with_dtype_sets_target_field():
    dtype = pa.int32()
    opts = ArrowCastOptions.check_arg(dtype)

    assert isinstance(opts, ArrowCastOptions)
    assert opts.target_field is not None
    assert opts.target_field.type == dtype
    assert opts.target_field.name == "int32"


# ---------------------------------------------------------------------------
# default_arrow_python_value / default_arrow_array
# ---------------------------------------------------------------------------
def test_default_arrow_array_nullable_vs_non_nullable():
    field_nullable = pa.field("x", pa.int32(), nullable=True)
    field_not_nullable = pa.field("y", pa.int32(), nullable=False)

    arr_null = default_arrow_array(field_nullable, length=3)
    assert isinstance(arr_null, pa.Array)
    assert arr_null.type == pa.int32()
    assert arr_null.null_count == 3

    arr_non_null = default_arrow_array(field_not_nullable, length=2)
    assert list(arr_non_null.to_pylist()) == [0, 0]


# ---------------------------------------------------------------------------
# cast_arrow_array tests
# ---------------------------------------------------------------------------

def test_cast_arrow_array_simple_numeric_cast():
    arr = pa.array([1, 2, None], type=pa.int32())
    target_field = pa.field("x", pa.float64(), nullable=True)

    opts = ArrowCastOptions.__safe_init__(target_field=target_field)
    casted = cast_arrow_array(arr, opts)

    assert isinstance(casted, pa.Array)
    assert casted.type == pa.float64()
    assert casted.to_pylist() == [1.0, 2.0, None]


def test_cast_arrow_array_fill_non_nullable_defaults():
    arr = pa.array([1, None, 3], type=pa.int32())
    target_field = pa.field("x", pa.int32(), nullable=False)

    opts = ArrowCastOptions.__safe_init__(target_field=target_field)
    casted = cast_arrow_array(arr, opts)

    assert casted.type == pa.int32()
    # null should be replaced by 0 (default for integer)
    assert casted.to_pylist() == [1, 0, 3]


def test_cast_arrow_array_safe_primitive_cast_enforced():
    arr = pa.array([128], type=pa.int32())
    target_field = pa.field("x", pa.int8(), nullable=True)

    opts = ArrowCastOptions.__safe_init__(target_field=target_field, safe=True)

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_array(arr, opts)


def test_cast_arrow_array_chunked_array_roundtrip():
    arr1 = pa.array([1, 2], type=pa.int32())
    arr2 = pa.array([3, 4], type=pa.int32())
    chunked = pa.chunked_array([arr1, arr2])

    target_field = pa.field("x", pa.int64(), nullable=False)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    casted = cast_arrow_array(chunked, opts)
    assert isinstance(casted, pa.ChunkedArray)
    assert casted.type == pa.int64()
    assert casted.to_pylist() == [1, 2, 3, 4]


def test_cast_arrow_array_struct_add_missing_field_defaults():
    struct_type_source = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
        ]
    )
    struct_type_target = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),  # new non-nullable field
        ]
    )

    arr = pa.array(
        [
            {"a": 1},
            {"a": None},
        ],
        type=struct_type_source,
    )

    target_field = pa.field("root", struct_type_target, nullable=False)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field, add_missing_columns=True)

    casted = cast_arrow_array(arr, opts)

    assert casted.type == struct_type_target
    result = casted.to_pylist()
    # b should be defaulted to 0 (non-nullable int)
    assert result == [
        {"a": 1, "b": 0},
        {"a": None, "b": 0},
    ]


def test_cast_arrow_array_map_to_struct_case_insensitive_and_defaults():
    map_type = pa.map_(pa.string(), pa.int32())
    arr = pa.array([
        {"A": 1},
        {"a": None},
    ], type=map_type)

    struct_type_target = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )
    target_field = pa.field("root", struct_type_target, nullable=False)

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_field,
        strict_match_names=False,
    )

    casted = cast_arrow_array(arr, opts)

    assert casted.type == struct_type_target
    assert casted.to_pylist() == [
        {"a": 1, "b": 0},
        {"a": None, "b": 0},
    ]


def test_cast_arrow_array_struct_to_map_preserves_values_and_nulls():
    struct_type = pa.struct(
        [
            pa.field("x", pa.int32(), nullable=True),
            pa.field("y", pa.int32(), nullable=True),
        ]
    )
    arr = pa.array(
        [
            {"x": 1, "y": 2},
            None,
        ],
        type=struct_type,
    )

    target_field = pa.field("root", pa.map_(pa.string(), pa.int32()), nullable=True)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    casted = cast_arrow_array(arr, opts)

    assert pa.types.is_map(casted.type)
    assert casted.to_pylist() == [
        [("x", 1), ("y", 2)],
        None,
    ]


def test_cast_arrow_array_list_of_struct_add_missing_field_defaults():
    struct_type_source = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
        ]
    )
    struct_type_target = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )

    list_source = pa.list_(struct_type_source)
    list_target = pa.list_(struct_type_target)

    arr = pa.array(
        [
            [{"a": 1}],
            None,
            [{"a": None}],
        ],
        type=list_source,
    )

    target_field = pa.field("root", list_target, nullable=True)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    casted = cast_arrow_array(arr, opts)

    assert casted.type == list_target
    assert casted.to_pylist() == [
        [{"a": 1, "b": 0}],
        None,
        [{"a": None, "b": 0}],
    ]


# ---------------------------------------------------------------------------
# cast_arrow_table tests
# ---------------------------------------------------------------------------

def test_cast_arrow_table_case_insensitive_column_match():
    table = pa.table({"A": [1, 2]})

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema, strict_match_names=False)

    casted = cast_arrow_table(table, opts)

    assert casted.schema.names == ["a"]
    assert casted.schema.field("a").type == pa.int64()
    assert casted.column("a").to_pylist() == [1, 2]


def test_cast_arrow_table_missing_column_raises_when_add_missing_false():
    table = pa.table({"a": [1, 2]})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=True),
        ]
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        add_missing_columns=False,
        strict_match_names=True,
    )

    with pytest.raises(pa.ArrowInvalid, match="Missing column b"):
        cast_arrow_table(table, opts)


def test_cast_arrow_table_add_missing_column_with_defaults():
    table = pa.table({"a": [1, 2]})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )

    opts = ArrowCastOptions.__safe_init__(
        target_field=target_schema,
        add_missing_columns=True,
        strict_match_names=True,
    )

    casted = cast_arrow_table(table, opts)

    assert casted.schema.names == ["a", "b"]
    assert casted.column("a").to_pylist() == [1, 2]
    # b should be defaulted to 0's
    assert casted.column("b").to_pylist() == [0, 0]


# ---------------------------------------------------------------------------
# cast_arrow_batch tests
# ---------------------------------------------------------------------------

def test_cast_arrow_batch_matches_table_cast():
    batch = pa.record_batch({"A": [1, 2]})
    table = pa.Table.from_batches([batch])

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema, strict_match_names=False)

    casted_table = cast_arrow_table(table, opts)
    casted_batch = cast_arrow_batch(batch, opts)

    # Compare via Table representation
    table_from_batch = pa.Table.from_batches([casted_batch])
    assert table_from_batch.equals(casted_table)


# ---------------------------------------------------------------------------
# RecordBatchReader casting tests
# ---------------------------------------------------------------------------

def _make_reader_from_table(table: pa.Table) -> pa.RecordBatchReader:
    batches = table.to_batches()
    return pa.RecordBatchReader.from_batches(table.schema, batches)


def test_cast_arrow_record_batch_reader_to_table():
    table = pa.table({"A": [1, 2, 3]})
    reader = _make_reader_from_table(table)

    target_schema = pa.schema(
        [pa.field("a", pa.int64(), nullable=False)],
    )
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema, strict_match_names=False)

    casted_reader = cast_arrow_record_batch_reader(reader, opts)
    casted_table = pa.Table.from_batches(list(casted_reader))

    assert casted_table.schema.names == ["a"]
    assert casted_table.column("a").to_pylist() == [1, 2, 3]


def test_cast_arrow_record_batch_reader_no_target_schema_returns_same_reader():
    table = pa.table({"A": [1, 2, 3]})
    reader = _make_reader_from_table(table)

    # options with no target_field -> target_schema is None
    opts = ArrowCastOptions.__safe_init__(target_field=None)
    casted_reader = cast_arrow_record_batch_reader(reader, opts)

    # Exhaust to Table and compare
    original_table = pa.Table.from_batches(_make_reader_from_table(table))
    casted_table = pa.Table.from_batches(casted_reader)

    assert casted_table.equals(original_table)


# ---------------------------------------------------------------------------
# Cross-container helper tests
# ---------------------------------------------------------------------------

def test_table_to_record_batch_and_back():
    table = pa.table({"a": [1, 2]})

    # Identity schema cast
    opts = ArrowCastOptions.__safe_init__(target_field=table.schema)

    batch = table_to_record_batch(table, opts)
    assert isinstance(batch, pa.RecordBatch)

    table_roundtrip = record_batch_to_table(batch, opts)
    assert table_roundtrip.equals(cast_arrow_table(table, opts))


def test_table_to_record_batch_reader_and_back():
    table = pa.table({"a": [1, 2, 3]})
    opts = ArrowCastOptions.__safe_init__(target_field=table.schema)

    reader = table_to_record_batch_reader(table, opts)
    assert isinstance(reader, pa.RecordBatchReader)

    table_roundtrip = record_batch_reader_to_table(reader, opts)
    assert table_roundtrip.equals(cast_arrow_table(table, opts))


def test_record_batch_to_record_batch_reader_and_back():
    batch = pa.record_batch({"a": [1, 2, 3]})
    opts = ArrowCastOptions.__safe_init__(target_field=batch.schema)

    reader = record_batch_to_record_batch_reader(batch, opts)
    assert isinstance(reader, pa.RecordBatchReader)

    batch_roundtrip = record_batch_reader_to_record_batch(reader, opts)
    # Compare via tables for simplicity
    table_from_original = pa.Table.from_batches([batch])
    table_from_roundtrip = pa.Table.from_batches([batch_roundtrip])

    assert table_from_roundtrip.equals(cast_arrow_table(table_from_original, opts))


# ---------------------------------------------------------------------------
# convert(...) API tests (using the registered converters)
# ---------------------------------------------------------------------------

def test_convert_array_to_array_uses_cast_arrow_array():
    arr = pa.array([1, None, 3], type=pa.int32())

    # Register_converter(pa.Array, pa.Array)(cast_arrow_array) is in the module
    result = convert(arr, pa.Array)

    assert isinstance(result, pa.Array)
    assert result.type == pa.int32()  # same type hint
    # non-nullable is not enforced here because target_hint is only pa.Array,
    # so behavior is basically identity
    assert result.to_pylist() == [1, None, 3]


def test_convert_chunked_array_to_chunked_array():
    arr1 = pa.array([1, 2], type=pa.int32())
    arr2 = pa.array([3, 4], type=pa.int32())
    chunked = pa.chunked_array([arr1, arr2])

    result = convert(chunked, pa.ChunkedArray)

    assert isinstance(result, pa.ChunkedArray)
    assert result.type == pa.int32()
    assert result.to_pylist() == [1, 2, 3, 4]


def test_convert_table_to_table():
    table = pa.table({"a": [1, 2]})
    # Target hint is pa.Table, the existing converter will run cast_arrow_table.
    # Since no explicit ArrowCastOptions is passed through convert, this should
    # behave as identity (DEFAULT_CAST_OPTIONS has no target_field).
    result = convert(table, pa.Table)

    assert isinstance(result, pa.Table)
    assert result.equals(table)


def test_convert_table_to_record_batch():
    table = pa.table({"a": [1, 2, 3]})

    result = convert(table, pa.RecordBatch)

    assert isinstance(result, pa.RecordBatch)
    assert result.num_rows == 3
    assert result.schema.names == ["a"]
    assert result.column(0).to_pylist() == [1, 2, 3]


def test_convert_record_batch_to_table():
    batch = pa.record_batch({"a": [1, 2, 3]})

    result = convert(batch, pa.Table)

    assert isinstance(result, pa.Table)
    assert result.schema.names == ["a"]
    assert result.column("a").to_pylist() == [1, 2, 3]


def test_convert_table_to_record_batch_reader_and_back_via_convert():
    table = pa.table({"a": [10, 20, 30]})

    reader = convert(table, pa.RecordBatchReader)
    assert isinstance(reader, pa.RecordBatchReader)

    table_back = convert(reader, pa.Table)
    assert isinstance(table_back, pa.Table)
    assert table_back.column("a").to_pylist() == [10, 20, 30]


def test_convert_respects_arrow_target_hint():
    arr = pa.array([1, 2, 3], type=pa.int32())

    converted = convert(arr, pa.float64())

    assert isinstance(converted, pa.Array)
    assert converted.type == pa.float64()


def test_convert_propagates_arrow_source_and_target_hints():
    arr = pa.array([1, 2, 3], type=pa.int32())
    target_hint = pa.field("a", pa.int64(), nullable=False)
    source_hint = pa.field("b", pa.int32(), nullable=True)

    received: dict[str, object] = {}

    # Temporarily replace the array->array converter to observe the options.
    from yggdrasil.types.cast import registry

    original_converter = registry._registry[(pa.Array, pa.Array)]

    def _spy(value, cast_options):  # type: ignore[override]
        received["cast_options"] = cast_options
        return value

    registry._registry[(pa.Array, pa.Array)] = _spy

    try:
        result = convert(
            arr,
            target_hint,
            source_field=source_hint,
        )
    finally:
        registry._registry[(pa.Array, pa.Array)] = original_converter

    assert isinstance(result, pa.Array)

    opts = received["cast_options"]
    assert isinstance(opts, ArrowCastOptions)
    assert opts.target_field.type == pa.int64()


def test_convert_record_batch_to_record_batch_reader_and_back_via_convert():
    batch = pa.record_batch({"a": [5, 6, 7]})

    reader = convert(batch, pa.RecordBatchReader)
    assert isinstance(reader, pa.RecordBatchReader)

    batch_back = convert(reader, pa.RecordBatch)
    assert isinstance(batch_back, pa.RecordBatch)

    assert batch_back.schema.names == ["a"]
    assert batch_back.column(0).to_pylist() == [5, 6, 7]


# ---------------------------------------------------------------------------
# pylist converters tests
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    x: int
    y: str


def test_pylist_to_arrow_table_infers_dataclass_schema():
    data = [_Point(1, "a"), _Point(2, "b")]

    table = pylist_to_arrow_table(data)

    assert table.schema.names == ["x", "y"]
    assert table.schema.field("x").type == pa.int64()
    assert table.schema.field("y").type == pa.string()
    assert table.column("x").to_pylist() == [1, 2]
    assert table.column("y").to_pylist() == ["a", "b"]


def test_pylist_to_arrow_table_empty_uses_target_schema():
    target_schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=True),
        ]
    )
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema)

    table = pylist_to_arrow_table([], opts)

    assert table.num_rows == 0
    assert table.schema.equals(target_schema)
    assert table.column("a").type == pa.int64()
    assert table.column("b").type == pa.string()


def test_pylist_to_arrow_table_preserves_nullable_none_values():
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=True)])
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema)

    table = pylist_to_arrow_table([{"a": 1}, {"a": None}], opts)

    assert table.schema.equals(target_schema)
    assert table.column("a").to_pylist() == [1, None]


def test_pylist_to_arrow_table_fills_non_nullable_none_values():
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    opts = ArrowCastOptions.__safe_init__(target_field=target_schema)

    table = pylist_to_arrow_table([{"a": None}, {"a": 3}], opts)

    # None should be replaced with the default integer value (0)
    assert table.column("a").to_pylist() == [0, 3]


def test_pylist_to_arrow_table_handles_none_and_dataclass_rows():
    data = [None, _Point(1, "a")]

    table = pylist_to_arrow_table(data)

    assert table.schema.names == ["x", "y"]
    assert table.column("x").to_pylist() == [None, 1]
    assert table.column("y").to_pylist() == [None, "a"]


def test_pylist_to_arrow_table_all_none_rows_use_target_schema_defaults():
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])

    table = pylist_to_arrow_table([None], target_schema)

    assert table.schema.equals(target_schema)
    assert table.column("a").to_pylist() == [0]


def test_pylist_to_record_batch_and_reader_via_convert():
    target_field = pa.field("value", pa.int64(), nullable=False)
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    batch = pylist_to_record_batch([{"value": 1}, {"value": 2}], opts)
    assert isinstance(batch, pa.RecordBatch)
    assert batch.schema.field("value").type == pa.int64()
    assert batch.column(0).to_pylist() == [1, 2]

    reader = convert([{"value": 3}, {"value": 4}], pa.RecordBatchReader, options=opts)
    table = pa.Table.from_batches(list(reader))
    assert table.schema.field("value").type == pa.int64()
    assert table.column("value").to_pylist() == [3, 4]


def test_cast_arrow_array_default_datetime_formats_to_utc_timestamp():
    values = [
        # Zoned with fraction seconds (T / space)
        "2024-01-02T03:04:05.123456Z",     # %Y-%m-%dT%H:%M:%S.%f%z
        "2024-01-02 03:04:05.123456+0100", # %Y-%m-%d %H:%M:%S.%f%z

        # Zoned without fraction seconds (T / space)
        "2024-01-02 03:04Z",            # %Y-%m-%dT%H:%M:%S%z
        "2024-01-02 03:04:05+0100",        # %Y-%m-%d %H:%M:%S%z

        # Zoned without seconds (T / space)
        "2024-01-02T03:04+0100",           # %Y-%m-%dT%H:%M%z
        "2024-01-02 03:04+0100",           # %Y-%m-%d %H:%M%z

        # Naive with fraction seconds (T / space)
        "2024-01-02T03:04:05.123456",      # %Y-%m-%dT%H:%M:%S.%f
        "2024-01-02 03:04:05.123456",      # %Y-%m-%d %H:%M:%S.%f

        # Naive without fraction seconds (T / space)
        "2024-01-02T03:04:05",             # %Y-%m-%dT%H:%M:%S
        "2024-01-02 03:04:05",             # %Y-%m-%d %H:%M:%S

        # Naive without seconds
        "2024-01-02 03:04",                # %Y-%m-%d %H:%M

        # Date only
        "2024-01-02",                      # %Y-%m-%d

        # Null-ish
        "",                                # empty -> null
        None,                              # explicit null
    ]

    arr = pa.array(values, type=pa.string())

    target_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=True)

    # No explicit datetime_formats: should fall back to DEFAULT_DATETIME_FORMATS
    opts = ArrowCastOptions.__safe_init__(target_field=target_field)

    casted = cast_arrow_array(arr, opts)

    assert casted.type == pa.timestamp("us", tz="UTC")

    result = casted.to_pylist()

    # IMPORTANT: Polars interprets %f as fractional seconds at nanosecond
    # precision, then we downcast to microseconds. So ".123456" becomes
    # 123456 ns -> 123 µs -> ".000123".
    expected = [
        # Zoned + fraction
        dt.datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=dt.timezone.utc),
        # 03:04:05.123456+0100 -> 02:04:05.123456Z -> 123 ns -> 123 µs
        dt.datetime(2024, 1, 2, 2, 4, 5, 123456, tzinfo=dt.timezone.utc),

        # Zoned no fraction
        dt.datetime(2024, 1, 2, 3, 4, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 2, 2, 4, 5, tzinfo=dt.timezone.utc),

        # Zoned no seconds
        dt.datetime(2024, 1, 2, 2, 4, 0, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 2, 2, 4, 0, tzinfo=dt.timezone.utc),

        # Naive with fraction (treated as UTC, same ns→µs behavior)
        dt.datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=dt.timezone.utc),

        # Naive no fraction
        dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),
        dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc),

        # Naive no seconds
        dt.datetime(2024, 1, 2, 3, 4, 0, tzinfo=dt.timezone.utc),

        # Date only -> midnight UTC
        dt.datetime(2024, 1, 2, 0, 0, 0, tzinfo=dt.timezone.utc),

        None,
        None,
    ]

    result_iso = [r.isoformat() if r is not None else None for r in result]
    expected_iso = [e.isoformat() if e is not None else None for e in expected]

    assert result_iso == expected_iso