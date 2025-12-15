from dataclasses import dataclass
from typing import Union

import pyarrow as pa
import pytest

from yggdrasil.types import convert
from yggdrasil.types.cast.arrow_cast import (
    arrow_schema_to_field,
    cast_arrow_array,
    cast_arrow_tabular,
    cast_arrow_record_batch_reader,
    default_arrow_array,
    table_to_record_batch,
    record_batch_to_table,
    table_to_record_batch_reader,
    record_batch_reader_to_table,
    record_batch_to_record_batch_reader,
    record_batch_reader_to_record_batch,
    pylist_to_record_batch,
)
from yggdrasil.types.cast.cast_options import CastOptions


@pytest.fixture
def ensure_any_to_arrow_scalar_has_options(monkeypatch):
    from yggdrasil.types.cast import arrow_cast as ac

    original = ac.any_to_arrow_scalar

    def _wrapped(value, options=None):
        if options is None:
            options = CastOptions()
        return original(value, options)

    monkeypatch.setattr(ac, "any_to_arrow_scalar", _wrapped)
    yield


# ---------------------------------------------------------------------------
# default_arrow_array / default_arrow_python_value
# ---------------------------------------------------------------------------


def test_default_arrow_array_handles_nullable_and_required_fields():
    nullable_field = pa.field("nullable", pa.int32(), nullable=True)
    required_field = pa.field("required", pa.int32(), nullable=False)

    nullable_array = default_arrow_array(nullable_field.type, nullable=True, size=3)
    assert isinstance(nullable_array, pa.Array)
    assert nullable_array.null_count == 3

    required_array = default_arrow_array(required_field.type, nullable=False, size=2)
    assert required_array.to_pylist() == [0, 0]


# ---------------------------------------------------------------------------
# cast_arrow_array
# ---------------------------------------------------------------------------


def test_cast_arrow_array_numeric_promotions_and_null_replacement():
    source = pa.array([1, None, 3], type=pa.int32())
    target_field = pa.field("value", pa.float64(), nullable=False)
    opts = CastOptions.safe_init(target_field=target_field)

    casted = cast_arrow_array(source, opts)

    assert casted.type == pa.float64()
    # Non-nullable float should replace None with 0.0
    assert casted.to_pylist() == [1.0, 0.0, 3.0]


def test_cast_arrow_array_enforces_safe_casts():
    source = pa.array([128], type=pa.int32())
    target_field = pa.field("value", pa.int8(), nullable=True)

    opts = CastOptions.safe_init(target_field=target_field, safe=True)

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_array(source, opts)


def test_cast_arrow_array_chunked_array_roundtrip():
    chunked = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])
    opts = CastOptions.safe_init(target_field=pa.field("value", pa.int64(), nullable=False))

    casted = cast_arrow_array(chunked, opts)

    assert isinstance(casted, pa.ChunkedArray)
    assert casted.type == pa.int64()
    assert casted.to_pylist() == [1, 2, 3, 4]


def test_cast_arrow_array_struct_adds_missing_nested_fields():
    source_struct = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field(
                "nested",
                pa.struct([
                    pa.field("x", pa.string(), nullable=True),
                ]),
                nullable=True,
            ),
        ]
    )
    target_struct = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field(
                "nested",
                pa.struct([
                    pa.field("x", pa.string(), nullable=True),
                    pa.field("y", pa.int64(), nullable=False),
                ]),
                nullable=False,
            ),
        ]
    )

    array = pa.array(
        [
            {"a": 1, "nested": {"x": "hello"}},
            {"a": None, "nested": None},
        ],
        type=source_struct,
    )

    opts = CastOptions.safe_init(
        target_field=pa.field("root", target_struct, nullable=False),
        add_missing_columns=True,
    )

    casted = cast_arrow_array(array, opts)

    assert casted.type == target_struct
    assert casted.to_pylist() == [
        {"a": 1, "nested": {"x": "hello", "y": 0}},
        {"a": None, "nested": {"x": "", "y": 0}},
    ]


def test_cast_arrow_array_map_to_struct_defaults_and_nulls():
    map_type = pa.map_(pa.string(), pa.int32())
    array = pa.array([{"a": 1}, None, {"a": None}], type=map_type)

    target_struct = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )
    opts = CastOptions.safe_init(
        target_field=pa.field("root", target_struct, nullable=True),
        strict_match_names=False,
    )

    casted = cast_arrow_array(array, opts)

    assert casted.type == target_struct
    assert casted.to_pylist() == [
        {"a": 1, "b": 0},
        None,
        {"a": None, "b": 0},
    ]


def test_cast_arrow_array_list_of_structs_with_missing_fields():
    source_struct = pa.struct([pa.field("a", pa.int32(), nullable=True)])
    target_struct = pa.struct(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )

    source_list = pa.list_(source_struct)
    target_list = pa.list_(target_struct)

    array = pa.array(
        [
            [{"a": 1}],
            None,
            [{"a": None}],
        ],
        type=source_list,
    )

    opts = CastOptions.safe_init(target_field=pa.field("root", target_list, nullable=True))

    casted = cast_arrow_array(array, opts)

    assert casted.type == target_list
    assert casted.to_pylist() == [
        [{"a": 1, "b": 0}],
        None,
        [{"a": None, "b": 0}],
    ]


# ---------------------------------------------------------------------------
# cast_arrow_table / cast_arrow_tabular
# ---------------------------------------------------------------------------


def test_cast_arrow_table_case_insensitive_column_match():
    table = pa.table({"A": [1, 2]})

    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema), strict_match_names=False
    )

    casted = cast_arrow_tabular(table, opts)

    assert casted.schema.names == ["a"]
    assert casted.column("a").to_pylist() == [1, 2]


def test_cast_arrow_table_adds_missing_columns_with_defaults():
    table = pa.table({"a": [1, 2]})
    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.int32(), nullable=False),
        ]
    )

    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema),
        add_missing_columns=True,
        strict_match_names=True,
    )

    casted = cast_arrow_tabular(table, opts)

    assert casted.schema.names == ["a", "b"]
    assert casted.column("a").to_pylist() == [1, 2]
    assert casted.column("b").to_pylist() == [0, 0]


def test_cast_arrow_tabular_record_batch_matches_table_behavior():
    batch = pa.record_batch({"A": [1, 2]})
    table = pa.Table.from_batches([batch])

    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema), strict_match_names=False
    )

    casted_table = cast_arrow_tabular(table, opts)
    casted_batch = cast_arrow_tabular(batch, opts)

    assert pa.Table.from_batches([casted_batch]).equals(casted_table)


# ---------------------------------------------------------------------------
# RecordBatchReader
# ---------------------------------------------------------------------------


def _make_reader_from_table(table: pa.Table) -> pa.RecordBatchReader:
    return pa.RecordBatchReader.from_batches(table.schema, table.to_batches())


def test_cast_arrow_record_batch_reader_applies_schema():
    table = pa.table({"A": [1, 2, 3]})
    reader = _make_reader_from_table(table)

    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])
    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema), strict_match_names=False
    )

    casted_reader = cast_arrow_record_batch_reader(reader, opts)
    casted_table = pa.Table.from_batches(list(casted_reader))

    assert casted_table.schema.names == ["a"]
    assert casted_table.column("a").to_pylist() == [1, 2, 3]


def test_cast_arrow_record_batch_reader_no_target_schema_passthrough():
    table = pa.table({"A": [1, 2, 3]})
    reader = _make_reader_from_table(table)

    opts = CastOptions.safe_init(target_field=None)
    casted_reader = cast_arrow_record_batch_reader(reader, opts)

    original_table = pa.Table.from_batches(_make_reader_from_table(table))
    casted_table = pa.Table.from_batches(casted_reader)

    assert casted_table.equals(original_table)


# ---------------------------------------------------------------------------
# Cross-container helper utilities
# ---------------------------------------------------------------------------

Tabular = Union[pa.Table, pa.RecordBatch]


def _make_options(data: Tabular, **kwargs) -> CastOptions:
    """
    Required init style:
        options = CastOptions.safe_init(target_field=table.schema, **kwargs)

    Note: for RecordBatch, .schema exists too.
    """
    return CastOptions.safe_init(source_field=data.schema, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "data",
    [
        pa.table({"a": [1, 2]}),
        pa.record_batch([pa.array([1, 2])], names=["a"]),
    ],
)
def test_cast_arrow_tabular_no_target_schema_returns_as_is(data: Tabular):
    options = _make_options(data, target_field=None)
    out = cast_arrow_tabular(data, options)
    assert out is data


@pytest.mark.parametrize(
    "cls, data",
    [
        (pa.Table, pa.table({"a": pa.array([], type=pa.int32())})),
        (pa.RecordBatch, pa.record_batch([pa.array([], type=pa.int32())], names=["a"])),
    ],
)
def test_cast_arrow_tabular_empty_rows_builds_empty_with_target_schema(cls, data):
    assert data.num_rows == 0

    target_schema = pa.schema(
        [
            pa.field("x", pa.int64(), nullable=False),
            pa.field("y", pa.string(), nullable=True),
        ]
    )

    options = _make_options(data, target_field=target_schema)
    out = cast_arrow_tabular(data, options)

    assert isinstance(out, cls)
    assert out.num_rows == 0
    assert out.schema == target_schema
    assert out.column(0).type == pa.int64()
    assert out.column(1).type == pa.string()


@pytest.mark.parametrize(
    "data",
    [
        pa.table({"a": pa.array([1, 2], type=pa.int32())}),
        pa.record_batch([pa.array([1, 2], type=pa.int32())], names=["a"]),
    ],
)
def test_cast_arrow_tabular_same_schema_returns_as_is(data: Tabular):
    options = _make_options(data, target_field=data.schema)
    out = cast_arrow_tabular(data, options)
    assert out is data


@pytest.mark.parametrize(
    "data",
    [
        pa.table({"a": pa.array([1, 2], type=pa.int32())}),
        pa.record_batch([pa.array([1, 2], type=pa.int32())], names=["a"]),
    ],
)
def test_cast_arrow_tabular_exact_name_match_and_type_cast(data: Tabular):
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=True)])
    options = _make_options(data, target_field=target_schema)

    out = cast_arrow_tabular(data, options)

    assert out.schema == target_schema
    col = out.column(0)
    assert col.type == pa.int64()
    assert col.to_pylist() == [1, 2]


def test_cast_arrow_tabular_case_insensitive_match_when_not_strict():
    data = pa.table({"A": pa.array([1, 2], type=pa.int32())})
    target_schema = pa.schema([pa.field("a", pa.int32(), nullable=True)])

    options = _make_options(
        data,
        target_field=target_schema,
        strict_match_names=False,
    )
    out = cast_arrow_tabular(data, options)

    assert out.schema == target_schema
    assert out.column(0).to_pylist() == [1, 2]


def test_cast_arrow_tabular_strict_name_mismatch_raises_when_missing_not_allowed():
    data = pa.table({"A": pa.array([1, 2], type=pa.int32())})
    target_schema = pa.schema([pa.field("a", pa.int32(), nullable=True)])

    options = _make_options(
        data,
        target_field=target_schema,
        strict_match_names=True,
        add_missing_columns=False,
    )

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_tabular(data, options)


def test_cast_arrow_tabular_missing_column_raises_when_add_missing_columns_false():
    data = pa.table({"a": pa.array([1, 2], type=pa.int32())})
    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("b", pa.string(), nullable=True),  # missing
        ]
    )

    options = _make_options(
        data,
        target_field=target_schema,
        add_missing_columns=False,
    )

    with pytest.raises(pa.ArrowInvalid):
        cast_arrow_tabular(data, options)


def test_cast_arrow_tabular_missing_column_filled_with_defaults_and_preserves_chunks_for_table():
    # Make chunking visible so cast_arrow_tabular picks up `chunks` from first column
    a = pa.chunked_array([pa.array([1, 2], type=pa.int32()), pa.array([3], type=pa.int32())])
    data = pa.table({"a": a})

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("missing", pa.int64(), nullable=False),
        ]
    )

    options = _make_options(
        data,
        target_field=target_schema,
        add_missing_columns=True,
    )
    out = cast_arrow_tabular(data, options)

    assert out.schema == target_schema
    missing_col = out.column(out.schema.get_field_index("missing"))

    assert isinstance(missing_col, pa.ChunkedArray)
    assert [len(c) for c in missing_col.chunks] == [2, 1]
    assert missing_col.type == pa.int64()
    assert missing_col.to_pylist() == [0, 0, 0]


def test_cast_arrow_tabular_missing_column_filled_with_defaults_recordbatch():
    data = pa.record_batch([pa.array([1, 2], type=pa.int32())], names=["a"])

    target_schema = pa.schema(
        [
            pa.field("a", pa.int32(), nullable=True),
            pa.field("missing", pa.string(), nullable=True),
        ]
    )

    options = _make_options(
        data,
        target_field=target_schema,
        add_missing_columns=True,
    )
    out = cast_arrow_tabular(data, options)

    assert out.schema == target_schema
    missing_col = out.column(out.schema.get_field_index("missing"))
    assert isinstance(missing_col, pa.Array)
    assert missing_col.type == pa.string()
    assert missing_col.to_pylist() == [None, None]


def test_cast_arrow_tabular_extra_columns_ignored_by_default():
    data = pa.table(
        {
            "a": pa.array([1, 2], type=pa.int32()),
            "extra": pa.array([9, 9], type=pa.int32()),
        }
    )
    target_schema = pa.schema([pa.field("a", pa.int32(), nullable=True)])

    options = _make_options(
        data,
        target_field=target_schema,
        allow_add_columns=False,
    )
    out = cast_arrow_tabular(data, options)

    assert out.schema == target_schema
    assert "extra" not in out.schema.names
    assert out.column(0).to_pylist() == [1, 2]


def test_cast_arrow_tabular_allow_add_columns_appends_extras_and_preserves_target_metadata():
    data = pa.table(
        {
            "a": pa.array([1, 2], type=pa.int32()),
            "extra": pa.array([9, 9], type=pa.int32()),
        }
    )
    target_schema = pa.schema(
        [pa.field("a", pa.int32(), nullable=True)],
        metadata={b"k": b"v"},
    )

    options = _make_options(
        data,
        target_field=target_schema,
        allow_add_columns=True,
    )
    out = cast_arrow_tabular(data, options)

    # Don't require exact equality: safe_init may add b"name": b"root"
    md = dict(out.schema.metadata or {})
    assert md.get(b"k") == b"v"

    assert out.schema.names == ["a", "extra"]
    assert out.column(out.schema.get_field_index("a")).to_pylist() == [1, 2]
    assert out.column(out.schema.get_field_index("extra")).to_pylist() == [9, 9]

# ---------------------------------------------------------------------------
# convert(...) API
# ---------------------------------------------------------------------------


def test_convert_array_to_array_uses_registered_converter():
    array = pa.array([1, None, 3], type=pa.int32())

    result = convert(array, pa.Array)

    assert isinstance(result, pa.Array)
    assert result.type == pa.int32()
    assert result.to_pylist() == [1, None, 3]


def test_convert_chunked_array_to_chunked_array():
    chunked = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4])])

    result = convert(chunked, pa.ChunkedArray)

    assert isinstance(result, pa.ChunkedArray)
    assert result.to_pylist() == [1, 2, 3, 4]


def test_convert_table_variants():
    table = pa.table({"a": [1, 2]})

    table_result = convert(table, pa.Table, options=arrow_schema_to_field(table.schema))
    assert table_result.equals(table)

    batch_result = convert(table, pa.RecordBatch, options=arrow_schema_to_field(table.schema))
    assert batch_result.schema.names == ["a"]
    assert batch_result.column(0).to_pylist() == [1, 2]

    reader_result = convert(table, pa.RecordBatchReader, options=arrow_schema_to_field(table.schema))
    roundtrip_table = convert(reader_result, pa.Table, options=arrow_schema_to_field(table.schema))
    assert roundtrip_table.column("a").to_pylist() == [1, 2]


def test_convert_respects_arrow_target_hint_and_options_propagation():
    array = pa.array([1, 2, 3], type=pa.int32())
    target_hint = pa.field("a", pa.int64(), nullable=False)
    source_hint = pa.field("b", pa.int32(), nullable=True)

    received: dict[str, object] = {}

    from yggdrasil.types.cast import registry

    original_converter = registry._registry[(pa.Array, pa.Array)]

    def _spy(value, options):  # type: ignore[override]
        received["options"] = options
        return value

    registry._registry[(pa.Array, pa.Array)] = _spy

    try:
        result = convert(
            array,
            pa.Array,
            source_arrow_field=source_hint,
            options=target_hint,
        )
    finally:
        registry._registry[(pa.Array, pa.Array)] = original_converter

    assert isinstance(result, pa.Array)

    opts = received["options"]
    assert isinstance(opts, CastOptions)
    assert opts.target_field.type == pa.int64()


def test_convert_record_batch_to_record_batch_reader_and_back():
    batch = pa.record_batch({"a": [5, 6, 7]})

    options = arrow_schema_to_field(batch.schema)
    reader = convert(batch, pa.RecordBatchReader, options=options)
    batch_back = convert(reader, pa.RecordBatch, options=options)

    assert batch_back.schema.names == ["a"]
    assert batch_back.column(0).to_pylist() == [5, 6, 7]


# ---------------------------------------------------------------------------
# pylist converters
# ---------------------------------------------------------------------------


@dataclass
class _Point:
    x: int
    y: str


_Point.__annotations__ = {"x": int, "y": str}


def test_pylist_to_arrow_table_infers_dataclass_and_handles_none_rows(ensure_any_to_arrow_scalar_has_options):
    data = [None, _Point(1, "a"), _Point(2, "b")]

    opts = CastOptions()
    batch = pylist_to_record_batch(data, opts)
    table = record_batch_to_table(batch, opts)

    assert table.schema.names == ["x", "y"]
    assert table.column("x").to_pylist() == [None, 1, 2]
    assert table.column("y").to_pylist() == [None, "a", "b"]


def test_pylist_to_arrow_table_empty_uses_target_schema_defaults(ensure_any_to_arrow_scalar_has_options):
    target_schema = pa.schema(
        [
            pa.field("a", pa.int64(), nullable=False),
            pa.field("b", pa.string(), nullable=True),
        ]
    )
    opts = CastOptions.safe_init(target_field=arrow_schema_to_field(target_schema))

    batch = pylist_to_record_batch([], opts)
    table = record_batch_to_table(batch, opts)

    assert table.num_rows == 0
    assert table.schema.equals(target_schema)


def test_pylist_to_arrow_table_preserves_nullable_values(ensure_any_to_arrow_scalar_has_options):
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=True)])
    opts = CastOptions.safe_init(target_field=arrow_schema_to_field(target_schema))

    batch = pylist_to_record_batch([{"a": 1}, {"a": None}], opts)
    table = record_batch_to_table(batch, opts)

    assert table.schema.equals(target_schema)
    assert table.column("a").to_pylist() == [1, None]


def test_pylist_to_arrow_table_all_none_rows_use_defaults(ensure_any_to_arrow_scalar_has_options):
    target_schema = pa.schema([pa.field("a", pa.int64(), nullable=False)])

    opts = CastOptions.safe_init(target_field=arrow_schema_to_field(target_schema))
    batch = pylist_to_record_batch([None], opts)
    table = record_batch_to_table(batch, opts)

    assert table.schema.equals(target_schema)
    assert table.column("a").to_pylist() == [0]


def test_pylist_to_record_batch_and_reader_via_convert(ensure_any_to_arrow_scalar_has_options):
    target_field = pa.field("value", pa.int64(), nullable=False)
    opts = CastOptions.safe_init(target_field=target_field)

    batch = pylist_to_record_batch([1, 2], opts)
    assert batch.column(0).to_pylist() == [1, 2]

    reader = record_batch_to_record_batch_reader(batch, opts)
    table = pa.Table.from_batches(list(reader))
    assert table.column("value").to_pylist() == [1, 2]

