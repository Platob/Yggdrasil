from __future__ import annotations

import pyarrow as pa
import pytest
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested.array import (
    ArrayType,
    cast_arrow_list_array,
    cast_arrow_map_array_to_list,
)
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import StructType, MapType
from yggdrasil.data.types.primitive import IntegerType, StringType


@pytest.fixture
def source_array_field() -> Field:
    return Field(
        name="source_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_array_field() -> Field:
    return Field(
        name="target_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StringType(),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_large_array_field() -> Field:
    return Field(
        name="target_large_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StringType(),
                nullable=True,
            ),
            list_size=None,
            large=True,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_fixed_array_field() -> Field:
    return Field(
        name="target_fixed_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StringType(),
                nullable=True,
            ),
            list_size=2,
            large=False,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_view_array_field() -> Field:
    return Field(
        name="target_view_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StringType(),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=True,
        ),
        nullable=True,
    )


@pytest.fixture
def source_map_field() -> Field:
    return Field(
        name="source_map",
        dtype=MapType(
            item_field=Field(
                name="entries",
                dtype=StructType(
                    fields=[
                        Field(
                            name="key",
                            dtype=StringType(),
                            nullable=False,
                        ),
                        Field(
                            name="value",
                            dtype=IntegerType(byte_size=8, signed=True),
                            nullable=True,
                        ),
                    ]
                ),
                nullable=False,
            ),
            keys_sorted=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_entries_array_field() -> Field:
    return Field(
        name="target_entries_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StructType(
                    fields=[
                        Field(
                            name="key",
                            dtype=StringType(),
                            nullable=False,
                        ),
                        Field(
                            name="value",
                            dtype=StringType(),
                            nullable=True,
                        ),
                    ]
                ),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def target_entries_large_array_field() -> Field:
    return Field(
        name="target_entries_large_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StructType(
                    fields=[
                        Field(
                            name="key",
                            dtype=StringType(),
                            nullable=False,
                        ),
                        Field(
                            name="value",
                            dtype=StringType(),
                            nullable=True,
                        ),
                    ]
                ),
                nullable=True,
            ),
            list_size=None,
            large=True,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def invalid_target_entries_scalar_array_field() -> Field:
    return Field(
        name="invalid_target_entries_scalar_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StringType(),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=False,
        ),
        nullable=True,
    )


@pytest.fixture
def invalid_target_entries_struct_one_field_array_field() -> Field:
    return Field(
        name="invalid_target_entries_struct_one_field_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StructType(
                    fields=[
                        Field(
                            name="key",
                            dtype=StringType(),
                            nullable=False,
                        ),
                    ]
                ),
                nullable=True,
            ),
            list_size=None,
            large=False,
            view=False,
        ),
        nullable=True,
    )


def test_array_type_from_arrow_type_list() -> None:
    dtype = pa.list_(pa.field("item", pa.int64(), nullable=True))

    result = ArrayType.from_arrow_type(dtype)

    assert isinstance(result, ArrayType)
    assert result.list_size is None
    assert result.large is False
    assert result.view is False
    assert result.to_arrow() == dtype


def test_array_type_from_arrow_type_large_list() -> None:
    dtype = pa.large_list(pa.field("item", pa.int64(), nullable=True))

    result = ArrayType.from_arrow_type(dtype)

    assert isinstance(result, ArrayType)
    assert result.list_size is None
    assert result.large is True
    assert result.view is False
    assert result.to_arrow() == dtype


def test_array_type_from_arrow_type_fixed_size_list() -> None:
    dtype = pa.list_(pa.field("item", pa.int64(), nullable=True), 2)

    result = ArrayType.from_arrow_type(dtype)

    assert isinstance(result, ArrayType)
    assert result.list_size == 2
    assert result.large is False
    assert result.view is False
    assert result.to_arrow() == dtype


def test_array_type_handles_dict() -> None:
    assert ArrayType.handles_dict({"id": int(DataTypeId.ARRAY)}) is True
    assert ArrayType.handles_dict({"name": "ARRAY"}) is True
    assert ArrayType.handles_dict({"name": "array"}) is True
    assert ArrayType.handles_dict({"name": "STRUCT"}) is False


def test_array_type_to_dict(
    source_array_field: Field,
) -> None:
    dtype: ArrayType = source_array_field.dtype

    result = dtype.to_dict()

    assert result["name"] == "ARRAY"
    assert result["item_field"]["name"] == "item"
    assert "list_size" not in result
    assert "large" not in result
    assert "view" not in result


def test_array_type_default_pyobj() -> None:
    dtype = ArrayType(
        item_field=Field(
            name="item",
            dtype=StringType(),
            nullable=True,
        )
    )

    assert dtype.default_pyobj(nullable=True) is None
    assert dtype.default_pyobj(nullable=False) == []


def test_cast_arrow_list_array_array_to_array(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
            [3, None],
            None,
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.ListArray)
    assert result.type == pa.list_(pa.string())
    assert result.to_pylist() == [
        ["1", "2"],
        ["3", None],
        None,
    ]


def test_cast_arrow_list_array_array_to_large_array(
    source_array_field: Field,
    target_large_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
            [3, None],
            None,
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_large_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.LargeListArray)
    assert result.type == pa.large_list(pa.string())
    assert result.to_pylist() == [
        ["1", "2"],
        ["3", None],
        None,
    ]


def test_cast_arrow_list_array_array_to_fixed_size_array(
    source_array_field: Field,
    target_fixed_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
            [3, None],
            None,
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_fixed_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.FixedSizeListArray)
    assert result.type == pa.list_(pa.string(), 2)
    assert result.to_pylist() == [
        ["1", "2"],
        ["3", None],
    ]


def test_cast_arrow_list_array_chunked_array_to_array(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    chunk_1 = pa.array(
        [
            [1, 2],
            None,
        ],
        type=pa.list_(pa.int64()),
    )
    chunk_2 = pa.array(
        [
            [3],
        ],
        type=pa.list_(pa.int64()),
    )
    array = pa.chunked_array(
        [chunk_1, chunk_2],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert isinstance(result, pa.ChunkedArray)
    assert result.type == pa.list_(pa.string())
    assert result.to_pylist() == [
        ["1", "2"],
        None,
        ["3"],
    ]


def test_cast_arrow_list_array_returns_original_when_target_field_is_none(
    source_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=None,
    )

    result = cast_arrow_list_array(array, options)

    assert result is array


def test_cast_arrow_list_array_raises_for_non_array_source(
    source_map_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1)],
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_list_array(array, options)


def test_cast_arrow_list_array_raises_for_view_target(
    source_array_field: Field,
    target_view_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_view_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_list_array(array, options)


def test_cast_arrow_map_array_to_list_map_to_entries_array(
    source_map_field: Field,
    target_entries_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1), ("b", 2)],
            [("c", None)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.ListArray)
    assert result.type == pa.list_(
        pa.struct(
            [
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string()),
            ]
        )
    )
    assert result.to_pylist() == [
        [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}],
        [{"key": "c", "value": None}],
        None,
    ]


def test_cast_arrow_map_array_to_list_map_to_large_entries_array(
    source_map_field: Field,
    target_entries_large_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1), ("b", 2)],
            [("c", None)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_large_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.LargeListArray)
    assert result.type == pa.large_list(
        pa.struct(
            [
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string()),
            ]
        )
    )
    assert result.to_pylist() == [
        [{"key": "a", "value": "1"}, {"key": "b", "value": "2"}],
        [{"key": "c", "value": None}],
        None,
    ]


def test_cast_arrow_map_array_to_list_chunked_map_to_entries_array(
    source_map_field: Field,
    target_entries_array_field: Field,
) -> None:
    chunk_1 = pa.array(
        [
            [("a", 1)],
            None,
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )
    chunk_2 = pa.array(
        [
            [("b", 2), ("c", 3)],
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )
    array = pa.chunked_array(
        [chunk_1, chunk_2],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=target_entries_array_field,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert isinstance(result, pa.ChunkedArray)
    assert result.type == pa.list_(
        pa.struct(
            [
                pa.field("key", pa.string(), nullable=False),
                pa.field("value", pa.string()),
            ]
        )
    )
    assert result.to_pylist() == [
        [{"key": "a", "value": "1"}],
        None,
        [{"key": "b", "value": "2"}, {"key": "c", "value": "3"}],
    ]


def test_cast_arrow_map_array_to_list_returns_original_when_target_field_is_none(
    source_map_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1)],
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=None,
    )

    result = cast_arrow_map_array_to_list(array, options)

    assert result is array


def test_cast_arrow_map_array_to_list_raises_for_non_map_source(
    source_array_field: Field,
    target_entries_array_field: Field,
) -> None:
    array = pa.array(
        [
            [1, 2],
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_entries_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)


def test_cast_arrow_map_array_to_list_raises_for_scalar_target_item_dtype(
    source_map_field: Field,
    invalid_target_entries_scalar_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1)],
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=invalid_target_entries_scalar_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)


def test_cast_arrow_map_array_to_list_raises_for_struct_target_item_wrong_arity(
    source_map_field: Field,
    invalid_target_entries_struct_one_field_array_field: Field,
) -> None:
    array = pa.array(
        [
            [("a", 1)],
        ],
        type=pa.map_(pa.string(), pa.int64()),
    )

    options = CastOptions(
        source_field=source_map_field,
        target_field=invalid_target_entries_struct_one_field_array_field,
    )

    with pytest.raises(pa.ArrowInvalid, match="Cannot cast"):
        cast_arrow_map_array_to_list(array, options)

# ---------------------------------------------------------------------------
# Construction / conversion surface
# ---------------------------------------------------------------------------


def test_array_type_from_item_field_sanitizes_negative_list_size() -> None:
    result = ArrayType.from_item_field(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=-1,
    )

    assert result.list_size is None


def test_array_type_from_item_field_preserves_zero_list_size() -> None:
    result = ArrayType.from_item_field(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=0,
        safe=True,
    )

    assert result.list_size == 0


def test_array_type_from_item_field_safe_mode_skips_field_rewrite() -> None:
    original = Field(name="custom", dtype=StringType(), nullable=False)
    result = ArrayType.from_item_field(item_field=original, safe=True)

    assert result.item_field is original


def test_array_type_from_item_field_non_safe_mode_renames_item() -> None:
    original = Field(name="custom", dtype=StringType(), nullable=False)
    result = ArrayType.from_item_field(item_field=original, safe=False)

    assert result.item_field.name == "item"


def test_array_type_from_arrow_type_list_view() -> None:
    dtype = pa.list_view(pa.field("item", pa.int64(), nullable=True))

    result = ArrayType.from_arrow_type(dtype)

    assert result.view is True
    assert result.large is False
    assert result.list_size is None


def test_array_type_from_arrow_type_large_list_view() -> None:
    dtype = pa.large_list_view(pa.field("item", pa.int64(), nullable=True))

    result = ArrayType.from_arrow_type(dtype)

    assert result.view is True
    assert result.large is True
    assert result.list_size is None


def test_array_type_from_arrow_type_rejects_non_list_dtype() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        ArrayType.from_arrow_type(pa.int64())


def test_array_type_handles_arrow_type_returns_false_for_non_list() -> None:
    assert ArrayType.handles_arrow_type(pa.int64()) is False
    assert ArrayType.handles_arrow_type(pa.map_(pa.string(), pa.int64())) is False
    assert ArrayType.handles_arrow_type(
        pa.struct([pa.field("a", pa.int64())])
    ) is False


def test_array_type_to_dict_includes_large_view_and_list_size() -> None:
    dtype = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=3,
        large=True,
        view=True,
    )

    payload = dtype.to_dict()

    assert payload["name"] == "ARRAY"
    assert payload["list_size"] == 3
    assert payload["large"] is True
    assert payload["view"] is True
    assert payload["item_field"]["dtype"]["name"] == "STRING"


def test_array_type_from_dict_roundtrip() -> None:
    original = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=3,
        large=True,
    )

    reborn = ArrayType.from_dict(original.to_dict())

    assert isinstance(reborn, ArrayType)
    assert reborn.list_size == 3
    assert reborn.large is True
    assert reborn.view is False
    assert reborn.item_field.name == "item"
    assert reborn.item_field.dtype.type_id == DataTypeId.STRING


def test_array_type_to_arrow_variants() -> None:
    dtype_default = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
    )
    dtype_large = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        large=True,
    )
    dtype_fixed = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=4,
    )
    dtype_view = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        view=True,
    )
    dtype_large_view = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        view=True,
        large=True,
    )

    assert pa.types.is_list(dtype_default.to_arrow())
    assert pa.types.is_large_list(dtype_large.to_arrow())
    assert pa.types.is_fixed_size_list(dtype_fixed.to_arrow())
    assert dtype_fixed.to_arrow().list_size == 4
    assert pa.types.is_list_view(dtype_view.to_arrow())
    assert pa.types.is_large_list_view(dtype_large_view.to_arrow())


def test_array_type_to_databricks_ddl() -> None:
    dtype = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
    )
    assert dtype.to_databricks_ddl().upper().startswith("ARRAY<")
    assert dtype.to_databricks_ddl().endswith(">")


def test_array_type_merge_with_same_id_keeps_list_size_when_one_fixed() -> None:
    left = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
        list_size=5,
    )
    right = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
    )

    merged = left._merge_with_same_id(right)
    assert merged.list_size == 5

    merged_reverse = right._merge_with_same_id(left)
    assert merged_reverse.list_size == 5


def test_array_type_merge_with_same_id_merges_items() -> None:
    left = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=False),
    )
    right = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
    )

    merged = left._merge_with_same_id(right)

    # merge_with on fields ORs the nullable flag.
    assert merged.item_field.nullable is True


def test_array_type_handles_dict_alternate_alias() -> None:
    assert ArrayType.handles_dict({"name": "NOT_AN_ARRAY"}) is False


def test_array_type_default_pyobj_variants() -> None:
    dtype = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
    )
    assert dtype.default_pyobj(nullable=True) is None
    assert dtype.default_pyobj(nullable=False) == []


def test_array_type_polars_roundtrip() -> None:
    polars = pytest.importorskip("polars")
    dtype = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(byte_size=8, signed=True), nullable=True),
    )

    polars_dtype = dtype.to_polars()
    assert isinstance(polars_dtype, polars.List)

    reborn = ArrayType.from_polars_type(polars_dtype)
    assert isinstance(reborn, ArrayType)
    assert reborn.item_field.dtype.type_id == DataTypeId.INTEGER


def test_array_type_from_polars_type_rejects_non_list() -> None:
    polars = pytest.importorskip("polars")
    with pytest.raises(TypeError, match="Unsupported Polars data type"):
        ArrayType.from_polars_type(polars.Int64())


def test_cast_arrow_list_array_preserves_nested_null_mask(
    source_array_field: Field,
    target_array_field: Field,
) -> None:
    array = pa.array(
        [
            None,
            [],
            [1, None, 3],
        ],
        type=pa.list_(pa.int64()),
    )

    options = CastOptions(
        source_field=source_array_field,
        target_field=target_array_field,
    )

    result = cast_arrow_list_array(array, options)

    assert result.to_pylist() == [None, [], ["1", None, "3"]]
    assert result.is_null().to_pylist() == [True, False, False]
