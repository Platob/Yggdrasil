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