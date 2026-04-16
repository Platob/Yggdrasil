"""Pure-Python tests for ArrayType — no engine round-trips.

This file covers construction, identity (handles_* / type_id),
dict serialisation round-trips, defaults, and merge semantics.
Engine-level casts live in the sibling test_arrow / test_polars /
test_pandas / test_spark modules.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.primitive import IntegerType, StringType


# ---------------------------------------------------------------------------
# from_item_field
# ---------------------------------------------------------------------------


def test_from_item_field_safe_mode_preserves_original_field() -> None:
    original = Field(name="custom", dtype=StringType(), nullable=False)

    result = ArrayType.from_item_field(item_field=original, safe=True)

    assert result.item_field is original


def test_from_item_field_non_safe_mode_renames_item_to_item() -> None:
    original = Field(name="custom", dtype=StringType(), nullable=False)

    result = ArrayType.from_item_field(item_field=original, safe=False)

    assert result.item_field.name == "item"


def test_from_item_field_sanitizes_negative_list_size_to_none() -> None:
    result = ArrayType.from_item_field(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=-1,
    )

    assert result.list_size is None


def test_from_item_field_preserves_zero_list_size() -> None:
    result = ArrayType.from_item_field(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=0,
        safe=True,
    )

    assert result.list_size == 0


# ---------------------------------------------------------------------------
# Type identity
# ---------------------------------------------------------------------------


def test_type_id_is_array() -> None:
    dtype = ArrayType.from_item_field(StringType().to_field())
    assert dtype.type_id == DataTypeId.ARRAY


def test_handles_dict_matches_by_id_and_name() -> None:
    assert ArrayType.handles_dict({"id": int(DataTypeId.ARRAY)}) is True
    assert ArrayType.handles_dict({"name": "ARRAY"}) is True
    assert ArrayType.handles_dict({"name": "array"}) is True


def test_handles_dict_rejects_other_types_and_unknown_names() -> None:
    assert ArrayType.handles_dict({"name": "STRUCT"}) is False
    assert ArrayType.handles_dict({"name": "NOT_AN_ARRAY"}) is False


def test_children_fields_is_the_item_field() -> None:
    dtype = ArrayType.from_item_field(StringType().to_field())
    children = dtype.children_fields
    assert len(children) == 1
    assert children[0].name == "item"


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_omits_default_flags() -> None:
    dtype = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
    )

    payload = dtype.to_dict()

    assert payload["name"] == "ARRAY"
    assert "list_size" not in payload
    assert "large" not in payload
    assert "view" not in payload


def test_to_dict_emits_all_flags_when_set() -> None:
    dtype = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=3,
        large=True,
        view=True,
    )

    payload = dtype.to_dict()

    assert payload["list_size"] == 3
    assert payload["large"] is True
    assert payload["view"] is True
    assert payload["item_field"]["dtype"]["name"] == "STRING"


def test_to_dict_from_dict_round_trip_preserves_all_fields() -> None:
    original = ArrayType(
        item_field=Field(name="item", dtype=StringType(), nullable=True),
        list_size=3,
        large=True,
    )

    rebuilt = ArrayType.from_dict(original.to_dict())

    assert isinstance(rebuilt, ArrayType)
    assert rebuilt.list_size == 3
    assert rebuilt.large is True
    assert rebuilt.view is False
    assert rebuilt.item_field.dtype.type_id == DataTypeId.STRING


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_pyobj_nullable_returns_none() -> None:
    dtype = ArrayType.from_item_field(StringType().to_field())
    assert dtype.default_pyobj(nullable=True) is None


def test_default_pyobj_non_nullable_returns_empty_list() -> None:
    dtype = ArrayType.from_item_field(StringType().to_field())
    assert dtype.default_pyobj(nullable=False) == []


# ---------------------------------------------------------------------------
# Databricks DDL
# ---------------------------------------------------------------------------


def test_to_databricks_ddl_wraps_item_type() -> None:
    dtype = ArrayType.from_item_field(StringType().to_field())
    ddl = dtype.to_databricks_ddl()
    assert ddl.upper().startswith("ARRAY<")
    assert ddl.endswith(">")


# ---------------------------------------------------------------------------
# Arrow type identity — pure Python, no arrays cast
# ---------------------------------------------------------------------------


def test_to_arrow_variants() -> None:
    item = Field(name="item", dtype=StringType(), nullable=True)

    assert pa.types.is_list(ArrayType(item_field=item).to_arrow())
    assert pa.types.is_large_list(ArrayType(item_field=item, large=True).to_arrow())
    fixed = ArrayType(item_field=item, list_size=4).to_arrow()
    assert pa.types.is_fixed_size_list(fixed)
    assert fixed.list_size == 4
    assert pa.types.is_list_view(ArrayType(item_field=item, view=True).to_arrow())
    assert pa.types.is_large_list_view(
        ArrayType(item_field=item, view=True, large=True).to_arrow()
    )


def test_handles_arrow_type_rejects_non_list_types() -> None:
    assert ArrayType.handles_arrow_type(pa.int64()) is False
    assert ArrayType.handles_arrow_type(pa.map_(pa.string(), pa.int64())) is False
    assert (
        ArrayType.handles_arrow_type(pa.struct([pa.field("a", pa.int64())])) is False
    )


def test_from_arrow_type_rejects_non_list_dtype() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        ArrayType.from_arrow_type(pa.int64())


def test_from_arrow_type_list_view_and_large_list_view_preserve_flags() -> None:
    view = ArrayType.from_arrow_type(
        pa.list_view(pa.field("item", pa.int64(), nullable=True))
    )
    assert view.view is True
    assert view.large is False

    large_view = ArrayType.from_arrow_type(
        pa.large_list_view(pa.field("item", pa.int64(), nullable=True))
    )
    assert large_view.view is True
    assert large_view.large is True


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_merge_with_same_id_keeps_list_size_from_either_side() -> None:
    left = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
        list_size=5,
    )
    right = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
    )

    assert left._merge_with_same_id(right).list_size == 5
    assert right._merge_with_same_id(left).list_size == 5


def test_merge_with_same_id_or_s_item_nullability() -> None:
    left = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=False),
    )
    right = ArrayType(
        item_field=Field(name="item", dtype=IntegerType(), nullable=True),
    )

    merged = left._merge_with_same_id(right)

    assert merged.item_field.nullable is True
