"""Pure-Python behavior of :class:`ArrayType`.

Construction, identity (handles_*, type_id, children), dict
serialization, defaults, DDL, Arrow type identity (no values cast),
and merge semantics. The engine-level cast paths live in the sibling
``test_arrow`` / ``test_polars`` / ``test_pandas`` / ``test_spark``
modules — this file deliberately stays single-process Python.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.primitive import IntegerType, StringType


# ---------------------------------------------------------------------------
# from_item — factory
# ---------------------------------------------------------------------------


class TestFromItem:

    def test_passes_through_item_field_unchanged_in_safe_mode(self) -> None:
        original = Field(name="custom", dtype=StringType(), nullable=False)

        result = ArrayType.from_item(item_field=original)

        assert result.item_field is original

    def test_negative_list_size_normalised_to_none(self) -> None:
        result = ArrayType.from_item(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
            list_size=-1,
        )

        assert result.list_size is None

    def test_zero_list_size_preserved_as_explicit_zero(self) -> None:
        result = ArrayType.from_item(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
            list_size=0,
        )

        assert result.list_size == 0


# ---------------------------------------------------------------------------
# Type identity — type_id, handlers, children
# ---------------------------------------------------------------------------


class TestIdentity:

    def test_type_id_is_array(self) -> None:
        dtype = ArrayType.from_item(StringType().to_field())

        assert dtype.type_id == DataTypeId.ARRAY

    def test_handles_dict_by_id_and_name(self) -> None:
        assert ArrayType.handles_dict({"id": int(DataTypeId.ARRAY)}) is True
        assert ArrayType.handles_dict({"name": "ARRAY"}) is True
        assert ArrayType.handles_dict({"name": "array"}) is True

    def test_handles_dict_rejects_other_type_ids(self) -> None:
        assert ArrayType.handles_dict({"name": "STRUCT"}) is False
        assert ArrayType.handles_dict({"name": "NOT_AN_ARRAY"}) is False

    def test_children_fields_is_just_the_item_field(self) -> None:
        dtype = ArrayType.from_item(StringType().to_field())

        children = dtype.children_fields
        assert len(children) == 1
        assert children[0].name == ""


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


class TestDictRoundTrip:

    def test_to_dict_omits_default_flags(self) -> None:
        dtype = ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
        )

        payload = dtype.to_dict()

        assert payload["name"] == "ARRAY"
        assert "list_size" not in payload
        assert "large" not in payload
        assert "view" not in payload

    def test_to_dict_emits_set_flags(self) -> None:
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

    def test_round_trip_preserves_flags(self) -> None:
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


class TestDefaults:

    def test_default_pyobj_nullable_is_none(self) -> None:
        dtype = ArrayType.from_item(StringType().to_field())

        assert dtype.default_pyobj(nullable=True) is None

    def test_default_pyobj_non_nullable_is_empty_list(self) -> None:
        dtype = ArrayType.from_item(StringType().to_field())

        assert dtype.default_pyobj(nullable=False) == []


# ---------------------------------------------------------------------------
# Databricks DDL
# ---------------------------------------------------------------------------


class TestDatabricksDdl:

    def test_wraps_inner_type_in_array(self) -> None:
        ddl = ArrayType.from_item(StringType().to_field()).to_spark_name()

        assert ddl.upper().startswith("ARRAY<")
        assert ddl.endswith(">")


# ---------------------------------------------------------------------------
# Arrow type identity — no values cast
# ---------------------------------------------------------------------------


class TestArrowTypeIdentity:

    def test_to_arrow_variants(self) -> None:
        item = Field(name="item", dtype=StringType(), nullable=True)

        assert pa.types.is_list(ArrayType(item_field=item).to_arrow())
        assert pa.types.is_large_list(
            ArrayType(item_field=item, large=True).to_arrow()
        )

        fixed = ArrayType(item_field=item, list_size=4).to_arrow()
        assert pa.types.is_fixed_size_list(fixed)
        assert fixed.list_size == 4

        assert pa.types.is_list_view(
            ArrayType(item_field=item, view=True).to_arrow()
        )
        assert pa.types.is_large_list_view(
            ArrayType(item_field=item, view=True, large=True).to_arrow()
        )

    def test_handles_arrow_type_rejects_non_lists(self) -> None:
        assert ArrayType.handles_arrow_type(pa.int64()) is False
        assert (
            ArrayType.handles_arrow_type(pa.map_(pa.string(), pa.int64())) is False
        )
        assert (
            ArrayType.handles_arrow_type(pa.struct([pa.field("a", pa.int64())]))
            is False
        )

    def test_from_arrow_type_rejects_non_list(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            ArrayType.from_arrow_type(pa.int64())

    def test_from_arrow_list_view_preserves_view_flag(self) -> None:
        view = ArrayType.from_arrow_type(
            pa.list_view(pa.field("item", pa.int64(), nullable=True))
        )

        assert view.view is True
        assert view.large is False

    def test_from_arrow_large_list_view_preserves_both_flags(self) -> None:
        large_view = ArrayType.from_arrow_type(
            pa.large_list_view(pa.field("item", pa.int64(), nullable=True))
        )

        assert large_view.view is True
        assert large_view.large is True


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


class TestMerge:

    def test_list_size_propagates_from_either_side(self) -> None:
        sized = ArrayType(
            item_field=Field(name="item", dtype=IntegerType(), nullable=True),
            list_size=5,
        )
        unsized = ArrayType(
            item_field=Field(name="item", dtype=IntegerType(), nullable=True),
        )

        assert sized._merge_with_same_id(unsized).list_size == 5
        assert unsized._merge_with_same_id(sized).list_size == 5

    def test_item_nullability_takes_strictest_side(self) -> None:
        strict = ArrayType(
            item_field=Field(name="item", dtype=IntegerType(), nullable=False),
        )
        loose = ArrayType(
            item_field=Field(name="item", dtype=IntegerType(), nullable=True),
        )

        merged = strict._merge_with_same_id(loose)

        assert merged.item_field.nullable is False
