"""Pure-Python behavior of :class:`StructType`.

No engine round-trips here — everything in this file is pure CPython
walking the dataclass: construction, identity, dict round-trip,
defaults, DDL, ``with_fields``, and the merge semantics that drive
schema reconciliation across the rest of the library.
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import DataType, Field
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.io.enums import Mode


# ---------------------------------------------------------------------------
# Construction & identity
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_post_init_freezes_fields_into_tuple(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=True),
            ]
        )
        assert isinstance(dtype.fields, tuple)

    def test_empty_struct_is_falsy(self) -> None:
        assert bool(StructType.empty()) is False

    def test_populated_struct_is_truthy(self, int64_type: IntegerType) -> None:
        dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
        assert bool(dtype) is True

    def test_children_fields_alias_fields(self, int64_type: IntegerType) -> None:
        fields = (
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=int64_type, nullable=True),
        )
        dtype = StructType(fields=fields)
        assert dtype.children_fields == dtype.fields

    def test_class_type_id_is_struct(self) -> None:
        assert StructType.class_type_id() == DataTypeId.STRUCT

    def test_instance_type_id_is_struct(self, int64_type: IntegerType) -> None:
        dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
        assert dtype.type_id == DataTypeId.STRUCT

    def test_to_struct_returns_self(self, int64_type: IntegerType) -> None:
        dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
        assert dtype.to_struct() is dtype


# ---------------------------------------------------------------------------
# Type detection (handles_*)
# ---------------------------------------------------------------------------


class TestHandlers:
    def test_handles_struct_arrow_type(self) -> None:
        assert StructType.handles_arrow_type(
            pa.struct([pa.field("a", pa.int64())])
        )

    @pytest.mark.parametrize("dtype", [pa.int64(), pa.list_(pa.int64()), pa.string()])
    def test_rejects_non_struct_arrow_types(self, dtype: pa.DataType) -> None:
        assert StructType.handles_arrow_type(dtype) is False

    @pytest.mark.parametrize(
        "payload",
        [
            {"id": int(DataTypeId.STRUCT)},
            {"name": "STRUCT"},
            {"name": "struct"},
        ],
    )
    def test_handles_struct_dict(self, payload: dict) -> None:
        assert StructType.handles_dict(payload) is True

    def test_rejects_non_struct_dict(self) -> None:
        assert StructType.handles_dict({"name": "ARRAY"}) is False


# ---------------------------------------------------------------------------
# Round-trips: arrow ⇄ struct, dict ⇄ struct
# ---------------------------------------------------------------------------


class TestArrowRoundTrip:
    def test_from_arrow_preserves_field_names_and_nullability(self) -> None:
        arrow_struct = pa.struct(
            [
                pa.field("x", pa.int64(), nullable=True),
                pa.field("y", pa.string(), nullable=False),
            ]
        )

        dtype = StructType.from_arrow_type(arrow_struct)

        assert isinstance(dtype, StructType)
        assert [f.name for f in dtype.fields] == ["x", "y"]
        assert [f.nullable for f in dtype.fields] == [True, False]

    def test_to_arrow_yields_struct_type(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[
                Field(name="x", dtype=int64_type, nullable=True),
                Field(name="y", dtype=int64_type, nullable=False),
            ]
        )

        produced = dtype.to_arrow()

        assert pa.types.is_struct(produced)
        assert produced.num_fields == 2
        assert produced.field(1).nullable is False

    def test_from_arrow_rejects_non_struct(self) -> None:
        with pytest.raises(TypeError, match="Unsupported Arrow data type"):
            StructType.from_arrow_type(pa.int64())


class TestDictRoundTrip:
    def test_to_dict_carries_fields(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        payload = dtype.to_dict()

        assert payload["name"] == "STRUCT"
        assert payload["fields"][0]["name"] == "a"

    def test_from_dict_round_trip(self, int64_type: IntegerType) -> None:
        original = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=False),
            ]
        )

        rebuilt = StructType.from_dict(original.to_dict())

        assert isinstance(rebuilt, StructType)
        assert [f.name for f in rebuilt.fields] == ["a", "b"]
        assert [f.nullable for f in rebuilt.fields] == [True, False]

    def test_from_dict_with_no_fields_returns_empty(self) -> None:
        rebuilt = StructType.from_dict({"name": "STRUCT"})

        assert isinstance(rebuilt, StructType)
        assert rebuilt.fields == ()


# ---------------------------------------------------------------------------
# Defaults & DDL
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_nullable_default_pyobj_is_none(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        assert dtype.default_pyobj(nullable=True) is None

    def test_non_nullable_default_pyobj_is_per_child_default(
        self, int64_type: IntegerType
    ) -> None:
        dtype = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=False),
            ]
        )
        default = dtype.default_pyobj(nullable=False)

        assert isinstance(default, dict)
        assert set(default.keys()) == {"a", "b"}


class TestDatabricksDdl:
    def test_quotes_simple_field_names(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[Field(name="plain", dtype=int64_type, nullable=True)]
        )
        ddl = dtype.to_databricks_ddl()

        assert ddl.startswith("STRUCT<")
        assert ddl.endswith(">")
        assert "`plain`:" in ddl

    def test_doubles_embedded_backticks(self, int64_type: IntegerType) -> None:
        dtype = StructType(
            fields=[Field(name="we`ird", dtype=int64_type, nullable=True)]
        )
        ddl = dtype.to_databricks_ddl()

        assert "`we``ird`:" in ddl


# ---------------------------------------------------------------------------
# with_fields
# ---------------------------------------------------------------------------


class TestWithFields:
    def test_inplace_mutates_existing_instance(
        self, int64_type: IntegerType
    ) -> None:
        dtype = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        replacement = [Field(name="b", dtype=int64_type, nullable=True)]

        same = dtype.with_fields(replacement, safe=True, inplace=True)

        assert same is dtype
        assert [f.name for f in dtype.fields] == ["b"]

    def test_non_inplace_returns_new_instance(
        self, int64_type: IntegerType
    ) -> None:
        dtype = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        replacement = [Field(name="b", dtype=int64_type, nullable=True)]

        new_dtype = dtype.with_fields(replacement, safe=True, inplace=False)

        assert new_dtype is not dtype
        assert [f.name for f in dtype.fields] == ["a"]
        assert [f.name for f in new_dtype.fields] == ["b"]

    def test_non_safe_coerces_dict_inputs(self) -> None:
        dtype = StructType(fields=[]).with_fields(
            [
                {
                    "name": "a",
                    "type_json": {"id": int(DataTypeId.INTEGER)},
                    "nullable": True,
                }
            ],
            safe=False,
            inplace=False,
        )

        assert len(dtype.fields) == 1
        assert dtype.fields[0].name == "a"
        assert dtype.fields[0].dtype.type_id == DataTypeId.INT64


# ---------------------------------------------------------------------------
# Merge semantics — the heart of cross-engine schema reconciliation.
# ---------------------------------------------------------------------------


class TestMerge:
    def test_matches_by_name_and_upcasts_dtypes(
        self,
        int64_type: IntegerType,
        string_type,
    ) -> None:
        left = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=False),
                Field(name="b", dtype=string_type, nullable=True),
            ]
        )
        right = StructType(
            fields=[
                Field(
                    name="a",
                    dtype=DataType.from_arrow_type(pa.int32()),
                    nullable=True,
                ),
                Field(
                    name="b",
                    dtype=string_type,
                    nullable=True,
                    metadata={"comment": "rhs"},
                ),
            ]
        )

        result = left._merge_with_same_id(right, upcast=True)

        assert isinstance(result, StructType)
        assert [f.name for f in result.fields] == ["a", "b"]
        # Left's non-nullable wins for column "a".
        assert result.fields[0].nullable is False
        assert result.fields[0].arrow_type == pa.int64()
        # Metadata picked up from right.
        assert result.fields[1].metadata == {b"comment": b"rhs"}

    def test_drops_left_only_fields_when_right_is_authoritative(
        self,
        int64_type: IntegerType,
        string_type,
    ) -> None:
        left = StructType(
            fields=[
                Field(name="left_only", dtype=int64_type, nullable=True),
                Field(name="shared", dtype=string_type, nullable=True),
            ]
        )
        right = StructType(
            fields=[Field(name="shared", dtype=string_type, nullable=True)]
        )

        result = left._merge_with_same_id(right)

        assert [f.name for f in result.fields] == ["shared"]

    def test_overwrite_mode_drops_right_only_fields(
        self,
        int64_type: IntegerType,
        string_type,
    ) -> None:
        left = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        right = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=string_type, nullable=True),
            ]
        )

        result = left._merge_with_same_id(right, mode=Mode.OVERWRITE)

        assert [f.name for f in result.fields] == ["a"]

    def test_append_mode_preserves_left_order_and_appends_new_right(
        self,
        int64_type: IntegerType,
        string_type,
        bool_type,
    ) -> None:
        left = StructType(
            fields=[
                Field(name="b", dtype=string_type, nullable=True),
                Field(name="a", dtype=int64_type, nullable=False),
            ]
        )
        right = StructType(
            fields=[
                Field(
                    name="a",
                    dtype=DataType.from_arrow_type(pa.int32()),
                    nullable=True,
                ),
                Field(name="c", dtype=bool_type, nullable=True),
            ]
        )

        result = left._merge_with_same_id(right, mode=Mode.APPEND, upcast=True)

        assert [f.name for f in result.fields] == ["a", "b", "c"]
        # "b" came from left untouched.
        assert result.fields[1].arrow_type == pa.string()
        assert result.fields[1].nullable is True

    def test_ignore_mode_returns_self_unchanged(
        self,
        int64_type: IntegerType,
    ) -> None:
        left = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )
        right = StructType(
            fields=[Field(name="b", dtype=int64_type, nullable=True)]
        )

        assert left._merge_with_same_id(right, mode=Mode.IGNORE) is left

    def test_rejects_non_struct_other(self, int64_type: IntegerType) -> None:
        left = StructType(
            fields=[Field(name="a", dtype=int64_type, nullable=True)]
        )

        with pytest.raises(TypeError, match="Cannot merge StructType with"):
            left._merge_with_same_id(DataType.from_arrow_type(pa.int64()))


# ---------------------------------------------------------------------------
# pretty_format — tiny smoke; format change shouldn't silently regress.
# ---------------------------------------------------------------------------


class TestPrettyFormat:
    def test_empty_struct(self) -> None:
        assert StructType.empty().pretty_format() == "struct<>"

    def test_populated_struct_includes_field_names(
        self, int64_type: IntegerType
    ) -> None:
        dtype = StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=True),
            ]
        )
        formatted = dtype.pretty_format()

        assert formatted.startswith("struct<")
        assert formatted.rstrip().endswith(">")
        assert "a" in formatted and "b" in formatted
