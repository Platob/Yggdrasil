"""Pure-Python tests for StructType — no engine round-trips.

Covers construction, identity, dict round-trips, defaults, DDL, the
``with_fields`` API, and merge semantics (which is the richest part
of StructType).
"""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import DataType, Field
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.io import SaveMode


# ---------------------------------------------------------------------------
# Construction / identity
# ---------------------------------------------------------------------------


def test_post_init_coerces_list_to_tuple(int64_type: IntegerType) -> None:
    dtype = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=int64_type, nullable=True),
        ]
    )

    assert isinstance(dtype.fields, tuple)


def test_children_fields_mirror_fields(int64_type: IntegerType) -> None:
    fields = (
        Field(name="a", dtype=int64_type, nullable=True),
        Field(name="b", dtype=int64_type, nullable=True),
    )
    dtype = StructType(fields=fields)

    assert dtype.children_fields == dtype.fields


def test_type_id_is_struct(int64_type: IntegerType) -> None:
    dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    assert dtype.type_id == DataTypeId.STRUCT


def test_handles_arrow_type_matrix() -> None:
    assert (
        StructType.handles_arrow_type(pa.struct([pa.field("a", pa.int64())])) is True
    )
    assert StructType.handles_arrow_type(pa.int64()) is False
    assert StructType.handles_arrow_type(pa.list_(pa.int64())) is False


def test_handles_dict_matrix() -> None:
    assert StructType.handles_dict({"id": int(DataTypeId.STRUCT)}) is True
    assert StructType.handles_dict({"name": "STRUCT"}) is True
    assert StructType.handles_dict({"name": "struct"}) is True
    assert StructType.handles_dict({"name": "ARRAY"}) is False


def test_from_arrow_type_round_trip_preserves_fields() -> None:
    arrow_struct = pa.struct(
        [
            pa.field("x", pa.int64(), nullable=True),
            pa.field("y", pa.string(), nullable=False),
        ]
    )

    dtype = StructType.from_arrow_type(arrow_struct)

    assert isinstance(dtype, StructType)
    assert [f.name for f in dtype.fields] == ["x", "y"]
    produced = dtype.to_arrow()
    assert pa.types.is_struct(produced)
    assert produced.num_fields == 2
    assert produced.field(1).nullable is False


def test_from_arrow_type_rejects_non_struct() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        StructType.from_arrow_type(pa.int64())


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_contains_fields(int64_type: IntegerType) -> None:
    dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])

    payload = dtype.to_dict()

    assert payload["name"] == "STRUCT"
    assert payload["fields"][0]["name"] == "a"


def test_from_dict_round_trip(int64_type: IntegerType) -> None:
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


def test_from_dict_empty_fields_returns_empty_struct() -> None:
    rebuilt = StructType.from_dict({"name": "STRUCT"})

    assert isinstance(rebuilt, StructType)
    assert rebuilt.fields == ()


# ---------------------------------------------------------------------------
# Defaults + DDL
# ---------------------------------------------------------------------------


def test_default_pyobj_variants(int64_type: IntegerType) -> None:
    dtype = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=int64_type, nullable=False),
        ]
    )

    assert dtype.default_pyobj(nullable=True) is None

    default = dtype.default_pyobj(nullable=False)
    assert isinstance(default, dict)
    assert set(default.keys()) == {"a", "b"}


def test_to_databricks_ddl_quotes_field_names(int64_type: IntegerType) -> None:
    dtype = StructType(
        fields=[Field(name="plain", dtype=int64_type, nullable=True)]
    )

    ddl = dtype.to_databricks_ddl()

    assert ddl.startswith("STRUCT<")
    assert ddl.endswith(">")
    assert "`plain`:" in ddl


def test_to_databricks_ddl_escapes_embedded_backticks(int64_type: IntegerType) -> None:
    dtype = StructType(
        fields=[Field(name="we`ird", dtype=int64_type, nullable=True)]
    )

    ddl = dtype.to_databricks_ddl()

    # Embedded backticks are doubled (SQL standard), so the quoted
    # identifier still parses.
    assert "`we``ird`:" in ddl


# ---------------------------------------------------------------------------
# with_fields / to_struct
# ---------------------------------------------------------------------------


def test_with_fields_not_inplace_returns_new_instance(int64_type: IntegerType) -> None:
    dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    replacement = [Field(name="b", dtype=int64_type, nullable=True)]

    new_dtype = dtype.with_fields(replacement, safe=True, inplace=False)

    assert new_dtype is not dtype
    assert [f.name for f in dtype.fields] == ["a"]
    assert [f.name for f in new_dtype.fields] == ["b"]


def test_with_fields_inplace_mutates(int64_type: IntegerType) -> None:
    dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    replacement = [Field(name="b", dtype=int64_type, nullable=True)]

    same = dtype.with_fields(replacement, safe=True, inplace=True)

    assert same is dtype
    assert [f.name for f in dtype.fields] == ["b"]


def test_with_fields_non_safe_coerces_inputs() -> None:
    dtype = StructType(fields=[])

    dtype = dtype.with_fields(
        [{"name": "a", "dtype": {"id": int(DataTypeId.INTEGER)}, "nullable": True}],
        safe=False,
        inplace=False,
    )

    assert len(dtype.fields) == 1
    assert dtype.fields[0].name == "a"
    assert dtype.fields[0].dtype.type_id == DataTypeId.INTEGER


def test_to_struct_returns_self(int64_type: IntegerType) -> None:
    dtype = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    assert dtype.to_struct() is dtype


# ---------------------------------------------------------------------------
# Merge semantics — the core of StructType's behavior.
# ---------------------------------------------------------------------------


def test_merge_same_id_matches_fields_by_name_and_upcasts(
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
            Field(name="a", dtype=DataType.from_arrow_type(pa.int32()), nullable=True),
            Field(
                name="b", dtype=string_type, nullable=True, metadata={"comment": "rhs"}
            ),
        ]
    )

    result = left._merge_with_same_id(right, upcast=True)

    assert isinstance(result, StructType)
    assert [f.name for f in result.fields] == ["a", "b"]
    assert result.fields[0].nullable is True
    assert result.fields[0].arrow_type == pa.int64()
    assert result.fields[1].metadata == {b"comment": b"rhs"}


def test_merge_same_id_appends_unmatched_fields_when_names_differ(
    int64_type: IntegerType,
    string_type,
) -> None:
    left = StructType(
        fields=[
            Field(name="left_a", dtype=int64_type, nullable=False),
            Field(name="left_b", dtype=string_type, nullable=True),
        ]
    )
    right = StructType(
        fields=[
            Field(
                name="right_a",
                dtype=DataType.from_arrow_type(pa.int32()),
                nullable=True,
            ),
            Field(
                name="right_b",
                dtype=DataType.from_arrow_type(pa.large_string()),
                nullable=True,
            ),
        ]
    )

    result = left._merge_with_same_id(right, upcast=True)

    assert [f.name for f in result.fields] == [
        "left_a",
        "left_b",
        "right_a",
        "right_b",
    ]
    assert result.fields[0].arrow_type == pa.int64()
    assert result.fields[0].nullable is False


def test_merge_same_id_keeps_left_only_fields(
    int64_type: IntegerType,
    string_type,
) -> None:
    left_only = Field(name="left_only", dtype=int64_type, nullable=True)
    shared = Field(name="shared", dtype=string_type, nullable=True)

    left = StructType(fields=[left_only, shared])
    right = StructType(
        fields=[Field(name="shared", dtype=string_type, nullable=True)]
    )

    result = left._merge_with_same_id(right)

    assert [f.name for f in result.fields] == ["left_only", "shared"]
    assert result.fields[0] == left_only


@pytest.mark.parametrize(
    "mode",
    [None, SaveMode.APPEND, SaveMode.UPSERT, SaveMode.AUTO],
)
def test_merge_same_id_appends_right_only_fields_for_append_like_modes(
    int64_type: IntegerType,
    string_type,
    mode,
) -> None:
    left = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    right = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="c", dtype=int64_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(right, mode=mode)

    assert [f.name for f in result.fields] == ["a", "b", "c"]


def test_merge_same_id_overwrite_mode_drops_right_only_fields(
    int64_type: IntegerType,
    string_type,
) -> None:
    left = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])
    right = StructType(
        fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(right, mode=SaveMode.OVERWRITE)

    assert [f.name for f in result.fields] == ["a"]


def test_merge_same_id_preserves_left_order_and_appends_new_right_fields(
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
            Field(name="a", dtype=DataType.from_arrow_type(pa.int32()), nullable=True),
            Field(name="c", dtype=bool_type, nullable=True),
        ]
    )

    result = left._merge_with_same_id(right, mode=SaveMode.APPEND, upcast=True)

    assert [f.name for f in result.fields] == ["b", "a", "c"]
    assert result.fields[1].arrow_type == pa.int64()
    assert result.fields[1].nullable is True


def test_merge_same_id_raises_for_non_struct_other(int64_type: IntegerType) -> None:
    left = StructType(fields=[Field(name="a", dtype=int64_type, nullable=True)])

    with pytest.raises(TypeError, match="Cannot merge StructType with"):
        left._merge_with_same_id(DataType.from_arrow_type(pa.int64()))
