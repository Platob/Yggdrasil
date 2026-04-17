"""Pure-Python tests for MapType — no engine round-trips."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.io import SaveMode  # noqa: F401 — kept for future merge-mode tests


# ---------------------------------------------------------------------------
# from_key_value
# ---------------------------------------------------------------------------


def test_from_key_value_forces_canonical_names_and_key_non_nullable(
    int64_type, string_type
) -> None:
    result = MapType.from_key_value(
        key_field=Field(name="nope", dtype=string_type, nullable=True),
        value_field=Field(name="also_nope", dtype=int64_type, nullable=True),
    )

    assert result.key_field.name == "key"
    assert result.key_field.nullable is False
    assert result.value_field.name == "value"
    assert result.value_field.nullable is True


def test_from_key_value_accepts_raw_dtypes(int64_type, string_type) -> None:
    result = MapType.from_key_value(
        key_field=string_type,
        value_field=int64_type,
        keys_sorted=True,
    )

    assert isinstance(result, MapType)
    assert result.keys_sorted is True
    assert result.key_field.dtype is string_type
    assert result.value_field.dtype is int64_type


# ---------------------------------------------------------------------------
# Type identity
# ---------------------------------------------------------------------------


def test_type_id_is_map(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type)
    assert dtype.type_id == DataTypeId.MAP


def test_children_fields_is_entries_struct(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type)

    children = dtype.children_fields

    assert len(children) == 1
    assert children[0].name == "entries"
    assert isinstance(children[0].dtype, StructType)


def test_handles_dict_matrix() -> None:
    assert MapType.handles_dict({"id": int(DataTypeId.MAP)}) is True
    assert MapType.handles_dict({"name": "MAP"}) is True
    assert MapType.handles_dict({"name": "map"}) is True
    assert MapType.handles_dict({"name": "ARRAY"}) is False


def test_handles_arrow_type_matrix() -> None:
    assert MapType.handles_arrow_type(pa.map_(pa.string(), pa.int64())) is True
    assert MapType.handles_arrow_type(pa.list_(pa.int64())) is False
    assert MapType.handles_arrow_type(pa.int64()) is False


# ---------------------------------------------------------------------------
# Dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_omits_keys_sorted_when_default(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type)

    payload = dtype.to_dict()

    assert payload["name"] == "MAP"
    assert "keys_sorted" not in payload
    assert payload["item_field"]["name"] == "entries"


def test_to_dict_includes_keys_sorted_when_true(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type, keys_sorted=True)

    assert dtype.to_dict()["keys_sorted"] is True


def test_from_dict_round_trip_preserves_key_value_and_sort(
    int64_type, string_type
) -> None:
    original = MapType.from_key_value(string_type, int64_type, keys_sorted=True)

    rebuilt = MapType.from_dict(original.to_dict())

    assert isinstance(rebuilt, MapType)
    assert rebuilt.keys_sorted is True
    assert rebuilt.key_field.dtype.type_id == string_type.type_id
    assert rebuilt.value_field.dtype.type_id == int64_type.type_id


# ---------------------------------------------------------------------------
# Defaults + DDL
# ---------------------------------------------------------------------------


def test_default_pyobj_variants(int64_type, string_type) -> None:
    dtype = MapType.from_key_value(string_type, int64_type)

    assert dtype.default_pyobj(nullable=True) is None
    assert dtype.default_pyobj(nullable=False) == {}


def test_to_databricks_ddl_uses_key_value_types(int64_type, string_type) -> None:
    ddl = MapType.from_key_value(string_type, int64_type).to_databricks_ddl()

    assert ddl.upper().startswith("MAP<")
    assert ddl.endswith(">")
    assert "," in ddl


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_with_same_id_upcasts_value_bytes(
    int64_type, int32_type, string_type
) -> None:
    left = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=False),
    )
    right = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int32_type, nullable=True),
    )

    result = left._merge_with_same_id(right, upcast=True)

    assert isinstance(result, MapType)
    assert result.key_field.arrow_type == pa.string()
    assert result.key_field.nullable is False
    assert result.value_field.arrow_type == pa.int64()
    assert result.value_field.nullable is True


@pytest.mark.parametrize(
    "left_sorted,right_sorted,expected",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_merge_with_same_id_ors_keys_sorted(
    int64_type, string_type, left_sorted, right_sorted, expected
) -> None:
    left = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=True),
        keys_sorted=left_sorted,
    )
    right = MapType.from_key_value(
        key_field=Field(name="key", dtype=string_type, nullable=False),
        value_field=Field(name="value", dtype=int64_type, nullable=True),
        keys_sorted=right_sorted,
    )

    assert left._merge_with_same_id(right).keys_sorted is expected


# ---------------------------------------------------------------------------
# Internal helper cache
# ---------------------------------------------------------------------------


def test_string_key_source_field_is_cached() -> None:
    from yggdrasil.data.types.nested.map import _string_key_source_field

    first = _string_key_source_field()
    second = _string_key_source_field()

    assert first is second
    assert first.name == "key"
    assert first.nullable is False
