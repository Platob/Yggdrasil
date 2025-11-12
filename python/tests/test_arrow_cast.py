from __future__ import annotations

import pathlib
import sys

import pyarrow as pa
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from yggdrasil.data.arrow.arrow_cast import (
    ARROW_CAST_REGISTRY,
    ArrowCastRegistry,
    ArrowCaster,
    ArrowUtility,
)


def test_ensure_arrow_field_wraps_datatype() -> None:
    source_type = pa.int32()

    field = ArrowUtility.ensure_arrow_field(source_type)

    assert isinstance(field, pa.Field)
    assert field.name == "value"
    assert field.type == source_type


def test_can_convert_nested_list_to_wider_integer() -> None:
    source_field = pa.field("numbers", pa.list_(pa.int32()))
    target_field = pa.field("numbers", pa.list_(pa.int64()))

    assert ArrowUtility.can_convert_arrow_fields(source_field, target_field)


def test_can_convert_struct_children() -> None:
    source_struct = pa.struct([
        pa.field("name", pa.string()),
        pa.field("age", pa.int32()),
    ])
    target_struct = pa.struct([
        pa.field("name", pa.large_string()),
        pa.field("age", pa.int64()),
    ])

    assert ArrowUtility.can_convert_arrow_fields(source_struct, target_struct)


def test_cast_array_promotes_integer_list() -> None:
    source_field = pa.field("numbers", pa.list_(pa.int32()))
    target_field = pa.field("numbers", pa.list_(pa.int64()))
    caster = ArrowCaster(source_field=source_field, target_field=target_field)

    values = pa.array([[1, 2, 3]], type=source_field.type)
    cast_values = caster.cast_array(values)

    assert cast_values.type == target_field.type
    assert cast_values.to_pylist() == [[1, 2, 3]]


def test_cast_scalar_to_target_field() -> None:
    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())
    caster = ArrowCaster(source_field=source_field, target_field=target_field)

    scalar = pa.scalar(42, type=source_field.type)
    cast_scalar = caster.cast_scalar(scalar)

    assert cast_scalar.type == target_field.type
    assert cast_scalar.as_py() == 42


def test_registry_builds_and_caches_caster() -> None:
    registry = ArrowCastRegistry()
    source_field = pa.field("value", pa.int32())
    target_field = pa.field("value", pa.int64())

    first = registry.get_or_build(source_field, target_field)
    second = registry.get_or_build(source_field, target_field)

    assert isinstance(first, ArrowCaster)
    assert first is second


def test_singleton_registry_is_shared_instance() -> None:
    instance_one = ArrowCastRegistry.instance()
    instance_two = ArrowCastRegistry.instance()

    assert instance_one is instance_two
    assert ARROW_CAST_REGISTRY is instance_one
