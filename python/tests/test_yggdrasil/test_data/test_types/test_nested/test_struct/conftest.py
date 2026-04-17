"""Shared fixtures for StructType tests."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import DataType, Field, Schema
from yggdrasil.data.types import IntegerType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.map import MapType
from yggdrasil.data.types.nested.struct import StructType


# ---------------------------------------------------------------------------
# Base dtypes
# ---------------------------------------------------------------------------


@pytest.fixture
def int64_type() -> IntegerType:
    return DataType.from_arrow_type(pa.int64())


@pytest.fixture
def int32_type() -> IntegerType:
    return DataType.from_arrow_type(pa.int32())


@pytest.fixture
def string_type():
    return DataType.from_arrow_type(pa.string())


@pytest.fixture
def bool_type():
    return DataType.from_arrow_type(pa.bool_())


# ---------------------------------------------------------------------------
# Canonical struct source/target fields
#
# The source has fields (a, b); the target re-orders them as (b, c, a) with
# an extra column ``c`` to exercise the "reorder + fill missing" logic
# common to every engine.
# ---------------------------------------------------------------------------


@pytest.fixture
def source_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="source_struct",
        dtype=StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=string_type, nullable=True),
            ]
        ),
        nullable=True,
    )


@pytest.fixture
def target_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="target_struct",
        dtype=StructType(
            fields=[
                Field(name="b", dtype=string_type, nullable=True),
                Field(name="c", dtype=int64_type, nullable=True),
                Field(name="a", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Map / list neighbours used by map->struct and list->struct casts.
# ---------------------------------------------------------------------------


@pytest.fixture
def source_map_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="source_map",
        dtype=MapType.from_key_value(
            key_field=Field(name="key", dtype=string_type, nullable=False),
            value_field=Field(name="value", dtype=int64_type, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def source_list_field(int64_type: IntegerType) -> Field:
    return Field(
        name="source_list",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=int64_type, nullable=True)
        ),
        nullable=True,
    )


@pytest.fixture
def target_list_to_struct_field(int64_type: IntegerType, string_type) -> Field:
    return Field(
        name="target_struct",
        dtype=StructType(
            fields=[
                Field(name="first", dtype=int64_type, nullable=True),
                Field(name="second", dtype=string_type, nullable=True),
                Field(name="third", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Schemas used by tabular casts.
# ---------------------------------------------------------------------------


@pytest.fixture
def source_tabular_schema(int64_type: IntegerType, string_type) -> Schema:
    return Schema(
        inner_fields=[
            Field(name="a", dtype=int64_type, nullable=True),
            Field(name="b", dtype=string_type, nullable=True),
        ]
    )


@pytest.fixture
def target_tabular_schema(int64_type: IntegerType, string_type) -> Schema:
    return Schema(
        inner_fields=[
            Field(name="b", dtype=string_type, nullable=True),
            Field(name="c", dtype=int64_type, nullable=True),
            Field(name="a", dtype=int64_type, nullable=True),
        ]
    )
