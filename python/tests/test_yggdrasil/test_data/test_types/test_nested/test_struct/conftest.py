"""Shared fixtures for StructType tests."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import DataType, Field, Schema
from yggdrasil.data.types import (
    DecimalType,
    IntegerType,
    TimestampType,
)
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


# ---------------------------------------------------------------------------
# Tricky-type schemas.
#
# These exercise the tabular cast on column shapes that are easy to get
# wrong: fixed-precision decimal (precision + scale must survive the
# rebuild), timestamp with a non-naive timezone (tz string must travel),
# a nested struct column whose own children also need reordering, and a
# list<struct> column (the list wrapper must not flatten the inner
# struct's children when they swap order).
#
# Same source/target pair drives every engine's test class so callers
# can compare cast behaviour across Arrow / Polars / Pandas / Spark
# against one fixture without re-declaring the schemas.
# ---------------------------------------------------------------------------


@pytest.fixture
def decimal_type() -> DecimalType:
    return DecimalType(precision=10, scale=2)


@pytest.fixture
def timestamp_utc_type() -> TimestampType:
    # us / UTC keeps it readable across Arrow + Polars + Spark; ns would
    # truncate on the Spark path.
    return DataType.from_arrow_type(pa.timestamp("us", tz="UTC"))


@pytest.fixture
def nested_struct_source_dtype(int64_type, string_type) -> StructType:
    return StructType(
        fields=[
            Field(name="x", dtype=int64_type, nullable=True),
            Field(name="y", dtype=string_type, nullable=True),
        ]
    )


@pytest.fixture
def nested_struct_target_dtype(int64_type, string_type) -> StructType:
    # Children swap order: the cast must reorder them inside every row.
    return StructType(
        fields=[
            Field(name="y", dtype=string_type, nullable=True),
            Field(name="x", dtype=int64_type, nullable=True),
        ]
    )


@pytest.fixture
def list_of_struct_source_dtype(nested_struct_source_dtype) -> ArrayType:
    return ArrayType.from_item(
        Field(name="item", dtype=nested_struct_source_dtype, nullable=True)
    )


@pytest.fixture
def list_of_struct_target_dtype(nested_struct_target_dtype) -> ArrayType:
    return ArrayType.from_item(
        Field(name="item", dtype=nested_struct_target_dtype, nullable=True)
    )


@pytest.fixture
def tricky_source_schema(
    int64_type: IntegerType,
    string_type,
    decimal_type: DecimalType,
    timestamp_utc_type: TimestampType,
    nested_struct_source_dtype: StructType,
    list_of_struct_source_dtype: ArrayType,
) -> Schema:
    return Schema(
        inner_fields=[
            # ``drop_me`` is absent from the target → must be selected
            # out of the result.
            Field(name="drop_me", dtype=int64_type, nullable=True),
            Field(name="amount", dtype=decimal_type, nullable=True),
            Field(name="ts", dtype=timestamp_utc_type, nullable=True),
            Field(name="nested", dtype=nested_struct_source_dtype, nullable=True),
            Field(name="items", dtype=list_of_struct_source_dtype, nullable=True),
            Field(name="name", dtype=string_type, nullable=True),
        ]
    )


@pytest.fixture
def tricky_target_schema(
    int64_type: IntegerType,
    string_type,
    decimal_type: DecimalType,
    timestamp_utc_type: TimestampType,
    nested_struct_target_dtype: StructType,
    list_of_struct_target_dtype: ArrayType,
) -> Schema:
    return Schema(
        inner_fields=[
            # Reordered + ``drop_me`` removed + ``missing`` appended.
            # ``nested`` and ``items`` re-declare with swapped child order
            # so the cast also has to descend into each row.
            Field(name="ts", dtype=timestamp_utc_type, nullable=True),
            Field(name="amount", dtype=decimal_type, nullable=True),
            Field(name="items", dtype=list_of_struct_target_dtype, nullable=True),
            Field(name="nested", dtype=nested_struct_target_dtype, nullable=True),
            Field(name="name", dtype=string_type, nullable=True),
            Field(name="missing", dtype=int64_type, nullable=True),
        ]
    )
