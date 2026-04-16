"""Shared fixtures for MapType tests."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data import DataType
from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import MapType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive import IntegerType


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
def source_map_field(int64_type: IntegerType, string_type) -> Field:
    """``map<string, int64>`` — canonical map source."""
    return Field(
        name="source_map",
        dtype=MapType.from_key_value(
            key_field=Field(name="key", dtype=string_type, nullable=False),
            value_field=Field(name="value", dtype=int64_type, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def target_map_field(string_type) -> Field:
    """``map<string, string>`` — cast destination."""
    return Field(
        name="target_map",
        dtype=MapType.from_key_value(
            key_field=Field(name="key", dtype=string_type, nullable=False),
            value_field=Field(name="value", dtype=string_type, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def source_list_of_struct_field(int64_type: IntegerType, string_type) -> Field:
    """``list<struct<key, value>>`` source for list->map casts."""
    entry_struct = StructType(
        fields=[
            Field(name="key", dtype=string_type, nullable=False),
            Field(name="value", dtype=int64_type, nullable=True),
        ]
    )
    return Field(
        name="source_entries",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=entry_struct, nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def source_struct_to_map_field(int64_type: IntegerType) -> Field:
    """``struct<a, b, c>`` source for struct->map casts (field names become keys)."""
    return Field(
        name="source_struct",
        dtype=StructType(
            fields=[
                Field(name="a", dtype=int64_type, nullable=True),
                Field(name="b", dtype=int64_type, nullable=True),
                Field(name="c", dtype=int64_type, nullable=True),
            ]
        ),
        nullable=True,
    )
