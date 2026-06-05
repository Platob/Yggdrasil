"""Shared fixtures for ArrayType tests.

Fixtures live here (rather than in the parent directory) so each
per-engine test module can focus on its own assertions without
redefining field scaffolding.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.data_field import Field
from yggdrasil.data.types.nested import MapType, StructType
from yggdrasil.data.types.nested.array import ArrayType
from yggdrasil.data.types.primitive import IntegerType, StringType


# ---------------------------------------------------------------------------
# Array-of-scalar fields
# ---------------------------------------------------------------------------


@pytest.fixture
def source_array_field() -> Field:
    """``list<int64>`` — the canonical source when casting arrays."""
    return Field(
        name="source_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=IntegerType(byte_size=8, signed=True),
                nullable=True,
            ),
        ),
        nullable=True,
    )


@pytest.fixture
def target_array_field() -> Field:
    """``list<string>`` — variable-length target for list-cast tests."""
    return Field(
        name="target_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def target_large_array_field() -> Field:
    """``large_list<string>``."""
    return Field(
        name="target_large_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
            large=True,
        ),
        nullable=True,
    )


@pytest.fixture
def target_fixed_array_field() -> Field:
    """``fixed_size_list<string, 2>``."""
    return Field(
        name="target_fixed_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
            list_size=2,
        ),
        nullable=True,
    )


@pytest.fixture
def target_view_array_field() -> Field:
    """``list_view<string>`` — intentionally unsupported as a cast target."""
    return Field(
        name="target_view_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
            view=True,
        ),
        nullable=True,
    )


# ---------------------------------------------------------------------------
# Map- and struct-shaped neighbours, used for map->list and list->list
# cast coverage.
# ---------------------------------------------------------------------------


@pytest.fixture
def source_map_field() -> Field:
    return Field(
        name="source_map",
        dtype=MapType(
            item_field=Field(
                name="entries",
                dtype=StructType(
                    fields=[
                        Field(name="key", dtype=StringType(), nullable=False),
                        Field(
                            name="value",
                            dtype=IntegerType(byte_size=8, signed=True),
                            nullable=True,
                        ),
                    ]
                ),
                nullable=False,
            ),
        ),
        nullable=True,
    )


def _entries_struct() -> StructType:
    return StructType(
        fields=[
            Field(name="key", dtype=StringType(), nullable=False),
            Field(name="value", dtype=StringType(), nullable=True),
        ]
    )


@pytest.fixture
def target_entries_array_field() -> Field:
    """``list<struct<key, value>>`` — the canonical map->list target."""
    return Field(
        name="target_entries_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=_entries_struct(), nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def target_entries_large_array_field() -> Field:
    return Field(
        name="target_entries_large_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=_entries_struct(), nullable=True),
            large=True,
        ),
        nullable=True,
    )


@pytest.fixture
def invalid_target_entries_scalar_array_field() -> Field:
    """``list<string>`` — rejected as a map->list target (item must be a 2-ary struct)."""
    return Field(
        name="invalid_target_entries_scalar_array",
        dtype=ArrayType(
            item_field=Field(name="item", dtype=StringType(), nullable=True),
        ),
        nullable=True,
    )


@pytest.fixture
def invalid_target_entries_struct_one_field_array_field() -> Field:
    """``list<struct<key>>`` — rejected: struct must have exactly 2 fields."""
    return Field(
        name="invalid_target_entries_struct_one_field_array",
        dtype=ArrayType(
            item_field=Field(
                name="item",
                dtype=StructType(
                    fields=[Field(name="key", dtype=StringType(), nullable=False)]
                ),
                nullable=True,
            ),
        ),
        nullable=True,
    )
