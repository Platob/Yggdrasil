"""Numeric :class:`DataType` family — integers, floats, decimals.

Public API stays flat: ``from .numeric import IntegerType``,
``Int32Type``, ``Float64Type``, ``DecimalType``, etc. The split
into submodules is an implementation detail — :mod:`base` for the
abstract :class:`NumericType` + shared cross-engine empty-string
normalization, :mod:`integer` / :mod:`floating_point` for each
sized family + dispatch registry, :mod:`decimal` for fixed-
precision decimals.
"""
from __future__ import annotations

from .base import NumericType
from .decimal import DecimalType
from .floating_point import (
    Float8Type,
    Float16Type,
    Float32Type,
    Float64Type,
    FloatingPointType,
)
from .integer import (
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    IntegerType,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
)


__all__ = [
    "NumericType",
    "IntegerType",
    "Int8Type",
    "Int16Type",
    "Int32Type",
    "Int64Type",
    "UInt8Type",
    "UInt16Type",
    "UInt32Type",
    "UInt64Type",
    "FloatingPointType",
    "Float8Type",
    "Float16Type",
    "Float32Type",
    "Float64Type",
    "DecimalType",
]
