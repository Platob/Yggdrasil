"""Dictionary-encoded / enum :class:`DataType` family.

Three flavors live here:

* :class:`DictionaryType` — the generic dictionary-encoded type, with
  any :class:`PrimitiveType` value type and an explicit ``categories``
  tuple. Maps to ``pa.dictionary(int32, value_type)`` on the Arrow
  side, ``pl.Enum`` on Polars when the value type is string, and
  degrades to ``value_type.to_spark()`` (with a ``CASE WHEN`` gate)
  on Spark.
* :class:`EnumType` — semantic enum subclass of ``DictionaryType``;
  same wire format, distinct ``DataTypeId`` so a column declared as
  ``enum`` round-trips through ``to_dict`` / ``from_dict``.
* :class:`StrEnumType` / :class:`IntEnumType` — typed convenience
  subclasses that pin ``value_type`` to ``StringType`` / ``IntegerType``
  and carry their own ``DataTypeId``. Mirror the stdlib ``StrEnum`` /
  ``IntEnum`` distinction at the schema layer.
"""
from __future__ import annotations

from .dictionary import DictionaryType
from .enum import EnumType, IntEnumType, StrEnumType


__all__ = [
    "DictionaryType",
    "EnumType",
    "StrEnumType",
    "IntEnumType",
]
