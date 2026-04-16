from __future__ import annotations

from enum import IntEnum

__all__ = [
    "DataTypeId",
]


class DataTypeId(IntEnum):
    OBJECT = 0
    NULL = 1
    BOOL = 2
    INTEGER = 3
    FLOAT = 4
    DECIMAL = 5
    DATE = 6
    TIME = 7
    TIMESTAMP = 8
    DURATION = 9
    BINARY = 10
    STRING = 11

    ARRAY = 32
    MAP = 33
    STRUCT = 34
    UNION = 35

    EXTENSION = 64
    DICTIONARY = 65
    JSON = 66
    ENUM = 67

