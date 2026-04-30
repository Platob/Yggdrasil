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

    DICTIONARY = 64
    JSON = 65
    ENUM = 66
    UNION = 67

    ARRAY = 100
    MAP = 101
    STRUCT = 102


    @property
    def is_primitive(self) -> bool:
        return 0 < self.value < 32

    @property
    def is_extension(self) -> bool:
        return 32 <= self.value < 100

    @property
    def is_any_or_null(self) -> bool:
        return self.value <= 1

    @property
    def is_nested(self) -> bool:
        return self.value >= 100

    @property
    def is_temporal(self) -> bool:
        return self.value in (6, 7, 8, 9)

