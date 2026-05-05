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

    # Specialized fixed-width numeric ids. Each one has its own concrete
    # ``DataType`` subclass (``Int8Type``, ``UInt32Type``, ``Float64Type``,
    # ...). The abstract ``INTEGER`` / ``FLOAT`` ids stay around for the
    # cases where the size isn't (yet) known — the matching abstract
    # ``IntegerType`` / ``FloatingPointType`` instance is what
    # ``__new__`` falls back to when no specialized class is registered.
    INT8 = 12
    INT16 = 13
    INT32 = 14
    INT64 = 15
    UINT8 = 16
    UINT16 = 17
    UINT32 = 18
    UINT64 = 19
    FLOAT16 = 20
    FLOAT32 = 21
    FLOAT64 = 22

    DICTIONARY = 64
    JSON = 65
    ENUM = 66
    UNION = 67

    ARRAY = 100
    MAP = 101
    STRUCT = 102


    @property
    def is_scalar(self) -> bool:
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

    @property
    def is_integer(self) -> bool:
        return self.value == 3 or 12 <= self.value <= 19

    @property
    def is_signed_integer(self) -> bool:
        return 12 <= self.value <= 15

    @property
    def is_unsigned_integer(self) -> bool:
        return 16 <= self.value <= 19

    @property
    def is_floating_point(self) -> bool:
        return self.value == 4 or 20 <= self.value <= 22

    @property
    def is_numeric(self) -> bool:
        return self.is_integer or self.is_floating_point or self.value == 5
