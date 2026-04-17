from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.types.primitive import (
    DurationType,
    IntegerType,
    TimestampType,
    TimeType,
)


def test_time_type_from_arrow_type_rejects_non_time() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        TimeType.from_arrow_type(pa.int64())


def test_time_type_from_arrow_type_accepts_time32_and_time64() -> None:
    t32 = TimeType.from_arrow_type(pa.time32("ms"))
    t64 = TimeType.from_arrow_type(pa.time64("us"))

    assert t32.byte_size == 4
    assert t32.unit == "ms"

    assert t64.byte_size == 8
    assert t64.unit == "us"


def test_timestamp_type_from_arrow_type_rejects_non_timestamp() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        TimestampType.from_arrow_type(pa.int64())


def test_timestamp_type_from_arrow_type_preserves_tz() -> None:
    ts = TimestampType.from_arrow_type(pa.timestamp("us", tz="UTC"))

    assert ts.unit == "us"
    assert ts.tz == "UTC"


def test_duration_type_from_arrow_type_rejects_non_duration() -> None:
    with pytest.raises(TypeError, match="Unsupported Arrow data type"):
        DurationType.from_arrow_type(pa.int64())


def test_duration_type_from_arrow_type_accepts_all_units() -> None:
    for unit in ("s", "ms", "us", "ns"):
        result = DurationType.from_arrow_type(pa.duration(unit))
        assert result.unit == unit
        assert result.byte_size == 8


def test_integer_type_str_signed() -> None:
    assert str(IntegerType(byte_size=1, signed=True)) == "int8"
    assert str(IntegerType(byte_size=2, signed=True)) == "int16"
    assert str(IntegerType(byte_size=4, signed=True)) == "int32"
    assert str(IntegerType(byte_size=8, signed=True)) == "int64"


def test_integer_type_str_unsigned() -> None:
    assert str(IntegerType(byte_size=1, signed=False)) == "uint8"
    assert str(IntegerType(byte_size=2, signed=False)) == "uint16"
    assert str(IntegerType(byte_size=4, signed=False)) == "uint32"
    assert str(IntegerType(byte_size=8, signed=False)) == "uint64"
