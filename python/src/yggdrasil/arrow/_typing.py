from __future__ import annotations

from typing import TypeAlias, Union

import pyarrow as pa

__all__ = [
    "ArrowDataType",
    "ArrowPrimitiveDataType",
    "ArrowTemporalDataType",
    "ArrowBinaryDataType",
    "ArrowNestedDataType",
    "ArrowArray",
    "ArrowNumericArray",
    "ArrowTemporalArray",
    "ArrowBinaryArray",
    "ArrowNestedArray",
    "ArrowArrayLike",
    "ArrowTabular",
]

# -------------------------------------------------------------------
# Concrete DataType classes inferred from pyarrow factory functions
# This is way more IDE-friendly than fake references like pa.BoolType
# -------------------------------------------------------------------

NullType = type(pa.null())
BoolType = type(pa.bool_())

Int8Type = type(pa.int8())
Int16Type = type(pa.int16())
Int32Type = type(pa.int32())
Int64Type = type(pa.int64())

UInt8Type = type(pa.uint8())
UInt16Type = type(pa.uint16())
UInt32Type = type(pa.uint32())
UInt64Type = type(pa.uint64())

Float16Type = type(pa.float16())
Float32Type = type(pa.float32())
Float64Type = type(pa.float64())

Decimal128Type = type(pa.decimal128(38, 10))
TimestampType = type(pa.timestamp("us"))

Date32Type = type(pa.date32())
Date64Type = type(pa.date64())
Time32Type = type(pa.time32("s"))
Time64Type = type(pa.time64("us"))
DurationType = type(pa.duration("us"))

BinaryType = type(pa.binary())
LargeBinaryType = type(pa.large_binary())
StringType = type(pa.string())
LargeStringType = type(pa.large_string())
FixedSizeBinaryType = type(pa.binary(16))

ListType = type(pa.list_(pa.int64()))
LargeListType = type(pa.large_list(pa.int64()))
StructType = type(pa.struct([("x", pa.int64())]))
MapType = type(pa.map_(pa.string(), pa.int64()))
DictionaryType = type(pa.dictionary(pa.int32(), pa.string()))

# sparse_union / dense_union both produce UnionType
UnionType = type(pa.sparse_union([pa.field("x", pa.int64())]))

ArrowPrimitiveDataType: TypeAlias = Union[
    NullType,
    BoolType,
    Int8Type,
    Int16Type,
    Int32Type,
    Int64Type,
    UInt8Type,
    UInt16Type,
    UInt32Type,
    UInt64Type,
    Float16Type,
    Float32Type,
    Float64Type,
    Decimal128Type,
]

ArrowTemporalDataType: TypeAlias = Union[
    TimestampType,
    Date32Type,
    Date64Type,
    Time32Type,
    Time64Type,
    DurationType,
]

ArrowBinaryDataType: TypeAlias = Union[
    BinaryType,
    LargeBinaryType,
    StringType,
    LargeStringType,
    FixedSizeBinaryType,
]

ArrowNestedDataType: TypeAlias = Union[
    ListType,
    LargeListType,
    StructType,
    MapType,
    DictionaryType,
    UnionType,
]

ArrowDataType: TypeAlias = Union[
    ArrowPrimitiveDataType,
    ArrowTemporalDataType,
    ArrowBinaryDataType,
    ArrowNestedDataType,
]

# -------------------------------------------------------------------
# Arrays
# These concrete array classes are exposed by pyarrow directly
# -------------------------------------------------------------------

ArrowNumericArray: TypeAlias = Union[
    pa.BooleanArray,
    pa.Int8Array,
    pa.Int16Array,
    pa.Int32Array,
    pa.Int64Array,
    pa.UInt8Array,
    pa.UInt16Array,
    pa.UInt32Array,
    pa.UInt64Array,
    pa.HalfFloatArray,
    pa.FloatArray,
    pa.DoubleArray,
    pa.Decimal128Array,
]

ArrowTemporalArray: TypeAlias = Union[
    pa.Date32Array,
    pa.Date64Array,
    pa.Time32Array,
    pa.Time64Array,
    pa.TimestampArray,
    pa.DurationArray,
]

ArrowBinaryArray: TypeAlias = Union[
    pa.BinaryArray,
    pa.LargeBinaryArray,
    pa.StringArray,
    pa.LargeStringArray,
    pa.FixedSizeBinaryArray,
]

ArrowNestedArray: TypeAlias = Union[
    pa.ListArray,
    pa.LargeListArray,
    pa.FixedSizeListArray,
    pa.StructArray,
    pa.MapArray,
    pa.DictionaryArray,
    pa.UnionArray,
]

ArrowArray: TypeAlias = Union[
    pa.NullArray,
    ArrowNumericArray,
    ArrowTemporalArray,
    ArrowBinaryArray,
    ArrowNestedArray,
]

ArrowArrayLike: TypeAlias = Union[
    ArrowArray,
    pa.ChunkedArray,
]

ArrowTabular: TypeAlias = Union[
    pa.Table,
    pa.RecordBatch,
]