"""Tests for XXHIntType and round-trip safe signed<->unsigned integer casts."""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.data_field import Field
from yggdrasil.data.types import IntegerType, XXHIntType
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.polars.tests import PolarsTestCase


_ROUND_TRIP_U64 = [0, 1, 42, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 1]
_ROUND_TRIP_I64 = [-(2**63), -(2**63) + 1, -1, 0, 1, 42, 2**63 - 1]


class TestXXHIntType:
    def test_type_id_is_integer(self):
        assert XXHIntType().type_id is DataTypeId.INTEGER

    def test_defaults_to_signed_int64(self):
        t = XXHIntType()
        assert t.signed is True
        assert t.byte_size == 8
        assert t.to_arrow() == pa.int64()

    def test_unsigned_variant_is_uint64(self):
        t = XXHIntType(signed=False)
        assert t.to_arrow() == pa.uint64()

    def test_str_repr_marks_intent(self):
        assert str(XXHIntType()) == "xxhint64"
        assert str(XXHIntType(signed=False)) == "xxhuint64"

    def test_to_dict_carries_xxhint_flag(self):
        d = XXHIntType().to_dict()
        assert d["xxhint"] is True
        assert d["name"] == "XXHINT64"

    def test_handles_arrow_type_defers_to_integer(self):
        # XXHIntType is caller-declared; integer64 alone shouldn't round-trip
        # into XXHIntType via from_arrow_type.
        assert XXHIntType.handles_arrow_type(pa.int64()) is False

    def test_from_arrow_type_accepts_int64_and_uint64(self):
        assert XXHIntType.from_arrow_type(pa.int64()).signed is True
        assert XXHIntType.from_arrow_type(pa.uint64()).signed is False

    def test_from_dict_marks_intent(self):
        d = {"id": int(DataTypeId.INTEGER), "name": "XXHINT64", "xxhint": True}
        t = XXHIntType.from_dict(d)
        assert isinstance(t, XXHIntType)
        assert t.signed is True

        d2 = {"id": int(DataTypeId.INTEGER), "name": "XXHUINT64", "xxhint": True}
        t2 = XXHIntType.from_dict(d2)
        assert t2.signed is False

    def test_autotag_includes_xxhint(self):
        tags = XXHIntType().autotag()
        assert tags.get(b"xxhint") == b"true"


class TestArrowIntegerRoundTrip(ArrowTestCase):
    def _cast(self, array, src_dtype, tgt_dtype, safe=True):
        src_f = Field(name="h", dtype=src_dtype)
        tgt_f = Field(name="h", dtype=tgt_dtype)
        opts = CastOptions(source_field=src_f, target_field=tgt_f, safe=safe)
        return tgt_f.cast_arrow_array(array, options=opts)

    def test_uint64_to_int64_preserves_bits(self):
        arr = pa.array(_ROUND_TRIP_U64, type=pa.uint64())
        casted = self._cast(
            arr,
            IntegerType(byte_size=8, signed=False),
            IntegerType(byte_size=8, signed=True),
            safe=True,
        )
        assert casted.type == pa.int64()
        # Values above INT64_MAX reinterpret via two's-complement.
        assert casted.to_pylist() == [
            v if v < 2**63 else v - 2**64 for v in _ROUND_TRIP_U64
        ]

    def test_int64_to_uint64_preserves_bits(self):
        arr = pa.array(_ROUND_TRIP_I64, type=pa.int64())
        casted = self._cast(
            arr,
            IntegerType(byte_size=8, signed=True),
            IntegerType(byte_size=8, signed=False),
            safe=True,
        )
        assert casted.type == pa.uint64()
        assert casted.to_pylist() == [
            v if v >= 0 else v + 2**64 for v in _ROUND_TRIP_I64
        ]

    def test_uint64_int64_round_trip(self):
        arr = pa.array(_ROUND_TRIP_U64, type=pa.uint64())
        u64 = IntegerType(byte_size=8, signed=False)
        i64 = IntegerType(byte_size=8, signed=True)
        as_i64 = self._cast(arr, u64, i64, safe=True)
        back = self._cast(as_i64, i64, u64, safe=True)
        assert back.to_pylist() == _ROUND_TRIP_U64

    def test_xxhint_target_preserves_bits(self):
        arr = pa.array(_ROUND_TRIP_U64, type=pa.uint64())
        casted = self._cast(
            arr,
            IntegerType(byte_size=8, signed=False),
            XXHIntType(signed=True),
            safe=True,
        )
        assert casted.type == pa.int64()
        assert casted.to_pylist() == [
            v if v < 2**63 else v - 2**64 for v in _ROUND_TRIP_U64
        ]

    def test_xxhint_round_trip_across_signedness(self):
        arr_u = pa.array(_ROUND_TRIP_U64, type=pa.uint64())
        as_signed = self._cast(
            arr_u,
            XXHIntType(signed=False),
            XXHIntType(signed=True),
            safe=True,
        )
        back = self._cast(
            as_signed,
            XXHIntType(signed=True),
            XXHIntType(signed=False),
            safe=True,
        )
        assert back.to_pylist() == _ROUND_TRIP_U64

    def test_uint8_to_int8_preserves_bits(self):
        arr = pa.array([0, 1, 127, 128, 255], type=pa.uint8())
        casted = self._cast(
            arr,
            IntegerType(byte_size=1, signed=False),
            IntegerType(byte_size=1, signed=True),
            safe=True,
        )
        assert casted.type == pa.int8()
        assert casted.to_pylist() == [0, 1, 127, -128, -1]


class TestPolarsIntegerRoundTrip(PolarsTestCase):
    def _cast(self, series, src_dtype, tgt_dtype, safe=True):
        src_f = Field(name="h", dtype=src_dtype)
        tgt_f = Field(name="h", dtype=tgt_dtype)
        opts = CastOptions(source_field=src_f, target_field=tgt_f, safe=safe)
        return tgt_f.cast_polars_series(series, options=opts)

    def test_uint64_to_int64_preserves_bits(self):
        pl = self.pl
        s = pl.Series("h", _ROUND_TRIP_U64, dtype=pl.UInt64)
        casted = self._cast(
            s,
            IntegerType(byte_size=8, signed=False),
            IntegerType(byte_size=8, signed=True),
            safe=True,
        )
        assert casted.dtype == pl.Int64
        assert casted.to_list() == [
            v if v < 2**63 else v - 2**64 for v in _ROUND_TRIP_U64
        ]

    def test_int64_to_uint64_preserves_bits(self):
        pl = self.pl
        s = pl.Series("h", _ROUND_TRIP_I64, dtype=pl.Int64)
        casted = self._cast(
            s,
            IntegerType(byte_size=8, signed=True),
            IntegerType(byte_size=8, signed=False),
            safe=True,
        )
        assert casted.dtype == pl.UInt64
        assert casted.to_list() == [
            v if v >= 0 else v + 2**64 for v in _ROUND_TRIP_I64
        ]

    def test_round_trip(self):
        pl = self.pl
        s = pl.Series("h", _ROUND_TRIP_U64, dtype=pl.UInt64)
        u64 = IntegerType(byte_size=8, signed=False)
        i64 = IntegerType(byte_size=8, signed=True)
        as_i64 = self._cast(s, u64, i64, safe=True)
        back = self._cast(as_i64, i64, u64, safe=True)
        assert back.to_list() == _ROUND_TRIP_U64

    def test_expr_round_trip(self):
        pl = self.pl
        df = pl.DataFrame({"h": pl.Series(_ROUND_TRIP_U64, dtype=pl.UInt64)})
        u64 = IntegerType(byte_size=8, signed=False)
        i64 = IntegerType(byte_size=8, signed=True)

        src_f = Field(name="h", dtype=u64)
        tgt_f = Field(name="h", dtype=i64)
        opts = CastOptions(source_field=src_f, target_field=tgt_f, safe=True)
        out = df.select(tgt_f.cast_polars_expr(pl.col("h"), options=opts))
        assert out.to_series().to_list() == [
            v if v < 2**63 else v - 2**64 for v in _ROUND_TRIP_U64
        ]
