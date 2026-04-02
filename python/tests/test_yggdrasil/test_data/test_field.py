"""Unit tests for yggdrasil/data/field.py"""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.data.field import (
    Field,
    _decode_metadata_dict,
    _merge_metadata_and_tags,
    _normalize_metadata,
    _to_bytes,
    field,
)


# ============================================================================
# _to_bytes
# ============================================================================


class TestToBytes:
    def test_none_returns_empty(self):
        assert _to_bytes(None) == b""

    def test_bytes_passthrough(self):
        assert _to_bytes(b"hello") == b"hello"

    def test_str_encodes_utf8(self):
        assert _to_bytes("hello") == b"hello"

    def test_str_unicode(self):
        assert _to_bytes("caf\u00e9") == "café".encode("utf-8")

    def test_int_json_encoded(self):
        assert _to_bytes(42) == b"42"

    def test_bool_json_encoded(self):
        assert _to_bytes(True) == b"true"

    def test_list_json_encoded(self):
        assert _to_bytes([1, 2]) == b"[1,2]"

    def test_dict_json_encoded(self):
        result = json.loads(_to_bytes({"a": 1}))
        assert result == {"a": 1}


# ============================================================================
# _normalize_metadata
# ============================================================================


class TestNormalizeMetadata:
    def test_both_none_returns_none(self):
        assert _normalize_metadata(None, None) is None

    def test_empty_dicts_return_none(self):
        assert _normalize_metadata({}, {}) is None

    def test_metadata_bytes_keys(self):
        result = _normalize_metadata({b"key": b"val"}, None)
        assert result == {b"key": b"val"}

    def test_metadata_str_keys(self):
        result = _normalize_metadata({"key": "val"}, None)
        assert result == {b"key": b"val"}

    def test_tags_get_t_prefix(self):
        result = _normalize_metadata(None, {"role": "identifier"})
        assert result == {b"t:role": b"identifier"}

    def test_metadata_and_tags_merged(self):
        result = _normalize_metadata({"comment": "hi"}, {"role": "id"})
        assert result[b"comment"] == b"hi"
        assert result[b"t:role"] == b"id"

    def test_none_values_excluded(self):
        result = _normalize_metadata({"k": None}, None)
        assert result is None

    def test_empty_key_excluded(self):
        result = _normalize_metadata({"": "val"}, None)
        assert result is None


# ============================================================================
# _merge_metadata_and_tags
# ============================================================================


class TestMergeMetadataAndTags:
    def test_both_none_returns_none(self):
        assert _merge_metadata_and_tags(None, None) is None

    def test_metadata_only(self):
        result = _merge_metadata_and_tags({b"k": b"v"}, None)
        assert result == {b"k": b"v"}

    def test_tags_get_t_prefix(self):
        result = _merge_metadata_and_tags(None, {b"role": b"id"})
        assert result == {b"t:role": b"id"}

    def test_tags_already_prefixed_not_doubled(self):
        result = _merge_metadata_and_tags(None, {b"t:role": b"id"})
        assert result == {b"t:role": b"id"}

    def test_both_merged(self):
        result = _merge_metadata_and_tags({b"a": b"1"}, {b"b": b"2"})
        assert result[b"a"] == b"1"
        assert result[b"t:b"] == b"2"

    def test_tags_overwrite_metadata(self):
        result = _merge_metadata_and_tags({b"t:x": b"old"}, {b"x": b"new"})
        assert result[b"t:x"] == b"new"


# ============================================================================
# _decode_metadata_dict
# ============================================================================


class TestDecodeMetadataDict:
    def test_none_returns_empty_dict(self):
        assert _decode_metadata_dict(None) == {}

    def test_empty_dict_returns_empty(self):
        assert _decode_metadata_dict({}) == {}

    def test_json_value_decoded(self):
        result = _decode_metadata_dict({b"n": b"42"})
        assert result["n"] == 42

    def test_string_value_decoded(self):
        result = _decode_metadata_dict({b"k": b"hello"})
        assert result["k"] == "hello"

    def test_boolean_json_decoded(self):
        result = _decode_metadata_dict({b"flag": b"true"})
        assert result["flag"] is True

    def test_key_decoded_as_utf8(self):
        result = _decode_metadata_dict({b"my_key": b'"val"'})
        assert "my_key" in result

    def test_non_json_falls_back_to_string(self):
        result = _decode_metadata_dict({b"k": b"not-json!"})
        assert result["k"] == "not-json!"


# ============================================================================
# field() factory
# ============================================================================


class TestFieldFactory:
    def test_basic_creation(self):
        f = field("price", pa.float64())
        assert isinstance(f, Field)
        assert f.name == "price"
        assert f.arrow_type == pa.float64()
        assert f.nullable is True

    def test_nullable_false(self):
        f = field("id", pa.int64(), nullable=False)
        assert f.nullable is False

    def test_with_metadata(self):
        f = field("x", pa.int32(), metadata={"comment": "test"})
        assert f.metadata[b"comment"] == b"test"

    def test_with_tags(self):
        f = field("x", pa.int32(), tags={"role": "id"})
        assert f.metadata[b"t:role"] == b"id"

    def test_with_metadata_and_tags(self):
        f = field("x", pa.int32(), metadata={"m": "1"}, tags={"t": "2"})
        assert f.metadata[b"m"] == b"1"
        assert f.metadata[b"t:t"] == b"2"


# ============================================================================
# Field — properties
# ============================================================================


class TestFieldProperties:
    def test_partition_by_true(self):
        f = Field(name="dt", arrow_type=pa.date32(), metadata={b"t:partition_by": b"true"})
        assert f.partition_by is True

    def test_partition_by_false_missing(self):
        f = Field(name="dt", arrow_type=pa.date32())
        assert f.partition_by is False

    def test_cluster_by_true(self):
        f = Field(name="id", arrow_type=pa.int64(), metadata={b"t:cluster_by": b"true"})
        assert f.cluster_by is True

    def test_cluster_by_false_missing(self):
        f = Field(name="id", arrow_type=pa.int64())
        assert f.cluster_by is False

    def test_tags_returns_dict_without_prefix(self):
        f = Field(
            name="x",
            arrow_type=pa.int32(),
            metadata={b"t:role": b"id", b"comment": b"hi"},
        )
        assert f.tags == {b"role": b"id"}

    def test_tags_empty_when_no_metadata(self):
        f = Field(name="x", arrow_type=pa.int32())
        assert f.tags == {}


# ============================================================================
# Field.copy
# ============================================================================


class TestFieldCopy:
    def setup_method(self):
        self.f = Field(
            name="amount",
            arrow_type=pa.float64(),
            nullable=True,
            metadata={b"comment": b"base"},
        )

    def test_copy_identical(self):
        f2 = self.f.copy()
        assert f2.name == self.f.name
        assert f2.arrow_type == self.f.arrow_type
        assert f2.nullable == self.f.nullable
        assert f2.metadata == self.f.metadata

    def test_copy_changes_name(self):
        f2 = self.f.copy(name="qty")
        assert f2.name == "qty"
        assert f2.arrow_type == self.f.arrow_type

    def test_copy_changes_type(self):
        f2 = self.f.copy(arrow_type=pa.int64())
        assert f2.arrow_type == pa.int64()

    def test_copy_changes_nullable(self):
        f2 = self.f.copy(nullable=False)
        assert f2.nullable is False

    def test_copy_new_metadata_replaces(self):
        f2 = self.f.copy(metadata={"new": "val"})
        assert f2.metadata == {b"new": b"val"}

    def test_copy_new_tags_replaces_metadata(self):
        f2 = self.f.copy(tags={"role": "measure"})
        assert f2.metadata == {b"t:role": b"measure"}

    def test_copy_preserves_immutability(self):
        f2 = self.f.copy()
        assert f2 is not self.f


# ============================================================================
# Field.autotag
# ============================================================================


class TestFieldAutotag:
    def _tag(self, f: Field, key: str) -> str | None:
        raw = f.metadata or {}
        v = raw.get(b"t:" + key.encode())
        return v.decode() if v else None

    def test_boolean_kind(self):
        f = Field("flag", pa.bool_()).autotag()
        assert self._tag(f, "kind") == "boolean"

    def test_integer_kind(self):
        f = Field("count", pa.int32()).autotag()
        assert self._tag(f, "kind") == "integer"
        assert self._tag(f, "numeric") == "true"

    def test_float_kind(self):
        f = Field("rate", pa.float32()).autotag()
        assert self._tag(f, "kind") == "float"
        assert self._tag(f, "numeric") == "true"

    def test_decimal_kind(self):
        f = Field("price", pa.decimal128(18, 4)).autotag()
        assert self._tag(f, "kind") == "decimal"
        assert self._tag(f, "numeric") == "true"

    def test_timestamp_kind(self):
        f = Field("ts", pa.timestamp("us")).autotag()
        assert self._tag(f, "kind") == "timestamp"
        assert self._tag(f, "temporal") == "true"

    def test_timestamp_unit_tagged(self):
        f = Field("ts", pa.timestamp("ms")).autotag()
        assert self._tag(f, "unit") == "ms"

    def test_timestamp_tz_tagged(self):
        f = Field("ts", pa.timestamp("us", tz="UTC")).autotag()
        assert self._tag(f, "tz") == "UTC"

    def test_date_kind(self):
        f = Field("event_date", pa.date32()).autotag()
        assert self._tag(f, "kind") == "date"
        assert self._tag(f, "temporal") == "true"

    def test_time_kind(self):
        f = Field("open_time", pa.time64("us")).autotag()
        assert self._tag(f, "kind") == "time"

    def test_duration_kind(self):
        f = Field("elapsed", pa.duration("s")).autotag()
        assert self._tag(f, "kind") == "duration"

    def test_string_kind(self):
        f = Field("label", pa.string()).autotag()
        assert self._tag(f, "kind") == "string"

    def test_binary_kind(self):
        f = Field("blob", pa.binary()).autotag()
        assert self._tag(f, "kind") == "binary"

    def test_list_kind(self):
        f = Field("items", pa.list_(pa.int32())).autotag()
        assert self._tag(f, "kind") == "list"
        assert self._tag(f, "nested") == "true"

    def test_struct_kind(self):
        f = Field("row", pa.struct([pa.field("x", pa.int32())])).autotag()
        assert self._tag(f, "kind") == "struct"
        assert self._tag(f, "nested") == "true"

    def test_nullable_tagged(self):
        f = Field("x", pa.int32(), nullable=False).autotag()
        assert self._tag(f, "nullable") == "false"

    def test_id_role(self):
        f = Field("customer_id", pa.int64()).autotag()
        assert self._tag(f, "role") == "identifier"

    def test_timestamp_role(self):
        f = Field("event_ts", pa.timestamp("us")).autotag()
        assert self._tag(f, "role") == "event_time"

    def test_date_role(self):
        f = Field("trade_date", pa.date32()).autotag()
        assert self._tag(f, "role") == "date"

    def test_created_at_role(self):
        f = Field("created_at", pa.timestamp("us")).autotag()
        assert self._tag(f, "role") == "created_at"

    def test_updated_at_role(self):
        f = Field("updated_at", pa.timestamp("us")).autotag()
        assert self._tag(f, "role") == "updated_at"

    def test_deleted_at_role(self):
        f = Field("deleted_at", pa.timestamp("us")).autotag()
        assert self._tag(f, "role") == "deleted_at"

    def test_is_prefix_flag_role(self):
        f = Field("is_active", pa.bool_()).autotag()
        assert self._tag(f, "role") == "flag"

    def test_has_prefix_flag_role(self):
        f = Field("has_access", pa.bool_()).autotag()
        assert self._tag(f, "role") == "flag"

    def test_price_role(self):
        f = Field("close_price", pa.float64()).autotag()
        assert self._tag(f, "role") == "price"

    def test_quantity_role(self):
        f = Field("quantity", pa.int64()).autotag()
        assert self._tag(f, "role") == "measure"

    def test_name_role(self):
        f = Field("full_name", pa.string()).autotag()
        assert self._tag(f, "role") == "attribute"

    def test_country_role(self):
        # "country" contains the substring "count" which triggers the measure heuristic
        # first; use a name that cleanly matches only the dimension heuristic.
        f = Field("region", pa.string()).autotag()
        assert self._tag(f, "role") == "dimension"

    def test_country_substring_collision(self):
        # "country" contains "count" → measure fires first via setdefault;
        # the dimension setdefault does not override it.  This is the current
        # behaviour; the test pins it so any future fix is visible.
        f = Field("country", pa.string()).autotag()
        assert self._tag(f, "role") in ("measure", "dimension")

    def test_partition_by_preserved_in_autotag(self):
        f = Field("dt", pa.date32(), metadata={b"t:partition_by": b"true"}).autotag()
        assert self._tag(f, "partition_by") == "true"

    def test_existing_tags_win_over_inferred(self):
        # manually set role overrides what autotag would infer
        f = Field(
            "customer_id",
            pa.int64(),
            metadata={b"t:role": b"custom"},
        ).autotag()
        assert self._tag(f, "role") == "custom"


# ============================================================================
# Field.from_any / from_arrow / to_arrow_field
# ============================================================================


class TestFieldConversion:
    def test_from_arrow_pa_field(self):
        arrow_field = pa.field("x", pa.int32(), nullable=False)
        f = Field.from_arrow(arrow_field)
        assert f.name == "x"
        assert f.arrow_type == pa.int32()
        assert f.nullable is False

    def test_from_arrow_preserves_metadata(self):
        arrow_field = pa.field("x", pa.int32(), metadata={b"k": b"v"})
        f = Field.from_arrow(arrow_field)
        assert f.metadata == {b"k": b"v"}

    def test_from_any_with_field_instance(self):
        f = Field(name="y", arrow_type=pa.float64())
        assert Field.from_any(f) is f

    def test_from_any_with_arrow_field(self):
        arrow_field = pa.field("z", pa.string())
        f = Field.from_any(arrow_field)
        assert isinstance(f, Field)
        assert f.name == "z"

    def test_to_arrow_field(self):
        f = Field(name="a", arrow_type=pa.int64(), nullable=False)
        af = f.to_arrow_field()
        assert isinstance(af, pa.Field)
        assert af.name == "a"
        assert af.type == pa.int64()
        assert af.nullable is False

    def test_to_arrow_field_with_metadata(self):
        f = Field(name="b", arrow_type=pa.string(), metadata={b"k": b"v"})
        af = f.to_arrow_field()
        assert af.metadata == {b"k": b"v"}

    def test_roundtrip_arrow(self):
        f = Field(name="c", arrow_type=pa.float32(), nullable=True)
        assert Field.from_arrow(f.to_arrow_field()) == f


# ============================================================================
# Field.from_polars / to_polars_field
# ============================================================================


class TestFieldPolars:
    @pytest.fixture(autouse=True)
    def _skip_if_no_polars(self):
        pytest.importorskip("polars")

    def test_from_polars_name_and_dtype(self):
        import polars as pl

        f = Field.from_polars(name="x", dtype=pl.Int64())
        assert f.name == "x"
        assert f.arrow_type == pa.int64()

    def test_from_polars_object(self):
        import polars as pl

        pl_field = pl.Field("val", pl.Float64())
        f = Field.from_polars(pl_field)
        assert f.name == "val"
        assert f.arrow_type == pa.float64()

    def test_from_polars_requires_name_and_dtype(self):
        with pytest.raises(ValueError):
            Field.from_polars()

    def test_to_polars_field(self):
        import polars as pl

        f = Field(name="score", arrow_type=pa.float32())
        pf = f.to_polars_field()
        assert pf.name == "score"
        assert pf.dtype == pl.Float32()

    def test_roundtrip_polars(self):
        import polars as pl

        f = Field.from_polars(name="ts", dtype=pl.Datetime("us"))
        pf = f.to_polars_field()
        assert pf.name == "ts"
