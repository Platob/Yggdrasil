"""Integration tests for :class:`SJsonType` and :class:`BJsonType`.

The cases line up the cross-engine surface yggdrasil promises for
JSON-shaped columns:

* parse / dispatch — ``DataType.from_str`` / ``from_dict`` / parser
  aliases land on the right class and round-trip through the dict
  exporter.
* arrow — encode (nested → JSON bytes / text) and decode (JSON →
  nested or primitive) via ``cast_arrow_array``.
* polars — same paths through ``cast_polars_series`` so the polars
  json codec is wired up.
* scalar / pyobj — ``_convert_pyobj`` accepts dicts/lists/strings
  and emits valid JSON for the target storage.
"""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from yggdrasil.arrow.tests import ArrowTestCase
from yggdrasil.data.data_field import Field
from yggdrasil.data.types import (
    ArrayType,
    BJsonType,
    DataType,
    DataTypeId,
    DecimalType,
    IntegerType,
    MapType,
    SJsonType,
    StringType,
    StructType,
)


_INT64 = IntegerType(byte_size=8, signed=True)


# ---------------------------------------------------------------------------
# Parser / dispatch
# ---------------------------------------------------------------------------


class TestJsonDispatch:

    def test_from_str_default_resolves_to_bjson(self) -> None:
        # Bare ``json`` defaults to the binary-shaped variant per the
        # canonical alias table.
        assert isinstance(DataType.from_str("json"), BJsonType)

    @pytest.mark.parametrize(
        "alias",
        ["sjson", "json_string", "json_text", "SJSON"],
    )
    def test_from_str_text_aliases_resolve_to_sjson(self, alias: str) -> None:
        assert isinstance(DataType.from_str(alias), SJsonType)

    @pytest.mark.parametrize(
        "alias",
        ["bjson", "jsonb", "json_binary", "BJSON"],
    )
    def test_from_str_binary_aliases_resolve_to_bjson(self, alias: str) -> None:
        assert isinstance(DataType.from_str(alias), BJsonType)

    def test_type_ids_distinct_and_in_extension_range(self) -> None:
        assert DataTypeId.SJSON.is_extension
        assert DataTypeId.BJSON.is_extension
        assert DataTypeId.SJSON.is_json
        assert DataTypeId.BJSON.is_json
        assert int(DataTypeId.SJSON) != int(DataTypeId.BJSON)

    def test_to_dict_roundtrip(self) -> None:
        s = SJsonType()
        b = BJsonType()
        assert DataType.from_dict(s.to_dict()) == s
        assert DataType.from_dict(b.to_dict()) == b

    def test_pretty_format(self) -> None:
        assert SJsonType().pretty_format() == "sjson"
        assert SJsonType(large=True).pretty_format() == "large_sjson"
        assert BJsonType().pretty_format() == "bjson"
        assert BJsonType(byte_size=16).pretty_format() == "bjson(16)"

    def test_storage_arrow_types(self) -> None:
        # SJSON rides on string storage; BJSON on binary.
        assert SJsonType().to_arrow() == pa.string()
        assert SJsonType(large=True).to_arrow() == pa.large_string()
        assert BJsonType().to_arrow() == pa.binary()
        assert BJsonType(large=True).to_arrow() == pa.large_binary()
        assert BJsonType(byte_size=8).to_arrow() == pa.binary(8)

    def test_handles_arrow_does_not_auto_infer(self) -> None:
        # A plain ``pa.string()`` column must NOT silently land as
        # SJsonType — JSON intent has to come from the user.
        assert SJsonType.handles_arrow_type(pa.string()) is False
        assert BJsonType.handles_arrow_type(pa.binary()) is False
        assert isinstance(DataType.from_arrow_type(pa.string()), StringType)


# ---------------------------------------------------------------------------
# Scalar conversion
# ---------------------------------------------------------------------------


class TestJsonConvertPyobj:

    def test_dict_to_sjson_text(self) -> None:
        out = SJsonType()._convert_pyobj({"a": 1, "b": [1, 2]})
        assert json.loads(out) == {"a": 1, "b": [1, 2]}

    def test_list_to_bjson_bytes(self) -> None:
        out = BJsonType()._convert_pyobj([1, 2, 3])
        assert isinstance(out, bytes)
        assert json.loads(out.decode()) == [1, 2, 3]

    def test_string_passthrough_is_kept_verbatim(self) -> None:
        # The bytes are already JSON text — passthrough preserves them
        # so a caller that already serialised upstream isn't punished
        # by a double-encode.
        assert SJsonType()._convert_pyobj('{"a":1}') == '{"a":1}'

    def test_bytes_decoded_for_sjson(self) -> None:
        assert SJsonType()._convert_pyobj(b'{"a":1}') == '{"a":1}'

    def test_default_pyobj_non_nullable_returns_json_null(self) -> None:
        assert SJsonType().default_pyobj(nullable=False) == "null"
        assert BJsonType().default_pyobj(nullable=False) == b"null"
        assert SJsonType().default_pyobj(nullable=True) is None
        assert BJsonType().default_pyobj(nullable=True) is None


# ---------------------------------------------------------------------------
# Arrow casts — encode / decode against nested types and primitives
# ---------------------------------------------------------------------------


class TestJsonArrowCasts(ArrowTestCase):

    def test_struct_to_sjson_encodes_to_string_array(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        src = self.pa.array(
            [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None],
            type=struct_dtype.to_arrow(),
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        self.assertEqual(out.type, self.pa.string())
        decoded = [None if v is None else json.loads(v) for v in out.to_pylist()]
        self.assertEqual(
            decoded, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, None]
        )

    def test_array_to_bjson_encodes_to_binary_array(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        src = self.pa.array(
            [[1, 2, 3], [4, 5], None], type=arr_dtype.to_arrow()
        )

        bjson = BJsonType()
        out = bjson.cast_arrow_array(
            src,
            source=Field("a", arr_dtype),
            target=Field("a", bjson),
        )

        self.assertEqual(out.type, self.pa.binary())
        decoded = [None if v is None else json.loads(v.decode()) for v in out.to_pylist()]
        self.assertEqual(decoded, [[1, 2, 3], [4, 5], None])

    def test_map_to_sjson_uses_object_encoding(self) -> None:
        map_dtype = MapType.from_key_value(
            key_field=Field("key", StringType()),
            value_field=Field("value", _INT64),
        )
        src = self.pa.array(
            [[("k", 1), ("m", 2)], [("z", 9)]], type=map_dtype.to_arrow()
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("m", map_dtype),
            target=Field("m", sjson),
        )

        decoded = [json.loads(v) for v in out.to_pylist()]
        self.assertEqual(decoded, [{"k": 1, "m": 2}, {"z": 9}])

    def test_sjson_to_struct_decodes(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        src = self.pa.array(
            ['{"a": 1, "b": "x"}', '{"a": 2, "b": "y"}'], type=self.pa.string()
        )

        out = struct_dtype.cast_arrow_array(
            src,
            source=Field("s", SJsonType()),
            target=Field("s", struct_dtype),
        )

        self.assertEqual(out.to_pylist(), [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])

    def test_bjson_to_array_roundtrips(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        bjson = BJsonType()

        src = self.pa.array(
            [b"[1,2,3]", b"[4,5]", None], type=self.pa.binary()
        )

        out = arr_dtype.cast_arrow_array(
            src,
            source=Field("a", bjson),
            target=Field("a", arr_dtype),
        )

        self.assertEqual(out.to_pylist(), [[1, 2, 3], [4, 5], None])

    def test_bjson_to_map_decodes_object(self) -> None:
        map_dtype = MapType.from_key_value(
            key_field=Field("key", StringType()),
            value_field=Field("value", _INT64),
        )
        src = self.pa.array([b'{"a": 1}', b'{"b": 2, "c": 3}'], type=self.pa.binary())

        out = map_dtype.cast_arrow_array(
            src,
            source=Field("m", BJsonType()),
            target=Field("m", map_dtype),
        )

        self.assertEqual(
            out.to_pylist(),
            [[("a", 1)], [("b", 2), ("c", 3)]],
        )

    def test_string_to_sjson_passes_storage_through(self) -> None:
        # SJSON over an existing string column is just a tag swap —
        # the cast must be the identity at the storage level.
        src = self.pa.array(["hello", "world", None], type=self.pa.string())

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", StringType()),
            target=Field("s", sjson),
        )

        self.assertEqual(out.to_pylist(), ["hello", "world", None])
        self.assertEqual(out.type, self.pa.string())

    def test_sjson_to_bjson_storage_swap(self) -> None:
        sjson = SJsonType()
        bjson = BJsonType()
        src = self.pa.array(['{"a":1}', '{"b":2}'], type=self.pa.string())

        out = bjson.cast_arrow_array(
            src,
            source=Field("j", sjson),
            target=Field("j", bjson),
        )

        self.assertEqual(out.type, self.pa.binary())
        self.assertEqual(
            [bytes(v) for v in out.to_pylist()],
            [b'{"a":1}', b'{"b":2}'],
        )


# ---------------------------------------------------------------------------
# Polars casts — same encode / decode round-trips on the polars side
# ---------------------------------------------------------------------------


pl = pytest.importorskip("polars", reason="polars not installed")


class TestJsonPolarsCasts:

    def test_struct_series_to_sjson_encodes_to_strings(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        s = pl.Series(
            "s", [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], dtype=struct_dtype.to_polars()
        )

        sjson = SJsonType()
        out = sjson.cast_polars_series(
            s,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        decoded = [json.loads(v) for v in out.to_list()]
        assert decoded == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        assert out.dtype == pl.String

    def test_sjson_series_to_struct_decodes(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        s = pl.Series("s", ['{"a": 1, "b": "x"}', '{"a": 2, "b": "y"}'], dtype=pl.String)

        out = struct_dtype.cast_polars_series(
            s,
            source=Field("s", SJsonType()),
            target=Field("s", struct_dtype),
        )

        assert out.to_list() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_bjson_series_to_array_decodes_via_string_bridge(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        s = pl.Series("a", [b"[1,2]", b"[3]"], dtype=pl.Binary)

        out = arr_dtype.cast_polars_series(
            s,
            source=Field("a", BJsonType()),
            target=Field("a", arr_dtype),
        )

        assert out.to_list() == [[1, 2], [3]]


# ---------------------------------------------------------------------------
# Flexible (safe=False) decode — bad rows null out instead of raising
# ---------------------------------------------------------------------------


class TestJsonArrowPermissive(ArrowTestCase):

    def test_invalid_row_raises_with_row_index_on_safe_true(self) -> None:
        # Strict mode: the previous behaviour raised a bare
        # ``json.JSONDecodeError`` that didn't say which row failed.
        # The new strict path raises ``pa.ArrowInvalid`` and names the
        # offending row index plus a preview of the bad value.
        # ``safe=True`` is opt-in now — the default flipped to ``False``,
        # so the test passes ``safe=True`` explicitly.
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        src = self.pa.array(
            ['[1, 2, 3]', '{not json', '[4]'], type=self.pa.string()
        )

        with self.assertRaises(self.pa.ArrowInvalid) as ctx:
            arr_dtype.cast_arrow_array(
                src,
                source=Field("a", SJsonType()),
                target=Field("a", arr_dtype),
                safe=True,
            )

        msg = str(ctx.exception)
        self.assertIn("row 1", msg)
        self.assertIn("{not json", msg)

    def test_invalid_row_nulls_out_on_safe_false_array(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        src = self.pa.array(
            ['[1, 2, 3]', '{not json', '[4]', None],
            type=self.pa.string(),
        )

        out = arr_dtype.cast_arrow_array(
            src,
            source=Field("a", SJsonType()),
            target=Field("a", arr_dtype),
            safe=False,
        )

        self.assertEqual(out.to_pylist(), [[1, 2, 3], None, [4], None])

    def test_invalid_row_nulls_out_on_safe_false_struct(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        src = self.pa.array(
            [
                '{"a": 1, "b": "x"}',
                'totally-not-json',
                '{"a": 2, "b": "y"}',
            ],
            type=self.pa.string(),
        )

        out = struct_dtype.cast_arrow_array(
            src,
            source=Field("s", SJsonType()),
            target=Field("s", struct_dtype),
            safe=False,
        )

        self.assertEqual(
            out.to_pylist(),
            [{"a": 1, "b": "x"}, None, {"a": 2, "b": "y"}],
        )

    def test_bjson_invalid_row_nulls_out_on_safe_false(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        src = self.pa.array(
            [b"[1,2]", b"not-json", b"[3]"], type=self.pa.binary()
        )

        out = arr_dtype.cast_arrow_array(
            src,
            source=Field("a", BJsonType()),
            target=Field("a", arr_dtype),
            safe=False,
        )

        self.assertEqual(out.to_pylist(), [[1, 2], None, [3]])


class TestJsonPolarsPermissive:

    def test_invalid_row_nulls_out_on_safe_false_array(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        s = pl.Series(
            "a",
            ['[1, 2]', '{nope', '[3]', None],
            dtype=pl.String,
        )

        out = arr_dtype.cast_polars_series(
            s,
            source=Field("a", SJsonType()),
            target=Field("a", arr_dtype),
            safe=False,
        )

        assert out.to_list() == [[1, 2], None, [3], None]

    def test_invalid_row_nulls_out_on_safe_false_struct(self) -> None:
        struct_dtype = StructType(
            fields=[Field("a", _INT64), Field("b", StringType())]
        )
        s = pl.Series(
            "s",
            ['{"a": 1, "b": "x"}', 'not json', '{"a": 2, "b": "y"}'],
            dtype=pl.String,
        )

        out = struct_dtype.cast_polars_series(
            s,
            source=Field("s", SJsonType()),
            target=Field("s", struct_dtype),
            safe=False,
        )

        assert out.to_list() == [
            {"a": 1, "b": "x"},
            None,
            {"a": 2, "b": "y"},
        ]


# ---------------------------------------------------------------------------
# Arrow JSON encode — edge-case types in struct payloads
# ---------------------------------------------------------------------------


class TestJsonArrowEncodeEdgeCases(ArrowTestCase):
    """Edge cases for ``cast_arrow_json_encode_array``.

    The encode path tries orjson first and falls back to stdlib json + the
    ``_json_default`` coercion for types orjson cannot natively handle.
    All three fallback types are exercised here.
    """

    def test_null_rows_remain_null_in_output(self) -> None:
        struct_dtype = StructType(
            fields=[Field("n", _INT64)]
        )
        src = self.pa.array(
            [{"n": 1}, None, {"n": 3}],
            type=struct_dtype.to_arrow(),
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        rows = out.to_pylist()
        self.assertIsNotNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertIsNotNone(rows[2])

    def test_decimal_value_in_struct_encodes_as_string(self) -> None:
        import decimal as _decimal

        src = self.pa.array(
            [{"amount": _decimal.Decimal("1.23")}],
            type=self.pa.struct([self.pa.field("amount", self.pa.decimal128(10, 2))]),
        )

        struct_dtype = StructType(
            fields=[Field("amount", DecimalType(precision=10, scale=2))]
        )
        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        decoded = json.loads(out[0].as_py())
        # Decimal serialised as string by _json_default fallback.
        self.assertEqual(decoded["amount"], "1.23")

    def test_empty_array_returns_empty_string_array(self) -> None:
        struct_dtype = StructType(
            fields=[Field("x", _INT64)]
        )
        src = self.pa.array([], type=struct_dtype.to_arrow())

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        self.assertEqual(len(out), 0)
        self.assertEqual(out.type, self.pa.string())

    def test_large_batch_all_nulls_encodes_correctly(self) -> None:
        struct_dtype = StructType(
            fields=[Field("n", _INT64)]
        )
        src = self.pa.array(
            [None] * 100,
            type=struct_dtype.to_arrow(),
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        self.assertEqual(len(out), 100)
        self.assertTrue(all(v is None for v in out.to_pylist()))

    def test_chunked_array_input_handled(self) -> None:
        struct_dtype = StructType(
            fields=[Field("n", _INT64)]
        )
        chunk1 = self.pa.array([{"n": 1}, {"n": 2}], type=struct_dtype.to_arrow())
        chunk2 = self.pa.array([{"n": 3}], type=struct_dtype.to_arrow())
        chunked = self.pa.chunked_array([chunk1, chunk2])

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            chunked,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        decoded = [json.loads(v) for v in out.to_pylist()]
        self.assertEqual(decoded, [{"n": 1}, {"n": 2}, {"n": 3}])


# ---------------------------------------------------------------------------
# JSON encode correctness — verify orjson output is valid JSON
# ---------------------------------------------------------------------------


class TestJsonEncodeOutputFormat(ArrowTestCase):
    """Verify that encoded rows parse as valid JSON and preserve value
    semantics regardless of which backend (orjson vs stdlib) handled them."""

    def test_list_of_ints_encodes_to_valid_json(self) -> None:
        arr_dtype = ArrayType.from_item(Field("item", _INT64))
        src = self.pa.array(
            [[1, 2, 3], [4, 5], None],
            type=arr_dtype.to_arrow(),
        )

        bjson = BJsonType()
        out = bjson.cast_arrow_array(
            src,
            source=Field("a", arr_dtype),
            target=Field("a", bjson),
        )

        rows = out.to_pylist()
        self.assertEqual(json.loads(rows[0]), [1, 2, 3])
        self.assertEqual(json.loads(rows[1]), [4, 5])
        self.assertIsNone(rows[2])

    def test_map_encodes_to_valid_json_object(self) -> None:
        map_dtype = MapType.from_key_value(
            key_field=Field("key", StringType()),
            value_field=Field("value", _INT64),
        )
        src = self.pa.array(
            [[("a", 1), ("b", 2)], [("z", 99)]],
            type=map_dtype.to_arrow(),
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("m", map_dtype),
            target=Field("m", sjson),
        )

        self.assertEqual(json.loads(out[0].as_py()), {"a": 1, "b": 2})
        self.assertEqual(json.loads(out[1].as_py()), {"z": 99})

    def test_unicode_strings_preserved_without_escaping(self) -> None:
        struct_dtype = StructType(
            fields=[Field("name", StringType())]
        )
        src = self.pa.array(
            [{"name": "héllo"}, {"name": "日本語"}],
            type=struct_dtype.to_arrow(),
        )

        sjson = SJsonType()
        out = sjson.cast_arrow_array(
            src,
            source=Field("s", struct_dtype),
            target=Field("s", sjson),
        )

        for raw in out.to_pylist():
            decoded = json.loads(raw)
            # Non-ASCII chars must survive the round-trip unescaped.
            self.assertNotIn("\\u", raw)
            self.assertIn(decoded["name"], {"héllo", "日本語"})
