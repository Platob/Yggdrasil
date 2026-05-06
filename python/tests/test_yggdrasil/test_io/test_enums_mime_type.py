"""Tests for yggdrasil.io.enums.mime_type."""

from __future__ import annotations

import pytest

from yggdrasil.data.enums.mime_type import MimeType, MimeTypes


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestGet:
    def test_lookup_by_value(self):
        assert MimeType.get("application/json") is MimeTypes.JSON

    def test_lookup_by_name_case_insensitive(self):
        assert MimeType.get("PARQUET") is MimeTypes.PARQUET

    def test_unknown_returns_none(self):
        assert MimeType.get("nope/nada") is None

    def test_non_string_returns_none(self):
        assert MimeType.get(42) is None  # type: ignore[arg-type]


class TestFromStr:
    def test_extension_with_dot(self):
        assert MimeType.from_str(".csv") is MimeTypes.CSV

    def test_extension_path_like(self):
        assert MimeType.from_str("data/file.csv") is MimeTypes.CSV

    def test_bare_name(self):
        assert MimeType.from_str("json") is MimeTypes.JSON

    def test_full_mime_value(self):
        assert MimeType.from_str("application/json") is MimeTypes.JSON

    def test_unknown_with_default(self):
        assert MimeType.from_str("nope/nada", default=None) is None

    def test_unknown_raises_without_default(self):
        with pytest.raises(ValueError):
            MimeType.from_str("nope/nada")


class TestFromMagic:
    def test_gzip_magic(self):
        assert MimeType.from_magic(b"\x1f\x8b\x08\x00") is MimeTypes.GZIP

    def test_zstd_magic(self):
        assert MimeType.from_magic(b"\x28\xb5\x2f\xfd") is MimeTypes.ZSTD

    def test_parquet_magic(self):
        assert MimeType.from_magic(b"PAR1") is MimeTypes.PARQUET

    def test_png_magic(self):
        assert MimeType.from_magic(b"\x89PNG\r\n\x1a\n") is MimeTypes.PNG

    def test_json_structural_sniff(self):
        assert MimeType.from_magic(b'{"a":1}') is MimeTypes.JSON

    def test_xml_structural_sniff(self):
        assert MimeType.from_magic(b"<?xml") is MimeTypes.XML

    def test_empty_with_default(self):
        assert MimeType.from_magic(b"", default=None) is None

    def test_empty_without_default_raises(self):
        with pytest.raises(ValueError):
            MimeType.from_magic(b"")


class TestParseMany:
    def test_accept_header_split(self):
        result = MimeType.parse_many("application/json, text/csv;q=0.8")
        assert MimeTypes.JSON in result
        assert MimeTypes.CSV in result

    def test_composite_format_plus_codec(self):
        result = MimeType.parse_many("text/csv+gzip")
        assert MimeTypes.CSV in result
        assert MimeTypes.GZIP in result

    def test_dotted_extension_chain(self):
        result = MimeType.parse_many("trades.parquet.zst")
        assert MimeTypes.PARQUET in result
        assert MimeTypes.ZSTD in result

    def test_none_returns_empty(self):
        assert MimeType.parse_many(None) == []

    def test_iterable_input(self):
        result = MimeType.parse_many(["json", "csv"])
        assert set(result) == {MimeTypes.JSON, MimeTypes.CSV}

    def test_dedup_preserves_first_seen(self):
        result = MimeType.parse_many(["json", "json", "json"])
        assert result == [MimeTypes.JSON]


class TestProperties:
    def test_codec_flag(self):
        assert MimeTypes.GZIP.is_codec
        assert not MimeTypes.JSON.is_codec

    def test_tabular_flag(self):
        assert MimeTypes.CSV.is_tabular
        assert not MimeTypes.JSON.is_tabular

    def test_extensions_present(self):
        assert "csv" in MimeTypes.CSV.extensions
        assert "parquet" in MimeTypes.PARQUET.extensions

    def test_octet_stream_is_any_bytes(self):
        assert MimeTypes.OCTET_STREAM.is_any_bytes is True
        assert MimeTypes.JSON.is_any_bytes is False

    def test_str_dunder_returns_value(self):
        assert str(MimeTypes.JSON) == "application/json"


class TestExtensionsFor:
    def test_known_mime(self):
        exts = MimeType.extensions_for(MimeTypes.JSON)
        assert "json" in exts

    def test_unknown_returns_empty(self):
        assert MimeType.extensions_for("not/a/mime") == []
