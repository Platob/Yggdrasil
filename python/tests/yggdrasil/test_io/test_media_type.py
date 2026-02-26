# tests/io/test_media_type.py
"""Unit tests for yggdrasil.io.enums.media_type — MediaType dataclass.

Coverage
--------
- Binary magic-byte detection for every _MAGIC table entry
- Text heuristics: JSON, NDJSON, CSV, TSV, XML, HTML, YAML, TOML
- Compressed streams: (OCTET_STREAM, codec) with cursor preservation
- All predicate properties
- Factory helpers: of(), without_codec(), with_codec(), from_extension()
- Extension properties: extension, codec_extension, full_extension
- __eq__ against str and MediaType; __hash__ consistency
- __str__ / __repr__
"""

from __future__ import annotations

import io

import pytest

from .conftest import SMALL, compress_gzip, compress_zstd

pytest.importorskip("yggdrasil")

from yggdrasil.io.enums.codec import Codec          # noqa: E402
from yggdrasil.io.enums.media_type import MediaType  # noqa: E402


# ===========================================================================
# Binary magic-byte detection
# ===========================================================================

class TestMediaTypeDetection:
    """Every entry in the _MAGIC table has a corresponding test."""

    # ── Apache columnar ────────────────────────────────────────────────────

    def test_parquet(self):
        mt = MediaType.from_bytes(b"PAR1" + b"\x00" * 28)
        assert mt.mime == MediaType.PARQUET
        assert mt.is_parquet

    def test_arrow_file(self):
        mt = MediaType.from_bytes(b"ARROW1\x00\x00" + b"\x00" * 24)
        assert mt.mime == MediaType.ARROW_FILE
        assert mt.is_arrow

    def test_arrow_stream(self):
        mt = MediaType.from_bytes(b"\xff\xff\xff\xff" + b"\x00" * 28)
        assert mt.mime == MediaType.ARROW_STREAM
        assert mt.is_arrow

    def test_orc(self):
        mt = MediaType.from_bytes(b"ORC" + b"\x00" * 29)
        assert mt.mime == MediaType.ORC

    def test_avro(self):
        mt = MediaType.from_bytes(b"Obj\x01" + b"\x00" * 28)
        assert mt.mime == MediaType.AVRO

    # ── Binary / container ────────────────────────────────────────────────

    def test_numpy(self):
        mt = MediaType.from_bytes(b"\x93NUMPY" + b"\x00" * 26)
        assert mt.mime == MediaType.NUMPY

    def test_sqlite(self):
        mt = MediaType.from_bytes(b"SQLite format 3\x00" + b"\x00" * 16)
        assert mt.mime == MediaType.SQLITE

    def test_hdf5(self):
        mt = MediaType.from_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 24)
        assert mt.mime == MediaType.HDF5

    def test_pdf(self):
        mt = MediaType.from_bytes(b"%PDF-1.7" + b"\x00" * 24)
        assert mt.mime == MediaType.PDF

    def test_zip(self):
        mt = MediaType.from_bytes(b"PK\x03\x04" + b"\x00" * 28)
        assert mt.mime == MediaType.ZIP

    def test_tar_at_offset(self):
        # TAR ustar magic lives at byte 257 — from_io must peek ≥ 262 bytes.
        # Build one full 512-byte TAR record block with the magic in place.
        header = b"\x00" * 257 + b"ustar" + b"\x00" * (512 - 257 - 5)
        mt = MediaType.from_bytes(header)
        assert mt.mime == MediaType.TAR

    # ── Image formats ─────────────────────────────────────────────────────

    def test_png(self):
        mt = MediaType.from_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
        assert mt.mime == MediaType.PNG
        assert mt.is_image

    def test_jpeg(self):
        mt = MediaType.from_bytes(b"\xff\xd8\xff" + b"\x00" * 29)
        assert mt.mime == MediaType.JPEG
        assert mt.is_image

    def test_gif87(self):
        mt = MediaType.from_bytes(b"GIF87a" + b"\x00" * 26)
        assert mt.mime == MediaType.GIF

    def test_gif89(self):
        mt = MediaType.from_bytes(b"GIF89a" + b"\x00" * 26)
        assert mt.mime == MediaType.GIF

    def test_tiff_little_endian(self):
        mt = MediaType.from_bytes(b"II\x2a\x00" + b"\x00" * 28)
        assert mt.mime == MediaType.TIFF

    def test_tiff_big_endian(self):
        mt = MediaType.from_bytes(b"MM\x00\x2a" + b"\x00" * 28)
        assert mt.mime == MediaType.TIFF

    def test_bmp(self):
        mt = MediaType.from_bytes(b"BM" + b"\x00" * 30)
        assert mt.mime == MediaType.BMP


# ===========================================================================
# Text heuristics
# ===========================================================================

class TestMediaTypeTextHeuristics:
    def test_json_object(self):
        mt = MediaType.from_bytes(b'{"symbol": "TTF", "price": 42.5}')
        assert mt.mime == MediaType.JSON
        assert mt.is_json_like

    def test_json_array(self):
        mt = MediaType.from_bytes(b'[{"a": 1}, {"b": 2}]')
        assert mt.mime == MediaType.JSON

    def test_ndjson_two_objects(self):
        mt = MediaType.from_bytes(b'{"a":1}\n{"b":2}\n')
        assert mt.mime == MediaType.NDJSON
        assert mt.is_json_like

    def test_csv(self):
        mt = MediaType.from_bytes(b"symbol,date,close\nTTF,2024-01-02,42.5\n")
        assert mt.mime == MediaType.CSV

    def test_tsv(self):
        mt = MediaType.from_bytes(b"symbol\tdate\tclose\nTTF\t2024-01-02\t42.5\n")
        assert mt.mime == MediaType.TSV

    def test_xml(self):
        mt = MediaType.from_bytes(b"<?xml version='1.0'?><root/>")
        assert mt.mime == MediaType.XML

    def test_html_doctype(self):
        mt = MediaType.from_bytes(b"<!DOCTYPE html><html><body></body></html>")
        assert mt.mime == MediaType.HTML

    def test_yaml_document_marker(self):
        mt = MediaType.from_bytes(b"---\nname: yggdrasil\nversion: 1\n")
        assert mt.mime == MediaType.YAML

    def test_toml_key_value(self):
        mt = MediaType.from_bytes(b'name = "yggdrasil"\nversion = "1.0"\n')
        assert mt.mime == MediaType.TOML

    def test_unknown_binary_fallback(self):
        mt = MediaType.from_bytes(b"\x00\x01\x02\x03\x04\x05")
        assert mt.mime == MediaType.OCTET_STREAM
        assert mt.is_unknown

    def test_empty_bytes_fallback(self):
        mt = MediaType.from_bytes(b"")
        assert mt.mime == MediaType.OCTET_STREAM


# ===========================================================================
# Compressed streams
# ===========================================================================

class TestMediaTypeCompressedStreams:
    def test_gzip_wrapping_parquet(self):
        mt = MediaType.from_bytes(compress_gzip(b"PAR1" + b"\x00" * 28))
        assert mt.codec is Codec.GZIP
        assert mt.mime == MediaType.OCTET_STREAM
        assert mt.is_compressed
        # codec is known, so the stream is NOT unknown
        assert not mt.is_unknown

    def test_zstd_stream(self):
        mt = MediaType.from_bytes(compress_zstd(SMALL))
        assert mt.codec is Codec.ZSTD

    def test_cursor_preserved_on_compressed_stream(self):
        data = compress_gzip(SMALL)
        stream = io.BytesIO(data)
        stream.seek(2)
        MediaType.from_io(stream)
        assert stream.tell() == 2


# ===========================================================================
# Predicate properties
# ===========================================================================

class TestMediaTypePredicates:
    @pytest.mark.parametrize("mime", [
        MediaType.PARQUET, MediaType.ARROW_FILE, MediaType.ARROW_STREAM,
        MediaType.ORC, MediaType.AVRO, MediaType.CSV, MediaType.TSV,
        MediaType.NDJSON,
    ])
    def test_is_tabular(self, mime):
        assert MediaType(mime).is_tabular

    @pytest.mark.parametrize("mime", [
        MediaType.PARQUET, MediaType.ARROW_FILE, MediaType.ARROW_STREAM,
        MediaType.ORC, MediaType.AVRO, MediaType.ICEBERG,
    ])
    def test_is_apache(self, mime):
        assert MediaType(mime).is_apache

    @pytest.mark.parametrize("mime", [
        MediaType.PNG, MediaType.JPEG, MediaType.GIF,
        MediaType.WEBP, MediaType.TIFF, MediaType.BMP,
    ])
    def test_is_image(self, mime):
        assert MediaType(mime).is_image

    @pytest.mark.parametrize("mime", [
        MediaType.ZIP, MediaType.TAR, MediaType.HDF5, MediaType.SQLITE,
    ])
    def test_is_archive(self, mime):
        assert MediaType(mime).is_archive

    @pytest.mark.parametrize("mime", [
        MediaType.JSON, MediaType.CSV, MediaType.TSV,
        MediaType.XML, MediaType.HTML, MediaType.YAML,
    ])
    def test_is_text(self, mime):
        assert MediaType(mime).is_text

    @pytest.mark.parametrize("mime", [
        MediaType.PARQUET, MediaType.ARROW_FILE, MediaType.CSV,
        MediaType.JSON, MediaType.NDJSON, MediaType.AVRO,
    ])
    def test_is_polars_readable(self, mime):
        assert MediaType(mime).is_polars_readable

    def test_png_is_not_polars_readable(self):
        assert not MediaType(MediaType.PNG).is_polars_readable

    def test_is_ipc_alias_for_arrow(self):
        assert MediaType(MediaType.ARROW_FILE).is_ipc

    def test_is_unknown_only_for_bare_octet_stream(self):
        assert MediaType(MediaType.OCTET_STREAM).is_unknown
        assert not MediaType(MediaType.OCTET_STREAM, codec=Codec.GZIP).is_unknown

    def test_recognised_format_is_not_unknown(self):
        assert not MediaType(MediaType.PARQUET).is_unknown

    def test_is_compressed_false_when_no_codec(self):
        assert not MediaType(MediaType.PARQUET).is_compressed

    def test_is_compressed_true_when_codec_set(self):
        assert MediaType(MediaType.PARQUET, codec=Codec.ZSTD).is_compressed


# ===========================================================================
# Factory helpers
# ===========================================================================

class TestMediaTypeFactoryHelpers:
    def test_of_with_string_codec(self):
        mt = MediaType.of(MediaType.PARQUET, codec="zstd")
        assert mt.codec is Codec.ZSTD
        assert mt.mime == MediaType.PARQUET

    def test_of_with_none_codec(self):
        mt = MediaType.of(MediaType.CSV)
        assert mt.codec is None

    def test_of_with_codec_instance(self):
        mt = MediaType.of(MediaType.CSV, codec=Codec.GZIP)
        assert mt.codec is Codec.GZIP

    def test_without_codec_strips(self):
        mt = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        stripped = mt.without_codec()
        assert stripped.codec is None
        assert stripped.mime == MediaType.PARQUET

    def test_without_codec_returns_self_when_already_none(self):
        mt = MediaType(MediaType.CSV)
        assert mt.without_codec() is mt   # no allocation

    def test_with_codec_sets_codec(self):
        mt = MediaType(MediaType.CSV).with_codec("gzip")
        assert mt.codec is Codec.GZIP

    def test_with_codec_none_strips(self):
        mt = MediaType(MediaType.CSV, codec=Codec.GZIP).with_codec(None)
        assert mt.codec is None

    def test_from_extension_parquet(self):
        mt = MediaType.from_extension(".parquet")
        assert mt.mime == MediaType.PARQUET
        assert mt.codec is None

    def test_from_extension_parquet_zst(self):
        mt = MediaType.from_extension(".parquet.zst")
        assert mt.mime == MediaType.PARQUET
        assert mt.codec is Codec.ZSTD

    def test_from_extension_csv_gz(self):
        mt = MediaType.from_extension("csv.gz")
        assert mt.mime == MediaType.CSV
        assert mt.codec is Codec.GZIP

    def test_from_extension_unknown(self):
        mt = MediaType.from_extension(".xyz123")
        assert mt.mime == MediaType.OCTET_STREAM

    @pytest.mark.parametrize("ext,expected_mime", [
        ("ipc",     MediaType.ARROW_FILE),
        ("feather", MediaType.ARROW_FILE),
        ("jsonl",   MediaType.NDJSON),
        ("pq",      MediaType.PARQUET),
        ("yml",     MediaType.YAML),
    ])
    def test_from_extension_aliases(self, ext, expected_mime):
        assert MediaType.from_extension(ext).mime == expected_mime


# ===========================================================================
# Extension properties
# ===========================================================================

class TestMediaTypeExtensionProperties:
    def test_extension_parquet(self):
        assert MediaType(MediaType.PARQUET).extension == "parquet"

    def test_extension_csv(self):
        assert MediaType(MediaType.CSV).extension == "csv"

    def test_extension_octet_stream_fallback(self):
        assert MediaType(MediaType.OCTET_STREAM).extension == "bin"

    def test_codec_extension_zstd(self):
        mt = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        assert mt.codec_extension == "zst"

    def test_codec_extension_gzip(self):
        assert MediaType(MediaType.CSV, codec=Codec.GZIP).codec_extension == "gz"

    def test_codec_extension_empty_when_no_codec(self):
        assert MediaType(MediaType.PARQUET).codec_extension == ""

    def test_full_extension_with_codec(self):
        mt = MediaType(MediaType.PARQUET, codec=Codec.GZIP)
        assert mt.full_extension == "parquet.gz"

    def test_full_extension_without_codec(self):
        assert MediaType(MediaType.CSV).full_extension == "csv"


# ===========================================================================
# Equality, hashing, and display
# ===========================================================================

class TestMediaTypeEqualityAndHash:
    def test_eq_against_mime_string(self):
        mt = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        assert mt == MediaType.PARQUET

    def test_eq_against_string_false(self):
        assert MediaType(MediaType.PARQUET) != MediaType.CSV

    def test_eq_full_instance_match(self):
        a = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        b = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        assert a == b

    def test_eq_different_codec_not_equal(self):
        a = MediaType(MediaType.PARQUET, codec=Codec.ZSTD)
        b = MediaType(MediaType.PARQUET, codec=Codec.GZIP)
        assert a != b

    def test_hash_equal_objects(self):
        a = MediaType(MediaType.CSV, codec=Codec.GZIP)
        b = MediaType(MediaType.CSV, codec=Codec.GZIP)
        assert hash(a) == hash(b)

    def test_hash_usable_as_dict_key(self):
        d = {MediaType(MediaType.PARQUET): "parquet_handler"}
        assert d[MediaType(MediaType.PARQUET)] == "parquet_handler"

    def test_str_returns_mime(self):
        assert str(MediaType(MediaType.CSV)) == MediaType.CSV

    def test_repr_no_codec(self):
        assert repr(MediaType(MediaType.CSV)) == "MediaType('text/csv')"

    def test_repr_with_codec_contains_codec_name(self):
        r = repr(MediaType(MediaType.CSV, codec=Codec.GZIP))
        assert "gzip" in r
        assert "text/csv" in r