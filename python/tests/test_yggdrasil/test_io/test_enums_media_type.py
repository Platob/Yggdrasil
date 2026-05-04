"""Tests for yggdrasil.io.enums.media_type."""

from __future__ import annotations

import gzip
import io

import pytest

from yggdrasil.io.enums.codec import GZIP, ZSTD
from yggdrasil.io.enums.media_type import MediaType, MediaTypes
from yggdrasil.io.enums.mime_type import MimeTypes


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPostInitNormalization:
    def test_codec_mime_type_promoted_to_codec_field(self):
        # MediaType(GZIP, codec=None) should normalize: mime → OCTET_STREAM,
        # codec → GZIP via __post_init__.
        mt = MediaType(MimeTypes.GZIP)
        assert mt.mime_type is MimeTypes.OCTET_STREAM
        assert mt.codec is GZIP

    def test_non_codec_mime_preserved(self):
        mt = MediaType(MimeTypes.JSON)
        assert mt.mime_type is MimeTypes.JSON
        assert mt.codec is None


# ---------------------------------------------------------------------------
# from_ dispatch
# ---------------------------------------------------------------------------


class TestFromDispatch:
    def test_passthrough(self):
        mt = MediaType(MimeTypes.JSON)
        assert MediaType.from_(mt) is mt

    def test_mime_type_input(self):
        mt = MediaType.from_(MimeTypes.PARQUET)
        assert mt.mime_type is MimeTypes.PARQUET

    def test_codec_only_input(self):
        mt = MediaType.from_(GZIP)
        assert mt.mime_type is MimeTypes.OCTET_STREAM
        assert mt.codec is GZIP

    def test_string_input(self):
        mt = MediaType.from_("application/json")
        assert mt.mime_type is MimeTypes.JSON


# ---------------------------------------------------------------------------
# Magic / IO sniffing
# ---------------------------------------------------------------------------


class TestFromMagic:
    def test_uncompressed_parquet_bytes(self):
        mt = MediaType.from_magic(b"PAR1" + b"\x00" * 60)
        assert mt.mime_type is MimeTypes.PARQUET
        assert mt.codec is None

    def test_gzip_wrapping_json_two_stage(self):
        body = gzip.compress(b'{"a":1}')
        mt = MediaType.from_magic(body)
        assert mt.codec is GZIP
        assert mt.mime_type is MimeTypes.JSON

    def test_unknown_with_default(self):
        assert MediaType.from_magic(b"~~~", default=None) is None

    def test_unknown_without_default_raises(self):
        with pytest.raises(ValueError):
            MediaType.from_magic(b"~~~")


class TestFromIO:
    def test_streaming_gzip_json(self):
        body = gzip.compress(b'{"a":1}')
        bio = io.BytesIO(body)
        bio.seek(3)  # caller cursor mid-stream
        mt = MediaType.from_io(bio)
        assert mt.codec is GZIP
        assert mt.mime_type is MimeTypes.JSON
        # Cursor is restored
        assert bio.tell() == 3


# ---------------------------------------------------------------------------
# from_url
# ---------------------------------------------------------------------------


class TestFromUrl:
    def test_url_with_extension(self):
        from yggdrasil.io.url import URL

        mt = MediaType.from_url(URL.from_str("/data/file.csv"))
        assert mt.mime_type is MimeTypes.CSV

    def test_dotted_chain_csv_gz(self):
        from yggdrasil.io.url import URL

        mt = MediaType.from_url(URL.from_str("/data/file.csv.gz"))
        assert mt.mime_type is MimeTypes.CSV
        assert mt.codec is GZIP

    def test_dotted_chain_snappy_parquet(self):
        # Spark / Databricks Delta convention: ``part-xxx.snappy.parquet``.
        # ``parquet`` wins as the outer format; ``snappy`` is parquet's
        # internal page codec, not an outer wrapper, so the resulting
        # MediaType has no outer codec — DeltaIO opens the file as a
        # plain parquet leaf and the parquet reader handles the page
        # decompression itself.
        from yggdrasil.io.url import URL

        mt = MediaType.from_url(URL.from_str("/data/part-00000.snappy.parquet"))
        assert mt.mime_type is MimeTypes.PARQUET
        assert mt.codec is None

    def test_dotted_chain_zstd_parquet(self):
        from yggdrasil.io.url import URL

        mt = MediaType.from_url(URL.from_str("/data/part-00000.zstd.parquet"))
        assert mt.mime_type is MimeTypes.PARQUET
        assert mt.codec is None


# ---------------------------------------------------------------------------
# Property accessors
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_octet(self):
        assert MediaType(MimeTypes.OCTET_STREAM).is_octet
        assert not MediaType(MimeTypes.JSON).is_octet

    def test_is_json(self):
        assert MediaType(MimeTypes.JSON).is_json

    def test_full_extension_with_codec(self):
        mt = MediaType(MimeTypes.CSV, codec=GZIP)
        assert mt.full_extension == "csv.gz"

    def test_full_extension_without_codec(self):
        mt = MediaType(MimeTypes.CSV)
        assert mt.full_extension == "csv"

    def test_full_mime_type_concat_codec(self):
        mt = MediaType(MimeTypes.CSV, codec=GZIP)
        full = mt.full_mime_type(concat_codec=True)
        assert "+gzip" in full.value

    def test_full_mime_type_no_concat(self):
        mt = MediaType(MimeTypes.CSV, codec=GZIP)
        assert mt.full_mime_type(concat_codec=False) is MimeTypes.CSV


class TestWithMethods:
    def test_with_codec(self):
        mt = MediaType(MimeTypes.CSV).with_codec(GZIP)
        assert mt.codec is GZIP

    def test_without_codec(self):
        mt = MediaType(MimeTypes.CSV, codec=GZIP).without_codec()
        assert mt.codec is None

    def test_with_mime_type(self):
        mt = MediaType(MimeTypes.JSON, codec=ZSTD).with_mime_type(MimeTypes.CSV)
        assert mt.mime_type is MimeTypes.CSV
        assert mt.codec is ZSTD


class TestMediaTypesPresets:
    def test_presets_defined(self):
        assert MediaTypes.OCTET_STREAM.mime_type is MimeTypes.OCTET_STREAM
        assert MediaTypes.PARQUET.mime_type is MimeTypes.PARQUET
        assert MediaTypes.JSON.mime_type is MimeTypes.JSON
        assert MediaTypes.ARROW_IPC.mime_type is MimeTypes.ARROW_IPC
