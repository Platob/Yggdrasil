"""Tests for yggdrasil.io.enums.codec."""

from __future__ import annotations

import pytest

from yggdrasil.data.enums.codec import (
    BROTLI,
    BZIP2,
    GZIP,
    LZ4,
    LZMA,
    SNAPPY,
    XZ,
    ZLIB,
    ZSTD,
    Codec,
    Codecs,
)
from yggdrasil.data.enums.mime_type import MimeTypes


PAYLOAD = b"Brent ICE front-month settle " * 200


class TestSingletons:
    def test_each_codec_unique(self):
        codecs = [GZIP, ZSTD, LZ4, BZIP2, XZ, SNAPPY, BROTLI, ZLIB, LZMA]
        assert len(set(codecs)) == len(codecs)

    def test_codecs_collection_aliases(self):
        assert Codecs.GZIP is GZIP
        assert Codecs.ZSTD is ZSTD


class TestCodecResolution:
    def test_from_codec_passthrough(self):
        assert Codec.from_(GZIP) is GZIP

    def test_from_short_name(self):
        assert Codec.from_("gzip") is GZIP
        assert Codec.from_("zstd") is ZSTD

    def test_from_short_name_case_insensitive(self):
        assert Codec.from_("GZIP") is GZIP

    def test_from_mime_value(self):
        assert Codec.from_("application/gzip") is GZIP

    def test_from_mime_object(self):
        assert Codec.from_mime(MimeTypes.GZIP) is GZIP

    def test_from_non_codec_mime_raises(self):
        with pytest.raises(ValueError):
            Codec.from_mime(MimeTypes.JSON)

    def test_from_non_codec_mime_default(self):
        assert Codec.from_mime(MimeTypes.JSON, default=None) is None

    def test_from_none_returns_default(self):
        assert Codec.from_(None, default=None) is None


class TestCodecMetadata:
    def test_name_matches_mime(self):
        assert GZIP.name == "gzip"
        assert ZSTD.name == "zstd"

    def test_mime_type_attribute(self):
        assert GZIP.mime_type is MimeTypes.GZIP

    def test_extensions_present(self):
        assert "gz" in GZIP.extensions

    def test_repr(self):
        assert "gzip" in repr(GZIP)


class TestCodecAll:
    def test_all_returns_known_codecs(self):
        all_codecs = Codec.all()
        assert {GZIP, ZSTD, LZ4, BZIP2, XZ, SNAPPY, BROTLI, ZLIB, LZMA} <= set(all_codecs)


class TestBytesRoundtrip:
    @pytest.mark.parametrize(
        "codec",
        [GZIP, BZIP2, XZ, ZLIB, LZMA],
        ids=lambda c: c.name,
    )
    def test_roundtrip_stdlib_codecs(self, codec):
        compressed = codec.compress_bytes(PAYLOAD)
        assert compressed != PAYLOAD
        assert codec.decompress_bytes(compressed) == PAYLOAD

    def test_zstd_roundtrip(self):
        zstd = pytest.importorskip("zstandard")
        del zstd
        assert ZSTD.decompress_bytes(ZSTD.compress_bytes(PAYLOAD)) == PAYLOAD

    def test_lz4_roundtrip(self):
        lz4 = pytest.importorskip("lz4.frame")
        del lz4
        assert LZ4.decompress_bytes(LZ4.compress_bytes(PAYLOAD)) == PAYLOAD


class TestStreamingFlag:
    def test_gzip_is_streaming(self):
        assert GZIP.is_streaming is True

    def test_snappy_is_not_streaming(self):
        # Snappy bindings don't expose streaming hooks.
        assert SNAPPY.is_streaming is False
