# tests/io/enums/test_codec.py
from __future__ import annotations

import io
import pytest

from yggdrasil.io.enums.codec import (
    Codec,
    detect,
    detect_bytes,
    GZIP,
    ZSTD,
    LZ4,
    BZIP2,
    XZ,
    SNAPPY,
    BROTLI,
    ZLIB,
    LZMA,
)
from yggdrasil.io.enums.mime_type import MimeType


# ----------------------------
# Parse / detect basics
# ----------------------------

@pytest.mark.parametrize(
    "s, expected",
    [
        ("gzip", GZIP),
        ("GZIP", GZIP),
        (".gz", GZIP),
        ("application/gzip", GZIP),
        ("zstd", ZSTD),
        (".zst", ZSTD),
        ("application/zstd", ZSTD),
        ("lz4", LZ4),
        (".lz4", LZ4),
        ("application/x-lz4", LZ4),
        ("bzip2", BZIP2),
        (".bz2", BZIP2),
        ("application/x-bzip2", BZIP2),
        ("xz", XZ),
        (".xz", XZ),
        ("application/x-xz", XZ),
        ("zlib", ZLIB),
        (".zlib", ZLIB),
        ("application/zlib", ZLIB),
        ("lzma", LZMA),
        (".lzma", LZMA),
        ("application/x-lzma", LZMA),
        ("brotli", BROTLI),
        (".br", BROTLI),
        ("application/x-brotli", BROTLI),
        ("snappy", SNAPPY),
        (".snappy", SNAPPY),
        ("application/x-snappy", SNAPPY),
    ],
)
def test_codec_parse_string(s: str, expected):
    assert Codec.parse(s) is expected


def test_codec_parse_defaults():
    assert Codec.parse(None) is None
    assert Codec.parse("nope") is None
    assert Codec.parse("nope", default=GZIP) is GZIP


def test_codec_from_mime():
    assert Codec.from_mime(MimeType.GZIP) is GZIP
    assert Codec.from_mime("application/gzip") is GZIP
    assert Codec.from_mime("application/octet-stream") is None


# ----------------------------
# detect_bytes: must match MimeType magic detection
# ----------------------------

@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"\x1f\x8b\x08\x00" + b"x" * 10, GZIP),
        (b"\x28\xb5\x2f\xfd" + b"x" * 10, ZSTD),
        (b"\x04\x22\x4d\x18" + b"x" * 10, LZ4),
        (b"BZh" + b"x" * 10, BZIP2),
        (b"\xfd\x37\x7a\x58\x5a\x00" + b"x" * 10, XZ),
        (b"\x78\x9c" + b"x" * 10, ZLIB),
        (b"\x78\x01" + b"x" * 10, ZLIB),
        (b"\x78\xda" + b"x" * 10, ZLIB),
    ],
)
def test_detect_bytes(payload: bytes, expected):
    assert detect_bytes(payload) is expected


def test_detect_bytes_unknown_returns_none():
    assert detect_bytes(b"\x00\x01\x02\x03\x04") is None


# ----------------------------
# detect(IO): non-consuming peek
# ----------------------------

def test_detect_io_preserves_cursor():
    data = b"\x1f\x8b\x08\x00" + b"x" * 100
    fh = io.BytesIO(data)
    fh.seek(0)
    pos = fh.tell()
    assert detect(fh) is GZIP
    assert fh.tell() == pos


# ----------------------------
# Roundtrip smoke tests (optional deps handled)
# ----------------------------

@pytest.mark.parametrize(
    "codec",
    [GZIP, ZLIB, BZIP2, XZ, LZMA],
)
def test_roundtrip_stdlib_codecs(codec):
    payload = b"arrow-go-brrr" * 1024
    assert codec.roundtrip(payload)


@pytest.mark.parametrize(
    "codec, module_name",
    [
        (ZSTD, "zstandard"),
        (LZ4, "lz4"),
        (BROTLI, "brotli"),
        (SNAPPY, "cramjam"),
    ],
)
def test_roundtrip_optional_codecs(codec, module_name):
    payload = b"arrow-go-brrr" * 1024
    assert codec.roundtrip(payload)


# ----------------------------
# Shared payload
# ----------------------------

_PAYLOAD = (
    b"START--" +
    (b"arrow-go-brrr|" * 8192) +
    b"--END"
)


def _expect(payload: bytes, n_start: int, n_end: int) -> tuple[bytes, bytes]:
    return payload[:n_start], (payload[-n_end:] if n_end else b"")


# ----------------------------
# Core behavior: bytes input
# ----------------------------

@pytest.mark.parametrize(
    "codec",
    [GZIP, ZLIB, BZIP2, XZ, LZMA],
)
def test_read_start_end_bytes_stdlib(codec):
    compressed = codec.compress_bytes(_PAYLOAD)
    got = codec.read_start_end(compressed, n_start=64, n_end=64)
    assert got == _expect(_PAYLOAD, 64, 64)


@pytest.mark.parametrize(
    "codec, dep",
    [
        (ZSTD, "zstandard"),
        (LZ4, "lz4"),
        (BROTLI, "brotli"),
        (SNAPPY, "cramjam"),
    ],
)
def test_read_start_end_bytes_optional(codec, dep):
    pytest.importorskip(dep)
    compressed = codec.compress_bytes(_PAYLOAD)
    got = codec.read_start_end(compressed, n_start=64, n_end=64)
    assert got == _expect(_PAYLOAD, 64, 64)


def test_read_start_end_zero_lengths():
    compressed = GZIP.compress_bytes(_PAYLOAD)
    start, end = GZIP.read_start_end(compressed, n_start=0, n_end=0)
    assert start == b""
    assert end == b""


def test_read_start_end_negative_raises():
    compressed = GZIP.compress_bytes(_PAYLOAD)
    with pytest.raises(ValueError):
        GZIP.read_start_end(compressed, n_start=-1, n_end=10)
    with pytest.raises(ValueError):
        GZIP.read_start_end(compressed, n_start=10, n_end=-1)


# ----------------------------
# File-like input: cursor preserved
# ----------------------------

@pytest.mark.parametrize(
    "codec",
    [GZIP, ZLIB, BZIP2, XZ, LZMA],
)
def test_read_start_end_io_preserves_cursor_stdlib(codec):
    compressed = codec.compress_bytes(_PAYLOAD)
    fh = io.BytesIO(compressed)
    fh.seek(5)
    pos = fh.tell()
    start, end = codec.read_start_end(fh, n_start=32, n_end=32)
    assert (start, end) == _expect(_PAYLOAD, 32, 32)
    assert fh.tell() == pos


@pytest.mark.parametrize(
    "codec, dep",
    [
        (ZSTD, "zstandard"),
        (LZ4, "lz4"),
        (BROTLI, "brotli"),
        (SNAPPY, "cramjam"),
    ],
)
def test_read_start_end_io_preserves_cursor_optional(codec, dep):
    pytest.importorskip(dep)
    compressed = codec.compress_bytes(_PAYLOAD)
    fh = io.BytesIO(compressed)
    fh.seek(3)
    pos = fh.tell()
    start, end = codec.read_start_end(fh, n_start=32, n_end=32)
    assert (start, end) == _expect(_PAYLOAD, 32, 32)
    assert fh.tell() == pos


# ----------------------------
# Sanity: detect functions still work
# ----------------------------

def test_detect_bytes_and_detect_io_smoke():
    gz = GZIP.compress_bytes(b"hello")
    assert detect_bytes(gz) is GZIP
    fh = io.BytesIO(gz)
    fh.seek(2)
    pos = fh.tell()
    assert detect(fh) is GZIP
    assert fh.tell() == pos


# ----------------------------
# Edge: small payload smaller than requested
# ----------------------------

def test_read_start_end_small_payload():
    payload = b"tiny"
    comp = GZIP.compress_bytes(payload)
    start, end = GZIP.read_start_end(comp, n_start=64, n_end=64)
    assert start == payload  # up to available
    assert end == payload