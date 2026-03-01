# tests/io/enums/test_media_type.py
from __future__ import annotations

import io
import pytest

from yggdrasil.io.enums.codec import GZIP, ZSTD
from yggdrasil.io.enums.media_type import MediaType
from yggdrasil.io.enums.mime_type import MimeType


def test_parse_passthrough():
    mt = MediaType(mime_type=MimeType.JSON, codec=None)
    assert MediaType.parse(mt) is mt


def test_parse_tuple_mime_codec():
    mt = MediaType.parse((MimeType.PARQUET, "gzip"))
    assert mt is not None
    assert mt.mime_type is MimeType.PARQUET
    assert mt.codec is GZIP


@pytest.mark.parametrize(
    "s, expected_mime, expected_codec",
    [
        ("application/json", MimeType.JSON, None),
        ("json", MimeType.JSON, None),
        (".json", MimeType.JSON, None),
        ("/tmp/x.json", MimeType.JSON, None),

        # codec-only strings => inner unknown => octet-stream + codec
        ("gzip", MimeType.OCTET_STREAM, GZIP),
        (".gz", MimeType.OCTET_STREAM, GZIP),
        ("application/gzip", MimeType.OCTET_STREAM, GZIP),

        ("parquet", MimeType.PARQUET, None),
        (".parquet", MimeType.PARQUET, None),
        ("/tmp/x.parquet", MimeType.PARQUET, None),

        # compound string forms (inner is knowable from the string)
        ("parquet+gzip", MimeType.PARQUET, GZIP),
        ("application/vnd.apache.parquet+gzip", MimeType.PARQUET, GZIP),
        ("file.parquet.gz", MimeType.PARQUET, GZIP),
        ("/tmp/file.parquet.gz", MimeType.PARQUET, GZIP),

        ("json.zst", MimeType.JSON, ZSTD),
    ],
)
def test_parse_str_variants(s: str, expected_mime: MimeType, expected_codec):
    mt = MediaType.parse_str(s)
    assert mt is not None
    assert mt.mime_type is expected_mime
    assert mt.codec is expected_codec


def test_parse_str_unknown_defaults():
    assert MediaType.parse_str("nope/nope", default=None) == MediaType(MimeType.OCTET_STREAM)
    d = MediaType(mime_type=MimeType.OCTET_STREAM, codec=None)
    assert MediaType.parse_str("nope/nope", default=d) is d


def test_parse_bytes_prefers_codec_wrapper():
    data = b"\x1f\x8b\x08\x00" + b"x" * 20
    mt = MediaType.parse_bytes(data)
    assert mt is not None
    assert mt.mime_type is MimeType.OCTET_STREAM
    assert mt.codec is GZIP


def test_parse_bytes_non_codec_detects_mime():
    data = b"PAR1" + b"x" * 100
    mt = MediaType.parse_bytes(data)
    assert mt is not None
    assert mt.mime_type is MimeType.PARQUET
    assert mt.codec is None


def test_parse_io_preserves_cursor_and_prefers_codec():
    fh = io.BytesIO(b"\x28\xb5\x2f\xfd" + b"x" * 200)  # zstd magic
    fh.seek(10)
    pos = fh.tell()
    mt = MediaType.parse_io(fh)
    assert mt is not None
    assert mt.mime_type is MimeType.OCTET_STREAM
    assert mt.codec is ZSTD
    assert fh.tell() == pos


def test_parse_dispatch():
    assert MediaType.parse(".parquet").mime_type is MimeType.PARQUET
    assert MediaType.parse(b"PAR1" + b"x" * 20).mime_type is MimeType.PARQUET

    fh = io.BytesIO(b"%PDF-" + b"x" * 100)
    assert MediaType.parse(fh).mime_type is MimeType.PDF