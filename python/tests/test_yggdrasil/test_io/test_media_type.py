# tests/io/enums/test_media_type.py
from __future__ import annotations

import io
import pytest

from yggdrasil.io.enums.codec import GZIP, ZSTD
from yggdrasil.io.enums.media_type import MediaType
from yggdrasil.io.enums.mime_type import MimeType, MimeTypes


def test_parse_passthrough():
    mt = MediaType(mime_type=MimeTypes.JSON, codec=None)
    assert MediaType.parse(mt) is mt


def test_parse_tuple_mime_codec():
    mt = MediaType.parse((MimeTypes.PARQUET, "gzip"))
    assert mt is not None
    assert mt.mime_type is MimeTypes.PARQUET
    assert mt.codec is GZIP


@pytest.mark.parametrize(
    "s, expected_mime, expected_codec",
    [
        ("application/json", MimeTypes.JSON, None),
        ("json", MimeTypes.JSON, None),
        (".json", MimeTypes.JSON, None),
        ("/tmp/x.json", MimeTypes.JSON, None),

        # codec-only strings => inner unknown => octet-stream + codec
        ("gzip", MimeTypes.OCTET_STREAM, GZIP),
        (".gz", MimeTypes.OCTET_STREAM, GZIP),
        ("application/gzip", MimeTypes.OCTET_STREAM, GZIP),

        ("parquet", MimeTypes.PARQUET, None),
        (".parquet", MimeTypes.PARQUET, None),
        ("/tmp/x.parquet", MimeTypes.PARQUET, None),

        # compound string forms (inner is knowable from the string)
        ("file.parquet.gz", MimeTypes.PARQUET, GZIP),
        ("/tmp/file.parquet.gz", MimeTypes.PARQUET, GZIP),

        ("json.zst", MimeTypes.JSON, ZSTD),
    ],
)
def test_parse_str_variants(s: str, expected_mime: MimeType, expected_codec):
    mt = MediaType.parse(s)
    assert mt is not None
    assert mt.mime_type is expected_mime
    assert mt.codec is expected_codec


def test_parse_str_unknown_defaults():
    assert MediaType.parse("nope/nope", default=None) == MediaType(MimeTypes.OCTET_STREAM)
    d = MediaType(mime_type=MimeTypes.OCTET_STREAM, codec=None)
    assert MediaType.parse("nope/nope", default=d) is d


def test_parse_bytes_non_codec_detects_mime():
    data = b"PAR1" + b"x" * 100
    mt = MediaType.parse(data)
    assert mt is not None
    assert mt.mime_type is MimeTypes.PARQUET
    assert mt.codec is None


def test_parse_dispatch():
    assert MediaType.parse(".parquet").mime_type is MimeTypes.PARQUET
    assert MediaType.parse(b"PAR1" + b"x" * 20).mime_type is MimeTypes.PARQUET

    fh = io.BytesIO(b"%PDF-" + b"x" * 100)
    assert MediaType.parse(fh).mime_type is MimeTypes.PDF