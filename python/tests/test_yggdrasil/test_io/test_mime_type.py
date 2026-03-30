# tests/io/enums/test_mime_type.py
from __future__ import annotations

import io
from pathlib import Path

import pytest

from yggdrasil.io.enums.mime_type import MimeType, MimeTypes


# ----------------------------
# get() + repr/str
# ----------------------------

@pytest.mark.parametrize(
    "inp, expected",
    [
        ("application/json", MimeTypes.JSON),
        ("Application/JSON", MimeTypes.JSON),
        (" json ", MimeTypes.JSON),            # name lookup
        ("JSON", MimeTypes.JSON),              # name lookup
        ("text/csv", MimeTypes.CSV),
        ("TEXT/CSV", MimeTypes.CSV),
        ("image/png", MimeTypes.PNG),
        ("application/gzip", MimeTypes.GZIP),
        ("application/zstd", MimeTypes.ZSTD),
        ("application/vnd.apache.parquet", MimeTypes.PARQUET),
        ("application/xml", MimeTypes.XML),
        # stripped prefix -> name
        ("application/parquet", MimeTypes.PARQUET),
        ("text/html", MimeTypes.HTML),
        ("image/jpeg", MimeTypes.JPEG),
    ],
)
def test_get_lax_lookup(inp: str, expected: MimeType):
    assert MimeType.get(inp) is expected


def test_get_non_string_returns_none():
    assert MimeType.get(None) is None
    assert MimeType.get(123) is None
    assert MimeType.get(b"application/json") is None


# ----------------------------
# parse_str(): extensions, paths, mime, name, heuristics
# ----------------------------

@pytest.mark.parametrize(
    "inp, expected",
    [
        (".parquet", MimeTypes.PARQUET),
        ("parquet", MimeTypes.PARQUET),
        ("PQ", MimeTypes.PARQUET),
        ("/tmp/file.parquet", MimeTypes.PARQUET),
        ("C:\\tmp\\file.parquet", MimeTypes.PARQUET),
        (".ipc", MimeTypes.ARROW_IPC),
        ("feather", MimeTypes.ARROW_IPC),
        ("/tmp/x.feather", MimeTypes.ARROW_IPC),
        (".orc", MimeTypes.ORC),
        ("avro", MimeTypes.AVRO),
        (".json", MimeTypes.JSON),
        (".jsonld", MimeTypes.NDJSON),
        (".csv", MimeTypes.CSV),
        (".tsv", MimeTypes.TSV),
        (".png", MimeTypes.PNG),
        (".jpeg", MimeTypes.JPEG),
        (".gz", MimeTypes.GZIP),
        (".zst", MimeTypes.ZSTD),
        (".lz4", MimeTypes.LZ4),
        (".bz2", MimeTypes.BZ2),
        (".xz", MimeTypes.XZ),
        (".zlib", MimeTypes.ZLIB),
        # container riders via extension
        (".xlsx", MimeTypes.XLSX),
        (".docx", MimeTypes.DOCX),
    ],
)
def test_parse_str_extension_and_paths(inp: str, expected: MimeType):
    assert MimeType.parse_str(inp) is expected


def test_parse_str_json_literal_heuristic():
    assert MimeType.parse_str('{"a":1}') is MimeTypes.JSON
    assert MimeType.parse_str("   [1,2,3]") is MimeTypes.JSON


def test_parse_str_default_when_unknown():
    assert MimeType.parse_str("nope/nope", default=None) is None
    assert MimeType.parse_str("nope/nope", default=MimeTypes.OCTET_STREAM) is MimeTypes.OCTET_STREAM


# ----------------------------
# parse_magic(): strong signatures
# ----------------------------

@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"\x1f\x8b\x08\x00" + b"x" * 20, MimeTypes.GZIP),
        (b"\x28\xb5\x2f\xfd" + b"x" * 20, MimeTypes.ZSTD),
        (b"\x04\x22\x4d\x18" + b"x" * 20, MimeTypes.LZ4),
        (b"BZh" + b"x" * 20, MimeTypes.BZ2),
        (b"\xfd\x37\x7a\x58\x5a\x00" + b"x" * 20, MimeTypes.XZ),
        (b"\x78\x9c" + b"x" * 20, MimeTypes.ZLIB),
        (b"\x78\x01" + b"x" * 20, MimeTypes.ZLIB),
        (b"\x78\xda" + b"x" * 20, MimeTypes.ZLIB),
        (b"PK\x03\x04" + b"x" * 20, MimeTypes.ZIP),
        (b"%PDF-" + b"x" * 20, MimeTypes.PDF),
        (b"SQLite format 3\x00" + b"x" * 20, MimeTypes.SQLITE),
        (b"\x89HDF\r\n\x1a\n" + b"x" * 20, MimeTypes.HDF5),
        (b"\x75\x73\x74\x61\x72" + b"x" * 20, MimeTypes.TAR),
        (b"PAR1" + b"x" * 20, MimeTypes.PARQUET),
        (b"ORC" + b"x" * 20, MimeTypes.ORC),
        (b"Obj\x01" + b"x" * 20, MimeTypes.AVRO),
        (b"ARROW1" + b"x" * 20, MimeTypes.ARROW_IPC),
        (b"\x89PNG\r\n\x1a\n" + b"x" * 20, MimeTypes.PNG),
        (b"\xff\xd8\xff" + b"x" * 20, MimeTypes.JPEG),
        (b"GIF89a" + b"x" * 20, MimeTypes.GIF),
        (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"x" * 20, MimeTypes.WEBP),
        (b"II*\x00" + b"x" * 20, MimeTypes.TIFF),
        (b"MM\x00*" + b"x" * 20, MimeTypes.TIFF),
        (b"BM" + b"x" * 20, MimeTypes.BMP),
        (b"\x93NUMPY" + b"x" * 20, MimeTypes.NUMPY),
    ],
)
def test_parse_magic_strong(payload: bytes, expected: MimeType):
    assert MimeType.parse_magic(payload) is expected


def test_parse_magic_default_when_unknown():
    assert MimeType.parse_magic(b"\x00\x01\x02", default=None) is None
    assert MimeType.parse_magic(b"\x00\x01\x02", default=MimeTypes.OCTET_STREAM) is MimeTypes.OCTET_STREAM


# ----------------------------
# parse(): dispatching (str/bytes/Path/IO)
# ----------------------------

def test_parse_dispatch_str_bytes_path_io():
    assert MimeType.parse(".parquet") is MimeTypes.PARQUET
    assert MimeType.parse(Path("/tmp/x.csv")) is MimeTypes.CSV
    assert MimeType.parse(b"\x1f\x8b\x08\x00xxxx") is MimeTypes.GZIP

    fh = io.BytesIO(b"PAR1" + b"x" * 100)
    pos = fh.tell()
    mt = MimeType.parse(fh)
    assert mt is MimeTypes.PARQUET
    assert fh.tell() == pos


# ----------------------------
# is_codec / is_tabular flags sanity
# ----------------------------

@pytest.mark.parametrize(
    "mt",
    [MimeTypes.GZIP, MimeTypes.ZSTD, MimeTypes.LZ4, MimeTypes.BZ2, MimeTypes.XZ, MimeTypes.ZLIB, MimeTypes.LZMA, MimeTypes.BROTLI, MimeTypes.SNAPPY],
)
def test_is_codec_true_for_codecs(mt: MimeType):
    assert mt.is_codec is True


@pytest.mark.parametrize("mt", [MimeTypes.CSV, MimeTypes.TSV, MimeTypes.NDJSON])
def test_is_tabular_true_for_rowish(mt: MimeType):
    assert mt.is_tabular is True


@pytest.mark.parametrize("mt", [MimeTypes.PARQUET, MimeTypes.ARROW_IPC, MimeTypes.PNG, MimeTypes.PDF])
def test_is_tabular_false_for_non_rowish(mt: MimeType):
    assert mt.is_tabular is False


# ----------------------------
# register_extension(s) behavior
# ----------------------------

def test_register_extension_overwrite_rules():
    # pick a new extension that's very unlikely to exist
    ext = "unit_mime_ext"

    # register
    MimeType.register_extension(ext, MimeTypes.JSON, overwrite=True)
    assert MimeType.parse_str("." + ext) is MimeTypes.JSON

    # without overwrite should error
    with pytest.raises(KeyError):
        MimeType.register_extension(ext, MimeTypes.CSV, overwrite=False)

    # with overwrite should replace
    MimeType.register_extension(ext, MimeTypes.CSV, overwrite=True)
    assert MimeType.parse_str(ext) is MimeTypes.CSV


def test_register_extensions_atomicity_fail_fast():
    # ensure a "fail" does not partially apply
    ext_ok = "unit_multi_ok"
    ext_bad = "unit_multi_bad"

    # clean (overwrite=True to avoid collisions if rerun)
    MimeType.register_extension(ext_ok, MimeTypes.JSON, overwrite=True)

    before = MimeType.registered_extensions()

    with pytest.raises(ValueError):
        MimeType.register_extensions(
            {
                ext_bad: "application/x-totally-not-a-real-mime",
                "unit_multi_ok2": MimeTypes.CSV,
            },
            overwrite=True,
        )

    after = MimeType.registered_extensions()
    assert after == before  # unchanged


def test_extensions_for():
    # basic: parquet has parquet + pq (from registry)
    exts = MimeType.extensions_for(MimeTypes.PARQUET)
    assert "parquet" in exts
    assert "pq" in exts

    # by string
    exts2 = MimeType.extensions_for("application/vnd.apache.parquet")
    assert exts2 == exts

    # unknown -> empty
    assert MimeType.extensions_for("application/x-nope") == []