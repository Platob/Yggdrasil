# tests/io/enums/test_mime_type.py
from __future__ import annotations

import io
from pathlib import Path

import pytest

from yggdrasil.io.enums.mime_type import MimeType


# ----------------------------
# get() + repr/str
# ----------------------------

def test_str_and_repr():
    assert str(MimeType.JSON) == "application/json"
    r = repr(MimeType.JSON)
    assert r.startswith("<MimeType.JSON:")
    assert "application/json" in r


@pytest.mark.parametrize(
    "inp, expected",
    [
        ("application/json", MimeType.JSON),
        ("Application/JSON", MimeType.JSON),
        (" json ", MimeType.JSON),            # name lookup
        ("JSON", MimeType.JSON),              # name lookup
        ("text/csv", MimeType.CSV),
        ("TEXT/CSV", MimeType.CSV),
        ("image/png", MimeType.PNG),
        ("application/gzip", MimeType.GZIP),
        ("application/zstd", MimeType.ZSTD),
        ("application/vnd.apache.parquet", MimeType.PARQUET),
        ("application/xml", MimeType.XML),
        # stripped prefix -> name
        ("application/parquet", MimeType.PARQUET),
        ("text/html", MimeType.HTML),
        ("image/jpeg", MimeType.JPEG),
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
        (".parquet", MimeType.PARQUET),
        ("parquet", MimeType.PARQUET),
        ("PQ", MimeType.PARQUET),
        ("/tmp/file.parquet", MimeType.PARQUET),
        ("C:\\tmp\\file.parquet", MimeType.PARQUET),
        (".ipc", MimeType.ARROW_IPC),
        ("feather", MimeType.ARROW_IPC),
        ("/tmp/x.feather", MimeType.ARROW_IPC),
        (".orc", MimeType.ORC),
        ("avro", MimeType.AVRO),
        (".json", MimeType.JSON),
        (".ndjson", MimeType.NDJSON),
        (".jsonl", MimeType.NDJSON),  # both registered
        (".csv", MimeType.CSV),
        (".tsv", MimeType.TSV),
        (".png", MimeType.PNG),
        (".jpeg", MimeType.JPEG),
        (".gz", MimeType.GZIP),
        (".zst", MimeType.ZSTD),
        (".lz4", MimeType.LZ4),
        (".bz2", MimeType.BZ2),
        (".xz", MimeType.XZ),
        (".zlib", MimeType.ZLIB),
        # container riders via extension
        (".xlsx", MimeType.XLSX),
        (".docx", MimeType.DOCX),
    ],
)
def test_parse_str_extension_and_paths(inp: str, expected: MimeType):
    assert MimeType.parse_str(inp) is expected


def test_parse_str_json_literal_heuristic():
    assert MimeType.parse_str('{"a":1}') is MimeType.JSON
    assert MimeType.parse_str("   [1,2,3]") is MimeType.JSON


def test_parse_str_default_when_unknown():
    assert MimeType.parse_str("nope/nope", default=None) is None
    assert MimeType.parse_str("nope/nope", default=MimeType.OCTET_STREAM) is MimeType.OCTET_STREAM


# ----------------------------
# parse_magic(): strong signatures
# ----------------------------

@pytest.mark.parametrize(
    "payload, expected",
    [
        (b"\x1f\x8b\x08\x00" + b"x" * 20, MimeType.GZIP),
        (b"\x28\xb5\x2f\xfd" + b"x" * 20, MimeType.ZSTD),
        (b"\x04\x22\x4d\x18" + b"x" * 20, MimeType.LZ4),
        (b"BZh" + b"x" * 20, MimeType.BZ2),
        (b"\xfd\x37\x7a\x58\x5a\x00" + b"x" * 20, MimeType.XZ),
        (b"\x78\x9c" + b"x" * 20, MimeType.ZLIB),
        (b"\x78\x01" + b"x" * 20, MimeType.ZLIB),
        (b"\x78\xda" + b"x" * 20, MimeType.ZLIB),
        (b"PK\x03\x04" + b"x" * 20, MimeType.ZIP),
        (b"%PDF-" + b"x" * 20, MimeType.PDF),
        (b"SQLite format 3\x00" + b"x" * 20, MimeType.SQLITE),
        (b"\x89HDF\r\n\x1a\n" + b"x" * 20, MimeType.HDF5),
        (b"\x75\x73\x74\x61\x72" + b"x" * 20, MimeType.TAR),
        (b"PAR1" + b"x" * 20, MimeType.PARQUET),
        (b"ORC" + b"x" * 20, MimeType.ORC),
        (b"Obj\x01" + b"x" * 20, MimeType.AVRO),
        (b"ARROW1" + b"x" * 20, MimeType.ARROW_IPC),
        (b"\x89PNG\r\n\x1a\n" + b"x" * 20, MimeType.PNG),
        (b"\xff\xd8\xff" + b"x" * 20, MimeType.JPEG),
        (b"GIF89a" + b"x" * 20, MimeType.GIF),
        (b"RIFF" + b"\x00" * 4 + b"WEBP" + b"x" * 20, MimeType.WEBP),
        (b"II*\x00" + b"x" * 20, MimeType.TIFF),
        (b"MM\x00*" + b"x" * 20, MimeType.TIFF),
        (b"BM" + b"x" * 20, MimeType.BMP),
        (b"\x93NUMPY" + b"x" * 20, MimeType.NUMPY),
    ],
)
def test_parse_magic_strong(payload: bytes, expected: MimeType):
    assert MimeType.parse_magic(payload) is expected


def test_parse_magic_weak_json_xml_fallbacks():
    assert MimeType.parse_magic(b"   {\"k\":1}") is MimeType.JSON
    assert MimeType.parse_magic(b"\n\n[1,2,3]") is MimeType.JSON
    assert MimeType.parse_magic(b"   <root/>") is MimeType.XML


def test_parse_magic_default_when_unknown():
    assert MimeType.parse_magic(b"\x00\x01\x02", default=None) is None
    assert MimeType.parse_magic(b"\x00\x01\x02", default=MimeType.OCTET_STREAM) is MimeType.OCTET_STREAM


# ----------------------------
# parse(): dispatching (str/bytes/Path/IO)
# ----------------------------

def test_parse_dispatch_str_bytes_path_io():
    assert MimeType.parse(".parquet") is MimeType.PARQUET
    assert MimeType.parse(Path("/tmp/x.csv")) is MimeType.CSV
    assert MimeType.parse(b"\x1f\x8b\x08\x00xxxx") is MimeType.GZIP

    fh = io.BytesIO(b"xxxPAR1" + b"x" * 100)
    fh.seek(3)  # ensure peek preserves cursor
    pos = fh.tell()
    mt = MimeType.parse(fh)
    assert mt is MimeType.PARQUET
    assert fh.tell() == pos


# ----------------------------
# is_codec / is_tabular flags sanity
# ----------------------------

@pytest.mark.parametrize(
    "mt",
    [MimeType.GZIP, MimeType.ZSTD, MimeType.LZ4, MimeType.BZ2, MimeType.XZ, MimeType.ZLIB, MimeType.LZMA, MimeType.BROTLI, MimeType.SNAPPY],
)
def test_is_codec_true_for_codecs(mt: MimeType):
    assert mt.is_codec is True


@pytest.mark.parametrize("mt", [MimeType.CSV, MimeType.TSV, MimeType.NDJSON])
def test_is_tabular_true_for_rowish(mt: MimeType):
    assert mt.is_tabular is True


@pytest.mark.parametrize("mt", [MimeType.PARQUET, MimeType.ARROW_IPC, MimeType.PNG, MimeType.PDF])
def test_is_tabular_false_for_non_rowish(mt: MimeType):
    assert mt.is_tabular is False


# ----------------------------
# register_extension(s) behavior
# ----------------------------

def test_register_extension_overwrite_rules():
    # pick a new extension that's very unlikely to exist
    ext = "unit_mime_ext"

    # register
    MimeType.register_extension(ext, MimeType.JSON, overwrite=True)
    assert MimeType.parse_str("." + ext) is MimeType.JSON

    # without overwrite should error
    with pytest.raises(KeyError):
        MimeType.register_extension(ext, MimeType.CSV, overwrite=False)

    # with overwrite should replace
    MimeType.register_extension(ext, MimeType.CSV, overwrite=True)
    assert MimeType.parse_str(ext) is MimeType.CSV


def test_register_extensions_atomicity_fail_fast():
    # ensure a "fail" does not partially apply
    ext_ok = "unit_multi_ok"
    ext_bad = "unit_multi_bad"

    # clean (overwrite=True to avoid collisions if rerun)
    MimeType.register_extension(ext_ok, MimeType.JSON, overwrite=True)

    before = MimeType.registered_extensions()

    with pytest.raises(ValueError):
        MimeType.register_extensions(
            {
                ext_bad: "application/x-totally-not-a-real-mime",
                "unit_multi_ok2": MimeType.CSV,
            },
            overwrite=True,
        )

    after = MimeType.registered_extensions()
    assert after == before  # unchanged


def test_extensions_for():
    # basic: parquet has parquet + pq (from registry)
    exts = MimeType.extensions_for(MimeType.PARQUET)
    assert "parquet" in exts
    assert "pq" in exts

    # by string
    exts2 = MimeType.extensions_for("application/vnd.apache.parquet")
    assert exts2 == exts

    # unknown -> empty
    assert MimeType.extensions_for("application/x-nope") == []