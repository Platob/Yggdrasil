"""Tabular IO over codec-tagged holders.

When a holder's :class:`MediaType` carries a codec (e.g. a
``LocalPath('data.csv.gz')`` whose URL inference says
``CSV + GZIP``), the Tabular leaf should transparently
decompress on read and compress on write — no extra glue at the
call site.

These tests verify the auto-handling against the four streamable
formats (CSV, NDJSON, Arrow IPC, Parquet) plus a sanity-check
fallback to uncompressed reads / writes.
"""
from __future__ import annotations

import gzip

import pyarrow as pa
import pytest

from yggdrasil.enums import MimeTypes
from yggdrasil.enums.codec import Codecs
from yggdrasil.enums.media_type import MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile
from yggdrasil.io.primitive.csv_file import CSVFile
from yggdrasil.io.primitive.json_file import JSONFile
from yggdrasil.io.primitive.ndjson_file import NDJSONFile
from yggdrasil.io.primitive.parquet_file import ParquetFile
from yggdrasil.io.primitive.xlsx_file import XLSXFile


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


def _seed_codec(holder: Memory, mime: MimeTypes, codec) -> Memory:
    """Stamp a media_type with codec onto the in-memory holder.

    This is what a path-bound holder learns from a ``.csv.gz``
    URL extension; for in-memory tests we pin it explicitly.
    """
    holder.media_type = MediaType(mime_type=mime, codec=codec)
    return holder


# ---------------------------------------------------------------------------
# Sanity — uncompressed Memory holders still round-trip
# ---------------------------------------------------------------------------


class TestUncompressedSanity:
    """No codec → ``_format_view`` is a plain view; ``_format_buffer``
    writes straight into self."""

    def test_csv_round_trip(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        assert not io.to_bytes().startswith(b"\x1f\x8b")  # not gzip
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_arrow_round_trip(self, table) -> None:
        io = ArrowIPCFile()
        io.write_arrow_table(table)
        assert io.read_arrow_table().equals(table)

    def test_parquet_round_trip(self, table) -> None:
        io = ParquetFile()
        io.write_arrow_table(table)
        assert io.read_arrow_table().equals(table)


# ---------------------------------------------------------------------------
# CSV + gzip (Memory holder, explicitly seeded)
# ---------------------------------------------------------------------------


class TestCsvGzipMemory:

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.CSV, Codecs.GZIP)
        io = CSVFile(holder=mem, owns_holder=False)
        io.write_arrow_table(table)

        # Bytes on the holder are gzip-compressed.
        assert mem.read_bytes()[:2] == b"\x1f\x8b"

        # Decompressed payload looks like CSV.
        decoded = gzip.decompress(mem.read_bytes())
        assert decoded.startswith(b'"id","name"')

        # Reading back transparently decompresses.
        reader = CSVFile(holder=mem, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# NDJSON + gzip
# ---------------------------------------------------------------------------


class TestNDJsonGzipMemory:

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.NDJSON, Codecs.GZIP)
        NDJSONFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        loaded = NDJSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC + zstd
# ---------------------------------------------------------------------------


class TestArrowIPCZstd:

    def test_round_trip(self, table) -> None:
        try:
            import zstandard  # noqa: F401
        except ImportError:
            pytest.skip("zstandard not installed")
        mem = _seed_codec(Memory(), MimeTypes.ARROW_IPC, Codecs.ZSTD)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)
        # Zstd magic bytes.
        assert mem.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
        loaded = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)


# ---------------------------------------------------------------------------
# CSV + gzip via LocalPath URL inference (.csv.gz)
# ---------------------------------------------------------------------------


class TestCsvGzipLocalPath:
    """The ``.csv.gz`` extension chain populates the holder's
    media_type with CSV + GZIP automatically; the leaf picks the
    codec up without any extra wiring at the call site."""

    def test_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "trades.csv.gz"))
        # The holder's media type already carries the codec — pinned
        # via the URL extension chain at construction.
        assert target.stat().media_type.codec is Codecs.GZIP

        CSVFile(holder=target, owns_holder=False).write_arrow_table(table)
        # On-disk bytes are gzip.
        assert target.read_bytes()[:2] == b"\x1f\x8b"
        # Decompressed contents look like CSV.
        with gzip.open(target.os_path, "rb") as fh:
            assert fh.read().startswith(b'"id","name"')

        # Read-back side decompresses transparently.
        loaded = CSVFile(holder=target, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_append_into_compressed_does_full_rewrite(
        self, tmp_path, table,
    ) -> None:
        from yggdrasil.enums import Mode
        from yggdrasil.io.primitive.csv_file import CsvOptions

        target = LocalPath(str(tmp_path / "trades.csv.gz"))
        CSVFile(holder=target, owns_holder=False).write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        CSVFile(holder=target, owns_holder=False).write_arrow_batches(
            more.to_batches(), options=CsvOptions(mode=Mode.APPEND),
        )
        loaded = CSVFile(holder=target, owns_holder=False).read_arrow_table()
        assert loaded.num_rows == 4
        # Header still appears exactly once after the rewrite.
        decoded = gzip.decompress(target.read_bytes()).decode("utf-8")
        assert decoded.count('"id","name"') == 1


# ---------------------------------------------------------------------------
# Compressed and uncompressed readers see the same content
# ---------------------------------------------------------------------------


class TestParityWithCodec:
    """A buffer written through CSVFile with a codec produces the same
    arrow rows as the same data written without one."""

    def test_csv_compressed_equals_uncompressed(self, table) -> None:
        plain = CSVFile()
        plain.write_arrow_table(table)
        plain_rows = plain.read_arrow_table().to_pylist()

        mem = _seed_codec(Memory(), MimeTypes.CSV, Codecs.GZIP)
        compressed = CSVFile(holder=mem, owns_holder=False)
        compressed.write_arrow_table(table)
        compressed_rows = compressed.read_arrow_table().to_pylist()

        assert plain_rows == compressed_rows


# ---------------------------------------------------------------------------
# JSON + gzip
# ---------------------------------------------------------------------------


class TestJsonGzipMemory:
    """Regression: an HTTP response body with
    ``Content-Encoding: gzip`` lands as ``application/json +
    application/gzip`` on the buffer's MediaType. JSONFile must peel
    the codec layer before parsing."""

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.JSON, Codecs.GZIP)
        JSONFile(holder=mem, owns_holder=False).write_arrow_table(table)
        # On-the-wire bytes are gzip-framed.
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        # Decompressed payload looks like a JSON array.
        decoded = gzip.decompress(mem.read_bytes())
        assert decoded.startswith(b"[")
        # Read-back transparently decompresses.
        loaded = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_reads_externally_gzipped_array(self) -> None:
        # Server-side gzip of a hand-rolled JSON array body — the
        # exact shape that triggered the original UnicodeDecodeError.
        body = b'[{"id":1,"name":"a"},{"id":2,"name":"b"}]'
        mem = _seed_codec(Memory(), MimeTypes.JSON, Codecs.GZIP)
        # Writing pre-compressed bytes via the raw BytesIO surface so
        # the test mirrors the HTTP response path (response body is
        # already gzip-framed; only the MediaType is stamped).
        BytesIO(holder=mem, owns_holder=False).write_bytes(gzip.compress(body))
        loaded = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2]

    def test_reads_externally_gzipped_ndjson_shape(self) -> None:
        # JSONFile also accepts NDJSON-shaped (newline-terminated) input.
        body = b'{"id":1,"name":"a"}\n{"id":2,"name":"b"}\n'
        mem = _seed_codec(Memory(), MimeTypes.JSON, Codecs.GZIP)
        BytesIO(holder=mem, owns_holder=False).write_bytes(gzip.compress(body))
        loaded = JSONFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2]


# ---------------------------------------------------------------------------
# XLSX + gzip
# ---------------------------------------------------------------------------


class TestXlsxGzipMemory:
    """An ``.xlsx.gz`` LocalPath stamps xlsx + gzip onto the holder.
    Reads should peel the codec before openpyxl opens the workbook;
    writes should compress the workbook ZIP into the holder."""

    def test_round_trip(self, table) -> None:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            pytest.skip("openpyxl not installed")
        mem = _seed_codec(Memory(), MimeTypes.XLSX, Codecs.GZIP)
        XLSXFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        # Decompressed payload is a ZIP (xlsx archive).
        decoded = gzip.decompress(mem.read_bytes())
        assert decoded[:2] == b"PK"
        loaded = XLSXFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Parquet + gzip codec wrapper
# ---------------------------------------------------------------------------


class TestParquetGzipMemory:
    """Parquet has its own internal compression, but the orthogonal
    holder-level codec still has to round-trip cleanly."""

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.PARQUET, Codecs.GZIP)
        ParquetFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        decoded = gzip.decompress(mem.read_bytes())
        # Parquet files start with the "PAR1" magic.
        assert decoded[:4] == b"PAR1"
        loaded = ParquetFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Arrow IPC + gzip
# ---------------------------------------------------------------------------


class TestArrowIPCGzipMemory:

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.ARROW_IPC, Codecs.GZIP)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        loaded = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert loaded.equals(table)
