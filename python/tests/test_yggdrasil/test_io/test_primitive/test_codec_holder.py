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

from yggdrasil.data.enums import MimeTypes
from yggdrasil.data.enums.codec import Codecs
from yggdrasil.data.enums.media_type import MediaType
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.primitive.csv_io import CsvIO
from yggdrasil.io.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.primitive.parquet_io import ParquetIO


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


def _seed_codec(holder: Memory, mime: MimeTypes, codec) -> Memory:
    """Stamp a media_type with codec onto the in-memory holder.

    This is what a path-bound holder learns from a ``.csv.gz``
    URL extension; for in-memory tests we pin it explicitly.
    """
    holder.stat().media_type = MediaType(mime_type=mime, codec=codec)
    return holder


# ---------------------------------------------------------------------------
# Sanity — uncompressed Memory holders still round-trip
# ---------------------------------------------------------------------------


class TestUncompressedSanity:
    """No codec → ``_format_view`` is a plain view; ``_format_buffer``
    writes straight into self."""

    def test_csv_round_trip(self, table) -> None:
        io = CsvIO()
        io.write_arrow_table(table)
        assert not io.to_bytes().startswith(b"\x1f\x8b")  # not gzip
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_arrow_round_trip(self, table) -> None:
        io = ArrowIPCIO()
        io.write_arrow_table(table)
        assert io.read_arrow_table().equals(table)

    def test_parquet_round_trip(self, table) -> None:
        io = ParquetIO()
        io.write_arrow_table(table)
        assert io.read_arrow_table().equals(table)


# ---------------------------------------------------------------------------
# CSV + gzip (Memory holder, explicitly seeded)
# ---------------------------------------------------------------------------


class TestCsvGzipMemory:

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.CSV, Codecs.GZIP)
        io = CsvIO(holder=mem, owns_holder=False)
        io.write_arrow_table(table)

        # Bytes on the holder are gzip-compressed.
        assert mem.read_bytes()[:2] == b"\x1f\x8b"

        # Decompressed payload looks like CSV.
        decoded = gzip.decompress(mem.read_bytes())
        assert decoded.startswith(b'"id","name"')

        # Reading back transparently decompresses.
        reader = CsvIO(holder=mem, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# NDJSON + gzip
# ---------------------------------------------------------------------------


class TestNDJsonGzipMemory:

    def test_round_trip(self, table) -> None:
        mem = _seed_codec(Memory(), MimeTypes.NDJSON, Codecs.GZIP)
        NDJsonIO(holder=mem, owns_holder=False).write_arrow_table(table)
        assert mem.read_bytes()[:2] == b"\x1f\x8b"
        loaded = NDJsonIO(holder=mem, owns_holder=False).read_arrow_table()
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
        ArrowIPCIO(holder=mem, owns_holder=False).write_arrow_table(table)
        # Zstd magic bytes.
        assert mem.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"
        loaded = ArrowIPCIO(holder=mem, owns_holder=False).read_arrow_table()
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

        CsvIO(holder=target, owns_holder=False).write_arrow_table(table)
        # On-disk bytes are gzip.
        assert target.read_bytes()[:2] == b"\x1f\x8b"
        # Decompressed contents look like CSV.
        with gzip.open(target.os_path, "rb") as fh:
            assert fh.read().startswith(b'"id","name"')

        # Read-back side decompresses transparently.
        loaded = CsvIO(holder=target, owns_holder=False).read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]

    def test_append_into_compressed_does_full_rewrite(
        self, tmp_path, table,
    ) -> None:
        from yggdrasil.data.enums import Mode
        from yggdrasil.io.primitive.csv_io import CsvOptions

        target = LocalPath(str(tmp_path / "trades.csv.gz"))
        CsvIO(holder=target, owns_holder=False).write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        CsvIO(holder=target, owns_holder=False).write_arrow_batches(
            more.to_batches(), options=CsvOptions(mode=Mode.APPEND),
        )
        loaded = CsvIO(holder=target, owns_holder=False).read_arrow_table()
        assert loaded.num_rows == 4
        # Header still appears exactly once after the rewrite.
        decoded = gzip.decompress(target.read_bytes()).decode("utf-8")
        assert decoded.count('"id","name"') == 1


# ---------------------------------------------------------------------------
# Compressed and uncompressed readers see the same content
# ---------------------------------------------------------------------------


class TestParityWithCodec:
    """A buffer written through CsvIO with a codec produces the same
    arrow rows as the same data written without one."""

    def test_csv_compressed_equals_uncompressed(self, table) -> None:
        plain = CsvIO()
        plain.write_arrow_table(table)
        plain_rows = plain.read_arrow_table().to_pylist()

        mem = _seed_codec(Memory(), MimeTypes.CSV, Codecs.GZIP)
        compressed = CsvIO(holder=mem, owns_holder=False)
        compressed.write_arrow_table(table)
        compressed_rows = compressed.read_arrow_table().to_pylist()

        assert plain_rows == compressed_rows
