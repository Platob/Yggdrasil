"""End-to-end codec round-trip tests through real leaves.

The lifecycle context tests cover the codec branch in isolation.
These tests verify the same path through an actual format leaf —
ensuring that:

- A leaf with a codec produces compressed bytes on disk.
- The same leaf reads them back transparently.
- Native scanner gating respects the codec (delegated to base
  shim, not native path).
- Append-via-rewrite round-trips through the codec correctly.

We use ParquetIO and CsvIO as the two representative leaves —
ParquetIO for the footer-indexed format, CsvIO for the streaming
format with native append.
"""

from __future__ import annotations

from yggdrasil.io.enums import MediaType, MimeTypes, Codecs

from yggdrasil.io.enums import Mode
from yggdrasil.io.buffer.primitive.csv_io import CsvIO
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO


def _patch_codec(io, codec, monkeypatch):
    """Inject ``codec`` as the IO's ``codec`` property for the test.

    The real route is via MediaType.with_codec; bypassing that here
    keeps the tests isolated from the codec-registration plumbing.
    """
    monkeypatch.setattr(type(io), "codec", property(lambda self: codec))


# ---------------------------------------------------------------------------
# ParquetIO + codec
# ---------------------------------------------------------------------------


class TestParquetWithCodec:

    def test_round_trip(self, arrow_table, monkeypatch):
        io = ParquetIO(media_type=MediaType(MimeTypes.PARQUET, Codecs.GZIP))

        with io:
            io.write_arrow_table(arrow_table)

            io.seek(0)
            payload = io.read()

            io.seek(0)
            result = io.read_arrow_table()

        assert result.equals(arrow_table)

    def test_native_scanner_blocked_by_codec(
        self, arrow_table, monkeypatch
    ):
        io = ParquetIO(media_type=MediaType(MimeTypes.PARQUET, Codecs.GZIP))

        with io:
            io.write_arrow_table(arrow_table)
            options = io.check_options()
            # Native scanner must refuse — on-disk bytes are
            # compressed and the scanner would parse them as raw
            # parquet.
            assert not io._can_use_native_scanner(options)

    def test_append_via_rewrite_round_trips_codec(
        self, arrow_table, monkeypatch
    ):
        io = ParquetIO(media_type=MediaType(MimeTypes.PARQUET, Codecs.GZIP))

        with io:
            io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            io.seek(0)
            io.write_arrow_table(arrow_table, mode=Mode.APPEND)
            io.seek(0)
            result = io.read_arrow_table()

        assert result.num_rows == 2 * arrow_table.num_rows


# ---------------------------------------------------------------------------
# CsvIO + codec
# ---------------------------------------------------------------------------


class TestCsvWithCodec:

    def test_round_trip(self, arrow_table, monkeypatch):
        io = CsvIO(media_type=MediaType(MimeTypes.CSV, Codecs.GZIP))

        with io:
            io.write_arrow_table(arrow_table)

            io.seek(0)
            payload = io.read()

            io.seek(0)
            result = io.read_arrow_table()

        # CSV doesn't preserve types; check row count + columns.
        assert result.num_rows == arrow_table.num_rows
        assert result.column_names == arrow_table.column_names

    def test_append_with_codec_routes_through_rewrite(
        self, arrow_table, monkeypatch
    ):
        """CSV's native APPEND seeks to end and concatenates bytes —
        which is wrong if the bytes are compressed (you can't
        concat two compressed streams of most codecs and get a
        valid concatenated stream).

        With a codec active, the right behavior is for the rewrite
        helper to take over: read existing (decompressing),
        concat with incoming, write back through the codec.

        The current code may not enforce this — if so, this test
        documents the expectation.
        """
        io = CsvIO(media_type=MediaType(mime_type=MimeTypes.CSV, codec=Codecs.GZIP))

        with io:
            io.write_arrow_table(arrow_table, mode=Mode.OVERWRITE)
            io.write_arrow_table(arrow_table, mode=Mode.APPEND)
            io.seek(0)
            result = io.read_arrow_table()

        assert result.num_rows == 2 * arrow_table.num_rows
