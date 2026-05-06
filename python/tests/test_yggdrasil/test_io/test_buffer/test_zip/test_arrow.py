"""ZipIO + ZipEntryIO: Arrow Table / RecordBatch round-trip.

Tabular writes route through :class:`NestedIO._write_arrow_batches`
which mints one :class:`ZipEntryIO` per chunk and lets the format
leaf (Arrow IPC by default) write into the entry's :class:`BytesIO`
buffer. Reads chain entries back into a flat batch stream.
"""

from __future__ import annotations

import zipfile

import pytest

from yggdrasil.io.buffer.nested import ZipIO, ZipOptions
from .._helpers import sample_batches, sample_table


class TestZipIOArrow:
    def test_write_creates_archive_with_batch_entries(self, tmp_path):
        path = tmp_path / "a.zip"
        ZipIO(path=str(path)).write_arrow_table(sample_table())

        with zipfile.ZipFile(str(path)) as zf:
            names = zf.namelist()
        assert names, "ZipIO write should produce at least one entry"
        assert all(n.startswith("batch-") for n in names)

    def test_round_trip_table(self, tmp_path):
        path = tmp_path / "a.zip"
        ZipIO(path=str(path)).write_arrow_table(sample_table())
        out = ZipIO(path=str(path)).read_arrow_table()
        assert out.num_rows == 3
        assert out.column_names == ["a", "b"]

    def test_round_trip_batches(self, tmp_path):
        path = tmp_path / "b.zip"
        ZipIO(path=str(path)).write_arrow_batches(iter(sample_batches()))
        out = list(ZipIO(path=str(path)).read_arrow_batches())
        assert sum(b.num_rows for b in out) == 3

    def test_collect_schema(self, tmp_path):
        path = tmp_path / "c.zip"
        ZipIO(path=str(path)).write_arrow_table(sample_table())
        schema = ZipIO(path=str(path)).collect_schema()
        assert list(schema.field_names()) == ["a", "b"]

