"""Tests for :class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.ARROW_IPC) is ArrowIPCFile

    def test_path_dispatches_arrow_ext(self, tmp_path) -> None:
        from yggdrasil.io.bytes_io import BytesIO

        b = BytesIO(path=str(tmp_path / "x.arrow"))
        assert isinstance(b, ArrowIPCFile)


class TestMemoryRoundTrip:

    def test_write_then_read_table(self) -> None:
        table = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)
        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.equals(table)

    def test_write_then_read_batches(self) -> None:
        batch_a = pa.record_batch({"x": [1, 2]})
        batch_b = pa.record_batch({"x": [3, 4]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter([batch_a, batch_b]),
        )
        got = list(ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_batches())
        combined = pa.Table.from_batches(got)
        assert combined.column("x").to_pylist() == [1, 2, 3, 4]


class TestRechunkThroughIPC:
    """Verify rechunk_arrow_batches → ArrowIPCFile round-trips preserve data."""

    def _wide_table(self, n: int) -> pa.Table:
        return pa.table({
            "id": list(range(n)),
            "val": [float(i) * 1.1 for i in range(n)],
            "label": [f"row-{i}" for i in range(n)],
        })

    def test_rechunked_batches_round_trip(self) -> None:
        from yggdrasil.arrow.cast import rechunk_arrow_batches, get_arrow_nbytes

        table = self._wide_table(500)
        total_bytes = get_arrow_nbytes(table)
        target = total_bytes // 5

        rechunked = list(rechunk_arrow_batches(
            table.to_batches(), byte_size=target,
        ))
        assert len(rechunked) > 1

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(rechunked),
        )

        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.to_pydict() == table.to_pydict()

    def test_rechunk_with_zstd_compression(self) -> None:
        from yggdrasil.arrow.cast import rechunk_arrow_batches, get_arrow_nbytes

        table = self._wide_table(200)
        target = get_arrow_nbytes(table) // 3

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            rechunk_arrow_batches(
                table.to_batches(), byte_size=target,
            ),
            compression="zstd",
        )

        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.to_pydict() == table.to_pydict()
        assert mem.size > 0

    def test_rechunk_produces_multiple_ipc_batches(self) -> None:
        from yggdrasil.arrow.cast import rechunk_arrow_batches, get_arrow_nbytes

        table = self._wide_table(300)
        target = get_arrow_nbytes(table) // 6

        rechunked = list(rechunk_arrow_batches(
            table.to_batches(), byte_size=target,
        ))
        assert len(rechunked) >= 3

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(rechunked),
        )

        got_batches = list(
            ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_batches()
        )
        assert len(got_batches) == len(rechunked)
        for written, read_back in zip(rechunked, got_batches):
            assert read_back.num_rows == written.num_rows

    def test_many_small_batches_coalesced_through_ipc(self) -> None:
        from yggdrasil.arrow.cast import rechunk_arrow_batches

        small = [pa.record_batch({"x": [i]}) for i in range(50)]
        rechunked = list(rechunk_arrow_batches(small, byte_size=10_000))
        assert len(rechunked) < 50

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(rechunked),
        )

        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        expected = pa.Table.from_batches(small)
        assert got.to_pydict() == expected.to_pydict()

    def test_rechunk_row_size_through_ipc(self) -> None:
        from yggdrasil.arrow.cast import rechunk_arrow_batches

        table = self._wide_table(25)
        rechunked = list(rechunk_arrow_batches(
            table.to_batches(), row_size=7,
        ))
        assert [b.num_rows for b in rechunked] == [7, 7, 7, 4]

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(rechunked),
        )
        got_batches = list(
            ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_batches()
        )
        assert [b.num_rows for b in got_batches] == [7, 7, 7, 4]
        got = pa.Table.from_batches(got_batches)
        assert got.to_pydict() == table.to_pydict()


class TestContentSizeRechunkThroughIPC:
    """Verify page-content-size rechunking in _combine_paginated_pages shape."""

    def _wide_table(self, n: int) -> pa.Table:
        return pa.table({
            "id": list(range(n)),
            "val": [float(i) * 1.1 for i in range(n)],
            "label": [f"row-{i:06d}" for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
        })

    @staticmethod
    def _rechunk(table: pa.Table, target: int) -> list[pa.RecordBatch]:
        total_rows = table.num_rows
        content_bytes = sum(b.serialize().size for b in table.to_batches())
        if total_rows > 0 and content_bytes > target:
            max_chunksize = max(1, total_rows * target // content_bytes)
            return table.to_batches(max_chunksize=max_chunksize)
        return table.to_batches()

    def test_splits_by_content_size(self) -> None:
        table = self._wide_table(10_000)
        target = 32 * 1024
        batches = self._rechunk(table, target)
        assert len(batches) > 1
        for batch in batches[:-1]:
            assert batch.serialize().size <= target * 2

    def test_round_trip_preserves_data(self) -> None:
        table = self._wide_table(5_000)
        batches = self._rechunk(table, 64 * 1024)

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(batches), compression="zstd",
        )
        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.to_pydict() == table.to_pydict()

    def test_small_table_stays_single_batch(self) -> None:
        table = self._wide_table(10)
        batches = self._rechunk(table, 128 * 1024 * 1024)
        assert len(batches) == 1
        assert batches[0].num_rows == 10

    def test_empty_table(self) -> None:
        table = pa.table({"a": pa.array([], type=pa.int64())})
        batches = self._rechunk(table, 1024)
        total = sum(b.num_rows for b in batches)
        assert total == 0


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.arrow"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().equals(table)
