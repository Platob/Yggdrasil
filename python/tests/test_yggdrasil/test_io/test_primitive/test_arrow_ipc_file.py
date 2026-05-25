"""Tests for :class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.io.memory import Memory
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.data.enums import MimeTypes

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


class TestCalibratedRechunkThroughIPC:
    """Verify _calibrate_ipc_chunksize produces precise serialized page sizes."""

    def _wide_table(self, n: int) -> pa.Table:
        return pa.table({
            "id": list(range(n)),
            "val": [float(i) * 1.1 for i in range(n)],
            "label": [f"row-{i:06d}" for i in range(n)],
            "flag": [i % 2 == 0 for i in range(n)],
        })

    def test_calibrated_chunksize_targets_serialized_size(self) -> None:
        import pyarrow.ipc as ipc
        from yggdrasil.http_.session import HTTPSession

        table = self._wide_table(10_000)
        target = 32 * 1024  # 32 KiB per serialized page

        chunksize = HTTPSession._calibrate_ipc_chunksize(table, target)
        batches = table.to_batches(max_chunksize=chunksize)
        assert len(batches) > 1

        for batch in batches[:-1]:
            sink = pa.BufferOutputStream()
            with ipc.new_stream(
                sink, batch.schema,
                options=ipc.IpcWriteOptions(compression="zstd"),
            ) as w:
                w.write_batch(batch)
            serialized = sink.getvalue().size
            assert serialized <= target * 2

    def test_calibrated_round_trip_preserves_data(self) -> None:
        from yggdrasil.http_.session import HTTPSession

        table = self._wide_table(5_000)
        target = 64 * 1024

        chunksize = HTTPSession._calibrate_ipc_chunksize(table, target)
        batches = table.to_batches(max_chunksize=chunksize)

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_batches(
            iter(batches), compression="zstd",
        )
        got = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert got.to_pydict() == table.to_pydict()

    def test_calibrated_vs_nbytes_precision(self) -> None:
        """Calibrated chunking should be closer to the target than nbytes-based."""
        import pyarrow.ipc as ipc
        from yggdrasil.arrow.cast import rechunk_arrow_batches, get_arrow_nbytes
        from yggdrasil.http_.session import HTTPSession

        table = self._wide_table(10_000)
        target = 64 * 1024

        # Calibrated approach
        chunksize = HTTPSession._calibrate_ipc_chunksize(table, target)
        calibrated_batches = table.to_batches(max_chunksize=chunksize)

        # nbytes-based approach
        nbytes_batches = list(rechunk_arrow_batches(
            table.to_batches(), byte_size=target,
        ))

        def _measure_serialized(batches):
            sizes = []
            for b in batches:
                sink = pa.BufferOutputStream()
                with ipc.new_stream(
                    sink, b.schema,
                    options=ipc.IpcWriteOptions(compression="zstd"),
                ) as w:
                    w.write_batch(b)
                sizes.append(sink.getvalue().size)
            return sizes

        cal_sizes = _measure_serialized(calibrated_batches)
        nb_sizes = _measure_serialized(nbytes_batches)

        cal_error = sum(abs(s - target) for s in cal_sizes[:-1]) / max(1, len(cal_sizes) - 1)
        nb_error = sum(abs(s - target) for s in nb_sizes[:-1]) / max(1, len(nb_sizes) - 1)

        assert cal_error <= nb_error

    def test_empty_table(self) -> None:
        from yggdrasil.http_.session import HTTPSession

        table = pa.table({"a": pa.array([], type=pa.int64())})
        chunksize = HTTPSession._calibrate_ipc_chunksize(table, 1024)
        assert chunksize == 1

    def test_tiny_table_single_batch(self) -> None:
        from yggdrasil.http_.session import HTTPSession

        table = self._wide_table(10)
        chunksize = HTTPSession._calibrate_ipc_chunksize(table, 128 * 1024 * 1024)
        assert chunksize >= 10
        batches = table.to_batches(max_chunksize=chunksize)
        assert len(batches) == 1
        assert batches[0].num_rows == 10


class TestLocalPathRoundTrip:

    def test_write_and_read(self, tmp_path) -> None:
        table = pa.table({"x": [10, 20, 30]})
        path = LocalPath(str(tmp_path / "out.arrow"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(table)
        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().equals(table)
