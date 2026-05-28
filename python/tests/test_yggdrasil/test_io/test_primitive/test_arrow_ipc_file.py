"""Tests for :class:`yggdrasil.io.primitive.arrow_ipc_file.ArrowIPCFile`."""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.io.holder import Holder
from yggdrasil.path.memory import Memory
from yggdrasil.path.local_path import LocalPath
from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCFile


class TestRegistration:

    def test_class_for_media_type(self) -> None:
        from yggdrasil.enums import MimeTypes

        assert Holder.class_for_media_type(MimeTypes.ARROW_IPC) is ArrowIPCFile

    def test_path_dispatches_arrow_ext(self, tmp_path) -> None:
        from yggdrasil.io.base import IO

        b = IO(path=str(tmp_path / "x.arrow"))
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


class TestWriteArrowTableBypassesBatchHook:
    """``ArrowIPCFile._write_arrow_table`` routes the "replace the
    buffer wholesale" shapes straight through ``writer.write_table``
    and only falls through to ``_write_arrow_batches`` for
    read-modify-rewrite merge cases and the guarded ``IGNORE`` /
    ``ERROR_IF_EXISTS`` paths."""

    @staticmethod
    def _counting_patch(monkeypatch):
        calls = {"n": 0}
        original = ArrowIPCFile._write_arrow_batches

        def counting(self, batches, options):
            calls["n"] += 1
            return original(self, batches, options)

        monkeypatch.setattr(ArrowIPCFile, "_write_arrow_batches", counting)
        return calls

    def test_overwrite_on_empty_skips_batch_hook(self, monkeypatch) -> None:
        calls = self._counting_patch(monkeypatch)
        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)

        assert calls["n"] == 0
        assert ArrowIPCFile(
            holder=mem, owns_holder=False,
        ).read_arrow_table().equals(table)

    def test_explicit_overwrite_on_nonempty_skips_batch_hook(
        self, monkeypatch,
    ) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [99]}), mode=Mode.OVERWRITE,
        )
        assert calls["n"] == 0
        out = ArrowIPCFile(
            holder=mem, owns_holder=False,
        ).read_arrow_table()
        assert out.column("id").to_pylist() == [99]

    def test_truncate_routes_to_fast_path(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [42]}), mode=Mode.TRUNCATE,
        )
        assert calls["n"] == 0
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert out.column("id").to_pylist() == [42]

    def test_append_to_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [4, 5]}), mode=Mode.APPEND,
        )
        assert calls["n"] >= 1
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4, 5]

    def test_append_to_empty_skips_batch_hook(self, monkeypatch) -> None:
        """APPEND on empty reduces to OVERWRITE — fast path is safe."""
        from yggdrasil.enums import Mode

        calls = self._counting_patch(monkeypatch)
        table = pa.table({"id": [1, 2, 3]})
        ArrowIPCFile(holder=Memory(), owns_holder=False).write_arrow_table(
            table, mode=Mode.APPEND,
        )
        assert calls["n"] == 0

    def test_auto_on_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        """AUTO without match_by resolves to APPEND on a non-empty
        buffer — must NOT clobber existing data."""
        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [4, 5]}),  # default mode = AUTO
        )
        assert calls["n"] >= 1
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert sorted(out.column("id").to_pylist()) == [1, 2, 3, 4, 5]

    def test_ignore_on_nonempty_uses_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)
        original_bytes = mem.to_bytes()

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [99]}), mode=Mode.IGNORE,
        )
        assert calls["n"] >= 1
        assert mem.to_bytes() == original_bytes

    def test_error_if_exists_on_nonempty_uses_batch_hook(
        self, monkeypatch,
    ) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        with pytest.raises(FileExistsError):
            ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
                pa.table({"id": [99]}), mode=Mode.ERROR_IF_EXISTS,
            )
        assert calls["n"] >= 1

    def test_upsert_with_match_by_uses_batch_hook(self, monkeypatch) -> None:
        from yggdrasil.enums import Mode

        seed = pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        calls = self._counting_patch(monkeypatch)
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            pa.table({"id": [2, 4], "v": ["B", "d"]}),
            mode=Mode.UPSERT, match_by=["id"],
        )
        assert calls["n"] >= 1
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        pairs = sorted(
            zip(out.column("id").to_pylist(), out.column("v").to_pylist())
        )
        assert pairs == [(1, "a"), (2, "B"), (3, "c"), (4, "d")]

    def test_target_schema_cast_applied_on_fast_path(self) -> None:
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.data_field import Field

        source = pa.table({"id": pa.array([1, 2, 3], type=pa.int64())})
        target = Field.from_(pa.schema([pa.field("id", pa.int32())]))

        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(
            source, options=CastOptions(target=target),
        )

        # Read back via stdlib IPC reader to confirm the cast committed.
        import pyarrow.ipc as _ipc
        reader = _ipc.RecordBatchFileReader(pa.BufferReader(mem.to_bytes()))
        assert reader.schema.field("id").type == pa.int32()

    def test_empty_table_fast_path(self) -> None:
        empty = pa.table({"id": pa.array([], type=pa.int64())})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(empty)

        reread = ArrowIPCFile(
            holder=mem, owns_holder=False,
        ).read_arrow_table()
        assert reread.num_rows == 0
        assert reread.schema.field("id").type == pa.int64()

    def test_cursor_opened_in_overwrite_mode_takes_fast_path(
        self, monkeypatch, tmp_path,
    ) -> None:
        """``path.open("wb")`` gives a cursor with parent.mode =
        OVERWRITE — ``holder_is_overwrite`` is True, so the override
        skips the merge path even when the underlying file has bytes."""
        path = LocalPath(str(tmp_path / "x.arrow"))
        with path.open("wb") as cursor:
            cursor.write_arrow_table(pa.table({"id": [1, 2, 3]}))

        calls = self._counting_patch(monkeypatch)
        with path.open("wb") as cursor:
            assert isinstance(cursor, ArrowIPCFile)
            cursor.write_arrow_table(pa.table({"id": [99]}))
        assert calls["n"] == 0

        with path.open("rb") as cursor:
            assert cursor.read_arrow_table().column("id").to_pylist() == [99]


class TestReadArrowTableBypassesBatchHook:
    """``ArrowIPCFile._read_arrow_table`` routes through
    :meth:`RecordBatchFileReader.read_all` — a single C++ call that
    decodes every batch into one :class:`pa.Table` — instead of
    streaming ``_read_arrow_batches`` and re-stitching via
    ``pa.Table.from_batches``."""

    @staticmethod
    def _counting_patch(monkeypatch):
        calls = {"n": 0}
        original = ArrowIPCFile._read_arrow_batches

        def counting(self, options):
            calls["n"] += 1
            return original(self, options)

        monkeypatch.setattr(ArrowIPCFile, "_read_arrow_batches", counting)
        return calls

    def test_read_arrow_table_skips_batch_hook(self, monkeypatch) -> None:
        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)

        calls = self._counting_patch(monkeypatch)
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert calls["n"] == 0
        assert out.equals(table)

    def test_row_limit_applied_on_fast_path(self) -> None:
        from yggdrasil.io.primitive.arrow_ipc_file import ArrowIPCOptions

        table = pa.table({"id": list(range(1000))})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(table)

        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table(
            options=ArrowIPCOptions(row_limit=42),
        )
        assert out.num_rows == 42
        assert out.column("id").to_pylist() == list(range(42))

    def test_target_projection_on_fast_path(self) -> None:
        """``options.target`` with a subset of columns drops the
        unused columns post-read (the IPC format reads all batches,
        but the projection still applies via apply_post_read_table)."""
        from yggdrasil.data.options import CastOptions
        from yggdrasil.data.data_field import Field

        seed = pa.table({"id": [1, 2, 3], "skip": ["a", "b", "c"], "keep": [10, 20, 30]})
        mem = Memory()
        ArrowIPCFile(holder=mem, owns_holder=False).write_arrow_table(seed)

        target = Field.from_(pa.schema([
            pa.field("id", pa.int64()),
            pa.field("keep", pa.int64()),
        ]))
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table(
            options=CastOptions(target=target),
        )
        assert out.column_names == ["id", "keep"]
        assert out.column("keep").to_pylist() == [10, 20, 30]

    def test_empty_file_falls_back_to_base(self, monkeypatch) -> None:
        # size == 0 path lands in the base for empty-schema synthesis.
        mem = Memory()
        calls = self._counting_patch(monkeypatch)
        out = ArrowIPCFile(holder=mem, owns_holder=False).read_arrow_table()
        assert calls["n"] >= 1
        assert out.num_rows == 0
