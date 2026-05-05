"""FolderIO: ``options.max_workers`` threaded operations.

Verifies that the threaded fan-out paths produce the same results
as the single-threaded baseline and actually run multiple threads
under the hood. Threads are observable via the active-set probe
inside the per-item callable — running the same callable inline
(``max_workers <= 1``) only ever sees one thread, while a fanned
run sees more than one when the workload is large enough.
"""

from __future__ import annotations

import threading
import time

import pyarrow as pa
import pytest

from yggdrasil.io.buffer.nested import FolderIO, FolderOptions
from yggdrasil.io.buffer.nested.base import _run_in_threads


class TestRunInThreads:
    def test_serial_runs_in_calling_thread(self):
        seen: list[int] = []

        def fn(x: int) -> int:
            seen.append(threading.get_ident())
            return x * 2

        out = _run_in_threads([1, 2, 3], fn, max_workers=0)
        assert out == [2, 4, 6]
        assert set(seen) == {threading.get_ident()}

    def test_parallel_uses_multiple_threads(self):
        active = set()
        lock = threading.Lock()

        def fn(x: int) -> int:
            with lock:
                active.add(threading.get_ident())
            # Hold long enough for the pool to genuinely overlap.
            time.sleep(0.05)
            return x

        out = _run_in_threads(list(range(8)), fn, max_workers=4)
        assert out == list(range(8))
        # The calling thread shouldn't be in the worker set.
        assert threading.get_ident() not in active
        assert len(active) > 1

    def test_empty_returns_empty(self):
        assert _run_in_threads([], lambda x: x, max_workers=4) == []

    def test_single_item_skips_pool(self):
        # One item path stays in the calling thread regardless of
        # ``max_workers`` — no executor overhead for trivial work.
        observed: list[int] = []

        def fn(x: int) -> int:
            observed.append(threading.get_ident())
            return x

        _run_in_threads([42], fn, max_workers=8)
        assert observed == [threading.get_ident()]

    def test_failure_propagates(self):
        def fn(x: int) -> int:
            if x == 3:
                raise RuntimeError("boom")
            return x

        with pytest.raises(RuntimeError, match="boom"):
            _run_in_threads([1, 2, 3, 4], fn, max_workers=4)


class TestThreadedClearChildren:
    def test_overwrite_with_threads_replaces_data(self, tmp_path):
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            pa.Table.from_pylist([{"a": 1}, {"a": 2}]),
        )
        folder.write_arrow_table(
            pa.Table.from_pylist([{"a": 9}]),
            options=FolderOptions(max_workers=4),
        )
        out = FolderIO(path=str(tmp_path)).read_arrow_table()
        assert out.column("a").to_pylist() == [9]


class TestThreadedPartitionedWrites:
    def _table(self) -> pa.Table:
        return pa.Table.from_pylist(
            [
                {"year": "2024", "month": "01", "v": 1},
                {"year": "2024", "month": "02", "v": 2},
                {"year": "2025", "month": "01", "v": 3},
                {"year": "2025", "month": "02", "v": 4},
                {"year": "2026", "month": "03", "v": 5},
            ]
        )

    def test_sorted_partitioned_write_threaded(self, tmp_path):
        folder = FolderIO(
            path=str(tmp_path),
            partition_columns=["year", "month"],
        )
        folder.write_arrow_table(
            self._table(),
            options=FolderOptions(
                partition_columns=["year", "month"],
                max_workers=4,
                sort_partitions=True,
            ),
        )
        out = FolderIO(
            path=str(tmp_path),
            partition_columns=["year", "month"],
        ).read_arrow_table().sort_by([("v", "ascending")])
        assert out.column("v").to_pylist() == [1, 2, 3, 4, 5]
        assert set(out.column_names) >= {"year", "month", "v"}

    def test_streaming_partitioned_write_threaded(self, tmp_path):
        folder = FolderIO(
            path=str(tmp_path),
            partition_columns=["year"],
        )
        folder.write_arrow_table(
            self._table(),
            options=FolderOptions(
                partition_columns=["year"],
                max_workers=3,
                sort_partitions=False,
            ),
        )
        out = FolderIO(
            path=str(tmp_path),
            partition_columns=["year"],
        ).read_arrow_table().sort_by([("v", "ascending")])
        assert out.column("v").to_pylist() == [1, 2, 3, 4, 5]


class TestThreadedSchemaCollection:
    def test_collect_schema_matches_serial(self, tmp_path):
        folder = FolderIO(path=str(tmp_path))
        folder.write_arrow_table(
            pa.Table.from_pylist(
                [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]
            ),
            options=FolderOptions(child_row_size=1),
        )

        serial = FolderIO(path=str(tmp_path)).collect_schema()
        threaded = FolderIO(path=str(tmp_path)).collect_schema(
            options=FolderOptions(max_workers=4),
        )
        assert list(serial) == list(threaded)


class TestThreadedUpsert:
    def test_partitioned_upsert_threaded(self, tmp_path):
        from yggdrasil.io.enums import Mode

        folder = FolderIO(
            path=str(tmp_path),
            partition_columns=["year"],
        )
        # Seed two partitions.
        folder.write_arrow_table(
            pa.Table.from_pylist(
                [
                    {"year": "2024", "id": 1, "v": 10},
                    {"year": "2024", "id": 2, "v": 20},
                    {"year": "2025", "id": 3, "v": 30},
                ]
            ),
        )
        # Upsert: update id=1 in 2024 and add id=4 in 2025; threaded.
        folder.write_arrow_table(
            pa.Table.from_pylist(
                [
                    {"year": "2024", "id": 1, "v": 99},
                    {"year": "2025", "id": 4, "v": 40},
                ]
            ),
            options=FolderOptions(
                mode=Mode.UPSERT,
                match_by_names=("id",),
                partition_columns=["year"],
                max_workers=4,
            ),
        )
        out = (
            FolderIO(path=str(tmp_path), partition_columns=["year"])
            .read_arrow_table()
            .sort_by([("id", "ascending")])
        )
        rows = out.select(["id", "v"]).to_pylist()
        assert rows == [
            {"id": 1, "v": 99},
            {"id": 2, "v": 20},
            {"id": 3, "v": 30},
            {"id": 4, "v": 40},
        ]
