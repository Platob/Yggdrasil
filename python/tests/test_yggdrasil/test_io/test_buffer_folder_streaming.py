"""FolderIO streaming-write + parallel-read coverage.

These tests exercise the canonical "tail this folder" pattern: a
producer writes batches in append mode while one or more consumers
read concurrently. The ``.ygg/`` sidecar surface (checkpoint log +
key/value metadata) is the durable coordination surface — readers
can pin a known state, and writers can emit progress markers without
contaminating the data folder.

The folder-root lock from ``concurrent=True`` keeps concurrent
writers from racing each other; reads are racy by construction
(append-only folders are eventually-consistent reads of whatever
finalized rename'd children exist), but the framing here verifies
no torn reads, no partial files, no schema drift.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time

import pyarrow as pa
import pytest

from yggdrasil.io.buffer.nested.folder_io import FolderIO
from yggdrasil.io.enums import Mode


_IS_WINDOWS = sys.platform.startswith("win")


def _make_table(start: int, n: int = 8) -> pa.Table:
    return pa.table({
        "id": pa.array(list(range(start, start + n)), type=pa.int64()),
        "tag": pa.array([f"row-{i}" for i in range(start, start + n)]),
    })


# ---------------------------------------------------------------------------
# .ygg sidecar — checkpoints + metadata
# ---------------------------------------------------------------------------


class TestYggSidecar:
    def test_ygg_directory_constants(self):
        assert FolderIO.YGG_DIR_NAME == ".ygg"
        assert FolderIO.CHECKPOINT_LOG_NAME == "checkpoints.jsonl"
        assert FolderIO.METADATA_DIR_NAME == "metadata"

    def test_list_checkpoints_empty_when_no_log(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            assert io.list_checkpoints() == []
            assert io.latest_checkpoint() is None

    def test_checkpoint_creates_log_under_ygg(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint("after-first-batch")
        assert (tmp_path / ".ygg" / "checkpoints.jsonl").exists()
        assert record["id"] == 1
        assert record["message"] == "after-first-batch"
        assert record["num_files"] >= 1

    def test_checkpoints_are_monotonic_and_ordered(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            for i in range(5):
                io.write_arrow_table(_make_table(i * 8), mode=Mode.APPEND)
                io.checkpoint(message=f"batch-{i}", batch=i)

            recs = io.list_checkpoints()
        assert [r["id"] for r in recs] == [1, 2, 3, 4, 5]
        # ``num_files`` grows with each appended batch.
        assert [r["num_files"] for r in recs] == sorted(
            r["num_files"] for r in recs
        )
        assert [r["batch"] for r in recs] == [0, 1, 2, 3, 4]

    def test_extra_kwargs_round_trip(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint(rows_seen=128, source="ingest-job-7")
        recs = io.list_checkpoints()
        assert recs == [record]
        assert recs[0]["rows_seen"] == 128
        assert recs[0]["source"] == "ingest-job-7"

    def test_corrupt_log_lines_are_skipped(self, tmp_path):
        ygg = tmp_path / ".ygg"
        ygg.mkdir()
        (ygg / "checkpoints.jsonl").write_bytes(
            b'{"id": 1, "ts": 1.0, "pid": 1, "files": [], "num_files": 0}\n'
            b"this is not json\n"
            b'{"id": 2, "ts": 2.0, "pid": 1, "files": [], "num_files": 0}\n'
            b"\n"
        )
        with FolderIO(path=str(tmp_path)) as io:
            recs = io.list_checkpoints()
        assert [r["id"] for r in recs] == [1, 2]

    def test_metadata_round_trip(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_metadata("schema_version", {"major": 1, "minor": 0})
            io.write_metadata("source-job", "ingest-7")
            io.write_metadata("counts.rows", 1024)

            assert io.read_metadata("schema_version") == {"major": 1, "minor": 0}
            assert io.read_metadata("source-job") == "ingest-7"
            assert io.read_metadata("counts.rows") == 1024
            assert io.read_metadata("missing") is None
            assert io.read_metadata("missing", default="fallback") == "fallback"
            assert io.list_metadata_keys() == sorted(
                ["schema_version", "source-job", "counts.rows"]
            )

    def test_metadata_overwrite_is_atomic_in_append_order(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_metadata("k", 1)
            io.write_metadata("k", 2)
            assert io.read_metadata("k") == 2

    @pytest.mark.parametrize(
        "bad",
        ["", "../escape", "with/slash", "..", ".", "with space", "key$"],
    )
    def test_metadata_keys_reject_path_traversal(self, tmp_path, bad):
        with FolderIO(path=str(tmp_path)) as io:
            with pytest.raises(ValueError):
                io.write_metadata(bad, "x")
            with pytest.raises(ValueError):
                io.read_metadata(bad)

    def test_ygg_folder_skipped_by_iter_children(self, tmp_path):
        """Adding sidecar metadata must not contaminate data
        enumeration — iter_children skips the ``.ygg/`` folder."""
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            io.checkpoint("c1")
            io.write_metadata("k", 1)

            children = list(io.iter_children())
            names = [c.path.name for c in children]
        assert ".ygg" not in names
        # The data file still appears.
        assert any(n.endswith(".parquet") for n in names)

    def test_read_arrow_table_ignores_ygg(self, tmp_path):
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4))
            io.checkpoint("with-meta")
            io.write_metadata("note", "hello")

            tbl = io.read_arrow_table()
        # The .ygg payload must not be merged into the data view.
        assert tbl.num_rows == 4
        assert set(tbl.column_names) == {"id", "tag"}


# ---------------------------------------------------------------------------
# Streaming write + parallel read
# ---------------------------------------------------------------------------


class TestStreamingFolderIO:
    def test_continuous_append_grows_row_count(self, tmp_path):
        """A sequential producer that appends N batches with a
        checkpoint after each one should see the row count grow
        monotonically with each post-checkpoint read."""
        n_batches = 6
        rows_per_batch = 8

        with FolderIO(path=str(tmp_path)) as writer:
            seen: list[int] = []
            for i in range(n_batches):
                writer.write_arrow_table(
                    _make_table(i * rows_per_batch, rows_per_batch),
                    mode=Mode.APPEND,
                )
                writer.checkpoint(batch=i)
                # Reader handle, fresh per iteration.
                with FolderIO(path=str(tmp_path)) as reader:
                    seen.append(reader.read_arrow_table().num_rows)

        assert seen == [
            (i + 1) * rows_per_batch for i in range(n_batches)
        ]

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_parallel_readers_see_consistent_data(self, tmp_path):
        """Several reader threads concurrently iterate the folder
        while a writer continuously appends. Every read either
        succeeds with a row count that's a multiple of ``rows_per_batch``
        (a clean snapshot) or raises a known IO error — never
        returns garbled rows."""
        rows_per_batch = 4
        n_batches = 12
        # Pre-populate so readers always have something to drain.
        with FolderIO(path=str(tmp_path)) as boot:
            boot.write_arrow_table(_make_table(0, rows_per_batch),
                                   mode=Mode.APPEND)

        stop = threading.Event()
        observed: list[int] = []
        observed_lock = threading.Lock()
        errors: list[Exception] = []

        def writer():
            with FolderIO(path=str(tmp_path), concurrent=True,
                          lock_wait=10.0) as w:
                for i in range(1, n_batches):
                    w.write_arrow_table(
                        _make_table(i * rows_per_batch, rows_per_batch),
                        mode=Mode.APPEND,
                    )
                    w.checkpoint(batch=i)
                    time.sleep(0.005)
            stop.set()

        def reader():
            while not stop.is_set():
                try:
                    with FolderIO(path=str(tmp_path)) as r:
                        n = r.read_arrow_table().num_rows
                    with observed_lock:
                        observed.append(n)
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
                    return
                time.sleep(0.001)

        wt = threading.Thread(target=writer)
        rs = [threading.Thread(target=reader) for _ in range(3)]
        wt.start()
        for r in rs:
            r.start()
        wt.join(timeout=30.0)
        for r in rs:
            r.join(timeout=30.0)

        assert errors == [], f"Reader saw errors: {errors[:3]}"
        # Every observation must be a multiple of rows_per_batch and
        # within the legal range — atomic finalize via stage+rename
        # guarantees we never see a fractional file.
        for n in observed:
            assert n % rows_per_batch == 0, n
            assert rows_per_batch <= n <= n_batches * rows_per_batch
        # The final read post-stop must equal the full total.
        with FolderIO(path=str(tmp_path)) as final:
            assert final.read_arrow_table().num_rows == (
                n_batches * rows_per_batch
            )

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_concurrent_checkpoints_serialise_via_log_lock(self, tmp_path):
        """Multiple threads calling ``checkpoint()`` concurrently
        must not tear each other's JSON lines, and every call must
        appear exactly once in the log."""
        n_threads = 8
        per_thread = 5

        with FolderIO(path=str(tmp_path)) as boot:
            boot.write_arrow_table(_make_table(0, 4))

        def worker(thread_id: int):
            with FolderIO(path=str(tmp_path)) as io:
                for i in range(per_thread):
                    io.checkpoint(thread=thread_id, seq=i)

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20.0)

        with FolderIO(path=str(tmp_path)) as r:
            recs = r.list_checkpoints()
        # Every thread × every seq is present, exactly once.
        assert len(recs) == n_threads * per_thread
        observed = {(rec["thread"], rec["seq"]) for rec in recs}
        expected = {
            (t, i) for t in range(n_threads) for i in range(per_thread)
        }
        assert observed == expected

    def test_checkpoint_records_current_filenames(self, tmp_path):
        """Each checkpoint snapshots what's visible at commit time —
        a downstream replay can reconstruct the folder's state at
        each commit point."""
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4), mode=Mode.APPEND)
            cp_a = io.checkpoint("after-a")
            io.write_arrow_table(_make_table(4, 4), mode=Mode.APPEND)
            cp_b = io.checkpoint("after-b")

        files_a = set(cp_a["files"])
        files_b = set(cp_b["files"])
        assert files_a.issubset(files_b)
        assert len(files_b - files_a) == 1
        # No sidecar entries leak into the file list.
        for f in files_b:
            assert not f.startswith(".")

    def test_metadata_survives_overwrite_of_data(self, tmp_path):
        """OVERWRITE mode clears data children but keeps the .ygg
        sidecar — checkpoints and metadata are out-of-band and
        shouldn't be wiped by a data refresh."""
        with FolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4), mode=Mode.APPEND)
            io.checkpoint("v1")
            io.write_metadata("flag", True)

            io.write_arrow_table(_make_table(100, 4), mode=Mode.OVERWRITE)

            assert io.read_metadata("flag") is True
            recs = io.list_checkpoints()
        assert len(recs) == 1
        assert recs[0]["message"] == "v1"
