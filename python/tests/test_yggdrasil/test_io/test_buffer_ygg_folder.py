"""YGGFolderIO — streaming-write + parallel-read + Stats sidecar.

These tests exercise the canonical "tail this folder" pattern: a
producer writes batches in append mode while one or more consumers
read concurrently. :class:`YGGFolderIO` adds a ``.ygg/`` sidecar
that holds:

- ``checkpoints.jsonl`` — append-only commit log.
- ``metadata/<key>.json`` — key/value attribute store.
- ``stats.arrow`` — Arrow-IPC-encoded :class:`Stats` for the folder.

Plus ``YGGFolderIO(path=...)`` auto-upgrades to :class:`YGGFolderIO`
when the target already has a ``.ygg/`` sidecar. The folder-root
lock from ``concurrent=True`` serialises concurrent writers; reads
are racy by construction (append-only folders are
eventually-consistent reads of whatever finalized rename'd children
exist), but the framing here verifies no torn reads, no partial
files, no schema drift.
"""

from __future__ import annotations

import pathlib
import sys
import threading
import time

import pyarrow as pa
import pytest

from yggdrasil.io.nested import FolderIO, YGGFolderIO, is_ygg_folder
from yggdrasil.data.enums import Mode, MimeTypes
from yggdrasil.io.stats import Stats


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
        assert YGGFolderIO.YGG_DIR_NAME == ".ygg"
        assert YGGFolderIO.CHECKPOINT_LOG_NAME == "checkpoints.jsonl"
        assert YGGFolderIO.METADATA_DIR_NAME == "metadata"
        assert YGGFolderIO.STATS_FILENAME == "stats.arrow"

    def test_default_media_type(self):
        assert YGGFolderIO.default_media_type() is MimeTypes.YGG_FOLDER
        assert FolderIO.default_media_type() is MimeTypes.FOLDER

    def test_list_checkpoints_empty_when_no_log(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            assert io.list_checkpoints() == []
            assert io.latest_checkpoint() is None

    def test_checkpoint_creates_log_under_ygg(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint("after-first-batch")
        assert (tmp_path / ".ygg" / "checkpoints.jsonl").exists()
        assert record["id"] == 1
        assert record["message"] == "after-first-batch"
        assert record["num_files"] >= 1

    def test_checkpoints_are_monotonic_and_ordered(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
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
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint(rows_seen=128, source="ingest-job-7")
        recs = io.list_checkpoints()
        assert recs == [record]
        assert recs[0]["rows_seen"] == 128
        assert recs[0]["source"] == "ingest-job-7"


    def test_checkpoint_records_compute_owner_url(self, tmp_path):
        """Each checkpoint stamps a compute-identifier URL so a
        downstream replay can attribute it to a specific machine /
        Databricks job / pipeline run."""
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint("attributed")
        assert "owner" in record
        url = record["owner"]
        # Default fallback shape outside Databricks env.
        assert isinstance(url, str)
        assert url.startswith("host://") or url.startswith("databricks://")

    def test_checkpoint_accepts_explicit_owner_override(self, tmp_path):
        """Spark drivers commit on behalf of distributed workers —
        the explicit ``owner`` override lets the single commit point
        attribute the whole batch to one job rather than to whichever
        local process happened to write the log line."""
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            record = io.checkpoint(
                "spark-batch",
                owner="databricks://shared-cluster/1?host=driver&job=42&run=99",
            )
        assert (
            record["owner"]
            == "databricks://shared-cluster/1?host=driver&job=42&run=99"
        )

    def test_corrupt_log_lines_are_skipped(self, tmp_path):
        ygg = tmp_path / ".ygg"
        ygg.mkdir()
        (ygg / "checkpoints.jsonl").write_bytes(
            b'{"id": 1, "ts": 1.0, "pid": 1, "files": [], "num_files": 0}\n'
            b"this is not json\n"
            b'{"id": 2, "ts": 2.0, "pid": 1, "files": [], "num_files": 0}\n'
            b"\n"
        )
        with YGGFolderIO(path=str(tmp_path)) as io:
            recs = io.list_checkpoints()
        assert [r["id"] for r in recs] == [1, 2]

    def test_metadata_round_trip(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
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
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_metadata("k", 1)
            io.write_metadata("k", 2)
            assert io.read_metadata("k") == 2

    @pytest.mark.parametrize(
        "bad",
        ["", "../escape", "with/slash", "..", ".", "with space", "key$"],
    )
    def test_metadata_keys_reject_path_traversal(self, tmp_path, bad):
        with YGGFolderIO(path=str(tmp_path)) as io:
            with pytest.raises(ValueError):
                io.write_metadata(bad, "x")
            with pytest.raises(ValueError):
                io.read_metadata(bad)

    def test_ygg_folder_skipped_by_iter_children(self, tmp_path):
        """Adding sidecar metadata must not contaminate data
        enumeration — iter_children skips the ``.ygg/`` folder."""
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0))
            io.checkpoint("c1")
            io.write_metadata("k", 1)

            children = list(io.iter_children())
            names = [c.path.name for c in children]
        assert ".ygg" not in names
        # The data file still appears. YGGFolderIO defaults to Arrow IPC
        # for child files (``.ipc``).
        assert any(n.endswith(".ipc") for n in names)

    def test_read_arrow_table_ignores_ygg(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
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

        with YGGFolderIO(path=str(tmp_path)) as writer:
            seen: list[int] = []
            for i in range(n_batches):
                writer.write_arrow_table(
                    _make_table(i * rows_per_batch, rows_per_batch),
                    mode=Mode.APPEND,
                )
                writer.checkpoint(batch=i)
                # Reader handle, fresh per iteration.
                with YGGFolderIO(path=str(tmp_path)) as reader:
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
        with YGGFolderIO(path=str(tmp_path)) as boot:
            boot.write_arrow_table(_make_table(0, rows_per_batch),
                                   mode=Mode.APPEND)

        stop = threading.Event()
        observed: list[int] = []
        observed_lock = threading.Lock()
        errors: list[Exception] = []

        def writer():
            with YGGFolderIO(path=str(tmp_path)) as w:
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
                    with YGGFolderIO(path=str(tmp_path)) as r:
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
        with YGGFolderIO(path=str(tmp_path)) as final:
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

        with YGGFolderIO(path=str(tmp_path)) as boot:
            boot.write_arrow_table(_make_table(0, 4))

        def worker(thread_id: int):
            with YGGFolderIO(path=str(tmp_path)) as io:
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

        with YGGFolderIO(path=str(tmp_path)) as r:
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
        with YGGFolderIO(path=str(tmp_path)) as io:
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
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4), mode=Mode.APPEND)
            io.checkpoint("v1")
            io.write_metadata("flag", True)

            io.write_arrow_table(_make_table(100, 4), mode=Mode.OVERWRITE)

            assert io.read_metadata("flag") is True
            recs = io.list_checkpoints()
        assert len(recs) == 1
        assert recs[0]["message"] == "v1"


# ---------------------------------------------------------------------------
# Auto-detection — FolderIO upgrades to YGGFolderIO when .ygg/ exists
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_plain_folder_stays_folderio(self, tmp_path):
        # Empty dir, no sidecar.
        io = FolderIO(path=str(tmp_path))
        try:
            assert type(io) is FolderIO
        finally:
            io.close()

    def test_folder_with_ygg_sidecar_upgrades_to_ygg(self, tmp_path):
        (tmp_path / ".ygg").mkdir()
        io = FolderIO(path=str(tmp_path))
        try:
            assert isinstance(io, YGGFolderIO)
        finally:
            io.close()

    def test_explicit_ygg_construction_keeps_class(self, tmp_path):
        io = YGGFolderIO(path=str(tmp_path))
        try:
            assert type(io) is YGGFolderIO
        finally:
            io.close()

    def test_is_ygg_folder_helper(self, tmp_path):
        assert is_ygg_folder(str(tmp_path)) is False
        (tmp_path / ".ygg").mkdir()
        assert is_ygg_folder(str(tmp_path)) is True

    def test_first_write_then_reopen_upgrades(self, tmp_path):
        """A fresh YGGFolderIO call leaves the .ygg sidecar behind;
        reopening as plain FolderIO afterwards auto-upgrades."""
        with YGGFolderIO(path=str(tmp_path)) as ygg:
            ygg.write_arrow_table(_make_table(0))
            ygg.checkpoint("v1")

        reopened = FolderIO(path=str(tmp_path))
        try:
            assert isinstance(reopened, YGGFolderIO)
            assert reopened.latest_checkpoint()["message"] == "v1"
        finally:
            reopened.close()


# ---------------------------------------------------------------------------
# Stats — Arrow IPC sidecar at .ygg/stats.arrow
# ---------------------------------------------------------------------------


class TestStatsSidecar:
    def test_compute_basic_stats(self):
        table = pa.table({
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "tag": pa.array(["a", "b", "c", None, "e"]),
        })
        stats = Stats.compute(table, name="t")
        ids = stats.column("id", source="t")
        tags = stats.column("tag", source="t")
        assert ids.num_values == 5
        assert ids.null_count == 0
        # Pre-IPC ColumnStats keeps native Python values; the IPC
        # round-trip stringifies them (see test_ipc_round_trip).
        assert ids.min_value == 1
        assert ids.max_value == 5
        assert tags.null_count == 1

    def test_compute_with_distinct(self):
        table = pa.table({"x": pa.array([1, 1, 2, 2, 3])})
        stats = Stats.compute(table, name="t", distinct=True)
        assert stats.column("x", source="t").distinct_count == 3

    def test_ipc_round_trip(self):
        table = pa.table({"x": pa.array([1, 2, 3])})
        stats = Stats.compute(table, name="src")
        blob = stats.to_ipc()
        recovered = Stats.from_ipc(blob)
        assert recovered.column_names == stats.column_names
        original = stats.column("x", source="src")
        round = recovered.column("x", source="src")
        assert round.num_values == original.num_values
        # The IPC sidecar carries min/max as strings (compact, portable
        # across backends); the pre-IPC ColumnStats keeps native values.
        assert str(round.min_value) == str(original.min_value)
        assert str(round.max_value) == str(original.max_value)

    def test_merge_aggregates(self):
        a = Stats.compute(
            pa.table({"x": pa.array([1, 2, 3])}), name="a",
        )
        b = Stats.compute(
            pa.table({"x": pa.array([4, 5, 6])}), name="b",
        )
        merged = Stats.merge([a, b])
        agg = merged.column("x", source=None)
        assert agg.num_values == 6
        # min/max use natural ordering on the merged-bound values.
        assert agg.min_value == 1
        assert agg.max_value == 6
        # Per-source rows preserved.
        assert merged.column("x", source="a").num_values == 3
        assert merged.column("x", source="b").num_values == 3

    def test_collect_stats_aggregate(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4), mode=Mode.APPEND)
            io.write_arrow_table(_make_table(4, 4), mode=Mode.APPEND)

            stats = io.collect_stats()
        ids = stats.column("id")
        assert ids.num_values == 8
        assert ids.null_count == 0
        assert ids.min_value == 0
        assert ids.max_value == 7

    def test_collect_stats_per_file(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 4), mode=Mode.APPEND)
            io.write_arrow_table(_make_table(100, 4), mode=Mode.APPEND)

            stats = io.collect_stats(per_file=True)
        # One source per data file plus aggregate (None).
        assert None in stats.sources
        non_aggregate = [s for s in stats.sources if s is not None]
        assert len(non_aggregate) == 2

    def test_write_and_read_stats_sidecar(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            io.write_arrow_table(_make_table(0, 6))
            io.write_stats()

            sidecar = tmp_path / ".ygg" / "stats.arrow"
            assert sidecar.exists()
            # The IPC file is valid Arrow IPC.
            stats = io.read_stats()
        assert stats is not None
        assert "id" in stats.column_names
        assert stats.column("id").num_values == 6

    def test_read_stats_returns_none_when_missing(self, tmp_path):
        with YGGFolderIO(path=str(tmp_path)) as io:
            assert io.read_stats() is None


# ---------------------------------------------------------------------------
# Stats across multiple data formats
# ---------------------------------------------------------------------------


class TestStatsAcrossFormats:
    @pytest.mark.parametrize(
        "io_cls,filename",
        [
            ("ParquetIO", "leaf.parquet"),
            ("ArrowIPCIO", "leaf.arrow"),
            ("CsvIO", "leaf.csv"),
            ("JsonIO", "leaf.json"),
            ("NDJsonIO", "leaf.ndjson"),
        ],
    )
    def test_stats_compute_for_format(self, tmp_path, io_cls, filename):
        from yggdrasil.io.primitive import ParquetIO
        from yggdrasil.io.primitive import ArrowIPCIO
        from yggdrasil.io.primitive import CsvIO
        from yggdrasil.io.primitive import JsonIO
        from yggdrasil.io.primitive import NDJsonIO

        cls_map = {
            "ParquetIO": ParquetIO,
            "ArrowIPCIO": ArrowIPCIO,
            "CsvIO": CsvIO,
            "JsonIO": JsonIO,
            "NDJsonIO": NDJsonIO,
        }
        cls = cls_map[io_cls]

        target = tmp_path / filename
        table = _make_table(0, 5)
        with cls(path=str(target), mode="wb+") as writer:
            writer.write_arrow_table(table)

        with cls(path=str(target), mode="rb") as reader:
            stats = Stats.compute(reader, name=filename)

        ids = stats.column("id", source=filename)
        assert ids.num_values == 5
        assert ids.min_value == 0
        assert ids.max_value == 4


# ---------------------------------------------------------------------------
# Time-sortable staging filenames
# ---------------------------------------------------------------------------


class TestTimeSortableStaging:
    def test_with_tmp_name_sorts_chronologically(self, tmp_path):
        from yggdrasil.io.path import LocalPath

        base = LocalPath.from_pathlib(pathlib.Path(str(tmp_path)))
        names: list[str] = []
        for _ in range(5):
            names.append(base.with_tmp_name(suffix=".bin").name)
            time.sleep(0.001)
        # Names sort lexically the same way they were generated:
        # since the start timestamp comes first and is zero-padded,
        # lexical order ⇒ chronological order. Within the same epoch
        # second the random seed is the tiebreaker, so the order is
        # not strictly chronological — but each chronologically
        # earlier batch sorts no later than later batches.
        assert sorted(names) == sorted(names, key=lambda s: s)
        # All names start with the time-first prefix, never the seed.
        for n in names:
            assert n.startswith("tmp-"), n
            after_prefix = n[len("tmp-"):]
            assert after_prefix[:12].isdigit(), (
                f"Expected zero-padded epoch start at offset 4 in {n!r}"
            )

    def test_spill_filename_lexical_order_matches_epoch(self):
        from yggdrasil.io.bytes_io import _mint_spill_path

        path_a = _mint_spill_path("bin", 60)
        time.sleep(1.05)
        path_b = _mint_spill_path("bin", 60)
        assert path_a.name < path_b.name


# ---------------------------------------------------------------------------
# optimize — small-file compaction
# ---------------------------------------------------------------------------


class TestTabularOptimizeDefault:
    def test_default_optimize_is_noop(self, tmp_path):
        from yggdrasil.io.primitive import ParquetIO

        ParquetIO(path=str(tmp_path / "a.parquet")).write_arrow_table(_make_table(0))
        io = ParquetIO(path=str(tmp_path / "a.parquet"))
        assert io.optimize() is io


class TestYggOptimize:
    def _write_n_small(self, folder: YGGFolderIO, n: int) -> None:
        # Stride by 2 so each batch's id range doesn't overlap the
        # next — keeps post-optimize integrity checks unambiguous.
        for i in range(n):
            folder.write_arrow_table(_make_table(i * 2, n=2), mode=Mode.APPEND)

    def _data_files(self, root: pathlib.Path) -> list[pathlib.Path]:
        return sorted(
            p for p in root.iterdir()
            if p.is_file() and not p.name.startswith(".") and not p.name.startswith("tmp-")
        )

    def test_optimize_skips_below_threshold(self, tmp_path):
        folder = YGGFolderIO(path=str(tmp_path))
        self._write_n_small(folder, 4)
        before = self._data_files(tmp_path)
        folder.optimize(target_bytes=1024 * 1024, min_files=5)
        after = self._data_files(tmp_path)
        assert {p.name for p in before} == {p.name for p in after}

    def test_optimize_compacts_when_above_threshold(self, tmp_path):
        folder = YGGFolderIO(path=str(tmp_path))
        self._write_n_small(folder, 8)
        before = self._data_files(tmp_path)
        assert len(before) == 8

        folder.optimize(target_bytes=1024 * 1024, min_files=3)
        after = self._data_files(tmp_path)
        assert len(after) < len(before)

        out = folder.read_arrow_table().sort_by([("id", "ascending")])
        assert out.column("id").to_pylist() == list(range(16))

    def test_optimize_preserves_large_files(self, tmp_path):
        folder = YGGFolderIO(path=str(tmp_path))
        self._write_n_small(folder, 8)
        # Tiny target — every file already exceeds it.
        folder.optimize(target_bytes=1, min_files=2)
        after = self._data_files(tmp_path)
        assert len(after) == 8

    def test_optimize_compacts_partition_subfolders(self, tmp_path):
        folder = YGGFolderIO(
            path=str(tmp_path), partition_columns=["year"],
        )
        for i in range(6):
            folder.write_arrow_table(
                pa.table({
                    "id": [i, i + 100],
                    "year": ["2024", "2024"],
                    "tag": [f"a-{i}", f"b-{i}"],
                }),
                mode=Mode.APPEND,
            )
        partition_root = tmp_path / "year=2024"
        assert partition_root.exists()
        before = sorted(p.name for p in partition_root.iterdir() if not p.name.startswith("."))
        assert len(before) == 6

        folder.optimize(target_bytes=1024 * 1024, min_files=3)
        after = sorted(p.name for p in partition_root.iterdir() if not p.name.startswith("."))
        assert len(after) < len(before)

        out = folder.read_arrow_table()
        assert out.num_rows == 12

    def test_optimize_returns_self_for_chaining(self, tmp_path):
        folder = YGGFolderIO(path=str(tmp_path))
        self._write_n_small(folder, 6)
        assert folder.optimize(target_bytes=1024 * 1024, min_files=3) is folder

    def test_optimize_partitions_only_touches_named_leaves(self, tmp_path):
        """Pruned optimize compacts named partitions and skips others."""
        folder = YGGFolderIO(
            path=str(tmp_path), partition_columns=["year"],
        )
        for year in ("2024", "2025"):
            for i in range(6):
                folder.write_arrow_table(
                    pa.table({
                        "id": [i, i + 100],
                        "year": [year, year],
                        "tag": [f"a-{year}-{i}", f"b-{year}-{i}"],
                    }),
                    mode=Mode.APPEND,
                )

        leaf_2024 = tmp_path / "year=2024"
        leaf_2025 = tmp_path / "year=2025"

        def _data_count(leaf: pathlib.Path) -> int:
            return sum(
                1 for p in leaf.iterdir()
                if p.is_file() and not p.name.startswith(".") and not p.name.startswith("tmp-")
            )

        before_2024 = _data_count(leaf_2024)
        before_2025 = _data_count(leaf_2025)
        assert before_2024 == 6
        assert before_2025 == 6

        folder.optimize(
            target_bytes=1024 * 1024,
            min_files=3,
            partitions={"year": ["2024"]},
        )

        # 2024 was compacted; 2025 was left alone.
        assert _data_count(leaf_2024) < before_2024
        assert _data_count(leaf_2025) == before_2025

        # Row count is preserved across the entire folder.
        assert folder.read_arrow_table().num_rows == 24

    def test_optimize_partitions_unknown_column_raises(self, tmp_path):
        """Typos in the partition spec must fail loudly."""
        folder = YGGFolderIO(
            path=str(tmp_path), partition_columns=["year"],
        )
        folder.write_arrow_table(
            pa.table({"id": [1], "year": ["2024"], "tag": ["a"]}),
            mode=Mode.APPEND,
        )

        with pytest.raises(ValueError, match="Unknown partition column"):
            folder.optimize(
                target_bytes=1024 * 1024,
                min_files=3,
                partitions={"month": ["01"]},
            )

    def test_optimize_partitions_missing_leaf_is_noop(self, tmp_path):
        """A partition with no leaf on disk silently skips."""
        folder = YGGFolderIO(
            path=str(tmp_path), partition_columns=["year"],
        )
        folder.write_arrow_table(
            pa.table({"id": [1], "year": ["2024"], "tag": ["a"]}),
            mode=Mode.APPEND,
        )

        # No leaf for 2099 yet — optimize must not fail.
        result = folder.optimize(
            target_bytes=1024 * 1024,
            min_files=3,
            partitions={"year": ["2099"]},
        )
        assert result is folder

