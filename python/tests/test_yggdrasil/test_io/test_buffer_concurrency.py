"""Tests for path-lock + stale-spill cleanup added in
``yggdrasil.io.buffer._concurrency``.

Locks have shared/exclusive semantics. The lock filename carries an
access-intent suffix (``-r.lock`` / ``-w.lock`` / ``-rw.lock``) so
external tooling can identify what kind of lock is held; readers and
writers therefore use *different* lock files by design — within each
kind the lock is correct (multiple readers coexist, multiple writers
serialise), but reader-vs-writer is not coordinated across modes.

The stale-spill cleaner inspects the TTL encoded by
``_mint_spill_path`` and deletes anything past its expiry, leaving
fresh files and unrelated names alone.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time

import pyarrow as pa
import pytest

from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer._concurrency import (
    FileLock,
    cleanup_stale_spill_files,
    lock_path_for,
    lock_suffix_for,
    maybe_cleanup_stale_spill_files,
)
import yggdrasil.io.buffer._concurrency as _concurrency
from yggdrasil.io.buffer.nested.folder_io import FolderIO
from yggdrasil.io.buffer.primitive.arrow_ipc_io import ArrowIPCIO
from yggdrasil.io.buffer.primitive.csv_io import CsvIO
from yggdrasil.io.buffer.primitive.json_io import JsonIO
from yggdrasil.io.buffer.primitive.ndjson_io import NDJsonIO
from yggdrasil.io.buffer.primitive.parquet_io import ParquetIO
from yggdrasil.io.fs import LocalPath


_IS_WINDOWS = sys.platform.startswith("win")


# ---------------------------------------------------------------------------
# Suffix + sidecar paths
# ---------------------------------------------------------------------------


class TestLockSuffix:
    def test_write_only_suffix(self):
        assert lock_suffix_for(read=False, write=True) == "-w"

    def test_read_only_suffix(self):
        assert lock_suffix_for(read=True, write=False) == "-r"

    def test_read_write_suffix(self):
        assert lock_suffix_for(read=True, write=True) == "-rw"

    def test_neither_defaults_to_write(self):
        assert lock_suffix_for(read=False, write=False) == "-w"

    def test_lock_path_for_uses_suffix(self, tmp_path):
        target = str(tmp_path / "data.bin")
        assert (
            lock_path_for(target, read=False, write=True)
            == str(tmp_path / ".data.bin-w.lock")
        )
        assert (
            lock_path_for(target, read=True, write=False)
            == str(tmp_path / ".data.bin-r.lock")
        )
        assert (
            lock_path_for(target, read=True, write=True)
            == str(tmp_path / ".data.bin-rw.lock")
        )


# ---------------------------------------------------------------------------
# FileLock primitives
# ---------------------------------------------------------------------------


class TestFileLock:
    def test_acquire_release_round_trip(self, tmp_path):
        lock = FileLock(str(tmp_path / "x.lock"))
        lock.acquire()
        assert lock.held
        lock.release()
        assert not lock.held
        assert not (tmp_path / "x.lock").exists()

    def test_context_manager(self, tmp_path):
        with FileLock(str(tmp_path / "x.lock")) as lock:
            assert lock.held
        assert not (tmp_path / "x.lock").exists()

    def test_idempotent_acquire(self, tmp_path):
        lock = FileLock(str(tmp_path / "x.lock"))
        lock.acquire()
        lock.acquire()  # no-op when already held
        assert lock.held
        lock.release()

    def test_idempotent_release(self, tmp_path):
        lock = FileLock(str(tmp_path / "x.lock"))
        lock.release()  # never acquired — must not raise
        lock.acquire()
        lock.release()
        lock.release()  # double-release — must not raise

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_zero_timeout_raises_when_held_exclusive(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()
        try:
            b = FileLock(path, timeout=0)
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_blocked_acquirer_succeeds_after_release(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()

        results: list[str] = []

        def waiter():
            with FileLock(path, timeout=2.0, poll=0.01):
                results.append("got-it")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        assert results == []
        a.release()
        t.join(timeout=2.0)
        assert results == ["got-it"]

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_shared_locks_coexist_on_same_file(self, tmp_path):
        path = str(tmp_path / "shared.lock")
        a = FileLock(path, shared=True)
        b = FileLock(path, shared=True)
        a.acquire()
        b.acquire()
        try:
            assert a.held and b.held
        finally:
            a.release()
            b.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_exclusive_blocks_shared_on_same_file(self, tmp_path):
        path = str(tmp_path / "x.lock")
        ex = FileLock(path, shared=False)
        ex.acquire()
        try:
            sh = FileLock(path, shared=True, timeout=0)
            with pytest.raises(TimeoutError):
                sh.acquire()
        finally:
            ex.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_shared_blocks_exclusive_on_same_file(self, tmp_path):
        path = str(tmp_path / "x.lock")
        sh = FileLock(path, shared=True)
        sh.acquire()
        try:
            ex = FileLock(path, shared=False, timeout=0)
            with pytest.raises(TimeoutError):
                ex.acquire()
        finally:
            sh.release()


# ---------------------------------------------------------------------------
# Path.lock surface
# ---------------------------------------------------------------------------


class TestPathLock:
    def _path(self, tmp_path, name="data.bin"):
        import pathlib
        return LocalPath.from_pathlib(pathlib.Path(str(tmp_path / name)))

    def test_lock_path_uses_mode_suffix(self, tmp_path):
        p = self._path(tmp_path)
        assert p.lock_path(read=False, write=True).endswith("-w.lock")
        assert p.lock_path(read=True, write=False).endswith("-r.lock")
        assert p.lock_path(read=True, write=True).endswith("-rw.lock")

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_path_write_lock_blocks_concurrent_writer(self, tmp_path):
        p = self._path(tmp_path)
        a = p.lock(write=True)
        a.acquire()
        try:
            b = p.lock(write=True, timeout=0)
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_path_read_lock_allows_concurrent_readers(self, tmp_path):
        p = self._path(tmp_path)
        a = p.lock(read=True, write=False)
        b = p.lock(read=True, write=False)
        a.acquire()
        b.acquire()
        try:
            assert a.held and b.held
        finally:
            a.release()
            b.release()


# ---------------------------------------------------------------------------
# BytesIO integration — opt-in via concurrent=True
# ---------------------------------------------------------------------------


class TestBytesIOPathLock:
    def test_concurrent_off_by_default(self, tmp_path):
        p = tmp_path / "out.bin"
        with BytesIO(path=str(p), mode="wb+") as buf:
            buf.write(b"x")
            # No lock file should be created — concurrent defaults to False.
            assert not any(
                str(name).endswith(".lock") for name in os.listdir(tmp_path)
            )

    def test_concurrent_write_creates_then_removes_lock_file(self, tmp_path):
        p = tmp_path / "out.bin"
        lock_target = str(tmp_path / ".out.bin-rw.lock")
        with BytesIO(path=str(p), mode="wb+", concurrent=True) as buf:
            buf.write(b"durable")
            # Lock file is present while the buffer is open.
            assert os.path.exists(lock_target)
        # Lock file gone after close.
        assert not os.path.exists(lock_target)
        assert p.read_bytes() == b"durable"

    def test_concurrent_read_only_uses_r_suffix(self, tmp_path):
        p = tmp_path / "in.bin"
        p.write_bytes(b"existing")
        lock_target = str(tmp_path / ".in.bin-r.lock")
        with BytesIO(path=str(p), mode="rb", concurrent=True) as buf:
            assert buf.read() == b"existing"
            # Read-only mode → ``-r.lock`` (shared).
            assert os.path.exists(lock_target)
        assert not os.path.exists(lock_target)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_concurrent_writers_serialise(self, tmp_path):
        p = tmp_path / "race.bin"
        # Hold the rw-lock from outside; a write attempt with a short
        # timeout must surface TimeoutError.
        outer = FileLock(str(tmp_path / ".race.bin-rw.lock"))
        outer.acquire()
        try:
            with pytest.raises(TimeoutError):
                BytesIO(
                    path=str(p), mode="wb+",
                    concurrent=True, lock_timeout=0.05,
                ).open()
        finally:
            outer.release()

        # After the holder releases, a fresh writer must succeed.
        with BytesIO(
            path=str(p), mode="wb+",
            concurrent=True, lock_timeout=1.0,
        ) as buf:
            buf.write(b"safe-write")
        assert p.read_bytes() == b"safe-write"

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_two_threads_writing_produce_one_winner(self, tmp_path):
        """Two threads racing the same target with concurrent=True
        interleave their writes through the lock; final content is
        always exactly one of the two payloads, never a mix."""
        p = tmp_path / "race.bin"
        payload_a = b"A" * 4096
        payload_b = b"B" * 4096

        def writer(payload, hold_seconds):
            with BytesIO(
                path=str(p), mode="wb+",
                concurrent=True, lock_timeout=5.0,
            ) as buf:
                buf.write(payload)
                time.sleep(hold_seconds)

        ta = threading.Thread(target=writer, args=(payload_a, 0.1))
        tb = threading.Thread(target=writer, args=(payload_b, 0.0))
        ta.start()
        time.sleep(0.02)
        tb.start()
        ta.join(timeout=5.0)
        tb.join(timeout=5.0)

        final = p.read_bytes()
        assert final in (payload_a, payload_b)

    def test_owned_spill_does_not_create_lock_file(self, tmp_path):
        # Self-owned spill (spilled-to-tempdir) shouldn't lock —
        # the path is unique by construction, even with concurrent=True.
        buf = BytesIO(b"Q" * 256, spill_bytes=8, concurrent=True)
        try:
            assert buf.spilled
            assert buf._owns_spill_path
            assert buf._path_lock is None
        finally:
            buf.close()


# ---------------------------------------------------------------------------
# In-memory threading.RLock guard
# ---------------------------------------------------------------------------


class TestMemoryThreadLock:
    def test_thread_lock_present_when_concurrent(self):
        b = BytesIO(b"x", concurrent=True)
        assert b._thread_lock is not None

    def test_thread_lock_absent_when_off(self):
        b = BytesIO(b"x")
        assert b._thread_lock is None

    def test_concurrent_writes_do_not_lose_bytes(self):
        """Many threads writing into an in-memory concurrent buffer
        produce a final size equal to the sum of every write — no
        torn writes, no lost bytes."""
        buf = BytesIO(b"", concurrent=True)
        chunk = b"x" * 64
        n_threads = 16
        writes_per_thread = 32

        def writer():
            for _ in range(writes_per_thread):
                buf.write(chunk)

        threads = [threading.Thread(target=writer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        expected = len(chunk) * n_threads * writes_per_thread
        assert buf.size == expected

    def test_concurrent_reader_writer_does_not_corrupt_offsets(self):
        """A reader and writer hammering the same buffer can't observe
        a torn cursor: every read returns either the empty tail or a
        slice of valid bytes, never an OOB or partial-byte panic."""
        buf = BytesIO(b"", concurrent=True)
        seen_errors: list[Exception] = []

        def writer():
            try:
                for i in range(500):
                    buf.write(f"{i:08d}".encode())
            except Exception as e:
                seen_errors.append(e)

        def reader():
            try:
                for _ in range(500):
                    buf.seek(0)
                    _ = buf.read(-1)
            except Exception as e:
                seen_errors.append(e)

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert seen_errors == []


# ---------------------------------------------------------------------------
# Stale spill-temp cleanup
# ---------------------------------------------------------------------------


def _make_spill_file(directory, *, end_epoch, ext="bin", seed="deadbeef0badf00d"):
    """Drop a fake spill file with a controlled TTL into *directory*."""
    start = end_epoch - 1
    name = f"tmp-{seed}-{start}-{end_epoch}.{ext}"
    full = os.path.join(directory, name)
    with open(full, "wb") as fh:
        fh.write(b"")
    return full


class TestCleanupStaleSpillFiles:
    def test_unlinks_expired_files(self, tmp_path):
        old1 = _make_spill_file(str(tmp_path), end_epoch=int(time.time()) - 100)
        old2 = _make_spill_file(
            str(tmp_path), end_epoch=int(time.time()) - 50, seed="aaaabbbbccccdddd",
        )
        assert os.path.exists(old1) and os.path.exists(old2)
        removed = cleanup_stale_spill_files(str(tmp_path))
        assert removed == 2
        assert not os.path.exists(old1)
        assert not os.path.exists(old2)

    def test_leaves_fresh_files_alone(self, tmp_path):
        fresh = _make_spill_file(str(tmp_path), end_epoch=int(time.time()) + 600)
        removed = cleanup_stale_spill_files(str(tmp_path))
        assert removed == 0
        assert os.path.exists(fresh)

    def test_ignores_files_not_matching_spill_pattern(self, tmp_path):
        unrelated = tmp_path / "user-data.parquet"
        unrelated.write_bytes(b"x")
        also_unrelated = tmp_path / "tmp-no-timestamps.bin"
        also_unrelated.write_bytes(b"x")
        old = _make_spill_file(str(tmp_path), end_epoch=int(time.time()) - 1)

        removed = cleanup_stale_spill_files(str(tmp_path))
        assert removed == 1
        assert unrelated.exists()
        assert also_unrelated.exists()
        assert not os.path.exists(old)

    def test_grace_period_keeps_recently_expired(self, tmp_path):
        recent_old = _make_spill_file(
            str(tmp_path), end_epoch=int(time.time()) - 10,
        )
        ancient = _make_spill_file(
            str(tmp_path),
            end_epoch=int(time.time()) - 1000,
            seed="0123456789abcdef",
        )
        removed = cleanup_stale_spill_files(str(tmp_path), grace_seconds=60)
        assert removed == 1
        assert os.path.exists(recent_old)
        assert not os.path.exists(ancient)

    def test_concurrent_cleaners_do_not_double_unlink(self, tmp_path):
        for i in range(5):
            _make_spill_file(
                str(tmp_path),
                end_epoch=int(time.time()) - 100 - i,
                seed=f"{i:016x}",
            )

        results: list[int] = []

        def run():
            results.append(cleanup_stale_spill_files(str(tmp_path)))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert sum(results) == 5
        leftovers = [n for n in os.listdir(str(tmp_path)) if n.startswith("tmp-")]
        assert leftovers == []

    def test_maybe_cleanup_throttles_in_process(self, tmp_path):
        path = _make_spill_file(str(tmp_path), end_epoch=int(time.time()) - 5)
        removed_first = maybe_cleanup_stale_spill_files(
            str(tmp_path), interval_s=0,
        )
        assert removed_first == 1
        assert not os.path.exists(path)

        path2 = _make_spill_file(
            str(tmp_path),
            end_epoch=int(time.time()) - 5,
            seed="ffffffffffffffff",
        )
        removed_second = maybe_cleanup_stale_spill_files(str(tmp_path))
        assert removed_second == 0
        assert os.path.exists(path2)

        removed_third = maybe_cleanup_stale_spill_files(
            str(tmp_path), interval_s=0,
        )
        assert removed_third == 1
        assert not os.path.exists(path2)


# ---------------------------------------------------------------------------
# Primitive IO integration — locking flows through BytesIO inheritance
# ---------------------------------------------------------------------------

# Each primitive leaf is a BytesIO subclass, so the ``concurrent``
# kwarg is inherited transparently. These tests exercise the full
# round-trip (write → read) under concurrent=True for the file
# formats we support, then verify the sidecar lock file appears for
# the right mode and disappears on close.

_TABLE = pa.table({"id": pa.array([1, 2, 3], type=pa.int64()),
                   "name": pa.array(["a", "b", "c"])})


def _expect_lock(target: str, *, suffix: str) -> str:
    parent = os.path.dirname(target)
    base = os.path.basename(target)
    return os.path.join(parent, f".{base}{suffix}.lock")


class TestPrimitiveIOConcurrency:

    @pytest.mark.parametrize(
        "io_cls,filename",
        [
            (ParquetIO, "data.parquet"),
            (ArrowIPCIO, "data.arrow"),
            (CsvIO, "data.csv"),
            (JsonIO, "data.json"),
            (NDJsonIO, "data.ndjson"),
        ],
    )
    def test_write_lock_sidecar_appears_during_write(
        self, tmp_path, io_cls, filename,
    ):
        target = tmp_path / filename
        lock_path = _expect_lock(str(target), suffix="-rw")
        with io_cls(path=str(target), mode="wb+", concurrent=True) as io:
            io.write_arrow_table(_TABLE)
            assert os.path.exists(lock_path)
        assert not os.path.exists(lock_path)
        assert target.exists() and target.stat().st_size > 0

    @pytest.mark.parametrize(
        "io_cls,filename",
        [
            (ParquetIO, "rt.parquet"),
            (ArrowIPCIO, "rt.arrow"),
            (CsvIO, "rt.csv"),
        ],
    )
    def test_concurrent_round_trip(self, tmp_path, io_cls, filename):
        target = tmp_path / filename
        with io_cls(path=str(target), mode="wb+", concurrent=True) as w:
            w.write_arrow_table(_TABLE)
        with io_cls(path=str(target), mode="rb", concurrent=True) as r:
            tbl = r.read_arrow_table()
        assert tbl.num_rows == _TABLE.num_rows
        assert tbl.column_names == _TABLE.column_names

    def test_no_lock_when_concurrent_false(self, tmp_path):
        target = tmp_path / "default.parquet"
        with ParquetIO(path=str(target), mode="wb+") as io:
            io.write_arrow_table(_TABLE)
        # No -*.lock files leftover.
        assert not any(
            n.startswith(".") and n.endswith(".lock")
            for n in os.listdir(tmp_path)
        )

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_concurrent_writers_serialise_via_lock(self, tmp_path):
        target = tmp_path / "race.parquet"
        # Hold the rw-lock externally; a concurrent ParquetIO writer
        # with a short timeout must surface TimeoutError.
        outer = FileLock(_expect_lock(str(target), suffix="-rw"))
        outer.acquire()
        try:
            with pytest.raises(TimeoutError):
                ParquetIO(
                    path=str(target),
                    mode="wb+",
                    concurrent=True,
                    lock_timeout=0.05,
                ).open()
        finally:
            outer.release()
        # After release, a fresh writer succeeds.
        with ParquetIO(
            path=str(target),
            mode="wb+",
            concurrent=True,
            lock_timeout=1.0,
        ) as io:
            io.write_arrow_table(_TABLE)
        assert target.exists()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_two_threads_write_parquet_one_winner(self, tmp_path):
        target = tmp_path / "thread-race.parquet"
        a = pa.table({"v": pa.array(["A"] * 10)})
        b = pa.table({"v": pa.array(["B"] * 10)})

        def write(payload, hold):
            with ParquetIO(
                path=str(target),
                mode="wb+",
                concurrent=True,
                lock_timeout=5.0,
            ) as io:
                io.write_arrow_table(payload)
                time.sleep(hold)

        ta = threading.Thread(target=write, args=(a, 0.1))
        tb = threading.Thread(target=write, args=(b, 0.0))
        ta.start()
        time.sleep(0.02)
        tb.start()
        ta.join(timeout=5.0)
        tb.join(timeout=5.0)

        # File holds exactly one of the two payloads (no torn parquet
        # footer, no mixed bytes).
        with ParquetIO(path=str(target), mode="rb") as r:
            final = r.read_arrow_table()
        assert final.num_rows == 10
        col = final["v"].to_pylist()
        assert col in (["A"] * 10, ["B"] * 10)


# ---------------------------------------------------------------------------
# Nested IO integration — folder-root lock
# ---------------------------------------------------------------------------


class TestNestedIOConcurrency:
    def _expect_folder_lock(self, folder: pathlib.Path) -> str:
        # FolderIO locks via Path.lock(read=True, write=True) which
        # produces ``<dir>/.<basename>-rw.lock``.
        parent = str(folder.parent)
        base = folder.name or "_"
        return os.path.join(parent, f".{base}-rw.lock")

    def test_concurrent_off_by_default_no_lock_file(self, tmp_path):
        folder = tmp_path / "store"
        folder.mkdir()
        with FolderIO(path=str(folder)) as io:
            assert io._path_lock is None
        assert not os.path.exists(self._expect_folder_lock(folder))

    def test_concurrent_acquires_and_releases_folder_lock(self, tmp_path):
        folder = tmp_path / "store"
        folder.mkdir()
        lock_path = self._expect_folder_lock(folder)
        with FolderIO(path=str(folder), concurrent=True) as io:
            assert io._path_lock is not None
            assert os.path.exists(lock_path)
        assert not os.path.exists(lock_path)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_two_concurrent_folder_handles_serialise(self, tmp_path):
        folder = tmp_path / "store"
        folder.mkdir()

        # First handle holds the rw-lock; the second must time out.
        a = FolderIO(path=str(folder), concurrent=True)
        a.open()
        try:
            b = FolderIO(path=str(folder), concurrent=True, lock_timeout=0.05)
            with pytest.raises(TimeoutError):
                b.open()
        finally:
            a.close()

        # Once a is closed, c can acquire.
        with FolderIO(
            path=str(folder),
            concurrent=True,
            lock_timeout=1.0,
        ) as c:
            assert c._path_lock is not None

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_threaded_folder_writers_do_not_overlap(self, tmp_path):
        """Two threaded FolderIO instances with concurrent=True against
        the same folder enter their critical sections sequentially,
        not in parallel — verified by tracking the high-water-mark of
        simultaneously-held locks."""
        folder = tmp_path / "store"
        folder.mkdir()

        active = 0
        peak = 0
        peak_lock = threading.Lock()

        def worker():
            nonlocal active, peak
            with FolderIO(
                path=str(folder),
                concurrent=True,
                lock_timeout=5.0,
            ) as io:
                with peak_lock:
                    active += 1
                    if active > peak:
                        peak = active
                # Hold for a moment so a parallel attempt would surface.
                time.sleep(0.05)
                with peak_lock:
                    active -= 1
                _ = io  # quiet unused-var

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert peak == 1, f"FolderIO concurrent=True peak overlap was {peak}"
