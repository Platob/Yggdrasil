"""Tests for path-lock + stale-spill cleanup added in
``yggdrasil.io.buffer._concurrency``.

Locks have shared/exclusive semantics. The lock filename carries an
access-intent token (``.r.lock`` / ``.w.lock`` / ``.rw.lock``) so
external tooling can identify what kind of lock is held; readers
and writers therefore use *different* lock files by design — within
each kind the lock is correct (multiple readers coexist, multiple
writers serialise), but reader-vs-writer is not coordinated across
modes.

Wait semantics are driven by a :class:`WaitingConfig` (``wait=N``
seconds, ``wait=None`` waits forever, ``wait=0`` raises on
contention). The stale-spill cleaner inspects the TTL encoded by
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

from yggdrasil.dataclasses.waiting import WaitingConfig
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.buffer._concurrency import (
    AbstractLock,
    AtomicLock,
    FileLock,
    _build_owner_payload,
    _host_from_owner_url,
    _parse_owner_payload,
    cleanup_stale_spill_files,
    compute_identifier_url,
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
        assert lock_suffix_for(read=False, write=True) == "w"

    def test_read_only_suffix(self):
        assert lock_suffix_for(read=True, write=False) == "r"

    def test_read_write_suffix(self):
        assert lock_suffix_for(read=True, write=True) == "rw"

    def test_neither_defaults_to_write(self):
        assert lock_suffix_for(read=False, write=False) == "w"

    def test_lock_path_for_uses_suffix(self, tmp_path):
        target = str(tmp_path / "data.bin")
        assert (
            lock_path_for(target, read=False, write=True)
            == str(tmp_path / ".data.bin.w.lock")
        )
        assert (
            lock_path_for(target, read=True, write=False)
            == str(tmp_path / ".data.bin.r.lock")
        )
        assert (
            lock_path_for(target, read=True, write=True)
            == str(tmp_path / ".data.bin.rw.lock")
        )


# ---------------------------------------------------------------------------
# Compute identifier URL — owner-info attribution for cross-host locks
# ---------------------------------------------------------------------------


_DATABRICKS_ENV_KEYS = (
    "DATABRICKS_RUNTIME_VERSION",
    "DB_CLUSTER_ID",
    "DATABRICKS_CLUSTER_ID",
    "DATABRICKS_JOB_ID",
    "DATABRICKS_JOB_RUN_ID",
    "DATABRICKS_RUN_ID",
    "DATABRICKS_TASK_KEY",
    "DATABRICKS_TASK_RUN_ID",
    "DB_NOTEBOOK_PATH",
    "DB_NOTEBOOK_ID",
)


@pytest.fixture
def _scrub_pipeline_env(monkeypatch):
    """Strip every Databricks-related env var so each test gets a
    deterministic baseline; tests that opt into pipeline detection
    set the vars they need explicitly."""
    for key in _DATABRICKS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


class TestComputeIdentifierURL:
    def test_default_is_host_pid_url(self, _scrub_pipeline_env):
        url = compute_identifier_url()
        assert url.startswith("host://")
        # Trailing path segment is the PID, query params absent.
        host_part, _, pid_part = url.partition("host://")[2].partition("/")
        assert host_part  # non-empty hostname
        assert pid_part.isdigit()
        assert int(pid_part) == os.getpid()

    def test_databricks_runtime_triggers_databricks_scheme(
        self, _scrub_pipeline_env,
    ):
        _scrub_pipeline_env.setenv("DATABRICKS_RUNTIME_VERSION", "14.3.x-scala2.12")
        url = compute_identifier_url()
        # Cluster id absent → falls back to literal "cluster" segment
        # but the scheme is still databricks://, signalling pipeline.
        assert url.startswith("databricks://")
        assert "host=" in url
        assert f"pid={os.getpid()}" in url

    def test_databricks_cluster_and_job_ids_make_it_into_url(
        self, _scrub_pipeline_env,
    ):
        _scrub_pipeline_env.setenv("DATABRICKS_RUNTIME_VERSION", "14.3.x")
        _scrub_pipeline_env.setenv("DB_CLUSTER_ID", "0501-abcdef-foo123")
        _scrub_pipeline_env.setenv("DATABRICKS_JOB_ID", "987")
        _scrub_pipeline_env.setenv("DATABRICKS_JOB_RUN_ID", "654321")
        _scrub_pipeline_env.setenv("DATABRICKS_TASK_KEY", "etl_step_1")
        url = compute_identifier_url()
        assert url.startswith("databricks://0501-abcdef-foo123/")
        assert "job=987" in url
        assert "run=654321" in url
        assert "task=etl_step_1" in url

    def test_databricks_notebook_tags_attach(self, _scrub_pipeline_env):
        _scrub_pipeline_env.setenv("DB_CLUSTER_ID", "0501-clusterX")
        _scrub_pipeline_env.setenv("DB_NOTEBOOK_PATH", "/Repos/team/notebook.py")
        _scrub_pipeline_env.setenv("DB_NOTEBOOK_ID", "12345")
        url = compute_identifier_url()
        # Path may be percent-encoded — match on the encoded segment.
        assert "notebook=" in url
        assert "notebook_id=12345" in url

    def test_url_pid_refreshes_per_call(self, _scrub_pipeline_env):
        """PID must be re-read per call so a forked child gets a fresh
        URL — caching at module load would mis-attribute children."""
        # Two consecutive calls in the same process produce the same
        # URL (pid stable); verify pid is in there for the round-trip.
        first = compute_identifier_url()
        second = compute_identifier_url()
        assert first == second
        assert str(os.getpid()) in first

    def test_owner_url_env_override_takes_precedence(self, _scrub_pipeline_env):
        """``$YGG_OWNER_URL`` is the propagation hook for shared-id
        coordination across processes (e.g. Spark driver → executors).
        It must override even Databricks pipeline detection."""
        _scrub_pipeline_env.setenv("DATABRICKS_RUNTIME_VERSION", "14.3.x")
        _scrub_pipeline_env.setenv("DB_CLUSTER_ID", "would-be-cluster")
        _scrub_pipeline_env.setenv(
            "YGG_OWNER_URL", "databricks://shared-cluster/1?host=driver&job=42",
        )
        url = compute_identifier_url()
        assert url == "databricks://shared-cluster/1?host=driver&job=42"

    def test_owner_url_env_override_sanitises_whitespace(self, _scrub_pipeline_env):
        """Lock payloads are whitespace-delimited — a pasted-in URL
        with a stray newline must not corrupt the third field."""
        _scrub_pipeline_env.setenv(
            "YGG_OWNER_URL", "  databricks://x/1\n?host=h  ",
        )
        url = compute_identifier_url()
        # Whitespace stripped at the edges, internal newline replaced.
        assert "\n" not in url
        assert " " not in url
        assert url.startswith("databricks://x/1")


class TestOwnerInfoPayload:
    def test_payload_round_trip_with_compute_url(self, _scrub_pipeline_env):
        payload = _build_owner_payload()
        pid, epoch, compute_url = _parse_owner_payload(payload)
        assert pid == os.getpid()
        assert epoch is not None and epoch > 0
        assert compute_url is not None and compute_url.startswith("host://")

    def test_parse_legacy_pid_only(self):
        pid, epoch, url = _parse_owner_payload(b"4321\n")
        assert pid == 4321
        assert epoch is None
        assert url is None

    def test_parse_legacy_pid_and_epoch(self):
        pid, epoch, url = _parse_owner_payload(b"4321 1700000000\n")
        assert pid == 4321
        assert epoch == 1700000000.0
        assert url is None

    def test_parse_garbage_returns_none(self):
        pid, epoch, url = _parse_owner_payload(b"\x00\x01garbage\xff")
        # First field "garbage" is not int-parseable → pid is None.
        assert pid is None
        assert epoch is None
        assert url is None

    def test_host_extracted_from_host_url(self):
        assert _host_from_owner_url("host://laptop-foo/12345") == "laptop-foo"

    def test_host_extracted_from_databricks_url(self):
        url = "databricks://cluster-x/1?host=worker-7&pid=1&job=99"
        assert _host_from_owner_url(url) == "worker-7"

    def test_host_extraction_handles_missing_host(self):
        # databricks URL without host= qs param → caller must skip.
        assert _host_from_owner_url("databricks://cluster-x/1?job=99") is None

    def test_host_extraction_handles_garbage(self):
        assert _host_from_owner_url(None) is None
        assert _host_from_owner_url("") is None
        assert _host_from_owner_url("not a url at all !!") is None


class TestFileLockOwnerInfo:
    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_owner_info_written_on_acquire(self, tmp_path):
        path = str(tmp_path / "data.lock")
        with FileLock(path):
            with open(path, "rb") as fh:
                head = fh.read(512)
        pid, epoch, compute_url = _parse_owner_payload(head)
        assert pid == os.getpid()
        assert epoch is not None and epoch > 0
        assert compute_url is not None
        # Default fallback URL form for non-Databricks env.
        assert compute_url.startswith("host://") or compute_url.startswith("databricks://")

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_stale_break_skipped_for_foreign_host(self, tmp_path):
        """A sidecar whose recorded host differs from ours must not be
        evaluated with ``os.kill(pid, 0)`` — local liveness probes
        say nothing about a remote holder. Direct call avoids the
        flock contention path which would overwrite the payload."""
        path = str(tmp_path / "data.lock")
        with open(path, "wb") as fh:
            fh.write(b"999999 1700000000 host://other-host/999999\n")
        FileLock(path)._try_break_stale_lock()
        # Foreign sidecar must survive — we can't conclude its holder
        # is dead from the local process table.
        assert os.path.exists(path)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_stale_break_unlinks_local_dead_holder(self, tmp_path):
        """Same hostname + a PID that lives in this process's table
        but is *us*: stale-break correctly skips (we're not dead).
        Same hostname + missing PID: the sidecar IS unlinked.
        """
        from yggdrasil.io.buffer._concurrency import _HOSTNAME

        path = str(tmp_path / "data.lock")
        # PID = our own → liveness check passes, sidecar persists.
        with open(path, "wb") as fh:
            fh.write(
                f"{os.getpid()} 1700000000 host://{_HOSTNAME}/{os.getpid()}\n"
                .encode("ascii")
            )
        FileLock(path)._try_break_stale_lock()
        assert os.path.exists(path)

        # Now overwrite with a definitely-dead PID we just reaped.
        proc = os.fork() if hasattr(os, "fork") else None
        if proc == 0:  # pragma: no cover — child path
            os._exit(0)
        if proc is None:
            pytest.skip("os.fork unavailable on this platform")
        pid_dead, _ = os.waitpid(proc, 0)
        with open(path, "wb") as fh:
            fh.write(
                f"{pid_dead} 1700000000 host://{_HOSTNAME}/{pid_dead}\n"
                .encode("ascii")
            )
        FileLock(path)._try_break_stale_lock()
        # Same host, dead PID → sidecar correctly unlinked.
        assert not os.path.exists(path)


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
    def test_wait_zero_raises_when_held_exclusive(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()
        try:
            b = FileLock(path, wait=0)
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_wait_seconds_blocks_then_succeeds_after_release(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()

        results: list[str] = []

        def waiter():
            with FileLock(
                path,
                wait=WaitingConfig(timeout=2.0, interval=0.01,
                                   backoff=1.0, max_interval=0.1),
            ):
                results.append("got-it")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        assert results == []
        a.release()
        t.join(timeout=2.0)
        assert results == ["got-it"]

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_wait_int_coerces_to_timeout(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()
        try:
            # ``wait=0.05`` → ~50ms timeout, then TimeoutError.
            b = FileLock(path, wait=0.05)
            t0 = time.monotonic()
            with pytest.raises(TimeoutError):
                b.acquire()
            assert time.monotonic() - t0 >= 0.04
        finally:
            a.release()

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
            sh = FileLock(path, shared=True, wait=0)
            with pytest.raises(TimeoutError):
                sh.acquire()
        finally:
            ex.release()

    def test_stale_probe_gate_first_iteration(self, tmp_path):
        """Iter 0 always probes — wait=0 callers must be able to break a
        dead-holder sidecar on their single attempt."""
        lock = FileLock(str(tmp_path / "x.lock"))
        assert lock._should_probe_stale(0) is True

    def test_stale_probe_gate_throttles_subsequent_iterations(self, tmp_path):
        """After iter 0 the gate combines an every-Nth-iteration count
        with a wall-clock floor; without backdating the timer no
        further probe should fire even at the count boundary."""
        lock = FileLock(str(tmp_path / "x.lock"))
        lock._should_probe_stale(0)  # primes the timer
        # Boundary iteration but the wall-clock floor blocks us.
        assert lock._should_probe_stale(_concurrency._STALE_PROBE_EVERY_N) is False
        # Non-boundary iterations short-circuit on the count alone.
        assert lock._should_probe_stale(1) is False
        assert lock._should_probe_stale(_concurrency._STALE_PROBE_EVERY_N - 1) is False
        # Once enough wall-clock passes, the boundary fires again.
        lock._last_stale_probe_at -= _concurrency._STALE_PROBE_MIN_INTERVAL_S * 2
        assert lock._should_probe_stale(_concurrency._STALE_PROBE_EVERY_N) is True

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_shared_blocks_exclusive_on_same_file(self, tmp_path):
        path = str(tmp_path / "x.lock")
        sh = FileLock(path, shared=True)
        sh.acquire()
        try:
            ex = FileLock(path, shared=False, wait=0)
            with pytest.raises(TimeoutError):
                ex.acquire()
        finally:
            sh.release()


# ---------------------------------------------------------------------------
# Path.lock surface
# ---------------------------------------------------------------------------


class TestPathLock:
    def _path(self, tmp_path, name="data.bin"):
        return LocalPath.from_pathlib(pathlib.Path(str(tmp_path / name)))

    def test_lock_path_uses_mode_suffix(self, tmp_path):
        p = self._path(tmp_path)
        assert p.lock_path(read=False, write=True).endswith(".w.lock")
        assert p.lock_path(read=True, write=False).endswith(".r.lock")
        assert p.lock_path(read=True, write=True).endswith(".rw.lock")

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_path_write_lock_blocks_concurrent_writer(self, tmp_path):
        p = self._path(tmp_path)
        a = p.lock(write=True)
        a.acquire()
        try:
            b = p.lock(write=True, wait=0)
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
            assert not any(
                str(name).endswith(".lock") for name in os.listdir(tmp_path)
            )

    def test_concurrent_write_creates_then_removes_lock_file(self, tmp_path):
        p = tmp_path / "out.bin"
        lock_target = str(tmp_path / ".out.bin.rw.lock")
        with BytesIO(path=str(p), mode="wb+", concurrent=True) as buf:
            buf.write(b"durable")
            assert os.path.exists(lock_target)
        assert not os.path.exists(lock_target)
        assert p.read_bytes() == b"durable"

    def test_concurrent_read_only_uses_r_suffix(self, tmp_path):
        p = tmp_path / "in.bin"
        p.write_bytes(b"existing")
        lock_target = str(tmp_path / ".in.bin.r.lock")
        with BytesIO(path=str(p), mode="rb", concurrent=True) as buf:
            assert buf.read() == b"existing"
            assert os.path.exists(lock_target)
        assert not os.path.exists(lock_target)

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_concurrent_writers_serialise(self, tmp_path):
        p = tmp_path / "race.bin"
        outer = FileLock(str(tmp_path / ".race.bin.rw.lock"))
        outer.acquire()
        try:
            with pytest.raises(TimeoutError):
                BytesIO(
                    path=str(p), mode="wb+",
                    concurrent=True, lock_wait=0.05,
                ).open()
        finally:
            outer.release()

        with BytesIO(
            path=str(p), mode="wb+",
            concurrent=True, lock_wait=1.0,
        ) as buf:
            buf.write(b"safe-write")
        assert p.read_bytes() == b"safe-write"

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_two_threads_writing_produce_one_winner(self, tmp_path):
        p = tmp_path / "race.bin"
        payload_a = b"A" * 4096
        payload_b = b"B" * 4096

        def writer(payload, hold_seconds):
            with BytesIO(
                path=str(p), mode="wb+",
                concurrent=True, lock_wait=5.0,
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
    """Drop a fake spill file with a controlled TTL into *directory*.

    Filename matches the time-sortable layout produced by
    ``_mint_spill_path``: ``tmp-{start}-{end}-{seed}.{ext}`` with
    zero-padded timestamps.
    """
    start = end_epoch - 1
    name = f"tmp-{start:012d}-{end_epoch:012d}-{seed}.{ext}"
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

_TABLE = pa.table({
    "id": pa.array([1, 2, 3], type=pa.int64()),
    "name": pa.array(["a", "b", "c"]),
})


def _expect_lock(target: str, *, suffix: str) -> str:
    parent = os.path.dirname(target)
    base = os.path.basename(target)
    return os.path.join(parent, f".{base}.{suffix}.lock")


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
        lock_path = _expect_lock(str(target), suffix="rw")
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
        assert not any(
            n.startswith(".") and n.endswith(".lock")
            for n in os.listdir(tmp_path)
        )

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_concurrent_writers_serialise_via_lock(self, tmp_path):
        target = tmp_path / "race.parquet"
        outer = FileLock(_expect_lock(str(target), suffix="rw"))
        outer.acquire()
        try:
            with pytest.raises(TimeoutError):
                ParquetIO(
                    path=str(target),
                    mode="wb+",
                    concurrent=True,
                    lock_wait=0.05,
                ).open()
        finally:
            outer.release()
        with ParquetIO(
            path=str(target),
            mode="wb+",
            concurrent=True,
            lock_wait=1.0,
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
                lock_wait=5.0,
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
        parent = str(folder.parent)
        base = folder.name or "_"
        return os.path.join(parent, f".{base}.rw.lock")

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

        a = FolderIO(path=str(folder), concurrent=True)
        a.open()
        try:
            b = FolderIO(path=str(folder), concurrent=True, lock_wait=0.05)
            with pytest.raises(TimeoutError):
                b.open()
        finally:
            a.close()

        with FolderIO(
            path=str(folder),
            concurrent=True,
            lock_wait=1.0,
        ) as c:
            assert c._path_lock is not None

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_threaded_folder_writers_do_not_overlap(self, tmp_path):
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
                lock_wait=5.0,
            ) as io:
                with peak_lock:
                    active += 1
                    if active > peak:
                        peak = active
                time.sleep(0.05)
                with peak_lock:
                    active -= 1
                _ = io

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert peak == 1, f"FolderIO concurrent=True peak overlap was {peak}"


# ---------------------------------------------------------------------------
# WaitingConfig integration — backoff actually applies
# ---------------------------------------------------------------------------


class TestWaitingConfigBackoff:
    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_waitingconfig_dict_is_accepted(self, tmp_path):
        """A dict ``wait`` arg is normalised through ``WaitingConfig.from_``."""
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()
        try:
            b = FileLock(path, wait={"timeout": 0.05, "interval": 0.005})
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_waitingconfig_instance_accepted(self, tmp_path):
        path = str(tmp_path / "x.lock")
        a = FileLock(path)
        a.acquire()
        try:
            cfg = WaitingConfig(timeout=0.05, interval=0.005,
                                backoff=1.5, max_interval=0.05, retries=2)
            b = FileLock(path, wait=cfg)
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    @pytest.mark.skipif(_IS_WINDOWS, reason="POSIX flock semantics")
    def test_wait_blocks_until_holder_releases(self, tmp_path):
        """When the holder releases mid-wait, the blocked acquirer
        finishes its current backoff sleep and then succeeds — no
        spurious TimeoutError."""
        path = str(tmp_path / "race.lock")
        a = FileLock(path)
        a.acquire()

        observed: list[float] = []

        def waiter():
            t0 = time.monotonic()
            with FileLock(path, wait=2.0):
                observed.append(time.monotonic() - t0)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.15)
        a.release()
        t.join(timeout=2.0)
        assert len(observed) == 1
        # The waiter should have been blocked for at least 0.15s.
        assert observed[0] >= 0.10


# ---------------------------------------------------------------------------
# AtomicLock — works on any Path that exposes 'xb' mode
# ---------------------------------------------------------------------------


class TestAtomicLock:
    def _sidecar(self, tmp_path, name=".x.rw.lock"):
        return LocalPath.from_pathlib(pathlib.Path(str(tmp_path / name)))

    def test_implements_lock_interface(self, tmp_path):
        lock = AtomicLock(self._sidecar(tmp_path))
        assert isinstance(lock, AbstractLock)
        assert lock.held is False

    def test_acquire_creates_sidecar(self, tmp_path):
        sidecar = self._sidecar(tmp_path)
        lock = AtomicLock(sidecar)
        lock.acquire()
        try:
            assert lock.held
            assert pathlib.Path(sidecar.full_path()).exists()
        finally:
            lock.release()
        assert not pathlib.Path(sidecar.full_path()).exists()

    def test_context_manager(self, tmp_path):
        sidecar = self._sidecar(tmp_path)
        with AtomicLock(sidecar):
            assert pathlib.Path(sidecar.full_path()).exists()
        assert not pathlib.Path(sidecar.full_path()).exists()

    def test_idempotent_acquire(self, tmp_path):
        lock = AtomicLock(self._sidecar(tmp_path))
        lock.acquire()
        lock.acquire()
        assert lock.held
        lock.release()

    def test_idempotent_release(self, tmp_path):
        lock = AtomicLock(self._sidecar(tmp_path))
        lock.release()  # never held
        lock.acquire()
        lock.release()
        lock.release()  # double-release

    def test_wait_zero_raises_when_held(self, tmp_path):
        sidecar = self._sidecar(tmp_path)
        a = AtomicLock(sidecar)
        a.acquire()
        try:
            b = AtomicLock(sidecar, wait=0)
            with pytest.raises(TimeoutError):
                b.acquire()
        finally:
            a.release()

    def test_wait_seconds_blocks_then_succeeds(self, tmp_path):
        sidecar = self._sidecar(tmp_path)
        a = AtomicLock(sidecar)
        a.acquire()

        results: list[str] = []

        def waiter():
            with AtomicLock(
                sidecar,
                wait=WaitingConfig(timeout=2.0, interval=0.01,
                                   backoff=1.0, max_interval=0.1),
            ):
                results.append("got-it")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        assert results == []
        a.release()
        t.join(timeout=2.0)
        assert results == ["got-it"]

    def test_two_atomic_locks_serialise(self, tmp_path):
        sidecar = self._sidecar(tmp_path)
        active = 0
        peak = 0
        peak_lock = threading.Lock()

        def worker():
            nonlocal active, peak
            with AtomicLock(sidecar, wait=5.0):
                with peak_lock:
                    active += 1
                    if active > peak:
                        peak = active
                time.sleep(0.05)
                with peak_lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
        assert peak == 1

    def test_stale_lock_is_force_unlinked(self, tmp_path):
        """A lingering sidecar past ``stale_after_seconds`` is
        force-unlinked by the next acquirer — covers the
        crashed-remote-writer recovery path."""
        sidecar_pl = pathlib.Path(str(tmp_path / "stale.rw.lock"))
        # Drop a "stale" sidecar with an old timestamp embedded.
        old_ts = int(time.time()) - 999
        sidecar_pl.write_bytes(f"99999999 {old_ts}\n".encode("ascii"))

        sidecar = LocalPath.from_pathlib(sidecar_pl)
        lock = AtomicLock(sidecar, wait=0, stale_after_seconds=1)
        # The first acquire should detect staleness, unlink, and succeed.
        lock.acquire()
        try:
            assert lock.held
        finally:
            lock.release()


# ---------------------------------------------------------------------------
# Path.lock dispatch — local → FileLock, remote → AtomicLock
# ---------------------------------------------------------------------------


class TestPathLockDispatch:
    def test_local_path_returns_filelock(self, tmp_path):
        p = LocalPath.from_pathlib(pathlib.Path(str(tmp_path / "x.bin")))
        lock = p.lock(write=True)
        assert isinstance(lock, FileLock)

    def test_non_local_path_returns_atomic_lock(self, tmp_path, monkeypatch):
        # Spoof a non-local LocalPath. The dispatch sees ``is_local=False``
        # and routes to AtomicLock; the sidecar is still resolved
        # against the same backend, which is what we want.
        p = LocalPath.from_pathlib(pathlib.Path(str(tmp_path / "x.bin")))
        monkeypatch.setattr(type(p), "is_local", property(lambda self: False))
        lock = p.lock(write=True)
        assert isinstance(lock, AtomicLock)
        # The sidecar is itself a yggdrasil Path so AtomicLock can call
        # ``open_io("xb")`` / ``unlink`` / ``stat`` on it.
        assert hasattr(lock.path, "full_path")
        assert lock.path.full_path().endswith(".x.bin.w.lock")
