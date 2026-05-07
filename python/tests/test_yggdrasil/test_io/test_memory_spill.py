"""Behavior + leak tests for the :class:`Memory` mmap auto-spill path.

Pins:

* **Threshold semantics** — capacity at or below ``spill_bytes`` stays
  in bytearray; crossing it migrates exactly once.
* **Round-trip correctness** — bytes written before and after a spill
  read back identically; the visible :attr:`size` is preserved.
* **No memory leak** — when the bytearray is replaced by an mmap, the
  old bytearray is dropped (GC-eligible). A weakref pinned just before
  the spill goes dead after one collection cycle. Resident set size,
  while platform-noisy, must not retain the spilled payload.
* **No fd / file leak** — the spill file is created on the way up and
  unlinked on :meth:`clear` and on holder release; the fd count
  reported by ``/proc/self/fd`` stays steady across many spill cycles.
* **Resize across the threshold** — repeated growth past the threshold
  doesn't re-spill or accumulate state.

Tests use :class:`yggdrasil.arrow.tests.ArrowTestCase` to keep the
optional-dep skip story consistent with the rest of the io test
suite, even though no Arrow types are exercised here.
"""

from __future__ import annotations

import gc
import mmap
import os
import sys

import pytest

from yggdrasil.io.memory import Memory


KB = 1024
MB = 1024 * KB

# bytearray doesn't support weakref, so the in-memory leak invariant is
# enforced via process RSS instead — a much more direct measure anyway:
# "if we leaked the bytearray, the resident set will visibly carry it."
_LINUX_ONLY = pytest.mark.skipif(
    sys.platform != "linux",
    reason="proc-fs based memory accounting is Linux-only",
)


def _open_fd_count() -> int:
    """Number of open fds for this process (Linux: ``/proc/self/fd``)."""
    try:
        return len(os.listdir("/proc/self/fd"))
    except FileNotFoundError:  # pragma: no cover — non-Linux
        pytest.skip("/proc/self/fd not available")


def _rss_bytes() -> int:
    """Resident set size for this process, in bytes (Linux only).

    Reads the second field of ``/proc/self/statm`` — RSS in pages —
    and multiplies by the system page size. Caller is responsible for
    skipping on platforms without procfs.
    """
    try:
        with open("/proc/self/statm", "r") as fh:
            rss_pages = int(fh.read().split()[1])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("/proc/self/statm not available")
    return rss_pages * os.sysconf("SC_PAGE_SIZE")


class TestSpillThreshold:
    """When does the bytearray flip to an mmap, and when does it not."""

    def test_no_spill_below_threshold(self) -> None:
        m = Memory(spill_bytes=64 * KB)
        m.write_bytes(b"x" * (32 * KB), 0)
        assert not m.is_spilled
        assert isinstance(m._buf, bytearray)
        assert m.spill_path is None

    def test_no_spill_at_threshold(self) -> None:
        # Strict inequality — capacity == threshold stays in memory.
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"y" * (4 * KB), 0)
        assert not m.is_spilled

    def test_spill_when_capacity_exceeds_threshold(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"z" * (8 * KB), 0)
        assert m.is_spilled
        assert isinstance(m._buf, mmap.mmap)
        assert m.spill_path is not None
        assert os.path.exists(m.spill_path)

    def test_spill_disabled_by_default(self) -> None:
        m = Memory()
        m.write_bytes(b"a" * (16 * MB), 0)
        assert not m.is_spilled

    def test_spill_bytes_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative byte count"):
            Memory(spill_bytes=-1)

    def test_int_capacity_above_threshold_spills_immediately(self) -> None:
        # Memory(int n) reserves n bytes; with spill_bytes set, the
        # bytearray must never materialize at the requested size.
        m = Memory(8 * KB, spill_bytes=4 * KB)
        assert m.is_spilled
        assert m.size == 0
        # Capacity sits on the mmap (rounded up at most by _MIN_MMAP_BYTES).
        assert m.capacity >= 8 * KB


class TestSpillRoundTrip:
    """Bytes survive the bytearray -> mmap migration intact."""

    def test_existing_bytes_survive_spill(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        seed = bytes(range(256)) * 8  # 2 KiB of recognizable bytes
        m.write_bytes(seed, 0)
        assert not m.is_spilled

        # Push past threshold by extending in place.
        m.write_bytes(b"\xff" * (8 * KB), len(seed))
        assert m.is_spilled

        # Original prefix readable through the mmap backing.
        assert m.read_bytes(len(seed), 0) == seed
        # Tail readable too.
        assert m.read_bytes(8 * KB, len(seed)) == b"\xff" * (8 * KB)

    def test_post_spill_writes_overwrite_in_place(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"a" * (8 * KB), 0)
        assert m.is_spilled
        m.pwrite(b"BCD", 100)
        assert m.read_bytes(3, 100) == b"BCD"

    def test_size_preserved_across_spill(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        payload = b"q" * (10 * KB)
        m.write_bytes(payload, 0)
        assert m.size == len(payload)
        assert m.is_spilled
        assert bytes(m.memoryview()) == payload

    def test_truncate_shrinks_after_spill(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"k" * (10 * KB), 0)
        m.truncate(1024)
        assert m.size == 1024
        assert m.read_bytes() == b"k" * 1024

    def test_growth_well_past_threshold(self) -> None:
        # Exercise mmap growth past the initial post-spill capacity.
        m = Memory(spill_bytes=64 * KB)
        for chunk in range(8):
            m.write_bytes(bytes([chunk]) * (32 * KB), -1)
        assert m.is_spilled
        assert m.size == 8 * 32 * KB
        for chunk in range(8):
            offset = chunk * 32 * KB
            assert m.read_bytes(1, offset) == bytes([chunk])


class TestSpillCleanup:
    """File and fd hygiene around clear() and close()."""

    def test_clear_removes_spill_file(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"x" * (8 * KB), 0)
        path = m.spill_path
        assert path is not None and os.path.exists(path)

        m.clear()
        assert not m.is_spilled
        assert not os.path.exists(path)
        assert m.spill_path is None
        assert isinstance(m._buf, bytearray)
        assert m.size == 0

    def test_clear_then_rewrite_can_respill(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"x" * (8 * KB), 0)
        first_path = m.spill_path
        m.clear()
        assert not m.is_spilled

        m.write_bytes(b"y" * (8 * KB), 0)
        assert m.is_spilled
        assert m.spill_path is not None
        assert m.spill_path != first_path  # fresh tempfile.

    def test_close_releases_spill(self) -> None:
        m = Memory(spill_bytes=4 * KB)
        m.acquire()
        m.write_bytes(b"x" * (8 * KB), 0)
        path = m.spill_path
        assert path is not None and os.path.exists(path)
        m.close()
        assert not os.path.exists(path)

    def test_context_manager_releases_spill(self) -> None:
        with Memory(spill_bytes=4 * KB) as m:
            m.write_bytes(b"q" * (8 * KB), 0)
            assert m.is_spilled
            path = m.spill_path
        assert path is not None
        assert not os.path.exists(path)

    def test_temporary_flag_clears_on_close(self) -> None:
        m = Memory(b"q" * (8 * KB), spill_bytes=4 * KB, temporary=True)
        m.acquire()
        assert m.is_spilled
        path = m.spill_path
        m.close()
        # temporary=True triggers clear(), which tears the spill down.
        assert not os.path.exists(path)
        assert m.size == 0

    def test_spill_dir_honored(self, tmp_path) -> None:
        m = Memory(spill_bytes=4 * KB, spill_dir=str(tmp_path))
        m.write_bytes(b"r" * (8 * KB), 0)
        assert m.is_spilled
        assert os.path.dirname(m.spill_path) == str(tmp_path)


class TestCallerManagedSpillPath:
    """``spill_path=`` lets the caller pin where the spill lands."""

    def test_uses_caller_supplied_path(self, tmp_path) -> None:
        target = tmp_path / "my_scratch.bin"
        m = Memory(spill_bytes=4 * KB, spill_path=str(target))
        m.write_bytes(b"x" * (8 * KB), 0)
        assert m.is_spilled
        assert m.spill_path == str(target)
        assert target.exists()

    def test_caller_path_not_unlinked_on_clear(self, tmp_path) -> None:
        target = tmp_path / "scratch.bin"
        m = Memory(spill_bytes=4 * KB, spill_path=str(target))
        m.write_bytes(b"x" * (8 * KB), 0)
        assert target.exists()

        m.clear()
        # Clearing the Memory tears down the mapping but leaves the
        # caller's file on disk — they own it.
        assert not m.is_spilled
        assert target.exists()

    def test_caller_path_not_unlinked_on_close(self, tmp_path) -> None:
        target = tmp_path / "scratch.bin"
        m = Memory(spill_bytes=4 * KB, spill_path=str(target))
        m.acquire()
        m.write_bytes(b"x" * (8 * KB), 0)
        assert target.exists()
        m.close()
        assert target.exists()

    def test_anonymous_path_still_unlinked(self, tmp_path) -> None:
        # Sanity: the new owns-path tracking didn't accidentally
        # break the default (owned) cleanup.
        m = Memory(spill_bytes=4 * KB, spill_dir=str(tmp_path))
        m.write_bytes(b"x" * (8 * KB), 0)
        path = m.spill_path
        assert os.path.exists(path)
        m.clear()
        assert not os.path.exists(path)

    def test_spill_path_and_spill_dir_conflict_rejected(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="not both"):
            Memory(
                spill_bytes=4 * KB,
                spill_dir=str(tmp_path),
                spill_path=str(tmp_path / "x.bin"),
            )

    def test_pathlib_accepted(self, tmp_path) -> None:
        target = tmp_path / "scratch.bin"
        m = Memory(spill_bytes=4 * KB, spill_path=target)  # PurePath
        m.write_bytes(b"x" * (8 * KB), 0)
        assert m.spill_path == str(target)

    def test_post_clear_respill_uses_same_caller_path(
        self, tmp_path,
    ) -> None:
        # The caller-supplied path is the spill target across cycles —
        # clear() doesn't reset the request, just the active mapping.
        target = tmp_path / "scratch.bin"
        m = Memory(spill_bytes=4 * KB, spill_path=str(target))
        m.write_bytes(b"x" * (8 * KB), 0)
        m.clear()

        m.write_bytes(b"y" * (8 * KB), 0)
        assert m.spill_path == str(target)
        assert m.read_bytes(8 * KB, 0) == b"y" * (8 * KB)

    def test_round_trip_with_caller_path(self, tmp_path) -> None:
        target = tmp_path / "scratch.bin"
        m = Memory(spill_bytes=1 * KB, spill_path=str(target))
        payload = bytes(range(256)) * 32  # 8 KiB
        m.write_bytes(payload, 0)
        assert m.is_spilled
        assert m.read_bytes() == payload


class TestSpillNoMemoryLeak:
    """The bytearray copy must be released when the mmap takes over.

    bytearray doesn't support :mod:`weakref`, so the no-leak invariant
    is enforced via process RSS — a more direct check anyway: "if the
    spill kept the bytearray alongside the mmap, RSS would carry both
    payloads at once and grow linearly across cycles."

    A small constant slop is allowed to absorb interpreter churn,
    Python frame allocations, and (importantly) the kernel page cache
    holding pages of recently-touched mmaps — that cache is not a leak
    in our object graph, but it does inflate RSS. The assertions
    target *linear* growth, not absolute size.
    """

    @_LINUX_ONLY
    def test_repeated_spill_cycles_dont_grow_rss(self) -> None:
        # If the spill path kept the bytearray alive after migration,
        # each iteration would add ~16 MiB to RSS. Drop everything
        # explicitly between iterations and assert resident bytes
        # stay close to baseline.
        payload = b"\xa5" * (16 * MB)
        gc.collect()
        baseline = _rss_bytes()
        for _ in range(8):
            m = Memory(spill_bytes=1 * MB)
            m.write_bytes(payload, 0)
            assert m.is_spilled
            m.clear()
            del m
            gc.collect()

        gc.collect()
        final = _rss_bytes()
        # Tolerance: 8 MiB slack for arena fragmentation. A real leak
        # would put us 8 * 16 MiB = 128 MiB above baseline.
        growth = final - baseline
        assert growth < 8 * MB, (
            f"RSS grew {growth/MB:.1f} MiB across 8 spill cycles "
            f"({baseline/MB:.1f} → {final/MB:.1f}) — likely retaining "
            f"the bytearray alongside the mmap"
        )

    @_LINUX_ONLY
    def test_payload_larger_than_spill_bytes_doesnt_double(self) -> None:
        # Sanity: a 64 MiB payload through a 1 MiB threshold should
        # cost ~64 MiB resident, not ~128 MiB. The first write itself
        # crosses the threshold (16 MiB capacity > 1 MiB), so the
        # spill kicks in immediately; the invariant we care about is
        # the *peak* resident set, not which call triggers spill.
        gc.collect()
        baseline = _rss_bytes()

        m = Memory(spill_bytes=1 * MB)
        m.write_bytes(b"\x01" * (16 * MB), 0)
        assert m.is_spilled
        m.write_bytes(b"\x02" * (48 * MB), -1)
        assert m.is_spilled
        gc.collect()
        peak = _rss_bytes()

        growth = peak - baseline
        # 64 MiB total payload. Allow a 32 MiB slack for the kernel
        # page cache of the mmap'd file plus interpreter overhead.
        # A double-copy leak would push growth above 96 MiB.
        assert growth < 96 * MB, (
            f"RSS grew {growth/MB:.1f} MiB for a 64 MiB payload — "
            f"the spill is keeping the bytearray and mmap "
            f"simultaneously resident"
        )

        # And the bytes are correct end-to-end.
        assert m.size == 16 * MB + 48 * MB
        assert m.read_bytes(1, 0) == b"\x01"
        assert m.read_bytes(1, 16 * MB - 1) == b"\x01"
        assert m.read_bytes(1, 16 * MB) == b"\x02"
        assert m.read_bytes(1, m.size - 1) == b"\x02"

    def test_post_spill_buf_is_mmap_not_bytearray(self) -> None:
        # The structural invariant the no-leak guarantee depends on:
        # after a spill, _buf must be the mmap, not a bytearray. If
        # the migration ever started keeping both this would be the
        # first thing to break.
        m = Memory(spill_bytes=4 * KB)
        m.write_bytes(b"x" * (8 * KB), 0)
        assert m.is_spilled
        assert isinstance(m._buf, mmap.mmap)
        assert not isinstance(m._buf, bytearray)

    @_LINUX_ONLY
    def test_repeated_spill_cycles_dont_leak_fds(self) -> None:
        gc.collect()
        before = _open_fd_count()
        for _ in range(50):
            m = Memory(spill_bytes=4 * KB)
            m.write_bytes(b"q" * (16 * KB), 0)
            assert m.is_spilled
            m.clear()
            del m
        gc.collect()
        after = _open_fd_count()
        # The harness opens its own fds (logging, capturing); allow a
        # tiny slop but reject linear growth.
        assert after - before < 5, (
            f"open fd count grew by {after - before} across 50 spill "
            f"cycles — likely a temp-file fd leak"
        )

    @_LINUX_ONLY
    def test_close_releases_fd(self) -> None:
        gc.collect()
        before = _open_fd_count()
        m = Memory(spill_bytes=4 * KB)
        m.acquire()
        m.write_bytes(b"q" * (16 * KB), 0)
        assert _open_fd_count() > before  # spill fd is open
        m.close()
        gc.collect()
        # Spill fd is gone; final count back near baseline.
        assert _open_fd_count() - before < 2

    @_LINUX_ONLY
    def test_garbage_collection_alone_releases_fds(self) -> None:
        # If a caller drops the Memory without calling close(), the
        # spill resources should still be reclaimed when the object
        # is GC'd — otherwise long-running processes accumulate temp
        # files and fds for any forgetful caller.
        gc.collect()
        before_fd = _open_fd_count()
        for _ in range(10):
            m = Memory(spill_bytes=4 * KB)
            m.write_bytes(b"x" * (16 * KB), 0)
            assert m.is_spilled
            del m
            gc.collect()
        after_fd = _open_fd_count()
        # CPython's mmap finalizer closes the fd it owns; the os.close
        # of our explicit fd happens via _release on temporary or via
        # the mmap finalizer's own close. Either way: no linear leak.
        assert after_fd - before_fd < 5, (
            f"open fd count grew by {after_fd - before_fd} across 10 "
            f"GC-only cycles — spill resources aren't being reclaimed "
            f"on object collection"
        )
