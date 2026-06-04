"""Concurrent Delta log reads — _read_cached_many fans independent GETs out.

Delta replay reads many small, independent files (commit JSONs, checkpoint
sidecars). On an object store each ``open()`` is a high-latency round-trip, so
they're fetched concurrently rather than serially. These tests pin that down
without touching a real object store.
"""
from __future__ import annotations

import io
import threading
from contextlib import contextmanager

from yggdrasil.io.delta import log as dlog
from yggdrasil.io.delta.log import DeltaLog, LogSegment


class FakePath:
    """Minimal Path stand-in: ``full_path()`` + a context-manager ``open()``."""

    def __init__(self, name, data=b"", *, missing=False, barrier=None, on_open=None):
        self._name = name
        self._data = data
        self._missing = missing
        self._barrier = barrier
        self._on_open = on_open

    def full_path(self):
        return self._name

    @contextmanager
    def open(self, mode="rb"):
        if self._on_open is not None:
            self._on_open(self._name)
        if self._barrier is not None:
            # If reads were serial, only one party ever arrives and this times
            # out — so a clean return proves the fan-out actually overlapped.
            self._barrier.wait(timeout=5)
        if self._missing:
            raise FileNotFoundError(self._name)
        yield io.BytesIO(self._data)


def _clear_cache():
    dlog._content_cache.clear()


def test_read_many_preserves_order():
    _clear_cache()
    paths = [FakePath(f"/k{i}", f"d{i}".encode()) for i in range(5)]
    assert dlog._read_cached_many(paths) == [b"d0", b"d1", b"d2", b"d3", b"d4"]


def test_read_many_missing_maps_to_none():
    _clear_cache()
    paths = [FakePath("/a", b"A"), FakePath("/b", missing=True), FakePath("/c", b"C")]
    assert dlog._read_cached_many(paths) == [b"A", None, b"C"]


def test_read_many_fetches_concurrently():
    _clear_cache()
    n = 4
    barrier = threading.Barrier(n)  # all n opens must be in flight at once
    paths = [FakePath(f"/p{i}", f"v{i}".encode(), barrier=barrier) for i in range(n)]
    # Would raise BrokenBarrierError on timeout if the reads were serialised.
    assert dlog._read_cached_many(paths) == [b"v0", b"v1", b"v2", b"v3"]


def test_read_many_uses_cache_and_skips_open():
    _clear_cache()
    dlog._content_cache["/cached"] = b"CACHED"
    opened: list[str] = []
    paths = [
        FakePath("/cached", b"x", on_open=opened.append),   # must NOT be opened
        FakePath("/fresh", b"FRESH", on_open=opened.append),
    ]
    assert dlog._read_cached_many(paths) == [b"CACHED", b"FRESH"]
    assert opened == ["/fresh"]
    # Fresh small read is now cached too.
    assert dlog._content_cache.get("/fresh") == b"FRESH"


def test_read_many_does_not_cache_oversized():
    _clear_cache()
    big = b"x" * (dlog._CONTENT_CACHE_MAX_BYTES + 1)
    assert dlog._read_cached_many([FakePath("/big", big)]) == [big]
    assert dlog._content_cache.get("/big") is None


def test_replay_raw_yields_commits_in_order():
    _clear_cache()
    c0 = FakePath("/0.json", b'{"add": {"path": "a"}}\n{"add": {"path": "b"}}')
    c1 = FakePath("/1.json", b'{"remove": {"path": "a"}}')
    seg = LogSegment(version=1, checkpoint_version=-1,
                     checkpoint_files=(), commit_files=(c0, c1))
    log = DeltaLog.__new__(DeltaLog)  # replay_raw only reads the segment
    out = list(log.replay_raw(seg))
    assert out == [
        {"add": {"path": "a"}},
        {"add": {"path": "b"}},
        {"remove": {"path": "a"}},
    ]
