"""Concurrent Delta log reads — _read_many fans independent GETs out.

Delta replay reads many small, independent files (commit JSONs, checkpoint
sidecars). On an object store each ``open()`` is a high-latency round-trip, so
they're fetched concurrently rather than serially. There is deliberately no
byte cache here — the parsed Snapshot is cached a layer up in DeltaFolder.
"""
from __future__ import annotations

import io
import threading
from contextlib import contextmanager

from yggdrasil.io.delta import log as dlog
from yggdrasil.io.delta.log import DeltaLog, LogSegment


class FakePath:
    """Minimal Path stand-in: ``full_path()`` + a context-manager ``open()``."""

    def __init__(self, name, data=b"", *, missing=False, barrier=None):
        self._name = name
        self._data = data
        self._missing = missing
        self._barrier = barrier

    def full_path(self):
        return self._name

    @contextmanager
    def open(self, mode="rb"):
        if self._barrier is not None:
            # If reads were serial, only one party ever arrives and this times
            # out — so a clean return proves the fan-out actually overlapped.
            self._barrier.wait(timeout=5)
        if self._missing:
            raise FileNotFoundError(self._name)
        yield io.BytesIO(self._data)


def test_read_many_preserves_order():
    paths = [FakePath(f"/k{i}", f"d{i}".encode()) for i in range(5)]
    assert dlog._read_many(paths) == [b"d0", b"d1", b"d2", b"d3", b"d4"]


def test_read_many_empty():
    assert dlog._read_many([]) == []


def test_read_many_missing_maps_to_none():
    paths = [FakePath("/a", b"A"), FakePath("/b", missing=True), FakePath("/c", b"C")]
    assert dlog._read_many(paths) == [b"A", None, b"C"]


def test_read_many_fetches_concurrently():
    n = 4
    barrier = threading.Barrier(n)  # all n opens must be in flight at once
    paths = [FakePath(f"/p{i}", f"v{i}".encode(), barrier=barrier) for i in range(n)]
    # Would raise BrokenBarrierError on timeout if the reads were serialised.
    assert dlog._read_many(paths) == [b"v0", b"v1", b"v2", b"v3"]


def test_read_many_does_not_cache():
    # No module-level byte cache exists after the rollback — only the parsed
    # Snapshot (in DeltaFolder) is cached.
    assert not hasattr(dlog, "_content_cache")
    assert not hasattr(dlog, "_pointer_cache")


def test_replay_raw_yields_commits_in_order():
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
