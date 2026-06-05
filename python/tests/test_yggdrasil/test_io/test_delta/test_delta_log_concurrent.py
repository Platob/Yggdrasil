"""Delta log byte cache + concurrent reads (yggdrasil.io.delta._cache).

Remote immutable log files go through a two-tier cache (small RAM LRU backed by
a local-disk store under ~/.cache); local tables and the mutable
``_last_checkpoint`` pointer read straight through (``cache=False``). Misses are
fetched concurrently. These tests isolate the disk tier to a temp dir and reset
the RAM tier per test.
"""
from __future__ import annotations

import io
import threading
from contextlib import contextmanager

import pytest

from yggdrasil.io.delta import _cache
from yggdrasil.io.delta.log import DeltaLog, LogSegment


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Per-test: fresh RAM LRU + a temp disk dir so we never touch ~/.cache."""
    monkeypatch.setattr(_cache, "_DISK_DIR", tmp_path / "delta-log")
    monkeypatch.setattr(_cache, "_ram", _cache._ByteLRU(_cache._RAM_MAX_BYTES,
                                                        _cache._RAM_ITEM_MAX_BYTES))
    yield


class FakePath:
    """Path stand-in: ``full_path()`` + a counting context-manager ``open()``."""

    def __init__(self, name, data=b"", *, missing=False, barrier=None):
        self._name = name
        self._data = data
        self._missing = missing
        self._barrier = barrier
        self.opens = 0

    def full_path(self):
        return self._name

    @contextmanager
    def open(self, mode="rb"):
        self.opens += 1
        if self._barrier is not None:
            # Serial reads would never gather all parties → timeout.
            self._barrier.wait(timeout=5)
        if self._missing:
            raise FileNotFoundError(self._name)
        yield io.BytesIO(self._data)


# -- RAM LRU ---------------------------------------------------------------

def test_byte_lru_evicts_oldest_over_budget():
    lru = _cache._ByteLRU(max_bytes=300, item_max=300)
    lru.put("a", b"x" * 100)
    lru.put("b", b"y" * 100)
    lru.put("c", b"z" * 100)      # full at 300
    lru.put("d", b"w" * 100)      # over 300 → evict LRU ("a")
    assert lru.get("a") is None
    assert lru.get("d") == b"w" * 100
    assert lru.get("c") == b"z" * 100


def test_byte_lru_skips_items_over_item_cap():
    lru = _cache._ByteLRU(max_bytes=1000, item_max=100)
    lru.put("big", b"x" * 101)    # too big for the RAM tier
    assert lru.get("big") is None


def test_byte_lru_get_marks_recently_used():
    lru = _cache._ByteLRU(max_bytes=300, item_max=300)
    lru.put("a", b"x" * 100)
    lru.put("b", b"y" * 100)
    lru.put("c", b"z" * 100)
    lru.get("a")                  # touch a → b is now LRU
    lru.put("d", b"w" * 100)      # evicts b, not a
    assert lru.get("a") is not None
    assert lru.get("b") is None


# -- two-tier read (cache=True) --------------------------------------------

def test_read_one_hits_ram_second_time():
    p = FakePath("/k", b"DATA")
    assert _cache.read_one(p, cache=True) == b"DATA"
    assert _cache.read_one(p, cache=True) == b"DATA"
    assert p.opens == 1           # second read served from RAM


def test_read_one_survives_ram_clear_via_disk(monkeypatch):
    p = FakePath("/k", b"DATA")
    assert _cache.read_one(p, cache=True) == b"DATA"   # populates RAM + disk
    # Simulate a fresh process: wipe RAM, keep disk.
    monkeypatch.setattr(_cache, "_ram", _cache._ByteLRU(_cache._RAM_MAX_BYTES,
                                                        _cache._RAM_ITEM_MAX_BYTES))
    assert _cache.read_one(p, cache=True) == b"DATA"
    assert p.opens == 1           # served from disk, no second source read


def test_cache_false_reads_through_without_caching():
    p = FakePath("/local", b"L")
    assert _cache.read_one(p, cache=False) == b"L"
    assert _cache.read_one(p, cache=False) == b"L"
    assert p.opens == 2           # read each time; nothing cached
    assert not (_cache._DISK_DIR.exists() and any(_cache._DISK_DIR.iterdir()))


def test_disk_disabled_when_ttl_zero(monkeypatch):
    monkeypatch.setattr(_cache, "_DISK_TTL", 0.0)
    _cache.read_one(FakePath("/k", b"D"), cache=True)
    assert not _cache._DISK_DIR.exists()      # nothing written to disk


# -- read_many (concurrent) ------------------------------------------------

def test_read_many_preserves_order():
    paths = [FakePath(f"/k{i}", f"d{i}".encode()) for i in range(5)]
    assert _cache.read_many(paths, cache=True) == [b"d0", b"d1", b"d2", b"d3", b"d4"]


def test_read_many_empty():
    assert _cache.read_many([], cache=True) == []


def test_read_many_missing_maps_to_none():
    paths = [FakePath("/a", b"A"), FakePath("/b", missing=True), FakePath("/c", b"C")]
    assert _cache.read_many(paths, cache=True) == [b"A", None, b"C"]


def test_read_many_fetches_misses_concurrently():
    n = 4
    barrier = threading.Barrier(n)
    paths = [FakePath(f"/p{i}", f"v{i}".encode(), barrier=barrier) for i in range(n)]
    # Would raise BrokenBarrierError on timeout if the misses were serialised.
    assert _cache.read_many(paths, cache=True) == [b"v0", b"v1", b"v2", b"v3"]


def test_read_many_cache_false_does_not_cache():
    paths = [FakePath(f"/h{i}", f"d{i}".encode()) for i in range(3)]
    _cache.read_many(paths, cache=False)
    _cache.read_many(paths, cache=False)
    assert all(p.opens == 2 for p in paths)   # no caching → re-read each time


# -- replay still correct through the cache --------------------------------

def test_replay_raw_yields_commits_in_order():
    c0 = FakePath("/0.json", b'{"add": {"path": "a"}}\n{"add": {"path": "b"}}')
    c1 = FakePath("/1.json", b'{"remove": {"path": "a"}}')
    seg = LogSegment(version=1, checkpoint_version=-1,
                     checkpoint_files=(), commit_files=(c0, c1))
    log = DeltaLog.__new__(DeltaLog)   # replay_raw only needs the segment + _remote
    log._remote = False                # read straight through (no caching)
    out = list(log.replay_raw(seg))
    assert out == [
        {"add": {"path": "a"}},
        {"add": {"path": "b"}},
        {"remove": {"path": "a"}},
    ]
