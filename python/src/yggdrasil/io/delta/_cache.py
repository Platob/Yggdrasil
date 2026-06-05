"""Two-tier byte cache for **immutable** Delta log files.

RAM is precious, so the in-memory tier is a tiny byte-bounded LRU (default 4 MB,
holding only the hottest small files — commit JSONs, manifests), backed by a
persistent local-disk store under ``~/.cache`` that holds the bulk (notably the
large checkpoint parquet) and survives process restarts, so repeated CLI runs /
job restarts don't re-GET the same files from S3.

Only immutable, version-addressed files go through here (commit JSONs, checkpoint
parquet/manifests, sidecars) — never the mutable ``_last_checkpoint`` pointer or
the directory listing (those stay listing-fresh). Local-filesystem sources are
read straight through (already on disk). All cache ops are best-effort: a miss or
an I/O error on the cache never breaks a read. The one staleness window is a
table dropped and *recreated at the same path* within the disk TTL.

Knobs (env): ``YGG_DELTA_RAM_CACHE_BYTES`` (default 4 MB),
``YGG_DELTA_CACHE_DIR`` (default ``$XDG_CACHE_HOME/yggdrasil/delta-log`` or
``~/.cache/...``), ``YGG_DELTA_DISK_CACHE_TTL`` (seconds, default 1 day; ``0``
disables disk), ``YGG_DELTA_DISK_CACHE_MAX_BYTES`` (default 1 GB).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import pathlib
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Iterable, List, Optional

if TYPE_CHECKING:
    from yggdrasil.path import Path

#: Cap on concurrent source GETs for cache misses.
_MAX_FETCH_WORKERS = 32

# -- RAM tier: keep it small; let disk hold the bulk. The per-item cap stops one
#    big blob (a checkpoint) from evicting many hot commit files.
_RAM_MAX_BYTES = int(os.environ.get("YGG_DELTA_RAM_CACHE_BYTES") or 4 * 1024 * 1024)
_RAM_ITEM_MAX_BYTES = max(1, _RAM_MAX_BYTES // 4)

# -- Disk tier under ~/.cache (XDG-aware), TTL- and size-bounded.
_DISK_DIR = pathlib.Path(
    os.environ.get("YGG_DELTA_CACHE_DIR")
    or (pathlib.Path(os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache"))
        / "yggdrasil" / "delta-log")
)
_DISK_TTL = float(os.environ.get("YGG_DELTA_DISK_CACHE_TTL") or 86400.0)
_DISK_MAX_BYTES = int(os.environ.get("YGG_DELTA_DISK_CACHE_MAX_BYTES") or 1024 * 1024 * 1024)


class _ByteLRU:
    """Thread-safe LRU bounded by total bytes (and a per-item byte cap)."""

    __slots__ = ("_max", "_item_max", "_d", "_bytes", "_lock")

    def __init__(self, max_bytes: int, item_max: int) -> None:
        self._max = max_bytes
        self._item_max = item_max
        self._d: "OrderedDict[str, bytes]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> "Optional[bytes]":
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
            return v

    def put(self, key: str, val: bytes) -> None:
        n = len(val)
        if n > self._item_max:          # too big for the RAM tier — disk only
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= len(old)
            self._d[key] = val
            self._bytes += n
            while self._bytes > self._max:
                _, evicted = self._d.popitem(last=False)
                self._bytes -= len(evicted)


_ram = _ByteLRU(_RAM_MAX_BYTES, _RAM_ITEM_MAX_BYTES)


def _key(path: "Path") -> str:
    fn = getattr(path, "full_path", None)
    full = fn() if callable(fn) else str(path)
    return hashlib.sha1(full.encode("utf-8")).hexdigest()


def _disk_get(key: str) -> "Optional[bytes]":
    if _DISK_TTL <= 0:
        return None
    f = _DISK_DIR / key
    try:
        if (time.time() - f.stat().st_mtime) > _DISK_TTL:
            f.unlink(missing_ok=True)
            return None
        return f.read_bytes()
    except OSError:
        return None


def _disk_put(key: str, raw: bytes) -> None:
    if _DISK_TTL <= 0:
        return
    try:
        _DISK_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _DISK_DIR / f".{key}.{os.getpid()}.tmp"
        tmp.write_bytes(raw)
        os.replace(tmp, _DISK_DIR / key)   # atomic publish (safe across procs)
        _maybe_prune()
    except OSError:
        pass


_puts_since_prune = 0
_prune_lock = threading.Lock()


def _maybe_prune() -> None:
    """Amortised disk cap: every ~64 puts, if the store is over the byte cap,
    delete oldest-by-mtime entries until under it. Never on the read path."""
    global _puts_since_prune
    with _prune_lock:
        _puts_since_prune += 1
        if _puts_since_prune < 64:
            return
        _puts_since_prune = 0
    try:
        entries = []
        total = 0
        for child in _DISK_DIR.iterdir():
            if child.name.endswith(".tmp"):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            entries.append((st.st_mtime, child, st.st_size))
            total += st.st_size
        if total <= _DISK_MAX_BYTES:
            return
        for _mtime, child, size in sorted(entries):   # oldest first
            if total <= _DISK_MAX_BYTES:
                break
            try:
                child.unlink()
                total -= size
            except OSError:
                pass
    except OSError:
        pass


def cached_read(path: "Path") -> bytes:
    """Two-tier read of an immutable log file; raises ``FileNotFoundError`` if
    the source is absent (so callers' existing handling still applies)."""
    if getattr(path, "is_local_path", False):
        with path.open("rb") as bio:           # already on local disk
            return bio.read()
    key = _key(path)
    hit = _ram.get(key)
    if hit is not None:
        return hit
    hit = _disk_get(key)
    if hit is not None:
        _ram.put(key, hit)
        return hit
    with path.open("rb") as bio:
        raw = bio.read()
    _disk_put(key, raw)
    _ram.put(key, raw)
    return raw


def cached_read_many(paths: "Iterable[Path]") -> "List[Optional[bytes]]":
    """Bytes for each path in order: RAM/disk hits resolved first, the remaining
    source misses fetched **concurrently**. Missing files map to ``None``."""
    paths = list(paths)
    if not paths:
        return []
    out: "List[Optional[bytes]]" = [None] * len(paths)
    # (index, path, key-or-None). key None ⇒ local source: read, don't cache.
    misses: "List[tuple]" = []
    for i, p in enumerate(paths):
        if getattr(p, "is_local_path", False):
            misses.append((i, p, None))
            continue
        key = _key(p)
        hit = _ram.get(key)
        if hit is None:
            hit = _disk_get(key)
        if hit is not None:
            _ram.put(key, hit)
            out[i] = hit
        else:
            misses.append((i, p, key))
    if not misses:
        return out

    def _fetch(item):
        i, p, key = item
        try:
            with p.open("rb") as bio:
                return i, key, bio.read()
        except FileNotFoundError:
            return i, key, None

    if len(misses) == 1:
        results = [_fetch(misses[0])]
    else:
        with cf.ThreadPoolExecutor(max_workers=min(len(misses), _MAX_FETCH_WORKERS)) as ex:
            results = list(ex.map(_fetch, misses))
    for i, key, raw in results:
        if raw is not None and key is not None:
            _disk_put(key, raw)
            _ram.put(key, raw)
        out[i] = raw
    return out
