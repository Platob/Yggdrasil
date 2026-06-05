"""Specialized, content-addressed local cache for HTTP responses.

The generic cache stores responses as a Hive-partitioned Arrow dataset and scans
it with a :class:`Predicate` on **every** lookup — the whole Folder / Tabular /
parquet pipeline for what is really a key→blob store. :class:`HttpResponseCache`
is the lean replacement for the **local** backend: it's a :class:`Folder` (so it
drops in wherever the cache expects a Folder/Tabular holder — ``cache_tabular``,
``local_cache_folder``, the cleanup daemon under ``~/.cache/http/response``), but
each response is stored as one tiny file named by the producing request's
``public_hash`` (sharded one byte deep): a 1-byte version, a JSON meta header
(status, received-at, headers, tags) and the raw body — **no Arrow** on the hot
path. A lookup is then a single O(1) file read + Response rebuild (the looked-up
request, same identity, is reattached), a write a single atomic file replace
(upsert). No dataset scan, partitioning, predicate, schema, or Arrow round-trip.

Over the on-disk store sits a small **byte-bounded RAM hot tier** (default 32 MB,
process-wide) keyed by ``public_hash``: the most-recent / most-reused responses
are served straight from memory, everything else from the per-key file — so the
cache's RAM can never balloon (an oversized response is kept disk-only rather
than evicting the hot set), and day-old files are pruned on construction.

A cache hit measures **several times faster than even a localhost HTTP call**,
with bounded memory vs the generic tabular cache — see
``benchmarks/http_/bench_response_cache.py``. The remote (Databricks Table)
backend is untouched and keeps the tabular path.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Optional

from yggdrasil.enums.mime_type import MimeTypes
from yggdrasil.path.folder import Folder

if TYPE_CHECKING:
    from yggdrasil.http_.request import PreparedRequest
    from yggdrasil.io.response import Response

__all__ = ["HttpResponseCache"]

#: Request match key — the per-response file name (== cache_config.MATCH_KEY).
_MATCH_KEY = "public_hash"
#: Column carrying the producing request's public_hash in a response batch.
_MATCH_COLUMN = "request_public_hash"
_MAX_WORKERS = 32

# In-RAM hot tier over disk: a byte-bounded LRU of the encoded records keyed by
# the request public_hash. The most-recent / most-reused responses are served
# straight from RAM; everything else falls through to the per-key file on disk.
# RAM stays capped (default 32 MB, process-wide; ``0`` disables) so the cache can
# never balloon — a single big response (> a quarter of the budget) is kept on
# disk only rather than evicting the whole hot set.
_RAM_MAX_BYTES = int(os.environ.get("YGG_HTTP_RAM_CACHE_BYTES") or 32 * 1024 * 1024)
_RAM_ITEM_MAX_BYTES = _RAM_MAX_BYTES // 4


class _ByteLRU:
    """Thread-safe LRU bounded by total bytes (and a per-item byte cap).

    Entries carry a creation time so a TTL :meth:`sweep` can *actively* drop the
    stale ones — ``get`` / ``put`` stay O(1) and never expire lazily on the hot
    path (that's the janitor's job)."""

    __slots__ = ("_max", "_item_max", "_d", "_bytes", "_lock")

    def __init__(self, max_bytes: int, item_max: int) -> None:
        from collections import OrderedDict
        self._max = max_bytes
        self._item_max = item_max
        self._d: "OrderedDict[int, tuple]" = OrderedDict()   # key -> (bytes, created_s)
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: int) -> "Optional[bytes]":
        if self._max <= 0:
            return None
        with self._lock:
            e = self._d.get(key)
            if e is None:
                return None
            self._d.move_to_end(key)
            return e[0]

    def put(self, key: int, val: bytes) -> None:
        n = len(val)
        if self._max <= 0 or n > self._item_max:   # too big for RAM → disk only
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= len(old[0])
            self._d[key] = (val, time.monotonic())
            self._bytes += n
            while self._bytes > self._max:
                _, (evicted, _ts) = self._d.popitem(last=False)
                self._bytes -= len(evicted)

    def sweep(self, max_age_s: float) -> int:
        """Actively drop entries older than *max_age_s*; returns the count."""
        if self._max <= 0 or max_age_s <= 0:
            return 0
        cutoff = time.monotonic() - max_age_s
        with self._lock:
            stale = [k for k, (_v, ts) in self._d.items() if ts < cutoff]
            for k in stale:
                val, _ts = self._d.pop(k)
                self._bytes -= len(val)
        return len(stale)


_ram = _ByteLRU(_RAM_MAX_BYTES, _RAM_ITEM_MAX_BYTES)

# On-disk record: ``\x01`` + uint32 meta-len + meta(JSON) + raw body. The meta
# is the minimum to rebuild a Response — status, received-at (µs), headers, tags
# — deliberately NOT Arrow: a per-response Arrow batch carries ~4 KB of framing
# even for a 16 B body and costs a full from_arrow_tabular rebuild on read. The
# producing request isn't stored either: a hit is keyed by the request's
# public_hash, so the *looked-up* request (same identity) is reattached on read.
_VERSION = 1

#: Cached responses auto-expire after this (default 1 day; ``0`` disables), on
#: both tiers. The janitor wakes every _SWEEP_INTERVAL to enforce it actively —
#: not lazily on access — so neither RAM nor disk lingers stale entries.
_CLEAN_TTL = float(os.environ.get("YGG_HTTP_CACHE_TTL") or 86400.0)
_SWEEP_INTERVAL = float(os.environ.get("YGG_HTTP_CACHE_SWEEP_INTERVAL") or 3600.0)
_roots: "set[str]" = set()
_janitor_lock = threading.Lock()
_janitor_started = False


def _prune_old(root: pathlib.Path, ttl: float) -> int:
    """Delete files older than *ttl* seconds under *root* and prune the empty
    shard dirs left behind. Best-effort; returns the count removed."""
    if ttl <= 0 or not root.exists():
        return 0
    cutoff = time.time() - ttl
    removed = 0
    for dirpath, _dirs, files in os.walk(str(root), topdown=False):
        for name in files:
            fp = os.path.join(dirpath, name)
            try:
                if os.stat(fp).st_mtime < cutoff:
                    os.unlink(fp)
                    removed += 1
            except OSError:
                pass
        if dirpath != str(root):
            try:
                os.rmdir(dirpath)
            except OSError:
                pass
    return removed


def _janitor_pass() -> None:
    """One active expiry sweep across every known disk root + the RAM tier."""
    for r in list(_roots):
        _prune_old(pathlib.Path(r), _CLEAN_TTL)
    _ram.sweep(_CLEAN_TTL)


def _start_janitor(root: pathlib.Path) -> None:
    """Register *root* and (once per process) start the background janitor — an
    immediate sweep, then one every _SWEEP_INTERVAL. A real daemon, not a
    lazy-on-access expiry, so day-old responses are reclaimed even in a
    long-lived process and never just linger in memory."""
    global _janitor_started
    if _CLEAN_TTL <= 0:
        return
    with _janitor_lock:
        _roots.add(str(root))
        if _janitor_started:
            return
        _janitor_started = True

    def _loop() -> None:
        while True:
            try:
                _janitor_pass()
            except Exception:        # never let the janitor die on a transient error
                pass
            time.sleep(_SWEEP_INTERVAL)

    threading.Thread(target=_loop, name="ygg-http-cache-janitor", daemon=True).start()


def _encode(status: int, received: "Any", headers: "Any", tags: "Any", body: bytes) -> bytes:
    import json
    micros = int(received.timestamp() * 1_000_000) if received is not None else 0
    meta = json.dumps(
        {"s": int(status or 0), "r": micros,
         "h": dict(headers or ()), "t": dict(tags or ())},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return bytes((_VERSION,)) + len(meta).to_bytes(4, "little") + meta + (body or b"")


def _decode(data: bytes, request: "Any") -> "Optional[Response]":
    import json
    from datetime import datetime, timezone
    from yggdrasil.io.response import Response

    if not data or data[0] != _VERSION:
        return None
    mlen = int.from_bytes(data[1:5], "little")
    meta = json.loads(data[5:5 + mlen])
    body = data[5 + mlen:]
    return Response(
        request=request,
        status_code=meta["s"],
        headers=meta["h"],
        tags=meta.get("t") or {},
        buffer=body,
        received_at=datetime.fromtimestamp(meta["r"] / 1_000_000, tz=timezone.utc),
    )


def _read_bytes(path: pathlib.Path) -> "Optional[bytes]":
    try:
        return path.read_bytes()
    except OSError:               # absent / unreadable ⇒ cache miss
        return None


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    # Caller (``_write_many``) has already created the shard dir.
    try:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)     # atomic publish (safe across procs)
    except OSError:               # cache is best-effort — never break a write
        pass


class HttpResponseCache(Folder):
    """Content-addressed local HTTP-response cache (one lightweight file per key)."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.RESPONSE_CACHE_FOLDER
    __slots__ = ("_root",)

    def __init__(self, data: Any = None, *, path: Any = None,
                 tabular_parent: Any = None, **kwargs: Any) -> None:
        super().__init__(data, path=path, tabular_parent=tabular_parent, **kwargs)
        self._root: "Optional[pathlib.Path]" = None
        try:                                   # auto-expire day-old responses
            _start_janitor(self.root)
        except Exception:                      # never let cleanup break construction
            pass

    @property
    def root(self) -> pathlib.Path:
        if self._root is None:
            full = self.path.full_path() if hasattr(self.path, "full_path") else str(self.path)
            self._root = pathlib.Path(full).expanduser()
        return self._root

    def _file(self, key: int) -> pathlib.Path:
        # Shard by the low byte so a high-volume API doesn't pile every response
        # into one directory.
        return self.root / f"{key & 0xFF:02x}" / f"{key}.arrow"

    def probe_hashes(self, requests: "Iterable[PreparedRequest]") -> "set[int]":
        """The subset of request ``public_hash`` keys that have a cached file —
        the cheap presence check the send pipeline uses to split hits/misses."""
        out: "set[int]" = set()
        for req in requests:
            key = req.match_value(_MATCH_KEY)
            if self._file(key).exists():
                out.add(key)
        return out

    # ------------------------------------------------------------------
    # Read — O(1) file lookup per request (no dataset scan / predicate)
    # ------------------------------------------------------------------
    def read_responses(
        self,
        requests: "Iterable[PreparedRequest]",
        *,
        config: "Any",
    ) -> "tuple[list, list]":
        """``(hits, misses)`` — each request's per-key file decoded to a
        :class:`Response` and accepted by ``config.filter_response``."""
        reqs = list(requests)
        if not reqs:
            return [], []

        def _load(req):
            key = req.match_value(_MATCH_KEY)
            data = _ram.get(key)                    # hot tier
            if data is None:
                data = _read_bytes(self._file(key))  # disk
                if data is not None:
                    _ram.put(key, data)
            if data is None:
                return req, None
            return req, _decode(data, req)

        if len(reqs) == 1:
            loaded = [_load(reqs[0])]
        else:
            with cf.ThreadPoolExecutor(max_workers=min(len(reqs), _MAX_WORKERS)) as ex:
                loaded = list(ex.map(_load, reqs))

        hits: list = []
        misses: list = []
        for req, resp in loaded:
            if resp is not None and config.filter_response(resp, request=req):
                hits.append(resp)
            else:
                misses.append(req)
        return hits, misses

    # ------------------------------------------------------------------
    # Write — one atomic file per response row (upsert by key)
    # ------------------------------------------------------------------
    def write_arrow(self, data: "Any") -> None:
        """Persist a response :class:`pa.RecordBatch`/``Table`` (or Spark frame),
        one file per row keyed by ``request_public_hash`` — overwriting the prior
        entry for that key (upsert)."""
        import pyarrow as pa

        if data is None:
            return
        if not isinstance(data, (pa.RecordBatch, pa.Table)):
            to_arrow = getattr(data, "toArrow", None)      # Spark frame
            if to_arrow is None:
                return
            data = to_arrow()
        batches = data.to_batches() if isinstance(data, pa.Table) else [data]
        for batch in batches:
            if batch.num_rows == 0:
                continue
            cols = set(batch.schema.names)
            if not ({_MATCH_COLUMN, "status_code", "headers", "body"} <= cols):
                continue
            # Pull the few columns we keep once (vectorised), then write one
            # lightweight file per row.
            keys = batch.column(_MATCH_COLUMN).to_pylist()
            statuses = batch.column("status_code").to_pylist()
            headers = batch.column("headers").to_pylist()
            bodies = batch.column("body").to_pylist()
            receiveds = (batch.column("received_at").to_pylist()
                         if "received_at" in cols else [None] * len(keys))
            tags = (batch.column("tags").to_pylist()
                    if "tags" in cols else [None] * len(keys))
            records = []
            for i, key in enumerate(keys):
                if key is None:
                    continue
                k = int(key)
                enc = _encode(statuses[i], receiveds[i], headers[i], tags[i], bodies[i])
                _ram.put(k, enc)                    # populate the hot tier too
                records.append((self._file(k), enc))
            self._write_many(records)

    @staticmethod
    def _write_many(records: "list") -> None:
        if not records:
            return
        for d in {f.parent for f, _ in records}:      # create the shard dirs once
            try:
                d.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass
        if len(records) == 1:
            _write_atomic(*records[0])
            return
        with cf.ThreadPoolExecutor(max_workers=min(len(records), _MAX_WORKERS)) as ex:
            list(ex.map(lambda item: _write_atomic(*item), records))
