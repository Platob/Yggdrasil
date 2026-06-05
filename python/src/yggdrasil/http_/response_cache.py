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

A cache hit measures **several times faster than even a localhost HTTP call**,
with lighter memory than the generic tabular cache — see
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

# On-disk record: ``\x01`` + uint32 meta-len + meta(JSON) + raw body. The meta
# is the minimum to rebuild a Response — status, received-at (µs), headers, tags
# — deliberately NOT Arrow: a per-response Arrow batch carries ~4 KB of framing
# even for a 16 B body and costs a full from_arrow_tabular rebuild on read. The
# producing request isn't stored either: a hit is keyed by the request's
# public_hash, so the *looked-up* request (same identity) is reattached on read.
_VERSION = 1

#: Cached responses auto-expire after this (default 1 day; ``0`` disables).
_CLEAN_TTL = float(os.environ.get("YGG_HTTP_CACHE_TTL") or 86400.0)
_cleaned_roots: "set[str]" = set()
_clean_lock = threading.Lock()


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


def _schedule_clean(root: pathlib.Path) -> None:
    """Once per root per process, prune day-old responses in a daemon thread —
    the local cache self-expires so it never grows without bound."""
    if _CLEAN_TTL <= 0:
        return
    key = str(root)
    with _clean_lock:
        if key in _cleaned_roots:
            return
        _cleaned_roots.add(key)
    threading.Thread(
        target=_prune_old, args=(root, _CLEAN_TTL),
        name="ygg-http-cache-clean", daemon=True,
    ).start()


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
            _schedule_clean(self.root)
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
            data = _read_bytes(self._file(req.match_value(_MATCH_KEY)))
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
            records = [
                (self._file(int(key)),
                 _encode(statuses[i], receiveds[i], headers[i], tags[i], bodies[i]))
                for i, key in enumerate(keys) if key is not None
            ]
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
