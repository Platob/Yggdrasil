"""Specialized, content-addressed local cache for HTTP responses.

The generic cache stores responses as a Hive-partitioned Arrow dataset and scans
it with a :class:`Predicate` on **every** lookup — the whole Folder / Tabular /
parquet pipeline for what is really a key→blob store. :class:`HttpResponseCache`
is the lean replacement for the **local** backend: it's a :class:`Folder` (so it
drops in wherever the cache expects a Folder/Tabular holder — ``cache_tabular``,
``local_cache_folder``, the cleanup daemon under ``~/.cache/http/response``), but
each response is stored as one small Arrow-IPC file named by the producing
request's ``public_hash`` (sharded one byte deep). A lookup is then a single
O(1) file open, a write a single atomic file replace (upsert) — no dataset scan,
partitioning, predicate, or schema cast.

The remote (Databricks Table) backend is untouched and keeps the tabular path.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import pathlib
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


def _ipc_to_table(data: bytes) -> "Any":
    import pyarrow as pa
    return pa.ipc.open_stream(pa.BufferReader(data)).read_all()


def _table_to_ipc(table: "Any") -> bytes:
    import pyarrow as pa
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _read_bytes(path: pathlib.Path) -> "Optional[bytes]":
    try:
        return path.read_bytes()
    except OSError:               # absent / unreadable ⇒ cache miss
        return None


def _write_atomic(path: pathlib.Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)     # atomic publish (safe across procs)
    except OSError:               # cache is best-effort — never break a write
        pass


class HttpResponseCache(Folder):
    """Content-addressed local HTTP-response cache (one Arrow-IPC file per key)."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.RESPONSE_CACHE_FOLDER
    __slots__ = ("_root",)

    def __init__(self, data: Any = None, *, path: Any = None,
                 tabular_parent: Any = None, **kwargs: Any) -> None:
        super().__init__(data, path=path, tabular_parent=tabular_parent, **kwargs)
        self._root: "Optional[pathlib.Path]" = None

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
        from yggdrasil.io.response import Response

        reqs = list(requests)
        if not reqs:
            return [], []

        def _load(req):
            data = _read_bytes(self._file(req.match_value(_MATCH_KEY)))
            if data is None:
                return req, None
            for resp in Response.from_arrow_tabular(_ipc_to_table(data)):
                return req, resp
            return req, None

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
        table = pa.Table.from_batches([data]) if isinstance(data, pa.RecordBatch) else data
        if table.num_rows == 0:
            return
        keys = table.column(_MATCH_COLUMN).to_pylist()
        for i, key in enumerate(keys):
            if key is None:
                continue
            _write_atomic(self._file(int(key)), _table_to_ipc(table.slice(i, 1)))
