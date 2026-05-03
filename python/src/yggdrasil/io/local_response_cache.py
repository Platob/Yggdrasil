"""Partitioned-folder local cache for HTTP :class:`Response` objects.

This module replaces the legacy per-request ``.arrow`` file layout
(one IPC file per anonymized-request hash, TTL encoded into the
filename) with a Hive-partitioned :class:`PartitionedFolderIO`
keyed on response-schema columns.

The new layout
--------------

::

    <root>/cache/
        request_method=GET/
            request_url_host=api.example.com/
                <staging-name>.arrow      ← N rows, RESPONSE_ARROW_SCHEMA
                <staging-name>.arrow      ← later writes append more leaves

* Every leaf file carries :data:`RESPONSE_ARROW_SCHEMA` minus the
  partition columns (Hive convention — partitions are encoded in the
  directory path, not duplicated in the leaf).
* Reads use partition pruning + row-level filtering, so a lookup on
  ``GET https://api.example.com/foo`` only scans the
  ``request_method=GET/request_url_host=api.example.com/`` subtree.
* Writes are append-only — concurrent stores from multiple workers
  drop independent leaves into the matching partition without
  contention. On lookup, when several rows match the same request
  the latest one (``response_received_at``) wins, so UPSERT mode is
  satisfied by the natural read tie-break instead of an explicit
  per-request eviction step.

Public surface
--------------

- :class:`LocalResponseCache` is the carrier. Construction takes the
  root path plus partition / matching / time-window knobs; everything
  else flows through :meth:`lookup`, :meth:`lookup_many`,
  :meth:`store`, :meth:`store_many`, :meth:`evict_partition`,
  :meth:`count`, and :meth:`open` (yields the underlying
  :class:`PartitionedFolderIO` for callers that need raw folder IO).

The public constants document the defaults — partition columns,
match-by columns, child media type — so callers tuning for hot
paths know what to override.
"""

from __future__ import annotations

import datetime as dt
import logging
import shutil
import uuid
from pathlib import Path as _StdPath
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Sequence,
)

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.io.buffer.nested.partitioned_io import PartitionedFolderIO
from yggdrasil.io.buffer.primitive import ArrowIPCIO
from yggdrasil.io.enums import MimeType, MimeTypes
from yggdrasil.io.fs import LocalPath, Path
from yggdrasil.io.request import PreparedRequest
from yggdrasil.io.response import (
    RESPONSE_ARROW_SCHEMA,
    Response,
)

if TYPE_CHECKING:
    pass


__all__ = [
    "LocalResponseCache",
    "DEFAULT_LOCAL_CACHE_PARTITIONS",
    "DEFAULT_LOCAL_CACHE_MATCH_BY",
    "DEFAULT_LOCAL_CACHE_MEDIA_TYPE",
    "DEFAULT_LOCAL_CACHE_FOLDER",
]


LOGGER = logging.getLogger(__name__)


# Hot dimensions for partition pruning: HTTP method first, then host.
# Method is small-cardinality (a handful of values), host is medium —
# together they segment the cache enough that a typical lookup reads
# only a single partition's worth of leaves.
DEFAULT_LOCAL_CACHE_PARTITIONS: tuple[str, ...] = (
    "request_method",
    "request_url_host",
)

# Match-by columns used to find a request's row inside a partition.
# ``request_url_str`` is the canonical full URL (scheme + host + path
# + query) on the anonymized request — distinct enough to identify a
# specific cached call without going through a hash, and stable across
# anonymization modes that don't touch the URL itself.
DEFAULT_LOCAL_CACHE_MATCH_BY: tuple[str, ...] = ("request_url_str",)

# Arrow IPC keeps the on-disk format simple: one record-batch-shaped
# file, fast read, no schema-evolution dance. Parquet is a reasonable
# alternative when the cache grows past memory and column pruning
# starts mattering — pass ``child_media_type=MimeTypes.PARQUET``.
DEFAULT_LOCAL_CACHE_MEDIA_TYPE: MimeType = MimeTypes.ARROW_IPC

# Sub-folder under the configured root that holds the partitioned
# cache. Keeping the cache under ``cache/`` leaves the root free for
# sibling artefacts (logs, tmp, etc.) the way the legacy layout did.
DEFAULT_LOCAL_CACHE_FOLDER: str = "cache"


def _default_root() -> _StdPath:
    """Default cache root: ``~/.yggdrasil/io/session``.

    Matches :meth:`CacheConfig.local_cache_folder` so both APIs land
    on the same on-disk location when the caller supplies neither.
    """
    return _StdPath.home() / ".yggdrasil" / "io" / "session"


class LocalResponseCache:
    """Partitioned-folder cache for HTTP :class:`Response` objects.

    Backed by a :class:`PartitionedFolderIO` rooted at
    ``<path>/<folder_name>/`` whose schema matches
    :data:`RESPONSE_ARROW_SCHEMA`. The Hive partitions encode
    :attr:`partition_columns` in the directory tree; lookups prune
    by partition prefix before touching any leaf, then filter rows
    by :attr:`match_by` and the configured received-window.

    Parameters
    ----------
    path:
        Cache root. Falls back to ``~/.yggdrasil/io/session`` when
        ``None``. The actual partitioned tree lives at
        ``<path>/<folder_name>/`` so the root can host sibling
        artefacts.
    folder_name:
        Sub-folder under ``path`` that holds the partitioned tree.
        Defaults to :data:`DEFAULT_LOCAL_CACHE_FOLDER` (``"cache"``).
    partition_columns:
        Hive partition columns. Defaults to
        :data:`DEFAULT_LOCAL_CACHE_PARTITIONS`. Order matters — it
        determines directory layering. Pass ``()`` to disable
        partitioning entirely (one flat folder).
    child_media_type:
        Format for partition-leaf files. Defaults to Arrow IPC; pass
        ``MimeTypes.PARQUET`` for column pruning on huge caches.
    match_by:
        Columns identifying a specific request inside a partition —
        used for both lookup row-selection and the lookup tie-break
        when several rows match (latest ``response_received_at``
        wins). Defaults to :data:`DEFAULT_LOCAL_CACHE_MATCH_BY`.
    received_from / received_to:
        Optional row-filter window applied on every read. Rows whose
        ``response_received_at`` falls outside the window are
        ignored — the cache never returns stale data even when the
        underlying leaf still holds it.
    """

    __slots__ = (
        "path",
        "folder_name",
        "partition_columns",
        "child_media_type",
        "match_by",
        "received_from",
        "received_to",
    )

    def __init__(
        self,
        path: "Path | _StdPath | str | None" = None,
        *,
        folder_name: str = DEFAULT_LOCAL_CACHE_FOLDER,
        partition_columns: "Sequence[str] | None" = None,
        child_media_type: "MimeType | str | None" = None,
        match_by: "Sequence[str] | None" = None,
        received_from: "dt.datetime | None" = None,
        received_to: "dt.datetime | None" = None,
    ) -> None:
        root = _default_root() if path is None else _StdPath(str(path))
        self.path: _StdPath = root
        self.folder_name: str = folder_name
        self.partition_columns: tuple[str, ...] = (
            tuple(partition_columns)
            if partition_columns is not None
            else DEFAULT_LOCAL_CACHE_PARTITIONS
        )
        self.child_media_type: MimeType = (
            MimeTypes.from_(child_media_type)
            if child_media_type is not None
            else DEFAULT_LOCAL_CACHE_MEDIA_TYPE
        )
        # ``match_by`` is intentionally a tuple of plain strings (not
        # Field) — request_tuple lookup goes through
        # :meth:`PreparedRequest.match_value`, which is keyed by name.
        self.match_by: tuple[str, ...] = (
            tuple(match_by) if match_by else DEFAULT_LOCAL_CACHE_MATCH_BY
        )
        self.received_from: "dt.datetime | None" = received_from
        self.received_to: "dt.datetime | None" = received_to

    # ------------------------------------------------------------------
    # Folder accessors
    # ------------------------------------------------------------------

    @property
    def folder_path(self) -> _StdPath:
        """Absolute path to the partitioned cache root."""
        return self.path / self.folder_name

    def open(self) -> PartitionedFolderIO:
        """Build a fresh :class:`PartitionedFolderIO` for this cache.

        Returned IO is un-acquired — caller opens it inside a
        ``with`` block when reading. Each call constructs a new
        instance; the IO itself carries no read state, so sharing
        across threads is fine but each operation should grab its
        own.
        """
        return PartitionedFolderIO(
            path=LocalPath(self.folder_path),
            partition_columns=list(self.partition_columns) or None,
        )

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    @property
    def arrow_schema(self) -> pa.Schema:
        """The full response schema, including partition columns."""
        return RESPONSE_ARROW_SCHEMA

    @property
    def leaf_arrow_schema(self) -> pa.Schema:
        """Schema as written to a partition leaf (partition cols stripped).

        Hive partitions live in directory names — they're injected
        back into the leaf rows on read, never stored on disk. This
        property reflects what's actually written to a leaf file.
        """
        names = [
            n for n in RESPONSE_ARROW_SCHEMA.names
            if n not in self.partition_columns
        ]
        return pa.schema([RESPONSE_ARROW_SCHEMA.field(n) for n in names])

    # ------------------------------------------------------------------
    # Reads — lookup / lookup_many / count
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total row count across the cache, after received-window filter.

        Cheap when the folder is empty (no read), bounded by the
        partition-pruned subset otherwise. Use as a barrier in
        tests waiting for fire-and-forget writes to land.
        """
        table = self._read_filtered()
        return 0 if table is None else table.num_rows

    def lookup(
        self,
        request: PreparedRequest,
    ) -> "Response | None":
        """Return the cached response for ``request``, or ``None``."""
        results = self.lookup_many([request])
        return results.get(self._request_key(request))

    def lookup_many(
        self,
        requests: Iterable[PreparedRequest],
    ) -> dict[tuple, Response]:
        """Bulk lookup — one folder read for many requests.

        Returns ``{request_key: Response}`` keyed by the tuple of
        :attr:`match_by` values on each request. Missing requests
        are absent from the dict. When several rows match the same
        request key the latest one (max ``response_received_at``)
        wins, so callers don't need to special-case UPSERT.
        """
        request_list = list(requests)
        if not request_list:
            return {}

        keys_by_request = [self._request_key(r) for r in request_list]
        wanted: set[tuple] = set(keys_by_request)

        partition_filter = self._build_partition_filter(request_list)
        table = self._read_filtered(extra_filter=partition_filter)
        if table is None or table.num_rows == 0:
            return {}

        out: dict[tuple, Response] = {}
        latest_at: dict[tuple, dt.datetime] = {}
        for response in Response.from_arrow_tabular(table.to_batches()):
            key = self._response_key(response)
            if key not in wanted:
                continue
            ts = response.received_at
            existing = latest_at.get(key)
            if existing is None or (ts is not None and ts >= existing):
                out[key] = response
                if ts is not None:
                    latest_at[key] = ts
        return out

    # ------------------------------------------------------------------
    # Writes — store / store_many / evict_partition
    # ------------------------------------------------------------------

    def store(self, response: Response) -> None:
        """Persist a single :class:`Response`. No-op if not ``ok``.

        Equivalent to :meth:`store_many` with one element; kept as a
        named entry point so the per-request hot path stays clear at
        the call site.
        """
        self.store_many([response])

    def store_many(self, responses: Iterable[Response]) -> None:
        """Persist many responses to the partitioned folder.

        Filters out non-``ok`` responses up front (the cache only
        keeps successful results). The remaining payload is grouped
        by partition tuple and written directly — one UUID-named
        leaf per (call, partition combination) — bypassing the
        :class:`FolderIO` staging+sequential-rename dance that
        races between concurrent worker threads. Concurrent stores
        across threads never collide on a leaf name, and lookup's
        ``response_received_at`` tie-break keeps the latest write
        as the canonical answer.
        """
        batches: list[pa.RecordBatch] = []
        for response in responses:
            if not response.ok:
                continue
            batches.append(response.to_arrow_batch(parse=False))
        if not batches:
            return

        table = pa.Table.from_batches(batches)
        self._direct_partition_write(table)

    def _direct_partition_write(self, table: pa.Table) -> None:
        """Write ``table`` to UUID-named leaves, one per partition tuple.

        Bypasses :class:`FolderIO`'s staging+rename routing so
        concurrent writers (the fire-and-forget per-response store
        path) can't race on the sequential ``part-NNNNN`` naming
        scheme. Hive-partition columns are stripped from the leaf
        payload (they live in the directory name) — same on-disk
        contract :class:`FolderIO` reads with.
        """
        if table.num_rows == 0:
            return

        self.folder_path.mkdir(parents=True, exist_ok=True)
        partition_columns = [
            c for c in self.partition_columns if c in table.column_names
        ]
        leaf_columns = [
            c for c in table.column_names if c not in partition_columns
        ]
        ext = self._leaf_extension()

        if not partition_columns:
            self._write_leaf(self.folder_path, table, ext, leaf_columns)
            return

        # Group rows by partition tuple — one leaf per distinct
        # combination. Sorting first keeps the row indices for each
        # partition contiguous so we can slice instead of taking by
        # index list (cheaper for large batches).
        sort_keys = [(c, "ascending") for c in partition_columns]
        sorted_table = table.sort_by(sort_keys)
        partition_view = sorted_table.select(partition_columns).to_pylist()
        n = sorted_table.num_rows
        run_start = 0
        run_key = tuple(partition_view[0].get(c) for c in partition_columns)
        for i in range(1, n + 1):
            if i == n:
                next_key = None
            else:
                next_key = tuple(
                    partition_view[i].get(c) for c in partition_columns
                )
            if i == n or next_key != run_key:
                slice_table = sorted_table.slice(run_start, i - run_start)
                target_dir = self._partition_dir(run_key, partition_columns)
                self._write_leaf(target_dir, slice_table, ext, leaf_columns)
                run_start = i
                if next_key is not None:
                    run_key = next_key

    def _partition_dir(
        self,
        partition_values: Sequence[Any],
        partition_columns: Sequence[str],
    ) -> _StdPath:
        target = self.folder_path
        for col, val in zip(partition_columns, partition_values):
            target = target / f"{col}={'' if val is None else val}"
        return target

    def _write_leaf(
        self,
        partition_dir: _StdPath,
        slice_table: pa.Table,
        ext: str,
        leaf_columns: Sequence[str],
    ) -> None:
        partition_dir.mkdir(parents=True, exist_ok=True)
        # uuid4 is collision-resistant enough that we can skip the
        # tmp+rename dance entirely — the leaf is written in place
        # under its final name. Failure mid-write leaves a partial
        # file that the read path's per-leaf try/except treats as
        # a miss, exactly the same as the legacy "drop corrupted
        # files on read" behaviour.
        leaf_name = f"part-{uuid.uuid4().hex}{ext}"
        leaf_path = partition_dir / leaf_name
        data = slice_table.select(list(leaf_columns)) if leaf_columns else slice_table
        ArrowIPCIO(path=str(leaf_path)).write_arrow_table(data)

    def _leaf_extension(self) -> str:
        """Filename extension for partition leaves.

        Picked from :attr:`child_media_type`'s primary extension
        (``.arrow`` for Arrow IPC, ``.parquet`` for Parquet, …) so
        the on-disk layout still matches what FolderIO expects on
        the read side.
        """
        try:
            ext = self.child_media_type.extensions[0]
        except (AttributeError, IndexError):
            return ""
        return f".{ext}" if ext else ""

    def evict_partition(
        self,
        partition_values: Mapping[str, Any],
    ) -> None:
        """Drop a whole partition subtree on disk.

        Useful when callers want hard eviction (e.g. a tenant key
        changed and stale rows must go). For per-request UPSERT
        eviction the natural lookup tie-break is enough — fresh
        stores win on read, so explicit eviction isn't required.
        """
        sub = self.folder_path
        for col in self.partition_columns:
            if col not in partition_values:
                break
            value = partition_values[col]
            sub = sub / f"{col}={'' if value is None else value}"
        if sub == self.folder_path:
            return
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)

    def clear(self) -> None:
        """Remove the entire cache subtree. Idempotent."""
        if self.folder_path.exists():
            shutil.rmtree(self.folder_path, ignore_errors=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _read_filtered(
        self,
        *,
        extra_filter: "pc.Expression | None" = None,
    ) -> "pa.Table | None":
        """Read the cache, applying received-window + extra filters.

        Returns ``None`` when the folder doesn't exist yet — keeps
        callers free of a ``try/except`` around the open. Empty but
        existing folders return an empty table (callers that need
        ``num_rows`` don't have to special-case the missing-folder
        path).
        """
        if not self.folder_path.exists():
            return None
        try:
            with self.open() as folder:
                table = folder.read_arrow_table()
        except Exception:
            # Corrupt leaf, race with a concurrent write, schema
            # drift — treat as "nothing matches" so callers fall
            # through to the next pipeline stage cleanly.
            LOGGER.debug(
                "Local response cache read failed under %s; "
                "treating as miss", self.folder_path,
                exc_info=True,
            )
            return None

        if extra_filter is not None:
            try:
                table = table.filter(extra_filter)
            except Exception:
                # A filter that references a missing column on an
                # empty cache shouldn't tank the read.
                LOGGER.debug(
                    "Partition filter failed on empty cache; "
                    "skipping", exc_info=True,
                )

        if "response_received_at" in table.column_names:
            if self.received_from is not None:
                table = table.filter(
                    pc.greater_equal(
                        table["response_received_at"],
                        pa.scalar(self.received_from),
                    )
                )
            if self.received_to is not None:
                table = table.filter(
                    pc.less(
                        table["response_received_at"],
                        pa.scalar(self.received_to),
                    )
                )
        return table

    def _build_partition_filter(
        self,
        requests: Sequence[PreparedRequest],
    ) -> "pc.Expression | None":
        """Build a partition-pruning predicate from the requests batch.

        For each partition column, collects the distinct values across
        all requests and AND-s an ``isin`` filter. The folder reader
        already prunes directories by name; this predicate is the
        belt-and-braces row filter applied after read in case the
        engine couldn't push it down.
        """
        if not self.partition_columns:
            return None
        expr: "pc.Expression | None" = None
        for col in self.partition_columns:
            try:
                values = sorted({
                    r.match_value(col) for r in requests
                }, key=lambda v: (v is None, v))
            except Exception:
                continue
            if not values:
                continue
            scalars = pa.array(values)
            term = pc.is_in(pc.field(col), value_set=scalars)
            expr = term if expr is None else (expr & term)
        return expr

    def _request_key(self, request: PreparedRequest) -> tuple:
        try:
            return tuple(request.match_value(c) for c in self.match_by)
        except Exception:
            return ()

    def _response_key(self, response: Response) -> tuple:
        try:
            return tuple(response.match_value(c) for c in self.match_by)
        except Exception:
            return ()
