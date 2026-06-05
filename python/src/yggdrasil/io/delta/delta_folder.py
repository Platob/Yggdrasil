"""DeltaFolder — :class:`Folder` over a Delta Lake table.

Full Delta read/write protocol: V1/V2 checkpoints, deletion vectors,
per-file stats, concurrent commit with exponential backoff, partition
pruning, APPEND / OVERWRITE / UPSERT / MERGE / IGNORE / ERROR_IF_EXISTS,
row-level delete via DV or rewrite, Spark mapInArrow integration.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import logging
import os
import random
import time
import urllib.parse
import uuid
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, List, Optional

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.path.folder import Folder, FolderOptions
from yggdrasil.io.parquet_file import ParquetFile, ParquetOptions
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.delta._names import format_commit_name
from yggdrasil.io.delta.checkpoint import update_last_checkpoint, write_checkpoint
from yggdrasil.io.delta.deletion_vector import (
    DeletionVector, decode_deletion_vector, encode_inline_deletion_vector,
    mask_batch_with_dv, write_uuid_deletion_vector,
)
from yggdrasil.io.delta.log import DeltaLog, LogSegment
from yggdrasil.io.delta.protocol import (
    AddFile, CommitInfo, DeltaAction,
    DomainMetadata, Metadata, Protocol, RemoveFile, Txn,
)
from yggdrasil.io.delta.schema_codec import (
    arrow_schema_to_spark_json, spark_json_to_arrow_schema,
    schema_to_spark_json, spark_json_to_schema,
)
from yggdrasil.io.delta.snapshot import Snapshot

if TYPE_CHECKING:
    from yggdrasil.execution.expr import Predicate

__all__ = ["ConcurrentDeltaCommitError", "DeltaFolder", "DeltaOptions"]

logger = logging.getLogger(__name__)


class ConcurrentDeltaCommitError(RuntimeError):
    """Raised on a Delta commit version race that can't be rebased.

    Two distinct causes, both surfaced through this one type:

    - **Logical conflict** — a concurrent writer touched the same files
      this operation depends on (e.g. two overwrites, or an append that
      raced a delete of a file we rewrote). Delta's protocol says these
      operations don't commute, so the loser must abort and let the
      caller decide. Carries :attr:`conflict` describing what clashed.
    - **Exhausted retries** — the version kept advancing under us faster
      than we could rebase. Carries ``conflict=None``.
    """

    def __init__(self, message: str, *, conflict: "Optional[str]" = None) -> None:
        super().__init__(message)
        self.conflict = conflict


@dataclasses.dataclass(frozen=True, slots=True)
class _CommitPlan:
    """Describes a write operation's intent for optimistic-concurrency rebase.

    The retry loop uses this to decide, on a version collision, whether the
    already-written data files can simply be *rebased* onto the new HEAD
    (cheap — just renumber the commit) or whether a concurrent writer
    created a genuine logical conflict that must abort.

    Fields:

    - ``build_actions`` — builds the action list against a base snapshot.
      Called once up front (attempt 0) and again only when a rebuild is
      forced (a real data-dependent operation that lost its base).
    - ``is_blind_append`` — the operation only *adds* files and reads
      nothing (plain APPEND). Concurrent appends commute under the Delta
      protocol, so a blind append never logically conflicts: on collision
      we keep the same AddFiles and bump the version.
    - ``read_file_paths`` — AddFile paths the operation read or depends on
      remaining present (the files UPSERT/DELETE/OVERWRITE rewrote). If a
      concurrent commit *removed* any of these, our rewrite is based on a
      stale view → conflict.
    - ``removes_all`` — the operation logically removes the entire table
      snapshot it saw (OVERWRITE). Any concurrent add/remove is a conflict
      because the overwrite's "replace everything I saw" no longer holds.
    """

    build_actions: "Callable[[Snapshot], list[DeltaAction]]"
    is_blind_append: bool
    read_file_paths: "frozenset[str]" = frozenset()
    removes_all: bool = False


_INLINE_DV_MAX_ROWS = 4096
_SIGNED_FOR_UINT = {8: pa.int8, 16: pa.int16, 32: pa.int32, 64: pa.int64}

#: How long a cached latest :class:`Snapshot` is trusted before
#: ``snapshot()`` re-checks the ``_delta_log`` (and incrementally applies any
#: new commits). Bounds how stale a long-lived reader can be vs. an external
#: writer without an explicit ``fresh=True`` / :meth:`refresh`. Configurable via
#: ``YGG_DELTA_SNAPSHOT_TTL`` — set ``0`` to re-list on **every** access (serve
#: the cached parse only while the listing is unchanged: freshest, at one LIST
#: per read); a larger value trades freshness for fewer LISTs on hot re-reads.
_SNAPSHOT_TTL = float(os.environ.get("YGG_DELTA_SNAPSHOT_TTL", "30") or 30)


@dataclasses.dataclass(frozen=True, slots=True)
class DeltaOptions(FolderOptions):
    version: Optional[int] = None
    checkpoint_interval: int = 10
    checkpoint_kind: str = "v1"
    operation: str = "WRITE"
    engine_info: str = "yggdrasil"
    txn_app_id: Optional[str] = None
    txn_version: Optional[int] = None
    min_reader_version: int = 1
    min_writer_version: int = 2
    delete_via_dv: bool = False
    #: When True (default) a write lands a new commit at version ``N+1`` in
    #: the ``_delta_log`` — the normal Delta path. When False, the data
    #: parquet files are still written into the table directory but **no
    #: commit is recorded**: the version is not bumped and the ``_delta_log``
    #: is left untouched. Lets a caller stage data files into a table
    #: location (for a later or external commit) without advancing the
    #: table's history or changing what a reader's snapshot sees. A delete /
    #: truncate — which can only take effect through a commit — is a no-op
    #: under this flag.
    write_new_version: bool = True
    commit_max_retries: int = 8
    commit_retry_backoff: float = 0.05
    commit_retry_jitter: float = 0.05
    commit_retry_max_delay: float = 1.0
    collect_stats: bool = True
    target_file_size: int = 128 * 1024 * 1024
    #: Columns to liquid-cluster the table by. Stamped into the
    #: ``delta.clustering`` domain-metadata + the ``clustering`` writer
    #: feature on create, and always given per-file stats (so the
    #: clustering-aware pruner can lean on them). Mirrors Databricks
    #: ``CREATE TABLE … CLUSTER BY (...)``.
    cluster_by: "Optional[tuple[str, ...]]" = None
    #: Cap on how many leading columns get per-file min/max stats —
    #: matches Databricks' ``delta.dataSkippingNumIndexedCols`` (32 by
    #: default). Clustering columns are always stat'd regardless of where
    #: they fall in the schema, because a predicate on a clustering column
    #: is the one that actually prunes (co-located rows ⇒ tight per-file
    #: ranges ⇒ most files excluded).
    stats_num_indexed_cols: int = 32


class DeltaFolder(Folder):
    """:class:`Folder` over a Delta Lake table at a :class:`Path`."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.DELTA_FOLDER
    __slots__ = ("_log", "_snapshot")

    @classmethod
    def options_class(cls):
        return DeltaOptions

    def __init__(self, data: Any = None, *, path: Any = None,
                 tabular_parent: Any = None, **kwargs: Any) -> None:
        super().__init__(data, path=path, tabular_parent=tabular_parent, **kwargs)
        self._log = DeltaLog(self.path)
        self._snapshot: "Optional[Snapshot]" = None
        self._snapshot_at: float = 0.0

    def __repr__(self) -> str:
        return f"DeltaFolder(path={self.path!r})"

    def refresh(self) -> "DeltaFolder":
        self._log.invalidate()
        self._snapshot = None
        self._snapshot_at = 0.0
        return self

    @property
    def log(self) -> DeltaLog:
        return self._log

    def snapshot(self, version: "Optional[int]" = None, *,
                 fresh: bool = False) -> Snapshot:
        """Collapsed table state at ``version`` (default: latest).

        The latest snapshot is cached for :data:`_SNAPSHOT_TTL` seconds. Past
        the TTL the cache is *advanced incrementally* — we re-list the
        ``_delta_log`` (always a fresh ListObjectsV2; there's no listing cache)
        and replay only the commits written since the cached version, instead
        of re-reading the checkpoint and every commit. ``fresh=True`` forces a
        full re-read from storage (used on the commit / write-credential
        paths, where a clean rebuild is cheaper to reason about than trusting
        the cached state). ``DeltaFolder`` is a location-keyed singleton, so
        without this an *external* commit (a warehouse / Spark INSERT, another
        writer) would be invisible until the process restarted.
        """
        if version is not None:
            return Snapshot.from_log(self._log, version)

        now = time.monotonic()
        if (not fresh and _SNAPSHOT_TTL > 0 and self._snapshot is not None
                and (now - self._snapshot_at) <= _SNAPSHOT_TTL):
            return self._snapshot

        # Re-list the log fresh, then either full-rebuild (fresh / cold) or
        # advance the cached snapshot by the new commits only.
        self._log.invalidate()
        base = self._snapshot
        if fresh or base is None:
            snap = Snapshot.from_log(self._log, None)
        else:
            latest = self._log.latest_version()
            if latest == base.version:
                snap = base                       # nothing new — just reset the clock
            elif latest > base.version:
                snap = base.advanced(self._log, self._log.commits_after(base.version), latest)
            else:
                snap = Snapshot.from_log(self._log, None)  # log shrank (recreated) → rebuild

        self._snapshot = snap
        self._snapshot_at = now
        return snap

    def _collect_schema(self, options: DeltaOptions):
        snap = self.snapshot(options.version)
        if snap.metadata is None or not snap.schema_string:
            from yggdrasil.data.schema import Schema
            return Schema.empty()
        return spark_json_to_schema(snap.schema_string)

    # ==================================================================
    # Read
    # ==================================================================

    def _read_arrow_batches(self, options: DeltaOptions) -> Iterator[pa.RecordBatch]:
        snap = self.snapshot(options.version)
        yield from self._read_snapshot_batches(snap, options)

    def _read_snapshot_batches(self, snap: Snapshot,
                               options: DeltaOptions) -> Iterator[pa.RecordBatch]:
        """Stream Arrow batches off a *pinned* :class:`Snapshot`.

        Split out from :meth:`_read_arrow_batches` so the polars IO-source
        plugin can close over a snapshot captured once at scan-build time and
        replay it on every ``collect`` without ``self`` re-resolving the log
        (no second listing per collect, an immutable consistent view). The
        Arrow read path passes the latest snapshot; both share the column
        pushdown below.
        """
        if snap.metadata is None:
            return

        partition_columns = snap.partition_columns
        full_schema = (spark_json_to_arrow_schema(snap.schema_string)
                       if snap.schema_string else None)

        # Column pushdown. ``read_columns()`` is the projection ∪ the
        # predicate's columns; ``None`` means "read everything" (unchanged
        # behaviour). We keep partition columns out of the leaf projection —
        # they're stripped from the data files on write and re-stamped below —
        # but keep predicate columns in so the row filter still sees them. The
        # final narrowing to the caller's output columns happens in the
        # post-read cast (``_read_arrow_table`` / the scan's ``select``).
        wanted = options.read_columns()
        partition_set = set(partition_columns)
        if wanted is not None and full_schema is not None:
            keep = [n for n in full_schema.names
                    if n in wanted and n not in partition_set]
            # Anything the predicate needs that the projection dropped is
            # already folded into ``read_columns()``, so ``keep`` is the full
            # set of leaf columns the read + filter require.
            # Push the projection into each parquet leaf so only those column
            # chunks are decoded off disk (ParquetFile honours a bound target
            # via ``_projection_columns``). When ``keep`` is empty — the caller
            # asked only for partition column(s), which live in the AddFile not
            # the leaf — read just the first physical column to learn the row
            # count cheaply; it's dropped when the partition columns are
            # stamped on and the cast projects to the output.
            if keep:
                target_schema = pa.schema([full_schema.field(n) for n in keep])
                leaf_proj = target_schema
            elif full_schema is not None and len(full_schema.names) > len(partition_set):
                first_physical = next(n for n in full_schema.names
                                      if n not in partition_set)
                target_schema = pa.schema([full_schema.field(first_physical)])
                leaf_proj = target_schema
            else:
                target_schema = full_schema
                leaf_proj = None
        else:
            target_schema = full_schema
            leaf_proj = None
        sidecar_cache: dict[str, bytes] = {}

        # Build row-level filter once
        row_filter: "Optional[Callable]" = None
        if options.predicate is not None:
            try:
                from yggdrasil.execution.expr import free_columns
                available = set(partition_columns)
                if target_schema is not None:
                    available |= set(target_schema.names)
                if set(free_columns(options.predicate)).issubset(available):
                    arrow_expr = options.predicate.to_arrow()
                    import pyarrow.dataset as pds
                    def row_filter(batch: pa.RecordBatch, _expr=arrow_expr) -> pa.RecordBatch:
                        if batch.num_rows == 0:
                            return batch
                        filtered = pds.dataset(pa.Table.from_batches([batch])).to_table(filter=_expr)
                        if filtered.num_rows == 0:
                            return pa.RecordBatch.from_pylist([], schema=batch.schema)
                        rebuilt = filtered.combine_chunks().to_batches()
                        return rebuilt[0] if rebuilt else pa.RecordBatch.from_pylist([], schema=batch.schema)
            except Exception:
                pass

        prune = _partition_prune_values(options.predicate, partition_columns)
        # Binding a target on the leaf options is what makes ``ParquetFile``
        # push ``columns=`` down to ``pq.ParquetFile`` — only the projected
        # column chunks are decoded off disk (lower bytes-read + peak RSS).
        # When no projection is requested ``leaf_proj`` is None ⇒ full read.
        leaf_opts = ParquetOptions.check(
            options=None, row_size=options.row_size,
            byte_size=options.byte_size, use_threads=options.use_threads,
            mode=Mode.READ_ONLY,
            target=leaf_proj,
        )

        candidate_adds = snap.prune_files(prune_values=prune)
        if options.predicate is not None:
            candidate_adds = _data_skip_adds(snap, candidate_adds, options.predicate)
        for add in candidate_adds:
            logger.debug(
                "DeltaFolder read: AddFile path=%s size=%d dv=%s",
                add.path, add.size, add.deletion_vector,
            )
            dv = decode_deletion_vector(add.deletion_vector, table_root=self.path,
                                        sidecar_cache=sidecar_cache)
            leaf = ParquetFile(holder=snap.resolve(add), owns_holder=False)
            base_offset = 0
            try:
                with leaf as opened:
                    for batch in opened._read_arrow_batches(leaf_opts):
                        # Databricks DBR auto-enables row tracking on
                        # DV tables, which both stamps internal
                        # ``_row-id-col-<uuid>`` /
                        # ``_row-commit-version-col-<uuid>`` columns
                        # onto post-feature AddFiles AND flips the
                        # nullability metadata on existing columns
                        # across files. Project to the table's
                        # canonical schema names AND re-wrap each
                        # batch with ``target_schema``'s field
                        # metadata so ``pa.Table.from_batches`` can
                        # concatenate cleanly across heterogeneous
                        # files.
                        if (target_schema is not None
                                and batch.schema != target_schema):
                            present = [
                                n for n in target_schema.names
                                if n in batch.schema.names
                            ]
                            if present:
                                batch = pa.record_batch(
                                    [batch.column(n) for n in present],
                                    schema=pa.schema([
                                        target_schema.field(n) for n in present
                                    ]),
                                )
                        masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                        base_offset += batch.num_rows
                        if masked.num_rows == 0:
                            continue
                        # Stamp partitions off the *full* schema so a
                        # projected-out partition column still gets its
                        # declared type (the narrowed ``target_schema`` no
                        # longer carries partition fields). The post-read cast
                        # drops any partition column the caller didn't ask for.
                        stamped = _stamp_partitions(masked, add.partition_values,
                                                    partition_columns, full_schema)
                        if row_filter is not None:
                            stamped = row_filter(stamped)
                            if stamped.num_rows == 0:
                                continue
                        yield stamped
            except FileNotFoundError:
                continue

    # ------------------------------------------------------------------
    # Polars — snapshot-self-contained, pruned, re-collectable lazy scan
    # ------------------------------------------------------------------

    def _scan_polars_frame(self, options: DeltaOptions) -> "Any":
        """Pure-lazy :class:`~polars.LazyFrame` that holds its own snapshot.

        Mirrors :meth:`Tabular._scan_polars_frame` (and the autonomous
        ``WarehouseStatementResult`` frame) but pins the resolved
        :class:`Snapshot` **once at scan-build time**:

        - The version is pinned to ``snap.version`` so the frame is a
          consistent, immutable view — a write that lands *after* the scan is
          built doesn't change what an earlier-built scan collects.
        - The deferred ``schema`` callable and the source generator both close
          over the captured ``snap``, never ``self``'s live state — so the
          frame can be returned, passed around, and collected on its own with
          no re-resolution of the ``_delta_log`` and no second snapshot
          listing per ``collect``.
        - Each collect streams the snapshot's AddFiles with column pushdown
          (``with_columns`` → Task A's leaf ``columns=``), predicate filter,
          and ``n_rows`` early-stop.

        ``is_pure=True`` is sound: a pinned version is immutable, so two reads
        of this frame inside one polars plan yield identical data.
        """
        from yggdrasil.lazy_imports import polars_module
        pl = polars_module()

        snap = self.snapshot(options.version)
        pinned = dataclasses.replace(options, version=snap.version)

        # Resolve the polars schema once from the pinned snapshot's
        # ``schema_string`` — no read, no live ``self`` lookup on collect.
        if snap.metadata is None or not snap.schema_string:
            from yggdrasil.data.schema import Schema
            pinned_schema = Schema.empty()
        else:
            pinned_schema = spark_json_to_schema(snap.schema_string)

        def schema() -> "Any":
            return pinned_schema.to_polars_schema()

        def source(with_columns, predicate, n_rows, batch_size):
            read_options = pinned
            if with_columns is not None:
                base = (pinned.target if pinned.target is not None
                        else pinned_schema)
                read_options = pinned.with_target(base.select(with_columns))
            remaining = n_rows
            for batch in self._read_snapshot_batches(snap, read_options):
                frame = pl.from_arrow(batch)
                if with_columns is not None:
                    frame = frame.select(with_columns)
                if predicate is not None:
                    frame = frame.filter(predicate)
                if remaining is not None:
                    if frame.height >= remaining:
                        yield frame.head(remaining)
                        return
                    remaining -= frame.height
                yield frame

        return pl.io.plugins.register_io_source(source, schema=schema, is_pure=True)

    def _read_polars_frame(self, options: DeltaOptions) -> "Any":
        # Eager read reuses the pruned Arrow path (Task A pushes the
        # projection into each parquet leaf) and ingests with ``rechunk=False``
        # so numeric columns stay zero-copy views over the Arrow buffers
        # instead of being memcpy'd into one contiguous chunk. Lower peak than
        # ``scan(...).collect()`` here: the IO-source plugin materializes a
        # polars frame *per batch* then concatenates, whereas ``from_arrow``
        # over the already-assembled table coalesces once.
        from yggdrasil.lazy_imports import polars_module
        return polars_module().from_arrow(
            self._read_arrow_table(options), rechunk=False,
        )

    # ==================================================================
    # Write
    # ==================================================================

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch],
                             options: DeltaOptions) -> None:
        mode = options.mode
        action = (Mode.OVERWRITE if mode in (Mode.OVERWRITE, Mode.TRUNCATE)
                  else Mode.UPSERT if mode in (Mode.UPSERT, Mode.MERGE)
                  else mode if mode in (Mode.IGNORE, Mode.ERROR_IF_EXISTS)
                  else Mode.APPEND)

        snap = self.snapshot(fresh=True)
        if action is Mode.IGNORE and snap.active_files:
            return
        if action is Mode.ERROR_IF_EXISTS and snap.active_files:
            raise FileExistsError(f"Delta table at {self.path!s} is non-empty; mode={options.mode!r}.")

        materialized: list[pa.RecordBatch] = list(batches)
        if not materialized and (action is not Mode.OVERWRITE or snap.metadata is None):
            return

        if action is Mode.UPSERT:
            self._commit_upsert(materialized, options=options, initial_snap=snap)
            return

        # Resolve schema + partitions
        is_initial = snap.metadata is None
        if is_initial:
            target_schema = materialized[0].schema if materialized else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        elif action is Mode.OVERWRITE and materialized:
            target_schema = materialized[0].schema
            partition_columns = list(self._infer_partition_columns(options)) or snap.partition_columns
        else:
            target_schema = (spark_json_to_arrow_schema(snap.schema_string)
                             if snap.schema_string
                             else (materialized[0].schema if materialized else pa.schema([])))
            partition_columns = snap.partition_columns

        new_adds = list(self._write_parts(iter(materialized),
                                          partition_columns=partition_columns,
                                          options=options)) if materialized else []

        def build(snap: Snapshot) -> "list[DeltaAction]":
            actions: list[DeltaAction] = []
            if snap.metadata is None:
                min_r, min_w = max(1, int(options.min_reader_version)), max(2, int(options.min_writer_version))
                rf, wf = [], []
                if options.checkpoint_kind == "v2":
                    min_r, min_w = max(min_r, 3), max(min_w, 7)
                    rf.append("v2Checkpoint"); wf.append("v2Checkpoint")
                if options.delete_via_dv:
                    min_r, min_w = max(min_r, 3), max(min_w, 7)
                    if "deletionVectors" not in rf: rf.append("deletionVectors")
                    if "deletionVectors" not in wf: wf.append("deletionVectors")
                cluster_cols = tuple(options.cluster_by or ())
                if cluster_cols:
                    # ``clustering`` is a writer-only feature gated on
                    # ``domainMetadata`` (where the cluster keys live) and
                    # reader/writer 3/7 — exactly what real Databricks
                    # stamps for ``CLUSTER BY``. Without these a 1/2 table
                    # can't legally carry a domainMetadata action.
                    min_r, min_w = max(min_r, 3), max(min_w, 7)
                    for feat in ("clustering", "domainMetadata"):
                        if feat not in wf: wf.append(feat)
                actions.append(Protocol(min_reader_version=min_r, min_writer_version=min_w,
                                        reader_features=rf, writer_features=wf))
                actions.append(Metadata(id=str(uuid.uuid4()),
                                        schema_string=arrow_schema_to_spark_json(target_schema),
                                        partition_columns=partition_columns,
                                        created_time=int(time.time() * 1000)))
                if cluster_cols:
                    # Mirror Databricks' physical shape: each key is its own
                    # single-element column path. ``Snapshot.clustering_columns``
                    # flattens these back to dotted names.
                    actions.append(DomainMetadata(
                        domain="delta.clustering",
                        configuration=ygg_json.dumps(
                            {"clusteringColumns": [[c] for c in cluster_cols],
                             "domainName": "delta.clustering"},
                            separators=(",", ":"), to_bytes=False),
                    ))
            elif action is Mode.OVERWRITE and materialized:
                actions.append(Metadata(id=snap.metadata.id,
                                        schema_string=arrow_schema_to_spark_json(target_schema),
                                        partition_columns=partition_columns,
                                        configuration=dict(snap.metadata.configuration),
                                        created_time=snap.metadata.created_time))
            if action is Mode.OVERWRITE:
                ts = int(time.time() * 1000)
                for path, add in snap.active_files.items():
                    actions.append(RemoveFile(path=path, deletion_timestamp=ts, data_change=True,
                                              extended_file_metadata=True,
                                              partition_values=dict(add.partition_values),
                                              size=int(add.size)))
            actions.extend(new_adds)
            if options.txn_app_id is not None and options.txn_version is not None:
                actions.append(Txn(app_id=options.txn_app_id, version=int(options.txn_version)))
            actions.append(self._build_commit_info(options=options, mode=action))
            return actions

        if action is Mode.OVERWRITE:
            plan = _CommitPlan(build_actions=build, is_blind_append=False,
                               read_file_paths=frozenset(snap.active_files.keys()),
                               removes_all=True)
        else:
            # APPEND / IGNORE / ERROR_IF_EXISTS that reach here only add
            # files — a blind append that commutes with any concurrent write.
            plan = _CommitPlan(build_actions=build, is_blind_append=True)
        self._with_commit_retry(build_actions=build, cleanup=None,
                                options=options, initial_snap=snap, plan=plan)

    def _commit_upsert(self, materialized: "list[pa.RecordBatch]", *,
                       options: DeltaOptions, initial_snap: Snapshot) -> None:
        match_by = list(options.match_by_keys or ())
        if not match_by:
            self._write_arrow_batches(materialized, dataclasses.replace(options, mode=Mode.APPEND))
            return

        is_initial = initial_snap.metadata is None
        if is_initial:
            target_schema = materialized[0].schema if materialized else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        else:
            target_schema = (spark_json_to_arrow_schema(initial_snap.schema_string)
                             if initial_snap.schema_string
                             else (materialized[0].schema if materialized else pa.schema([])))
            partition_columns = initial_snap.partition_columns

        incoming_adds = list(self._write_parts(iter(materialized),
                                               partition_columns=partition_columns,
                                               options=options)) if materialized else []
        incoming_keys = Folder._collect_keys_from_batches(materialized, match_by)
        rewrite_state: dict[str, list[AddFile]] = {"current": []}

        def build(snap: Snapshot) -> "list[DeltaAction]":
            removes, rewrites = [], []
            ts = int(time.time() * 1000)
            for add in list(snap.active_files.values()):
                matched, survivors = self._partition_file_for_keys(
                    snap.resolve(add), add=add, match_by=match_by, incoming_keys=incoming_keys)
                if not matched:
                    continue
                if survivors:
                    survivor_batches = self._read_indexed_batches(
                        leaf=ParquetFile(holder=snap.resolve(add), owns_holder=False),
                        indices=survivors, partition_columns=partition_columns,
                        partition_values=dict(add.partition_values))
                    rewrites.extend(self._write_parts(iter(survivor_batches),
                                                      partition_columns=partition_columns, options=options))
                removes.append(RemoveFile(path=add.path, deletion_timestamp=ts, data_change=True,
                                          extended_file_metadata=True, partition_values=dict(add.partition_values),
                                          size=int(add.size), deletion_vector=add.deletion_vector))
            rewrite_state["current"] = rewrites
            actions: list[DeltaAction] = []
            if snap.metadata is None:
                min_r, min_w = max(1, int(options.min_reader_version)), max(2, int(options.min_writer_version))
                actions.append(Protocol(min_reader_version=min_r, min_writer_version=min_w))
                actions.append(Metadata(id=str(uuid.uuid4()),
                                        schema_string=arrow_schema_to_spark_json(target_schema),
                                        partition_columns=partition_columns,
                                        created_time=int(time.time() * 1000)))
            actions.extend(removes)
            actions.extend(rewrites)
            actions.extend(incoming_adds)
            if options.txn_app_id is not None and options.txn_version is not None:
                actions.append(Txn(app_id=options.txn_app_id, version=int(options.txn_version)))
            actions.append(self._build_commit_info(
                options=options, mode=Mode.APPEND,
                operation_parameters={"mode": "upsert"},
                is_blind_append=False,
            ))
            return actions

        def cleanup() -> None:
            for add in rewrite_state["current"]:
                try: (self.path / add.path).unlink(missing_ok=True)
                except Exception: pass
            rewrite_state["current"] = []

        # UPSERT reads + rewrites the files matching the incoming keys. On a
        # collision the rebaser re-scans against the rival's new files (so a
        # key landing in a concurrently-added file is still matched) and only
        # fails if the rival removed a file we also rewrote.
        plan = _CommitPlan(
            build_actions=build, is_blind_append=False,
            read_file_paths=frozenset(initial_snap.active_files.keys()),
        )
        self._with_commit_retry(build_actions=build, cleanup=cleanup,
                                options=options, initial_snap=initial_snap, plan=plan)

    def _with_commit_retry(self, *, build_actions: "Callable[[Snapshot], list[DeltaAction]]",
                           cleanup: "Optional[Callable[[], None]]",
                           options: DeltaOptions, initial_snap: Snapshot,
                           plan: "Optional[_CommitPlan]" = None) -> None:
        """Optimistic-concurrency commit loop with rebase-on-conflict.

        Attempt 0 builds the action set against ``initial_snap`` and tries an
        atomic create at ``version+1``. If that version was taken by a
        concurrent writer we **rebase** instead of blindly redoing the write:

        - We read exactly the actions the rival committed between our base
          version and the new HEAD and ask Delta's conflict rules whether
          they commute with ours (:meth:`_rebase_actions`).
        - A *blind append* always commutes — concurrent appends are
          independent, so we keep the same already-written AddFiles and just
          renumber the commit at the new HEAD. No re-read, no rewrite.
        - A data-dependent op (UPSERT / OVERWRITE / DELETE) commutes only
          when the rival didn't touch the files we read; if it did, that's a
          genuine logical conflict and we raise straight away rather than
          burning the whole retry budget. When it *does* commute but our
          rewrite needs to see the rival's new files (it doesn't, for the
          rewrite-in-place ops we model), we rebuild against the new HEAD.

        ``ConcurrentDeltaCommitError`` is reserved for true logical
        conflicts and exhausted retries.

        When ``options.write_new_version`` is False the data files written by
        the caller are kept on disk but no commit is recorded — we return
        without bumping the version or touching the ``_delta_log``.
        """
        if not options.write_new_version:
            return

        max_retries = max(0, int(options.commit_max_retries or 0))
        backoff = float(options.commit_retry_backoff or 0.0)
        jitter = float(options.commit_retry_jitter or 0.0)
        max_delay = float(options.commit_retry_max_delay or 0.0)

        # Base snapshot + the actions we want to land, built once. ``plan`` is
        # None for the back-compat callers (delete path) — they fall back to
        # the rebuild-every-attempt behaviour.
        base_snap = initial_snap
        pending_actions: "Optional[list[DeltaAction]]" = (
            build_actions(base_snap) if plan is not None else None
        )

        for attempt in range(max_retries + 1):
            if attempt == 0:
                snap = base_snap
                actions = pending_actions if pending_actions is not None else build_actions(snap)
            elif plan is not None:
                # Rebase the already-built actions onto the fresh HEAD.
                snap = self.snapshot(fresh=True)
                actions = self._rebase_actions(
                    plan=plan, pending=pending_actions,  # type: ignore[arg-type]
                    base_snap=base_snap, head_snap=snap, cleanup=cleanup,
                    options=options,
                )
                if actions is None:
                    # Commutes but needs a rebuild against the new HEAD.
                    if cleanup is not None:
                        cleanup()
                    actions = build_actions(snap)
                    pending_actions = actions
                base_snap = snap
            else:
                snap = self.snapshot(fresh=True)
                actions = build_actions(snap)
            next_version = (snap.version + 1) if snap.metadata is not None else 0
            try:
                self._commit_atomic(next_version, actions)
            except FileExistsError:
                if plan is None and cleanup is not None:
                    cleanup()
                self._log.invalidate()
                self._snapshot = None
                if attempt == max_retries:
                    if plan is not None and cleanup is not None:
                        cleanup()
                    raise ConcurrentDeltaCommitError(
                        f"Failed to commit at {self.path!s} after {attempt + 1} attempts.")
                if attempt > 0 and backoff > 0:
                    delay = min(backoff * (2 ** (attempt - 1)), max_delay) if max_delay > 0 else backoff * (2 ** (attempt - 1))
                    if jitter > 0: delay += random.uniform(0, jitter)
                    time.sleep(delay)
                continue

            self._log.extend_listing(format_commit_name(next_version))
            self._snapshot = None
            interval = int(options.checkpoint_interval or 0)
            if interval > 0 and (next_version + 1) % interval == 0:
                snap = self.snapshot(next_version, fresh=True)
                result = write_checkpoint(snap, log_path=self._log.log_path, kind=options.checkpoint_kind)
                if result is not None:
                    size, sidecar_files = result
                    update_last_checkpoint(log_path=self._log.log_path, version=next_version,
                                           size=size, kind=options.checkpoint_kind,
                                           sidecar_files=sidecar_files)
            return

    def _winning_actions(self, base_version: int, head_version: int) -> "list[DeltaAction]":
        """Replay only the commits a rival landed in ``(base, head]``.

        These are the "winning" commits we lost the race to — we read their
        Add/Remove footprint to decide whether our pending commit still
        commutes. We replay each commit JSON individually (no checkpoint
        collapse) so we see the exact per-commit actions, not a merged view.
        """
        out: "list[DeltaAction]" = []
        for v in range(base_version + 1, head_version + 1):
            seg = LogSegment(version=v, checkpoint_version=v - 1,
                             checkpoint_files=(),
                             commit_files=(self._log.log_path / format_commit_name(v),))
            out.extend(self._log.replay(seg))
        return out

    def _rebase_actions(self, *, plan: "_CommitPlan", pending: "list[DeltaAction]",
                        base_snap: Snapshot, head_snap: Snapshot,
                        cleanup: "Optional[Callable[[], None]]",
                        options: DeltaOptions) -> "Optional[list[DeltaAction]]":
        """Decide how to land *pending* onto the advanced *head_snap*.

        Returns the (possibly unchanged) action list to commit at the new
        HEAD when our op commutes with the rival's, ``None`` to signal "needs
        a rebuild against the new HEAD", or raises
        :class:`ConcurrentDeltaCommitError` on a genuine logical conflict.

        Conflict rules follow the Delta optimistic-concurrency protocol:

        - *Blind append* (``is_blind_append``): always commutes. Concurrent
          appends are independent — neither reads nor removes the other's
          files — so we keep the same AddFiles and just commit at the new
          version. This is the hot path (N writers all appending).
        - *Overwrite* (``removes_all``): the operation's contract is "replace
          everything I saw". If a rival added or removed any file after our
          base, that contract is void → conflict.
        - *Rewrite-in-place* (UPSERT / non-DV DELETE, via
          ``read_file_paths``): conflicts only if the rival *removed* a file
          we also rewrote/removed (double-remove of the same file is a
          write-write conflict) or removed a file we read. Otherwise the
          rival only added new files we never inspected, so our rewrite is
          still valid against them and we commute — but our own RemoveFiles
          must still be present at HEAD, which they are (we only removed files
          that existed at base and the rival didn't touch), so we keep the
          actions as-is.
        """
        winning = self._winning_actions(base_snap.version, head_snap.version)
        rival_adds = {a.path for a in winning if isinstance(a, AddFile)}
        rival_removes = {a.path for a in winning if isinstance(a, RemoveFile)}

        if plan.is_blind_append:
            logger.debug(
                "DeltaFolder rebase at %r: blind append commutes past %d rival commit(s) "
                "(+%d/-%d files), landing at v%d",
                self.path, head_snap.version - base_snap.version,
                len(rival_adds), len(rival_removes), head_snap.version + 1,
            )
            return pending

        if plan.removes_all:
            if rival_adds or rival_removes:
                raise ConcurrentDeltaCommitError(
                    f"Concurrent commit at {self.path!s} changed files under an "
                    f"OVERWRITE (rival added {len(rival_adds)}, removed "
                    f"{len(rival_removes)}); the overwrite's snapshot is stale.",
                    conflict="overwrite-vs-concurrent-write",
                )
            return pending

        # Rewrite-in-place (UPSERT / non-DV DELETE). A rival removing a file
        # we also removed is an unambiguous write-write conflict — both
        # writers tried to supersede the same data, and committing ours would
        # silently lose the rival's edit (or double-remove). Abort.
        our_removes = {a.path for a in pending if isinstance(a, RemoveFile)}
        clash = (plan.read_file_paths & rival_removes) | (our_removes & rival_removes)
        if clash:
            raise ConcurrentDeltaCommitError(
                f"Concurrent commit at {self.path!s} removed {len(clash)} file(s) "
                f"this operation also rewrote/removed; the merge would lose a write.",
                conflict="rewrite-vs-concurrent-remove",
            )
        # The rival only *added* files (or removed files we never touched).
        # Those new files may contain keys our UPSERT must match, so we can't
        # blindly reuse the base-version action set — rebuild the rewrite
        # against the new HEAD (signalled by returning None). Cheaper than a
        # full redo only when there's no conflict, and correct.
        if rival_adds or rival_removes:
            logger.debug(
                "DeltaFolder rebase at %r: rewrite re-scans against %d new rival "
                "commit(s) (+%d/-%d files) before landing at v%d",
                self.path, head_snap.version - base_snap.version,
                len(rival_adds), len(rival_removes), head_snap.version + 1,
            )
            return None
        return pending

    # ==================================================================
    # Helpers
    # ==================================================================

    def _infer_partition_columns(self, options: DeltaOptions) -> "List[str]":
        target = options.target
        if target is None:
            return []
        return [f.name for f in getattr(target, "fields", ())
                if getattr(f, "_tag_flag", lambda _: False)(b"partition_by")]

    def _write_parts(self, batches: "Iterator[pa.RecordBatch]", *,
                     partition_columns: "List[str]",
                     options: DeltaOptions) -> "Iterator[AddFile]":
        # Bucket batches by partition key
        buckets: "dict[tuple, list[pa.RecordBatch]]" = {}
        for batch in batches:
            if batch.num_rows == 0:
                continue
            if not partition_columns:
                buckets.setdefault((), []).append(batch)
            elif not all(c in batch.schema.names for c in partition_columns):
                buckets.setdefault(tuple(None for _ in partition_columns), []).append(batch)
            else:
                table = pa.Table.from_batches([batch])
                cols = [table.column(c).to_pylist() for c in partition_columns]
                key_indices: "dict[tuple, list[int]]" = {}
                for row_idx, key in enumerate(zip(*cols)):
                    key_indices.setdefault(key, []).append(row_idx)
                for key, indices in key_indices.items():
                    sub = table.take(pa.array(indices, type=pa.int64())).combine_chunks()
                    for sb in sub.to_batches():
                        buckets.setdefault(key, []).append(sb)

        def _hq(v: Any) -> str:
            return urllib.parse.quote(str(v), safe="") if v is not None else "__HIVE_DEFAULT_PARTITION__"

        for key, sub_batches in buckets.items():
            kv = dict(zip(partition_columns, key))
            target_dir = self.path
            for col in partition_columns:
                target_dir = target_dir / f"{col}={_hq(kv[col])}"
            target_dir.mkdir(parents=True, exist_ok=True)

            stem = f"part-{int(time.time() * 1000)}-{os.urandom(8).hex()}.parquet"
            file_path = target_dir / stem

            # Strip partition columns + coerce to Delta-physical types.
            payload_batches: "list[pa.RecordBatch]" = []
            for sb in sub_batches:
                drop = [c for c in partition_columns if c in sb.schema.names]
                sb = sb.drop_columns(drop) if drop else sb
                payload_batches.append(_delta_physical_batch(sb))

            with ParquetFile(holder=file_path, owns_holder=False) as opened:
                opened._write_arrow_batches(payload_batches, ParquetOptions(mode=Mode.OVERWRITE))

            parts = [f"{col}={_hq(kv[col])}" for col in partition_columns]
            parts.append(stem)
            yield AddFile(
                path="/".join(parts),
                partition_values={c: (str(kv[c]) if kv[c] is not None else None) for c in partition_columns},
                size=int(file_path.size),
                modification_time=int(time.time() * 1000),
                data_change=True,
                stats=(_collect_stats(
                    payload_batches,
                    cluster_by=options.cluster_by,
                    num_indexed_cols=options.stats_num_indexed_cols,
                ) if getattr(options, "collect_stats", True) else None),
            )

    def _build_commit_info(
        self,
        *,
        options: "DeltaOptions",
        mode: "Mode",
        operation: "str | None" = None,
        operation_parameters: "dict[str, Any] | None" = None,
        is_blind_append: "bool | None" = None,
    ) -> CommitInfo:
        """Build a :class:`CommitInfo` action from the active options + mode.

        Centralizes the payload shape the three commit sites
        (write / upsert / delete) used to build inline so callers and
        tests have a single helper to reach for. ``operation`` /
        ``engineInfo`` / ``timestamp`` come from *options* and the
        process clock; ``operationParameters.mode`` and
        ``isBlindAppend`` derive from *mode* unless overridden.

        The explicit *operation* argument wins over ``options.operation``
        so callers that always want a fixed label (e.g. the DELETE
        retain path) don't need to mutate the returned payload.
        """
        params = operation_parameters
        if params is None:
            params = {"mode": mode.name.lower()}
        if is_blind_append is None:
            is_blind_append = mode is Mode.APPEND
        op_label = operation if operation is not None else (
            str(options.operation or "WRITE")
        )
        return CommitInfo(payload={
            "timestamp": int(time.time() * 1000),
            "operation": op_label,
            "operationParameters": params,
            "engineInfo": str(options.engine_info or "yggdrasil"),
            "isBlindAppend": bool(is_blind_append),
        })

    def _commit_atomic(self, version: int, actions: "Iterable[DeltaAction]") -> None:
        self._log.log_path.mkdir(parents=True, exist_ok=True)
        commit_path = self._log.log_path / format_commit_name(version)
        body = ("\n".join(ygg_json.dumps(a.to_action(), separators=(",", ":"),
                                         ensure_ascii=False, to_bytes=False)
                          for a in actions) + "\n").encode("utf-8")

        if getattr(commit_path, "is_local_path", False):
            fd = os.open(commit_path.full_path(), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                mv = memoryview(body)
                while mv:
                    n = os.write(fd, mv)
                    if n <= 0: raise OSError(f"os.write returned {n}")
                    mv = mv[n:]
            finally:
                os.close(fd)
            return

        # On S3, use a conditional create-if-absent (``If-None-Match: *``) so
        # the commit is genuinely atomic: two writers racing for the same
        # version both attempt the PUT, exactly one gets 200 and the other a
        # 412 → FileExistsError → rebase. A plain ``exists()``-then-``put``
        # has a TOCTOU window that loses writes under contention.
        http = getattr(getattr(commit_path, "s3_bucket", None), "http", None)
        key = getattr(commit_path, "key", None)
        if http is not None and key is not None:
            http.put(key, body, content_type="application/json", if_none_match=True)
            return

        if commit_path.exists():
            raise FileExistsError(f"Delta commit v{version} already exists at {commit_path!s}")
        with commit_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    # ==================================================================
    # Row-level delete
    # ==================================================================

    def _delete(
        self,
        predicate: "Predicate" = None,
        *,
        remove_path: bool = False,
        recursive: bool = True,
        files_only: bool = False,
        wait: Any = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        fresher_than: Any = None,
        older_than: Any = None,
        **kwargs: Any,
    ) -> int:
        # Path-removal mode (``remove`` / ``unlink``) physically deletes the
        # whole table directory — ``_delta_log`` and all — via Path; that's
        # distinct from a row delete / logical truncate (``predicate``-driven
        # RemoveFile commits, below), which keeps the table and its history.
        if remove_path:
            return super()._delete(
                remove_path=True, recursive=recursive, files_only=files_only,
                missing_ok=missing_ok, wait=wait,
                fresher_than=fresher_than, older_than=older_than,
            )

        options = self.check_options(kwargs.pop("options", None), **kwargs)
        snap = self.snapshot(fresh=True)
        if snap.metadata is None:
            return 0
        # A row delete / truncate only takes effect through a commit
        # (RemoveFile + survivor AddFile). With write_new_version disabled we
        # may not record one, so it's a no-op — return before rewriting any
        # survivor files.
        if not options.write_new_version:
            return 0

        sidecar_cache: dict[str, bytes] = {}
        ts = int(time.time() * 1000)
        new_actions: list[DeltaAction] = []
        deleted = 0

        for add_path, add in list(snap.active_files.items()):
            leaf = ParquetFile(holder=snap.resolve(add), owns_holder=False)
            existing_dv = decode_deletion_vector(add.deletion_vector, table_root=self.path,
                                                  sidecar_cache=sidecar_cache)
            survivors, file_deleted = self._partition_file_rows(
                leaf=leaf, predicate=predicate, existing_dv=existing_dv)
            if not file_deleted:
                continue

            deleted += len(file_deleted)
            pv = dict(add.partition_values)

            if options.delete_via_dv:
                prev = existing_dv.deleted_rows if existing_dv is not None else set()
                rows = sorted(set(file_deleted) | prev)
                dv = (encode_inline_deletion_vector(rows) if len(rows) <= _INLINE_DV_MAX_ROWS
                      else write_uuid_deletion_vector(rows, table_root=self.path))
                new_actions.append(RemoveFile(path=add_path, deletion_timestamp=ts, data_change=True,
                                              extended_file_metadata=True, partition_values=pv,
                                              size=int(add.size), deletion_vector=add.deletion_vector))
                new_actions.append(AddFile(path=add_path, partition_values=pv, size=int(add.size),
                                           modification_time=ts, data_change=True,
                                           stats=add.stats, deletion_vector=dv))
            else:
                fresh_adds = list(self._write_parts(
                    iter(self._read_indexed_batches(leaf=leaf, indices=survivors,
                                                    partition_columns=snap.partition_columns,
                                                    partition_values=pv)),
                    partition_columns=snap.partition_columns, options=options))
                new_actions.append(RemoveFile(path=add_path, deletion_timestamp=ts, data_change=True,
                                              extended_file_metadata=True, partition_values=pv,
                                              size=int(add.size)))
                new_actions.extend(fresh_adds)

        if not new_actions:
            return 0

        if options.delete_via_dv and snap.protocol is not None:
            if "deletionVectors" not in (snap.protocol.writer_features or []):
                new_actions.insert(0, Protocol(
                    min_reader_version=max(snap.protocol.min_reader_version, 3),
                    min_writer_version=max(snap.protocol.min_writer_version, 7),
                    reader_features=sorted({*(snap.protocol.reader_features or []), "deletionVectors"}),
                    writer_features=sorted({*(snap.protocol.writer_features or []), "deletionVectors"})))

        new_actions.append(self._build_commit_info(
            options=options, mode=Mode.OVERWRITE,
            operation="DELETE",
            operation_parameters={},
            is_blind_append=False,
        ))
        next_version = snap.version + 1
        self._commit_atomic(next_version, new_actions)
        self._log.extend_listing(format_commit_name(next_version))
        self._snapshot = None
        interval = int(options.checkpoint_interval or 0)
        if interval > 0 and (next_version + 1) % interval == 0:
            ck_snap = self.snapshot(next_version, fresh=True)
            result = write_checkpoint(ck_snap, log_path=self._log.log_path, kind=options.checkpoint_kind)
            if result is not None:
                update_last_checkpoint(log_path=self._log.log_path, version=next_version,
                                       size=result[0], kind=options.checkpoint_kind,
                                       sidecar_files=result[1])
        return deleted

    _ROW_INDEX_COL = "__yggdrasil_dv_row_index__"

    def _partition_file_rows(self, *, leaf: ParquetFile, predicate: "Predicate" = None,
                             existing_dv: "Optional[DeletionVector]") -> "tuple[list[int], list[int]]":
        already_masked = existing_dv.deleted_rows if existing_dv is not None else set()
        kept, all_visible, total = [], [], 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(ParquetOptions()):
                n = batch.num_rows
                if n == 0: continue
                visible = [i for i in range(n) if (total + i) not in already_masked]
                if not visible:
                    total += n; continue
                vis_table = pa.Table.from_batches([batch]).take(pa.array(visible, type=pa.int64()))
                idx_col = pa.array([total + i for i in visible], type=pa.int64())
                tagged = vis_table.append_column(self._ROW_INDEX_COL, idx_col)
                all_visible.extend(idx_col.to_pylist())
                # ``None`` predicate → no filter, every visible row matches.
                matched = tagged if predicate is None else predicate.filter_arrow_table(tagged)
                if matched.num_rows:
                    kept.extend(matched.column(self._ROW_INDEX_COL).to_pylist())
                total += n
        deleted_set = set(kept)
        return [i for i in all_visible if i not in deleted_set], [i for i in all_visible if i in deleted_set]

    def _partition_file_for_keys(self, file_path: "Any", *, add: AddFile,
                                 match_by: "List[str]", incoming_keys: "set[tuple]") -> "tuple[bool, list[int]]":
        existing_dv = decode_deletion_vector(add.deletion_vector, table_root=self.path)
        already_masked = existing_dv.deleted_rows if existing_dv is not None else set()
        leaf = ParquetFile(holder=file_path, owns_holder=False)
        survivors, matched, total = [], False, 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(ParquetOptions()):
                n = batch.num_rows
                if n == 0: continue
                if not all(c in batch.schema.names for c in match_by):
                    total += n; continue
                cols = [batch.column(c).to_pylist() for c in match_by]
                for i, key in enumerate(zip(*cols)):
                    abs_idx = total + i
                    if abs_idx in already_masked: continue
                    if key in incoming_keys: matched = True
                    else: survivors.append(abs_idx)
                total += n
        return matched, survivors

    def _read_indexed_batches(self, *, leaf: ParquetFile, indices: "List[int]",
                              partition_columns: "List[str]",
                              partition_values: "dict[str, Optional[str]]") -> "List[pa.RecordBatch]":
        if not indices: return []
        with leaf as opened:
            batches = list(opened._read_arrow_batches(ParquetOptions()))
        if not batches: return []
        sub = pa.Table.from_batches(batches).take(pa.array(indices, type=pa.int64()))
        return [_stamp_partitions(b, partition_values, partition_columns, None)
                for b in sub.to_batches() if b.num_rows > 0]

    # ==================================================================
    # Spark
    # ==================================================================

    def _read_spark_frame(self, options: DeltaOptions) -> "Any":
        import pickle
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(options.spark_session, create=True, import_error=True)
        # Pin the snapshot version so the distributed frame reads a consistent
        # view: the AddFiles + schema captured here, pickled per-file, make the
        # frame self-contained — the ``mapInArrow`` closure references only the
        # pickled payload (root + add + projection + partition info), never
        # ``self``.
        snap = self.snapshot(options.version)
        full_schema_obj = self._collect_schema(dataclasses.replace(options, version=snap.version))
        if snap.metadata is None or not snap.active_files:
            return spark.createDataFrame([], schema=full_schema_obj.to_spark_schema())
        full_schema = spark_json_to_arrow_schema(snap.schema_string) if snap.schema_string else None
        if full_schema is None:
            return spark.createDataFrame([], schema=full_schema_obj.to_spark_schema())
        partition_columns = snap.partition_columns
        partition_set = set(partition_columns)

        # Column pushdown (mirror of the Arrow read). ``read_columns()`` is the
        # projection ∪ predicate columns; partition columns are stamped, not
        # read off the leaf. Narrow both the per-leaf projection (pushed into
        # the closure's ``columns=``) and the result ``spark_schema``.
        wanted = options.read_columns()
        if wanted is not None:
            leaf_proj = [n for n in full_schema.names
                         if n in wanted and n not in partition_set]
            arrow_proj = pa.schema([full_schema.field(n) for n in leaf_proj]) if leaf_proj else None
            # The *output* columns are the projection only (``column_names``);
            # ``wanted`` (= ``read_columns()``) additionally carries predicate
            # columns that must be *read* but not returned. Drop those from the
            # result schema. The mapInArrow closure still yields the predicate
            # columns, so narrow each yielded batch down to ``output_cols``.
            output_cols = [n for n in full_schema.names
                           if options.column_names is None or n in options.column_names]
            spark_schema = full_schema_obj.select(output_cols).to_spark_schema()
            output_set = frozenset(output_cols)
        else:
            arrow_proj = None
            spark_schema = full_schema_obj.to_spark_schema()
            output_set = None

        prune = _partition_prune_values(options.predicate, partition_columns)
        candidate = snap.prune_files(prune_values=prune)
        if options.predicate is not None:
            candidate = _data_skip_adds(snap, candidate, options.predicate)
        active_adds = list(candidate)
        if not active_adds:
            return spark.createDataFrame([], schema=spark_schema)

        blobs = [pickle.dumps((self.path, add, partition_columns, full_schema, arrow_proj, output_set))
                 for add in active_adds]
        try: parallelism = max(spark.sparkContext.defaultParallelism, 1)
        except Exception: parallelism = 4
        leaf_df = spark.createDataFrame(
            pa.table({"_pkl": pa.array(blobs, type=pa.binary())}),
        ).coalesce(min(len(blobs), parallelism))

        def _read_delta_files(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            import pickle as _pkl
            from yggdrasil.io.parquet_file import ParquetFile as _PF, ParquetOptions as _PO
            from yggdrasil.io.delta.deletion_vector import decode_deletion_vector as _dv, mask_batch_with_dv as _mask
            from yggdrasil.enums import Mode as _M
            for batch in batches:
                for blob in batch.column("_pkl").to_pylist():
                    root, add, pcols, full_sch, proj, out_set = _pkl.loads(blob)
                    dv = _dv(add.deletion_vector, table_root=root)
                    # Bind the projection as the leaf options' target → only
                    # the requested column chunks are decoded off disk.
                    leaf_opts = _PO.check(options=None, mode=_M.READ_ONLY, target=proj)
                    base = 0
                    with _PF(holder=root / add.path, owns_holder=False) as f:
                        for rb in f._read_arrow_batches(leaf_opts):
                            m = _mask(rb, dv, base_offset=base); base += rb.num_rows
                            if m.num_rows == 0:
                                continue
                            stamped = _stamp_partitions(m, add.partition_values, pcols, full_sch)
                            # Drop predicate-only columns the projection didn't
                            # ask for so the batch matches ``spark_schema``
                            # (mapInArrow binds by position).
                            if out_set is not None:
                                keep = [n for n in stamped.schema.names if n in out_set]
                                if keep != stamped.schema.names:
                                    stamped = stamped.select(keep)
                            yield stamped
        return leaf_df.mapInArrow(_read_delta_files, schema=spark_schema)

    def _write_spark_frame(self, frame: "Any", options: DeltaOptions) -> None:
        for method in ("toArrow", "toArrowBatchIterator"):
            fn = getattr(frame, method, None)
            if not callable(fn): continue
            try:
                result = fn()
                self._write_arrow_batches(
                    result.to_batches() if isinstance(result, pa.Table) else list(result), options)
                return
            except Exception: continue
        self._write_arrow_batches(pa.Table.from_pandas(frame.toPandas()).to_batches(), options)

    def iter_children(self) -> "Iterator":
        snap = self.snapshot()
        for add in snap.active_files.values():
            yield ParquetFile(
                holder=snap.resolve(add),
                owns_holder=False,
                tabular_parent=self,
            )


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _stamp_partitions(batch: pa.RecordBatch, values: "dict[str, Optional[str]]",
                      columns: "List[str]", target_schema: "Optional[pa.Schema]") -> pa.RecordBatch:
    if not columns or not values:
        return batch
    existing = set(batch.schema.names)
    for col in columns:
        if col in existing: continue
        raw = values.get(col)
        arrow_type, target_field = pa.string(), None
        if target_schema is not None:
            idx = target_schema.get_field_index(col)
            if idx >= 0:
                target_field = target_schema.field(idx)
                arrow_type = target_field.type
        value = raw
        if raw is not None and raw != "":
            try: value = pa.scalar(raw).cast(arrow_type).as_py()
            except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError): pass
        else:
            value = None
        batch = batch.append_column(
            target_field if target_field is not None else pa.field(col, arrow_type, nullable=True),
            pa.array([value] * batch.num_rows, type=arrow_type))
    # Partition columns were appended at the end; restore the declared
    # ``target_schema`` order so positional consumers line up. Spark's
    # ``mapInArrow`` binds the yielded batch to its result schema *by
    # position*, so an out-of-order partition column would silently swap
    # data between columns (e.g. a partition ``region`` reading the ``val``
    # values). Name-based consumers (the Arrow read path) are unaffected.
    if target_schema is not None:
        ordered = [f.name for f in target_schema if f.name in set(batch.schema.names)]
        if ordered and ordered != batch.schema.names:
            batch = batch.select(ordered)
    return batch


def _delta_to_physical_type(t: pa.DataType) -> "Optional[pa.DataType]":
    """Map an Arrow type to the type Delta parquet stores it as, or ``None``
    when it already matches.

    Two invariants the Delta protocol + the Spark/Photon parquet reader
    require (and that the ``deltalake`` Rust writer enforces):

    - **Timestamps are microseconds.** Delta's ``timestamp`` /
      ``timestamp_ntz`` map to parquet ``TIMESTAMP(MICROS)``; a
      nanosecond (or second) unit makes Databricks' reader reject the
      file outright (``Unsupported time unit in Parquet TimestampType``).
      The zone is preserved — ``timestamp[ns, UTC]`` → ``timestamp[us, UTC]``.
    - **Unsigned integers don't exist in Spark.** Reinterpret them as the
      same-width signed type so the value round-trips bit-for-bit.
    """
    if pa.types.is_timestamp(t) and t.unit != "us":
        return pa.timestamp("us", tz=t.tz)
    if pa.types.is_unsigned_integer(t):
        return _SIGNED_FOR_UINT[t.bit_width]()
    return None


def _delta_physical_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Coerce *batch* into the physical types Delta parquet must store.

    No-op (returns *batch* unchanged) when every column already matches,
    so the common all-micros / signed case pays nothing.
    """
    targets = [_delta_to_physical_type(f.type) for f in batch.schema]
    if not any(targets):
        return batch
    arrays = [
        batch.column(i).cast(tgt, safe=False) if tgt is not None else batch.column(i)
        for i, tgt in enumerate(targets)
    ]
    fields = [
        pa.field(f.name, targets[i], nullable=f.nullable, metadata=f.metadata)
        if targets[i] is not None else f
        for i, f in enumerate(batch.schema)
    ]
    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


def _collect_stats(batches: "list[pa.RecordBatch]", *,
                   cluster_by: "Optional[Iterable[str]]" = None,
                   num_indexed_cols: int = 32) -> "Optional[str]":
    """Collect Delta data-skipping stats for *batches*.

    Databricks only indexes the leading ``delta.dataSkippingNumIndexedCols``
    columns (32 by default) — stat'ing a 500-column table in full bloats
    every commit for skips that rarely fire. We mirror that cap, *but*
    clustering columns are always stat'd even when they fall past the cut:
    liquid clustering co-locates rows by those columns, so a per-file
    min/max on a clustering column is tight (a file holds a small,
    contiguous slice of the key space) and a predicate on it excludes most
    files. A min/max on an arbitrary unclustered column is loose — the
    values are scattered across every file — so it rarely prunes anything.
    Spending the stat budget on clustering columns is where the skip win
    actually comes from.
    """
    if not batches: return None
    total_rows = sum(b.num_rows for b in batches)
    schema = batches[0].schema
    min_vals, max_vals, null_counts = {}, {}, {}

    cluster_set = {c for c in (cluster_by or ())}
    cap = max(0, int(num_indexed_cols))
    # Leading-N indexed columns plus every clustering column, regardless of
    # position. A clustering column past the cap still earns its stat.
    indexed_names: "set[str]" = set()
    for i, f in enumerate(schema):
        if i < cap or f.name in cluster_set:
            indexed_names.add(f.name)

    def _sv(val: Any, t: pa.DataType) -> Any:
        # Delta data-skipping compares stats *as JSON strings* against the
        # same-formatted value, so the format must match what Databricks /
        # Spark emit or pruning silently misfires (and cross-readers can
        # reject the table). Timestamps: ISO-8601 UTC, millisecond
        # precision, trailing ``Z`` for the instant (``timestamp``) type;
        # no ``Z`` for the wall-clock (``timestamp_ntz``) type. Dates:
        # ``yyyy-MM-dd``.
        if isinstance(val, datetime.datetime):
            aware = pa.types.is_timestamp(t) and t.tz is not None
            if aware:
                base = val.astimezone(datetime.timezone.utc).replace(tzinfo=None)
                return base.strftime("%Y-%m-%dT%H:%M:%S.") + f"{base.microsecond // 1000:03d}Z"
            naive = val.replace(tzinfo=None) if val.tzinfo is not None else val
            return naive.strftime("%Y-%m-%dT%H:%M:%S.") + f"{naive.microsecond // 1000:03d}"
        if isinstance(val, datetime.date):
            return val.isoformat()
        if isinstance(val, bytes): return val.hex()
        if isinstance(val, decimal.Decimal): return str(val)
        return val

    for field in schema:
        if field.name not in indexed_names:
            continue
        t = field.type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_string(t)
                or pa.types.is_large_string(t) or pa.types.is_date(t) or pa.types.is_timestamp(t)
                or pa.types.is_decimal(t) or pa.types.is_boolean(t)):
            continue
        col_min = col_max = None; col_nulls = 0
        for batch in batches:
            col = batch.column(field.name)
            col_nulls += col.null_count
            if col.null_count == len(col): continue
            try:
                mm = pc.min_max(col)
                v_min, v_max = mm["min"].as_py(), mm["max"].as_py()
                if v_min is not None and (col_min is None or v_min < col_min): col_min = v_min
                if v_max is not None and (col_max is None or v_max > col_max): col_max = v_max
            except Exception: continue
        if col_min is not None: min_vals[field.name] = _sv(col_min, t)
        if col_max is not None: max_vals[field.name] = _sv(col_max, t)
        null_counts[field.name] = col_nulls

    stats: dict[str, Any] = {"numRecords": total_rows}
    if min_vals: stats["minValues"] = min_vals
    if max_vals: stats["maxValues"] = max_vals
    if null_counts: stats["nullCount"] = null_counts
    # ``tightBounds`` tells readers the min/max are exact (computed over
    # live rows), not loosened by a deletion vector — Databricks always
    # stamps it, and skipping treats its absence conservatively.
    stats["tightBounds"] = True
    return ygg_json.dumps(stats, separators=(",", ":"), to_bytes=False)


def _partition_prune_values(predicate: "Predicate", partition_columns: "List[str]") -> "Optional[dict]":
    if predicate is None or not partition_columns: return None
    from yggdrasil.execution.expr import extract_partition_filters
    return extract_partition_filters(predicate, partition_columns) or None


# ---------------------------------------------------------------------------
# Data-skipping — drop files whose per-column stats can't satisfy a predicate
# ---------------------------------------------------------------------------

#: Column constraints we can prove against AddFile min/max stats. Maps a
#: column to a list of ``(op, value)`` lower/upper bounds. Only emitted for
#: leaf comparisons under a top-level conjunction — an ``OR`` anywhere on the
#: path makes the bound non-binding, so we drop it (conservative: keep file).
def _extract_range_constraints(predicate: "Predicate") -> "Optional[dict[str, list[tuple[str, Any]]]]":
    """Pull ``column <op> literal`` bounds out of a conjunctive predicate.

    Returns a ``{column: [(op, value), ...]}`` map of constraints joined by
    AND, or ``None`` when nothing useful could be extracted. Disjunctions,
    negations, and non-comparable shapes contribute no constraints (so the
    file is conservatively kept). Only ``=`` / ``<`` / ``<=`` / ``>`` /
    ``>=`` / ``IN`` / ``BETWEEN`` against a column on one side and a scalar
    literal on the other are understood.
    """
    from yggdrasil.execution.expr.nodes import (
        Between, Column, Comparison, InList, Literal, Logical,
    )
    from yggdrasil.execution.expr.operators import CompareOp, LogicalOp

    out: "dict[str, list[tuple[str, Any]]]" = {}

    def _add(col: str, op: str, value: Any) -> None:
        out.setdefault(col, []).append((op, value))

    def _walk(node: Any) -> None:
        if isinstance(node, Logical):
            # Only AND propagates bounds — under OR a file can match via the
            # other branch, so neither side's bound is binding.
            if node.op is LogicalOp.AND:
                for child in node.operands:
                    _walk(child)
            return
        if isinstance(node, Comparison):
            left, right = node.left, node.right
            if isinstance(left, Column) and isinstance(right, Literal):
                col, val, op = left.name, right.value, node.op
            elif isinstance(right, Column) and isinstance(left, Literal):
                # Flip the operator when the column is on the right.
                col, val = right.name, left.value
                op = {CompareOp.LT: CompareOp.GT, CompareOp.GT: CompareOp.LT,
                      CompareOp.LE: CompareOp.GE, CompareOp.GE: CompareOp.LE}.get(
                          node.op, node.op)
            else:
                return
            _add(col, op.value, val)
            return
        if isinstance(node, Between):
            if getattr(node, "negated", False):
                return  # NOT BETWEEN can't tighten a single file's bounds.
            target, lo, hi = node.target, node.low, node.high
            if (isinstance(target, Column) and isinstance(lo, Literal)
                    and isinstance(hi, Literal)):
                _add(target.name, ">=", lo.value)
                _add(target.name, "<=", hi.value)
            return
        if isinstance(node, InList):
            if getattr(node, "negated", False):
                return
            # ``InList.values`` are raw Python scalars, not Literal nodes.
            values = list(getattr(node, "values", ()) or ())
            target = node.target
            if isinstance(target, Column) and values:
                _add(target.name, "in", values)
            return
        # Unknown shape — contributes nothing.

    _walk(predicate)
    return out or None


def _bound_excludes(lo: Any, hi: Any, bounds: "list[tuple[str, Any]]") -> bool:
    """True when min/max ``[lo, hi]`` can't satisfy any of *bounds* (joined by AND).

    A single unsatisfiable bound is enough to exclude the file — the
    constraints are ANDed, so one provably-empty range kills the file.
    """
    for op, val in bounds:
        try:
            if op == "=":
                if val < lo or val > hi:
                    return True
            elif op == ">":
                if hi <= val:
                    return True
            elif op == ">=":
                if hi < val:
                    return True
            elif op == "<":
                if lo >= val:
                    return True
            elif op == "<=":
                if lo > val:
                    return True
            elif op == "in":
                if all(v < lo or v > hi for v in val):
                    return True
        except TypeError:
            # Mixed/incomparable types — keep the file.
            continue
    return False


def _stats_exclude_file(add: AddFile, constraints: "dict[str, list[tuple[str, Any]]]",
                        skippable: "frozenset[str]",
                        cluster_cols: "tuple[str, ...]" = ()) -> bool:
    """True when *add*'s per-column min/max stats prove it holds no matching row.

    Conservative: any missing stat / un-comparable value / unknown op leaves
    the file in. Only columns in *skippable* (numeric + string, where the
    JSON stat value compares directly to the predicate literal) participate.

    *cluster_cols* (in cluster-key order) are checked **first** and
    short-circuit: liquid clustering co-locates rows by those keys, so each
    file holds a small, contiguous slice of the clustering key-space and its
    min/max on a clustering column is tight. A predicate on a clustering
    column is therefore the one most likely to exclude a file, and proving it
    empty lets us skip without touching any other column's stats. Pruning on
    an arbitrary unclustered column rarely fires — its values are scattered
    across every file, so almost every file's [min, max] spans the literal.
    """
    if not add.stats:
        return False
    try:
        stats = ygg_json.loads(add.stats)
    except Exception:
        return False
    mins = stats.get("minValues") or {}
    maxs = stats.get("maxValues") or {}

    # Clustering keys first (in key order) — tightest bounds, most likely to
    # exclude, and short-circuiting saves work on the wide unclustered tail.
    ordered_cols: "list[str]" = []
    seen: "set[str]" = set()
    for col in cluster_cols:
        if col in constraints and col not in seen:
            ordered_cols.append(col); seen.add(col)
    for col in constraints:
        if col not in seen:
            ordered_cols.append(col); seen.add(col)

    for col in ordered_cols:
        if col not in skippable:
            continue
        lo = mins.get(col)
        hi = maxs.get(col)
        if lo is None or hi is None:
            continue
        if _bound_excludes(lo, hi, constraints[col]):
            return True
    return False


def _data_skip_adds(snap: Snapshot, adds: "Iterable[AddFile]",
                    predicate: "Predicate") -> "Iterator[AddFile]":
    """Yield only the files in *adds* whose stats don't exclude *predicate*.

    Skipping is restricted to numeric + string columns (the JSON stat value
    is directly order-comparable to the predicate literal). Partition columns
    are handled separately by :meth:`Snapshot.prune_files` and excluded here.
    """
    constraints = _extract_range_constraints(predicate)
    if not constraints:
        yield from adds
        return
    target_schema = (spark_json_to_arrow_schema(snap.schema_string)
                     if snap.schema_string else None)
    partition_columns = set(snap.partition_columns)
    cluster_cols = tuple(snap.clustering_columns)
    skippable: set[str] = set()
    if target_schema is not None:
        for f in target_schema:
            if f.name in partition_columns:
                continue
            if (pa.types.is_integer(f.type) or pa.types.is_floating(f.type)
                    or pa.types.is_string(f.type) or pa.types.is_large_string(f.type)
                    or pa.types.is_date(f.type) or pa.types.is_timestamp(f.type)
                    or pa.types.is_decimal(f.type)):
                skippable.add(f.name)
    frozen = frozenset(skippable)
    if not frozen:
        yield from adds
        return
    # Only carry clustering keys the predicate actually constrains and that we
    # can compare against stats — those drive the short-circuit in
    # ``_stats_exclude_file``.
    active_cluster = tuple(c for c in cluster_cols if c in constraints and c in frozen)
    kept = skipped = 0
    for add in adds:
        if _stats_exclude_file(add, constraints, frozen, active_cluster):
            skipped += 1
            continue
        kept += 1
        yield add
    if skipped:
        logger.debug(
            "DeltaFolder data-skipping at %r: kept %d file(s), skipped %d via stats "
            "(clustering keys in predicate: %r)",
            snap.table_root, kept, skipped, list(active_cluster),
        )
