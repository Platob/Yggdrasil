"""DeltaFolder — :class:`FolderPath` over a Delta Lake table.

The leaf orchestrates four subsystems documented in this package:

- :class:`yggdrasil.io.nested.delta.log.DeltaLog` resolves the
  snapshot version and yields the action stream for a
  :class:`LogSegment`.
- :class:`yggdrasil.io.nested.delta.snapshot.Snapshot` reduces the
  actions into the active file set + metadata + protocol + txn map +
  domain metadata.
- :mod:`yggdrasil.io.nested.delta.deletion_vector` decodes per-file
  DV blobs (read path) and encodes new DVs (write path), masking
  rows out of each :class:`pyarrow.RecordBatch` for the read.
- :mod:`yggdrasil.io.nested.delta.checkpoint` writes V1 and V2
  checkpoints and updates ``_last_checkpoint``.

What changes vs :class:`FolderPath`
---------------------------------

- **Children** come from the snapshot, not :func:`Path.iterdir`. We
  never list the table root for parquet parts — Delta's whole point
  is that the log is authoritative, and listing the root on a remote
  store is the most expensive metadata round trip there is.
- **Reads** push partition pruning into the snapshot itself
  (``options.prune_values``) and pass the row-level
  ``options.predicate`` through to the leaf parquet's reader. DVs
  attached to an :class:`AddFile` decode lazily and mask rows on the
  way out.
- **Writes** mint a parquet under ``<root>/`` (or under
  ``<col>=<val>/`` when partitioned), then atomically commit the new
  ``add`` actions as ``<version>.json`` in ``_delta_log``. ``OVERWRITE``
  emits ``remove`` actions for every active file in the prior
  snapshot before the ``add``s; ``APPEND`` only emits ``add``s.
- **Deletes** by predicate either mark rows via a DV (when the
  table opts into the ``deletionVectors`` writer feature) or rewrite
  the affected parquet parts with the surviving rows (the no-DV
  default — semantically identical, just heavier on bytes).
- **Checkpoints** fire automatically every
  :attr:`DeltaOptions.checkpoint_interval` commits — V1 by default
  (one parquet covering the full action set), opt-in V2 via
  ``checkpoint_kind="v2"``.

Caching
-------

Every :class:`DeltaFolder` carries a :class:`DeltaLog` instance whose
listing + ``_last_checkpoint`` reads are memoized. A read pass that
includes a schema collect, a row count, and a batch scan does
exactly **one** ``_delta_log`` listing and **one** ``_last_checkpoint``
fetch. :meth:`refresh` clears the cache when the caller knows the
table moved underneath them.

Engine bridges
--------------

:class:`DeltaFolder` inherits :class:`FolderPath` -> :class:`Tabular`, so
``read_polars_frame`` / ``read_pandas_frame`` / ``read_spark_frame``
work without any per-engine plumbing here. The Arrow batch stream
:meth:`_read_arrow_batches` produces routes through
:mod:`yggdrasil.data.cast` for the engine the caller asked for —
reads go Arrow → engine, writes go engine → Arrow → DeltaFolder. That
keeps a single code path for partition pruning, DV masking, and
checkpoint replay regardless of which engine the caller is using.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, List, Optional

import pyarrow as pa

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.nested.folder_path import FolderPath, FolderOptions
from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import format_commit_name
from yggdrasil.io.nested.delta.checkpoint import (
    update_last_checkpoint,
    write_checkpoint,
)
from yggdrasil.io.nested.delta.deletion_vector import (
    DeletionVector,
    decode_deletion_vector,
    encode_inline_deletion_vector,
    mask_batch_with_dv,
    write_uuid_deletion_vector,
)
from yggdrasil.io.nested.delta.log import DeltaLog
from yggdrasil.io.nested.delta.protocol import (
    AddFile,
    CommitInfo,
    DeltaAction,
    DeletionVectorDescriptor,
    Metadata,
    Protocol,
    RemoveFile,
    Txn,
)
from yggdrasil.io.nested.delta.schema_codec import (
    arrow_schema_to_spark_json,
    spark_json_to_arrow_schema,
)
from yggdrasil.io.nested.delta.snapshot import Snapshot

if TYPE_CHECKING:
    from yggdrasil.execution.expr import Predicate

__all__ = ["ConcurrentDeltaCommitError", "DeltaFolder", "DeltaOptions"]


class ConcurrentDeltaCommitError(RuntimeError):
    """Raised when retries are exhausted on a Delta commit version race.

    A commit conflicts when version ``N`` is taken by another writer
    between the moment we resolve our snapshot and the moment we try
    to atomically create ``_delta_log/<N>.json``. The retry loop
    refreshes the log, re-bases the action set against the new HEAD,
    and tries the next version up to
    :attr:`DeltaOptions.commit_max_retries` times. This exception is
    surfaced when that budget is exhausted — the caller can either
    accept the loss (their write didn't land) or retry the whole
    operation themselves with a longer budget.
    """


# Inline DVs are cheap to read but bloat the JSON commit. Above this
# row count we spill to a UUID sidecar even when the caller hasn't
# asked for one explicitly. 4096 = one Roaring array container, which
# is also the threshold the Delta spec calls out for the "stay
# inline" cutoff in its DV writer guidance.
_INLINE_DV_MAX_ROWS = 4096


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DeltaOptions(FolderOptions):
    """:class:`FolderOptions` extended with Delta-specific knobs."""

    #: Pin the read to a specific version. ``None`` = HEAD.
    version: Optional[int] = None
    #: Number of commits between automatic checkpoints. Set to 0 to
    #: disable. The default (10) matches Spark's ``delta.checkpointInterval``.
    checkpoint_interval: int = 10
    #: ``"v1"`` (single parquet) or ``"v2"`` (manifest + sidecars).
    #: Snapshots can read either regardless of writer choice.
    checkpoint_kind: str = "v1"
    #: Operation name stamped into the ``commitInfo`` action.
    operation: str = "WRITE"
    #: Optional engine name written into ``commitInfo.engineInfo``.
    engine_info: str = "yggdrasil"
    #: Application id for idempotent-writes; pairs with ``txn_version``.
    txn_app_id: Optional[str] = None
    txn_version: Optional[int] = None
    #: Min reader/writer versions a fresh table writes into its
    #: ``protocol`` action. Bumped automatically when an enabled
    #: feature requires it (DV → reader 3 / writer 7).
    min_reader_version: int = 1
    min_writer_version: int = 2
    #: When ``True``, :meth:`DeltaFolder._delete` marks rows via a deletion
    #: vector on the existing parquet rather than rewriting the file.
    #: Forces the protocol to declare the ``deletionVectors`` feature.
    delete_via_dv: bool = False
    #: Maximum number of retries on a concurrent-commit version race.
    #: The commit JSON is created with ``O_EXCL`` (or the remote
    #: backend's equivalent) so a race fails fast with
    #: :class:`FileExistsError`; we then refresh the log, rebuild the
    #: action set against the new HEAD, and retry. Set to 0 to fail
    #: fast on the first conflict.
    commit_max_retries: int = 8
    #: Base sleep (seconds) before the first *backed-off* retry; the
    #: actual delay is ``base * 2**(attempt - 1) + jitter``. The first
    #: conflict retries immediately (no sleep) — the race is usually
    #: just two writers landing in the same millisecond, so an extra
    #: round trip is cheaper than a 50 ms idle. Subsequent retries
    #: back off exponentially to stop a tight contention loop from
    #: amplifying.
    commit_retry_backoff: float = 0.05
    #: Upper bound (seconds) on the per-attempt random jitter added to
    #: the backoff. Stays small so the retry pause stays predictable.
    commit_retry_jitter: float = 0.05
    #: Hard cap on the per-attempt backoff before jitter. Stops the
    #: exponential from blowing up under sustained contention — once
    #: the delay hits this ceiling, every further retry waits at most
    #: this long (still with jitter on top).
    commit_retry_max_delay: float = 1.0


# ---------------------------------------------------------------------------
# DeltaFolder
# ---------------------------------------------------------------------------


class DeltaFolder(FolderPath):
    """:class:`FolderPath` over a Delta Lake table at a :class:`Path`."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.DELTA_FOLDER

    __slots__ = ("_log", "_snapshot")

    @classmethod
    def options_class(cls):
        return DeltaOptions

    # ==================================================================
    # Construction
    # ==================================================================

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        tabular_parent: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            data,
            path=path,
            tabular_parent=tabular_parent,
            **kwargs,
        )
        self._log = DeltaLog(self.path)
        self._snapshot: "Optional[Snapshot]" = None

    def __repr__(self) -> str:
        return f"DeltaFolder(path={self.path!r})"

    # ==================================================================
    # Cache control
    # ==================================================================

    def refresh(self) -> "DeltaFolder":
        """Drop cached log + snapshot. Next read re-fetches everything."""
        self._log.invalidate()
        self._snapshot = None
        return self

    @property
    def log(self) -> DeltaLog:
        return self._log

    def snapshot(
        self,
        version: "Optional[int]" = None,
        *,
        fresh: bool = False,
    ) -> Snapshot:
        """Materialize the snapshot at *version* (or HEAD), with caching.

        Repeat calls collapse to one log walk. ``fresh=True`` forces a
        re-resolve — useful between writes when the caller knows a new
        version landed.

        Only the HEAD snapshot is cached: time-travel reads at an
        explicit version pin are rare, and conflating them with the
        HEAD slot would resurrect a stale snapshot the next time a
        caller asks for ``None``. The HEAD cache is invalidated by
        :meth:`refresh` (which the writer calls after every commit).
        """
        if version is not None:
            return Snapshot.from_log(self._log, version)
        if not fresh and self._snapshot is not None:
            return self._snapshot
        self._snapshot = Snapshot.from_log(self._log, None)
        return self._snapshot

    # ==================================================================
    # Schema introspection — cheap, just decodes the snapshot metadata
    # ==================================================================

    def _collect_schema(self, options: DeltaOptions):
        from yggdrasil.data.schema import Schema

        snap = self.snapshot(options.version)
        if snap.metadata is None or not snap.schema_string:
            return Schema.empty()
        arrow_schema = spark_json_to_arrow_schema(snap.schema_string)
        return Schema.from_arrow(arrow_schema)

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: DeltaOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow batches from the snapshot's active files.

        Pipeline:

        1. Resolve snapshot at ``options.version`` (HEAD by default).
        2. Filter active files via partition pruning before any
           parquet open. The accepted-value sets come from
           ``options.prune_values`` directly *and* from
           :func:`extract_partition_filters` walking
           ``options.predicate`` for the partition columns — so a
           caller who passes only a ``Predicate`` still gets file-
           level skipping for free.
        3. Read each parquet through :class:`ParquetFile` so codec /
           memory-map / native pushdown all work as usual.
        4. Mask rows with the file's :class:`DeletionVector` when one
           is present.
        5. Re-attach partition columns from :class:`AddFile.partition_values`
           (Hive convention — partition values aren't stored in the
           parquet payload).
        """
        snap = self.snapshot(options.version)
        if snap.metadata is None:
            return  # empty / uninitialized table

        partition_columns = snap.partition_columns
        target_schema = (
            spark_json_to_arrow_schema(snap.schema_string)
            if snap.schema_string
            else None
        )

        # Threading through one shared cache lets multiple files that
        # reference the same DV sidecar collapse to one window read.
        sidecar_cache: dict[str, bytes] = {}

        prune = _merge_prune_with_predicate(
            options.prune_values, options.predicate, partition_columns,
        )
        for add in snap.prune_files(prune_values=prune):
            try:
                yield from self._read_one_add(
                    add,
                    snap=snap,
                    options=options,
                    partition_columns=partition_columns,
                    target_schema=target_schema,
                    sidecar_cache=sidecar_cache,
                )
            except FileNotFoundError:
                # The active set referenced a parquet that vacuum
                # already pulled out from under us. Skip it rather
                # than crash the read — the snapshot is stale, the
                # caller should ``refresh()`` and retry.
                continue

    def _read_one_add(
        self,
        add: AddFile,
        *,
        snap: Snapshot,
        options: DeltaOptions,
        partition_columns: List[str],
        target_schema: "Optional[pa.Schema]",
        sidecar_cache: dict,
    ) -> Iterator[pa.RecordBatch]:
        file_path = snap.resolve(add)
        leaf = ParquetFile(holder=file_path, owns_holder=False)

        # Decode the file's DV once (shared sidecar cache amortizes
        # multiple files that point at the same sidecar window).
        dv: Optional[DeletionVector] = decode_deletion_vector(
            add.deletion_vector,
            table_root=self.path,
            sidecar_cache=sidecar_cache,
        )

        # Project ParquetOptions out of the DeltaOptions — the parquet
        # leaf only knows about its own knobs.
        leaf_options = ParquetOptions.check(
            options=None,
            row_size=options.row_size,
            byte_size=options.byte_size,
            use_threads=options.use_threads,
            predicate=options.predicate,
            mode=Mode.READ_ONLY,
        )

        # Row-level filter — applied after partition stamping so the
        # predicate can reference both partition columns (carried on
        # the AddFile, not in the parquet payload) and data columns.
        # The partition-pruning extractor already dropped files whose
        # partition value was rejected; this is the residual non-
        # partition filter (the ``id > 1`` in
        # ``region == "us" AND id > 1``) plus the partition predicate
        # re-run as a sanity pass.
        row_filter = _arrow_row_filter_for(
            options.predicate, partition_columns, target_schema,
        )

        base_offset = 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(leaf_options):
                masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                base_offset += batch.num_rows
                if masked.num_rows == 0:
                    continue
                stamped = self._stamp_partitions(
                    masked,
                    add.partition_values,
                    partition_columns,
                    target_schema,
                )
                if row_filter is not None:
                    stamped = row_filter(stamped)
                    if stamped.num_rows == 0:
                        continue
                yield stamped

    @staticmethod
    def _stamp_partitions(
        batch: pa.RecordBatch,
        values: "dict[str, Optional[str]]",
        columns: List[str],
        target_schema: "Optional[pa.Schema]",
    ) -> pa.RecordBatch:
        """Re-attach partition columns to a batch.

        Hive convention — Delta stores partition values *only* in the
        :class:`AddFile`, never in the parquet payload. We resurrect
        them with the dtype from ``target_schema`` so the rebuilt batch
        types correctly without an extra cast pass.
        """
        if not columns or not values:
            return batch
        existing = set(batch.schema.names)
        for col in columns:
            if col in existing:
                continue
            raw = values.get(col)
            arrow_type: pa.DataType = pa.string()
            target_field: "Optional[pa.Field]" = None
            if target_schema is not None:
                idx = target_schema.get_field_index(col)
                if idx >= 0:
                    target_field = target_schema.field(idx)
                    arrow_type = target_field.type
            value = _coerce_partition(raw, arrow_type)
            arr = pa.array([value] * batch.num_rows, type=arrow_type)
            arrow_field = (
                target_field
                if target_field is not None
                else pa.field(col, arrow_type, nullable=True)
            )
            batch = batch.append_column(arrow_field, arr)
        return batch

    # ==================================================================
    # Write path
    # ==================================================================

    def _resolve_action(self, mode: Mode) -> Mode:
        """Pick the disposition for a write call.

        Overrides :meth:`FolderPath._resolve_action` to keep
        :data:`Mode.UPSERT` / :data:`Mode.MERGE` as distinct actions —
        the parent collapses both to ``APPEND`` because plain folders
        don't have row-level identity, but Delta does (via
        ``options.match_by_keys``), and the merge path here re-writes
        the affected parquets so a key collision actually means
        "incoming wins."
        """
        if mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.UPSERT
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        # AUTO / APPEND / anything unrecognised → APPEND.
        return Mode.APPEND

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: DeltaOptions,
    ) -> None:
        """Persist a batch stream as a Delta commit.

        Disposition matrix:

        - OVERWRITE / TRUNCATE — emit a ``remove`` per active file from
          the prior snapshot, then ``add`` the freshly-written parquet
          parts. The new commit replaces the active set.
        - APPEND — emit only ``add``s. Existing files stay live.
        - UPSERT / MERGE — key-aware: ``options.match_by_keys`` names
          the columns to dedup on. Existing files that contain a
          matching row get rewritten with the survivors (rows whose
          keys aren't in the incoming set); incoming rows are written
          as fresh parts. Without keys it collapses to APPEND.
        - IGNORE — skip when the table already has at least one
          active file.
        - ERROR_IF_EXISTS — raise when the table already has at least
          one active file.

        On a brand-new table (no log directory yet), we write a
        ``protocol`` + ``metaData`` action ahead of the file ``add``s
        so the first commit is self-contained.

        Concurrent writers
        ------------------

        The commit JSON is created with exclusive-create semantics
        (``O_EXCL`` on local; the backend's equivalent on remote) so a
        version race fails fast with :class:`FileExistsError`. The
        retry loop refreshes the log, rebuilds the action set against
        the new HEAD, and tries the next available version, up to
        :attr:`DeltaOptions.commit_max_retries`. Pure APPEND retries
        are cheap (only the version number changes); OVERWRITE re-
        derives its ``remove`` set from the fresh snapshot; UPSERT
        re-runs the affected-file detection so a concurrent commit
        that touched the same file is correctly rebased over.
        """
        action = self._resolve_action(options.mode)

        snap = self.snapshot(fresh=True)

        if action is Mode.IGNORE and snap.active_files:
            return
        if action is Mode.ERROR_IF_EXISTS and snap.active_files:
            raise FileExistsError(
                f"Delta table at {self.path!s} is non-empty; refusing to "
                f"write under mode={options.mode!r}."
            )

        # Materialize the stream once — UPSERT needs to walk it twice
        # (once for keys, once for the incoming-data parquet write),
        # OVERWRITE/APPEND only needs it once but the materialization
        # cost is tiny next to the parquet write.
        materialized: list[pa.RecordBatch] = list(batches)

        # An empty OVERWRITE on a populated table is still meaningful
        # (drop the active set). Every other mode bails on no data.
        if not materialized and (action is not Mode.OVERWRITE or snap.metadata is None):
            return

        if action is Mode.UPSERT:
            self._commit_upsert(materialized, options=options, initial_snap=snap)
            return

        self._commit_simple(
            materialized,
            options=options,
            initial_snap=snap,
            action=action,
        )

    # ------------------------------------------------------------------
    # Simple (APPEND / OVERWRITE) commit path with retry
    # ------------------------------------------------------------------

    def _commit_simple(
        self,
        materialized: "list[pa.RecordBatch]",
        *,
        options: DeltaOptions,
        initial_snap: Snapshot,
        action: Mode,
    ) -> None:
        """Commit an APPEND or OVERWRITE with concurrency-safe retries.

        Parquet parts are written **once** up front (their filenames
        carry a uuid, so they can't collide with any concurrent
        writer). The retry loop only rebuilds the action list — for
        APPEND that's a no-op besides the version bump, for OVERWRITE
        the ``remove`` set is rederived against the fresh HEAD so a
        concurrent append that landed in the meantime is correctly
        replaced.
        """
        is_initial = initial_snap.metadata is None

        # Resolve schema + partitioning. On the first commit the
        # incoming batch's schema is the table's schema; on a follow-up
        # we trust the table's recorded ``schemaString``.
        if is_initial:
            target_schema = materialized[0].schema if materialized else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        else:
            target_schema = (
                spark_json_to_arrow_schema(initial_snap.schema_string)
                if initial_snap.schema_string
                else (materialized[0].schema if materialized else pa.schema([]))
            )
            partition_columns = initial_snap.partition_columns

        # Write the new parquet parts once. Each filename is uuid-
        # tagged, so the same set of adds can be committed at any
        # eventual version without conflict.
        new_adds: list[AddFile] = []
        if materialized:
            new_adds = list(
                self._write_parts(
                    iter(materialized),
                    partition_columns=partition_columns,
                    options=options,
                )
            )

        def build(snap: Snapshot) -> "list[DeltaAction]":
            actions: list[DeltaAction] = []
            if snap.metadata is None:
                actions.extend(
                    self._initial_protocol_metadata(
                        options=options,
                        target_schema=target_schema,
                        partition_columns=partition_columns,
                    )
                )
            if action is Mode.OVERWRITE:
                actions.extend(self._removes_for_snapshot(snap))
            actions.extend(new_adds)
            self._maybe_append_txn(actions, options)
            actions.append(self._build_commit_info(options=options, mode=action))
            return actions

        # Cleanup is a no-op for APPEND/OVERWRITE: the parquet parts
        # we wrote are valid for any version we end up landing at, so
        # we never delete them between attempts.
        self._with_commit_retry(
            build_actions=build,
            cleanup=None,
            options=options,
            initial_snap=initial_snap,
        )

    # ------------------------------------------------------------------
    # Upsert (key-aware merge) commit path with retry
    # ------------------------------------------------------------------

    def _commit_upsert(
        self,
        materialized: "list[pa.RecordBatch]",
        *,
        options: DeltaOptions,
        initial_snap: Snapshot,
    ) -> None:
        """Key-aware upsert: incoming rows win on key collision.

        Pipeline per attempt:

        1. Walk the active set, find files whose contents include a
           row whose key tuple appears in the incoming batches.
        2. Rewrite each affected file with the surviving rows (key
           NOT in incoming) into a fresh parquet, emit a ``remove``
           for the old AddFile.
        3. Append the incoming batches as a new parquet group.

        Without ``options.match_by_keys`` we can't tell rows apart by
        identity — fall back to plain APPEND so the call still does
        something useful.
        """
        match_by = list(options.match_by_keys or ())
        if not match_by:
            self._commit_simple(
                materialized,
                options=options,
                initial_snap=initial_snap,
                action=Mode.APPEND,
            )
            return

        is_initial = initial_snap.metadata is None
        if is_initial:
            target_schema = materialized[0].schema if materialized else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        else:
            target_schema = (
                spark_json_to_arrow_schema(initial_snap.schema_string)
                if initial_snap.schema_string
                else (materialized[0].schema if materialized else pa.schema([]))
            )
            partition_columns = initial_snap.partition_columns

        # The incoming-data parquets are deterministic (uuid-named) and
        # safe across retries — write once.
        incoming_adds: list[AddFile] = []
        if materialized:
            incoming_adds = list(
                self._write_parts(
                    iter(materialized),
                    partition_columns=partition_columns,
                    options=options,
                )
            )

        # Collect the set of incoming key tuples once. The set is
        # invariant across retries (we don't re-evaluate the predicate
        # against a different snapshot — we evaluate it against the
        # incoming rows, which don't change).
        incoming_keys = FolderPath._collect_keys_from_batches(materialized, match_by)

        # Each retry attempt may produce a different rewrite set if a
        # concurrent writer landed an add/remove in between. Track the
        # rewrites so we can clean them up before the next attempt.
        rewrite_state: dict[str, list[AddFile]] = {"current": []}

        def build(snap: Snapshot) -> "list[DeltaAction]":
            removes: list[RemoveFile] = []
            rewrites: list[AddFile] = []
            for add in list(snap.active_files.values()):
                file_path = snap.resolve(add)
                matched, survivors = self._partition_file_for_keys(
                    file_path,
                    add=add,
                    match_by=match_by,
                    incoming_keys=incoming_keys,
                )
                if not matched:
                    continue
                if survivors:
                    survivor_batches = self._read_indexed_batches(
                        leaf=ParquetFile(holder=file_path, owns_holder=False),
                        indices=survivors,
                        partition_columns=partition_columns,
                        partition_values=dict(add.partition_values),
                    )
                    fresh = list(
                        self._write_parts(
                            iter(survivor_batches),
                            partition_columns=partition_columns,
                            options=options,
                        )
                    )
                    rewrites.extend(fresh)
                removes.append(self._build_remove(add))

            # Stash for cleanup if the commit ends up retried.
            rewrite_state["current"] = rewrites

            actions: list[DeltaAction] = []
            if snap.metadata is None:
                actions.extend(
                    self._initial_protocol_metadata(
                        options=options,
                        target_schema=target_schema,
                        partition_columns=partition_columns,
                    )
                )
            actions.extend(removes)
            actions.extend(rewrites)
            actions.extend(incoming_adds)
            self._maybe_append_txn(actions, options)
            actions.append(self._build_commit_info(options=options, mode=Mode.UPSERT))
            return actions

        def cleanup() -> None:
            for add in rewrite_state["current"]:
                try:
                    (self.path / add.path).unlink(missing_ok=True)
                except Exception:
                    pass
            rewrite_state["current"] = []

        self._with_commit_retry(
            build_actions=build,
            cleanup=cleanup,
            options=options,
            initial_snap=initial_snap,
        )

    # ------------------------------------------------------------------
    # Generic commit-with-retry loop
    # ------------------------------------------------------------------

    def _with_commit_retry(
        self,
        *,
        build_actions: "Callable[[Snapshot], list[DeltaAction]]",
        cleanup: "Optional[Callable[[], None]]",
        options: DeltaOptions,
        initial_snap: Snapshot,
    ) -> None:
        """Drive a commit through the version-race retry budget.

        ``build_actions`` is called with the snapshot the next commit
        rebases onto — the first attempt sees ``initial_snap``,
        subsequent attempts see whatever fresh HEAD won the race. The
        callable must produce a complete action list for that snapshot
        (including ``protocol`` / ``metaData`` for an initial commit
        and ``commitInfo`` at the tail).

        ``cleanup`` (when provided) runs between failed attempts. The
        UPSERT path uses it to delete rewrite parquets that won't
        apply to the next snapshot; APPEND/OVERWRITE leave it as
        ``None`` because their adds are valid at any version.
        """
        max_retries = max(0, int(options.commit_max_retries or 0))
        backoff = float(options.commit_retry_backoff or 0.0)
        jitter = float(options.commit_retry_jitter or 0.0)
        max_delay = float(options.commit_retry_max_delay or 0.0)
        last_exc: "Optional[FileExistsError]" = None
        for attempt in range(max_retries + 1):
            snap = initial_snap if attempt == 0 else self.snapshot(fresh=True)
            actions = build_actions(snap)

            next_version = (snap.version + 1) if snap.metadata is not None else 0
            try:
                self._commit_atomic(next_version, actions)
            except FileExistsError as exc:
                last_exc = exc
                if cleanup is not None:
                    cleanup()
                self._log.invalidate()
                self._snapshot = None
                if attempt == max_retries:
                    raise ConcurrentDeltaCommitError(
                        f"Failed to commit Delta change at {self.path!s} "
                        f"after {attempt + 1} attempts; concurrent writers "
                        f"keep winning the version race. Increase "
                        f"DeltaOptions.commit_max_retries (currently "
                        f"{max_retries}) if your contention level is "
                        f"genuinely this high."
                    ) from last_exc
                # First conflict retries immediately — the race is
                # usually two writers landing in the same ms, the
                # winning version is already visible on the next log
                # listing, and an idle 50ms throws away the speedup
                # from refreshing the snapshot right away. Subsequent
                # retries back off exponentially, capped at
                # ``max_delay`` so sustained contention doesn't blow
                # the budget.
                if attempt == 0 or backoff <= 0:
                    delay = 0.0
                else:
                    delay = backoff * (2 ** (attempt - 1))
                    if max_delay > 0 and delay > max_delay:
                        delay = max_delay
                if jitter > 0:
                    delay += random.uniform(0, jitter)
                if delay > 0:
                    time.sleep(delay)
                continue

            self.refresh()

            interval = int(options.checkpoint_interval or 0)
            if interval > 0 and (next_version + 1) % interval == 0:
                self._write_checkpoint(next_version, kind=options.checkpoint_kind)
            return

    # ------------------------------------------------------------------
    # Action-list helpers shared by the simple + upsert paths
    # ------------------------------------------------------------------

    def _initial_protocol_metadata(
        self,
        *,
        options: DeltaOptions,
        target_schema: pa.Schema,
        partition_columns: "List[str]",
    ) -> "list[DeltaAction]":
        """Build the protocol + metaData pair for a brand-new table."""
        min_r, min_w, reader_features, writer_features = self._resolve_protocol(
            options,
            has_dv=False,
        )
        return [
            Protocol(
                min_reader_version=min_r,
                min_writer_version=min_w,
                reader_features=reader_features,
                writer_features=writer_features,
            ),
            # Delta convention: schemaString covers *all* columns,
            # partition columns included; only the parquet payload
            # drops them.
            Metadata(
                id=str(uuid.uuid4()),
                schema_string=arrow_schema_to_spark_json(target_schema),
                partition_columns=partition_columns,
                created_time=int(time.time() * 1000),
            ),
        ]

    def _maybe_append_txn(
        self,
        actions: "list[DeltaAction]",
        options: DeltaOptions,
    ) -> None:
        if options.txn_app_id is not None and options.txn_version is not None:
            actions.append(
                Txn(app_id=options.txn_app_id, version=int(options.txn_version)),
            )

    def _build_remove(
        self,
        add: AddFile,
        *,
        snap: "Optional[Snapshot]" = None,
    ) -> RemoveFile:
        ts = int(time.time() * 1000)
        return RemoveFile(
            path=add.path,
            deletion_timestamp=ts,
            data_change=True,
            extended_file_metadata=True,
            partition_values=dict(add.partition_values),
            size=int(add.size),
            deletion_vector=add.deletion_vector,
        )

    # ==================================================================
    # Helpers — write path
    # ==================================================================

    def _infer_partition_columns(self, options: DeltaOptions) -> "List[str]":
        """Pull partition columns from ``options.target`` tags."""
        target = options.target
        if target is None:
            return []
        names: list[str] = []
        for f in getattr(target, "fields", ()):
            try:
                if f._tag_flag(b"partition_by"):
                    names.append(f.name)
            except AttributeError:
                continue
        return names

    def _resolve_protocol(
        self,
        options: DeltaOptions,
        *,
        has_dv: bool,
    ) -> "tuple[int, int, list[str], list[str]]":
        """Pick a protocol shape for a fresh table.

        We don't enable any opt-in feature by default; callers turn
        them on explicitly via ``DeltaOptions.checkpoint_kind="v2"``
        (V2 checkpoint reader feature) or by writing files with DV
        descriptors (which would require ``deletionVectors`` writer
        feature).
        """
        min_r = max(1, int(options.min_reader_version))
        min_w = max(2, int(options.min_writer_version))
        reader_features: list[str] = []
        writer_features: list[str] = []
        if options.checkpoint_kind == "v2":
            min_r = max(min_r, 3)
            min_w = max(min_w, 7)
            reader_features.append("v2Checkpoint")
            writer_features.append("v2Checkpoint")
        if has_dv or options.delete_via_dv:
            min_r = max(min_r, 3)
            min_w = max(min_w, 7)
            if "deletionVectors" not in reader_features:
                reader_features.append("deletionVectors")
            if "deletionVectors" not in writer_features:
                writer_features.append("deletionVectors")
        return min_r, min_w, reader_features, writer_features

    def _removes_for_snapshot(self, snap: Snapshot) -> "Iterator[RemoveFile]":
        ts = int(time.time() * 1000)
        for path, add in snap.active_files.items():
            yield RemoveFile(
                path=path,
                deletion_timestamp=ts,
                data_change=True,
                extended_file_metadata=True,
                partition_values=dict(add.partition_values),
                size=int(add.size),
            )

    def _build_commit_info(
        self,
        *,
        options: DeltaOptions,
        mode: Mode,
    ) -> CommitInfo:
        return CommitInfo(
            payload={
                "timestamp": int(time.time() * 1000),
                "operation": str(options.operation or "WRITE"),
                "operationParameters": {"mode": mode.name.lower()},
                "engineInfo": str(options.engine_info or "yggdrasil"),
                "isBlindAppend": mode is Mode.APPEND,
            }
        )

    def _write_parts(
        self,
        batches: "Iterator[pa.RecordBatch]",
        *,
        partition_columns: "List[str]",
        options: DeltaOptions,
    ) -> "Iterator[AddFile]":
        """Spill *batches* into one parquet per partition group + emit Adds.

        Unpartitioned table → exactly one parquet under the table root.
        Partitioned table → one parquet per (col, val, …) tuple, written
        under ``<col>=<val>/.../``. Hive convention: partition columns
        are dropped from the parquet payload.

        Unsigned-integer columns are reinterpreted as their same-width
        signed counterpart on the way in (``uint8`` → ``int8`` via
        two's-complement, ``safe=False``). Delta has no native unsigned
        types, and the schema codec already declares the column as a
        signed primitive — the cast keeps the on-disk parquet payload
        in sync with the declared :attr:`schemaString` without widening
        storage. ``max(uint8)`` lands as ``-1``; reading the column
        back yields the same signed value, and a caller that needs
        the original unsigned interpretation casts ``int → uint`` at
        the same width with ``safe=False`` to recover it.
        """
        # Bucket every batch by its partition-tuple. Materializing once
        # is unavoidable when we need a parquet-per-bucket — pyarrow
        # parquet writers are file-scoped, not partition-aware.
        buckets: "dict[tuple, list[pa.RecordBatch]]" = {}
        for batch in batches:
            if batch.num_rows == 0:
                continue
            for key, sub in _split_batch(batch, partition_columns):
                buckets.setdefault(key, []).append(sub)

        for key, sub_batches in buckets.items():
            kv = dict(zip(partition_columns, key))
            target_dir = self.path
            for col in partition_columns:
                target_dir = target_dir / f"{col}={_quote(kv[col])}"
            target_dir.mkdir(parents=True, exist_ok=True)

            stem = f"part-{int(time.time() * 1000)}-{os.urandom(8).hex()}.parquet"
            file_path = target_dir / stem

            # Drop partition columns from the parquet payload, then flip
            # any unsigned-integer payload columns to their signed
            # counterparts.
            payload_batches: "list[pa.RecordBatch]" = []
            for sb in sub_batches:
                drop = [c for c in partition_columns if c in sb.schema.names]
                stripped = sb.drop_columns(drop) if drop else sb
                payload_batches.append(_reinterpret_unsigned_as_signed(stripped))

            leaf = ParquetFile(holder=file_path, owns_holder=False)
            with leaf as opened:
                opened._write_arrow_batches(
                    payload_batches,
                    ParquetOptions(mode=Mode.OVERWRITE),
                )

            size = int(file_path.size)
            relative = self._table_relative(partition_columns, kv, stem)
            yield AddFile(
                path=relative,
                partition_values={c: _str_or_none(kv[c]) for c in partition_columns},
                size=size,
                modification_time=int(time.time() * 1000),
                data_change=True,
            )

    @staticmethod
    def _table_relative(
        partition_columns: "List[str]",
        kv: "dict[str, Any]",
        stem: str,
    ) -> str:
        """Build the table-relative path Delta records in :class:`AddFile`."""
        parts: list[str] = []
        for col in partition_columns:
            parts.append(f"{col}={_quote(kv[col])}")
        parts.append(stem)
        return "/".join(parts)

    # ==================================================================
    # Commit + checkpoint
    # ==================================================================

    def _commit_atomic(
        self,
        version: int,
        actions: "Iterable[DeltaAction]",
    ) -> None:
        """Atomically write ``_delta_log/<version>.json``.

        Local paths use ``open(O_CREAT | O_EXCL)`` so two processes
        racing on the same version see exactly one winner — the loser
        gets :class:`FileExistsError`. Remote paths fall back to
        check-then-write; the race window is small (one network
        round trip) and the retry loop in :meth:`_with_commit_retry`
        absorbs the rare case where both writers slip through.

        Atomicity is the foundation of Delta's optimistic-concurrency
        story: readers see commits land in version order, and a writer
        that loses the race retries its action set at the next free
        version. Writes that *would* clobber an existing version
        always raise — never silently overwrite.
        """
        self._log.log_path.mkdir(parents=True, exist_ok=True)
        commit_path = self._log.log_path / format_commit_name(version)

        lines: list[str] = []
        for a in actions:
            payload = a.to_action()
            line = ygg_json.dumps(
                payload,
                separators=(",", ":"),
                ensure_ascii=False,
                to_bytes=False,
            )
            lines.append(line)
        body = ("\n".join(lines) + "\n").encode("utf-8")

        if getattr(commit_path, "is_local_path", False):
            full = commit_path.full_path()
            # ``O_EXCL`` is the POSIX atomic-create primitive — the
            # kernel guarantees nobody else can land on the same path
            # between our open() and the write. Linux + macOS both
            # honour it on local filesystems; a NFS mount older than
            # v3 might not, but Delta isn't really supported there
            # anyway.
            fd = os.open(full, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                # ``os.write`` may return short on huge buffers; loop
                # until the whole body is on disk so a partial write
                # can't leak past the close().
                offset = 0
                while offset < len(body):
                    written = os.write(fd, body[offset:])
                    if written <= 0:
                        raise OSError(
                            f"os.write returned {written} on "
                            f"{full!r}; refusing to land a torn "
                            f"commit JSON."
                        )
                    offset += written
            finally:
                os.close(fd)
            return

        # Remote path — the backend's :meth:`Path.open` doesn't
        # uniformly support exclusive create. Best-effort: probe for
        # an existing version, then write. The retry loop catches the
        # race window because the next snapshot read picks up the
        # competing commit.
        if commit_path.exists():
            raise FileExistsError(
                f"Delta commit at version {version} already exists at "
                f"{commit_path!s}; concurrent writer landed first."
            )
        with commit_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    def _write_checkpoint(self, version: int, *, kind: str = "v1") -> None:
        """Snapshot the table at *version* into a checkpoint file."""
        snap = self.snapshot(version, fresh=True)
        size = write_checkpoint(snap, log_path=self._log.log_path, kind=kind)
        if size is None:
            return
        update_last_checkpoint(
            log_path=self._log.log_path,
            version=version,
            size=size,
            kind=kind,
        )

    # ==================================================================
    # Row-level delete — DV-based or rewrite, picked from options
    # ==================================================================

    def _delete(self, predicate: "Predicate", options: DeltaOptions) -> int:
        """Drop rows matching *predicate* and emit a Delta commit.

        Two strategies, picked from ``options.delete_via_dv``:

        - ``False`` (default) — read each affected file, write a fresh
          parquet with the surviving rows, emit a ``remove`` for the
          old file + ``add`` for the new one. Heavy on bytes (a single
          deleted row triggers a full file rewrite) but works on every
          Delta protocol version.
        - ``True`` — keep the original parquet, emit a ``remove`` for
          the old AddFile, then re-``add`` the same path with a
          freshly-written deletion vector descriptor. Cheap on bytes;
          requires the ``deletionVectors`` writer feature, which we
          add to the protocol when the strategy is opted in.
        """
        snap = self.snapshot(fresh=True)
        if snap.metadata is None:
            return 0

        sidecar_cache: dict[str, bytes] = {}
        ts = int(time.time() * 1000)

        new_actions: list[DeltaAction] = []
        deleted = 0
        any_change = False

        for add_path, add in list(snap.active_files.items()):
            file_path = snap.resolve(add)
            leaf = ParquetFile(holder=file_path, owns_holder=False)
            existing_dv = decode_deletion_vector(
                add.deletion_vector,
                table_root=self.path,
                sidecar_cache=sidecar_cache,
            )

            survivors, file_deleted_rows = self._partition_file_rows(
                leaf=leaf,
                predicate=predicate,
                existing_dv=existing_dv,
            )

            if not file_deleted_rows:
                continue

            deleted += len(file_deleted_rows)
            any_change = True

            partition_columns = snap.partition_columns
            partition_values = dict(add.partition_values)

            if options.delete_via_dv:
                # The DV needs the union of the previously-masked rows
                # and the new ones the predicate matched — readers
                # apply a single DV per file, not a stack.
                prev_masked = (
                    existing_dv.deleted_rows if existing_dv is not None else set()
                )
                deleted_rows = sorted(set(file_deleted_rows) | prev_masked)
                dv = self._mint_dv(deleted_rows)

                new_actions.append(
                    RemoveFile(
                        path=add_path,
                        deletion_timestamp=ts,
                        data_change=True,
                        extended_file_metadata=True,
                        partition_values=partition_values,
                        size=int(add.size),
                        deletion_vector=add.deletion_vector,
                    )
                )
                new_actions.append(
                    AddFile(
                        path=add_path,
                        partition_values=partition_values,
                        size=int(add.size),
                        modification_time=ts,
                        data_change=True,
                        deletion_vector=dv,
                    )
                )
            else:
                # Rewrite the surviving rows into a fresh parquet, drop
                # the original.
                survivor_batches = self._read_indexed_batches(
                    leaf=leaf,
                    indices=survivors,
                    partition_columns=partition_columns,
                    partition_values=partition_values,
                )
                fresh_adds = list(
                    self._write_parts(
                        iter(survivor_batches),
                        partition_columns=partition_columns,
                        options=options,
                    )
                )
                new_actions.append(
                    RemoveFile(
                        path=add_path,
                        deletion_timestamp=ts,
                        data_change=True,
                        extended_file_metadata=True,
                        partition_values=partition_values,
                        size=int(add.size),
                    )
                )
                new_actions.extend(fresh_adds)

        if not any_change:
            return 0

        if options.delete_via_dv and snap.protocol is not None:
            need_dv = "deletionVectors" not in (snap.protocol.writer_features or [])
            if need_dv:
                new_proto = Protocol(
                    min_reader_version=max(snap.protocol.min_reader_version, 3),
                    min_writer_version=max(snap.protocol.min_writer_version, 7),
                    reader_features=sorted(
                        {
                            *(snap.protocol.reader_features or []),
                            "deletionVectors",
                        }
                    ),
                    writer_features=sorted(
                        {
                            *(snap.protocol.writer_features or []),
                            "deletionVectors",
                        }
                    ),
                )
                new_actions.insert(0, new_proto)

        new_actions.append(
            self._build_commit_info(options=options, mode=Mode.OVERWRITE),
        )

        next_version = snap.version + 1
        self._commit_atomic(next_version, new_actions)
        self.refresh()

        interval = int(options.checkpoint_interval or 0)
        if interval > 0 and (next_version + 1) % interval == 0:
            self._write_checkpoint(next_version, kind=options.checkpoint_kind)
        return deleted

    #: Reserved column name used internally to recover absolute row
    #: indices through a predicate filter. Picked to be unlikely to
    #: collide with a real column — Delta column names are typically
    #: lowercase tokens, this one is namespaced + dunder.
    _ROW_INDEX_COL = "__yggdrasil_dv_row_index__"

    def _partition_file_rows(
        self,
        *,
        leaf: ParquetFile,
        predicate: "Predicate",
        existing_dv: "Optional[DeletionVector]",
    ) -> "tuple[list[int], list[int]]":
        """Return ``(survivor_abs_indices, deleted_abs_indices)`` for a parquet.

        Tags every batch row with its absolute file index in a reserved
        column, applies the predicate's filter, and reads the
        surviving indices off the result. The complement (filtered
        from the full set) is the deleted list. Rows already masked
        by ``existing_dv`` are excluded from both buckets — they're
        already gone, the predicate doesn't see them.
        """
        already_masked = existing_dv.deleted_rows if existing_dv is not None else set()

        kept_indices: list[int] = []
        all_visible: list[int] = []
        total = 0

        with leaf as opened:
            for batch in opened._read_arrow_batches(ParquetOptions()):
                n = batch.num_rows
                if n == 0:
                    continue
                # Build the absolute-index column for this batch; mask
                # out rows that are already DV-deleted so the predicate
                # filter never considers them.
                visible_local = [
                    i for i in range(n) if (total + i) not in already_masked
                ]
                if not visible_local:
                    total += n
                    continue
                base_table = pa.Table.from_batches([batch])
                visible_table = base_table.take(
                    pa.array(visible_local, type=pa.int64()),
                )
                idx_col = pa.array(
                    [total + i for i in visible_local],
                    type=pa.int64(),
                )
                tagged = visible_table.append_column(
                    self._ROW_INDEX_COL,
                    idx_col,
                )
                all_visible.extend(idx_col.to_pylist())
                # ``predicate`` semantics (matches :meth:`Tabular.delete`):
                # rows where the predicate is True are the rows the
                # caller wants to delete. ``filter_arrow_table`` keeps
                # rows where the predicate is True — so the returned
                # rows ARE the deletion set, and the complement is
                # the survivor set.
                matched = predicate.filter_arrow_table(tagged)
                if matched.num_rows:
                    kept_indices.extend(matched.column(self._ROW_INDEX_COL).to_pylist())
                total += n

        deleted_set = set(kept_indices)
        survivors: list[int] = [i for i in all_visible if i not in deleted_set]
        deleted: list[int] = [i for i in all_visible if i in deleted_set]
        return survivors, deleted

    def _partition_file_for_keys(
        self,
        file_path: "Any",
        *,
        add: AddFile,
        match_by: "List[str]",
        incoming_keys: "set[tuple]",
    ) -> "tuple[bool, list[int]]":
        """Walk *file_path*, return ``(matched, survivor_indices)``.

        ``matched`` is True iff at least one row in the file (after
        applying any existing deletion vector) carries a key tuple
        that appears in ``incoming_keys``. ``survivor_indices`` is the
        absolute row indices that should *survive* the upsert — those
        whose keys aren't in the incoming set, and that aren't already
        masked by the file's existing DV. If ``matched`` is False the
        caller skips the file entirely (no remove + no rewrite).

        Files whose schema doesn't include every key column are
        treated as un-matchable — there's no row in such a file whose
        key could collide with an incoming row, so we leave them
        alone.
        """
        sidecar_cache: dict[str, bytes] = {}
        existing_dv = decode_deletion_vector(
            add.deletion_vector,
            table_root=self.path,
            sidecar_cache=sidecar_cache,
        )
        already_masked = existing_dv.deleted_rows if existing_dv is not None else set()

        leaf = ParquetFile(holder=file_path, owns_holder=False)
        survivors: list[int] = []
        matched = False
        total = 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(ParquetOptions()):
                n = batch.num_rows
                if n == 0:
                    continue
                if not all(c in batch.schema.names for c in match_by):
                    # Schema mismatch — partition column or rename
                    # drift. The file can't have a matching key, so
                    # skip it entirely (won't appear in removes).
                    total += n
                    continue
                cols = [batch.column(c).to_pylist() for c in match_by]
                for i, key in enumerate(zip(*cols)):
                    abs_idx = total + i
                    if abs_idx in already_masked:
                        continue
                    if key in incoming_keys:
                        matched = True
                    else:
                        survivors.append(abs_idx)
                total += n
        return matched, survivors

    def _read_indexed_batches(
        self,
        *,
        leaf: ParquetFile,
        indices: "List[int]",
        partition_columns: "List[str]",
        partition_values: "dict[str, Optional[str]]",
    ) -> "List[pa.RecordBatch]":
        """Read *indices* from *leaf*'s parquet, restamping partitions."""
        if not indices:
            return []
        with leaf as opened:
            batches = list(opened._read_arrow_batches(ParquetOptions()))
        if not batches:
            return []
        table = pa.Table.from_batches(batches)
        sub = table.take(pa.array(indices, type=pa.int64()))
        out: List[pa.RecordBatch] = []
        for batch in sub.to_batches():
            if batch.num_rows == 0:
                continue
            # Re-attach partition columns so the rewritten parquet
            # carries them — :meth:`_write_parts` immediately drops
            # them again before the parquet write, so the wire shape
            # matches the original.
            stamped = self._stamp_partitions(
                batch,
                partition_values,
                partition_columns,
                target_schema=None,
            )
            out.append(stamped)
        return out

    def _mint_dv(self, deleted_rows: "List[int]") -> DeletionVectorDescriptor:
        """Encode *deleted_rows* into the right DV storage shape.

        Tiny DVs go inline (Z85 in the action); larger ones spill to a
        UUID-named sidecar under the table root. The threshold matches
        the Delta DV writer guidance — 4096 rows = one Roaring array
        container.
        """
        if len(deleted_rows) <= _INLINE_DV_MAX_ROWS:
            return encode_inline_deletion_vector(deleted_rows)
        return write_uuid_deletion_vector(deleted_rows, table_root=self.path)

    # ==================================================================
    # FolderPath surface — children = active files
    # ==================================================================

    def iter_children(self) -> "Iterator":
        """Yield one :class:`ParquetFile` per active file in the snapshot.

        Override of :meth:`FolderPath.iter_children`: we never list the
        physical folder. The snapshot is the source of truth.
        """
        snap = self.snapshot()
        for add in snap.active_files.values():
            file_path = snap.resolve(add)
            yield self.adopt_child(ParquetFile(holder=file_path, owns_holder=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(
    first: pa.RecordBatch,
    rest: "Iterator[pa.RecordBatch]",
) -> "Iterator[pa.RecordBatch]":
    yield first
    yield from rest


def _quote(value: Any) -> str:
    """URL-quote a partition value Hive-style.

    Delta uses URL-encoded path fragments for partition values so
    ``/`` and ``=`` can appear in the data without breaking the
    directory hierarchy.
    """
    import urllib.parse

    if value is None:
        return "__HIVE_DEFAULT_PARTITION__"
    return urllib.parse.quote(str(value), safe="")


def _str_or_none(value: Any) -> "Optional[str]":
    if value is None:
        return None
    return str(value)


def _coerce_partition(raw: "Optional[str]", arrow_type: pa.DataType) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return pa.scalar(raw).cast(arrow_type).as_py()
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        return raw


#: pyarrow's :func:`pa.types.is_uintN` family doesn't expose a direct
#: ``unsigned → signed`` shortcut, so we route through ``bit_width`` to
#: pick the matching signed factory. The dict is small enough to inline,
#: but extracting it keeps the cast loop readable and lints clean.
_SIGNED_FOR_UINT_BITS = {
    8: pa.int8,
    16: pa.int16,
    32: pa.int32,
    64: pa.int64,
}


def _reinterpret_unsigned_as_signed(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Cast every unsigned-integer column to its same-width signed type.

    ``safe=False`` triggers pyarrow's two's-complement reinterpretation:
    the underlying bit pattern survives, only the type changes. So
    ``uint8(255)`` arrives as ``int8(-1)``, ``uint64(2**63)`` as
    ``int64(-2**63)``, etc. Pass-through when the batch holds no
    unsigned columns — the common case for a freshly-built Arrow
    table from a database read.
    """
    if not any(pa.types.is_unsigned_integer(f.type) for f in batch.schema):
        return batch

    new_arrays: list[pa.Array] = []
    new_fields: list[pa.Field] = []
    for i, field in enumerate(batch.schema):
        if pa.types.is_unsigned_integer(field.type):
            signed = _SIGNED_FOR_UINT_BITS[field.type.bit_width]()
            new_arrays.append(batch.column(i).cast(signed, safe=False))
            new_fields.append(
                pa.field(
                    field.name,
                    signed,
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            new_arrays.append(batch.column(i))
            new_fields.append(field)
    return pa.RecordBatch.from_arrays(new_arrays, schema=pa.schema(new_fields))


def _arrow_row_filter_for(
    predicate: "Predicate",
    partition_columns: "List[str]",
    target_schema: "Optional[pa.Schema]",
) -> "Optional[Callable[[pa.RecordBatch], pa.RecordBatch]]":
    """Compile ``predicate`` to a per-batch pyarrow filter, or ``None``.

    Returns ``None`` (caller skips filtering) when:

    - ``predicate`` is ``None`` — nothing to filter.
    - One of the predicate's free columns is missing from the
      stamped batch schema (partition columns + ``target_schema``
      columns). Matches the documented contract on
      :attr:`CastOptions.predicate`: "missing inputs can't yield a
      coherent boolean, and the alternative ('drop everything') is
      almost always wrong for heterogeneous-source folders."

    Otherwise compiles the predicate to a
    :class:`pyarrow.compute.Expression` once and returns a closure
    that runs the C++ filter on each :class:`pa.RecordBatch`.
    Compiled once per :meth:`_read_arrow_batches` invocation so the
    per-batch work is just the kernel call.
    """
    if predicate is None:
        return None
    try:
        from yggdrasil.execution.expr import free_columns
    except ImportError:
        return None

    referenced = set(free_columns(predicate))
    if target_schema is not None:
        available = set(target_schema.names) | set(partition_columns)
    else:
        available = set(partition_columns)
    if not referenced.issubset(available):
        # Predicate touches a column the snapshot's schema doesn't
        # carry. Degrade to "accept everything" rather than raise —
        # heterogeneous-source folders need that contract.
        return None

    try:
        arrow_expr = predicate.to_arrow()
    except Exception:
        # Predicate compiles to a shape pyarrow can't lift (rare:
        # custom node types, unsupported dtype). Degrade gracefully.
        return None

    import pyarrow.dataset as pds

    def _filter(batch: "pa.RecordBatch") -> "pa.RecordBatch":
        if batch.num_rows == 0:
            return batch
        filtered = pds.dataset(pa.Table.from_batches([batch])).to_table(
            filter=arrow_expr,
        )
        if filtered.num_rows == 0:
            return pa.RecordBatch.from_pylist([], schema=batch.schema)
        # ``combine_chunks`` re-materialises into a single chunk so
        # ``to_batches`` yields exactly one batch — saves the caller
        # from re-stitching when filter survival is partial.
        rebuilt = filtered.combine_chunks().to_batches()
        return rebuilt[0] if rebuilt else pa.RecordBatch.from_pylist(
            [], schema=batch.schema,
        )

    return _filter


def _merge_prune_with_predicate(
    explicit: "Optional[dict]",
    predicate: "Predicate",
    partition_columns: "List[str]",
) -> "Optional[dict]":
    """Combine caller-supplied ``prune_values`` with predicate-extracted hints.

    The result is what :meth:`Snapshot.prune_files` consumes — a
    ``Mapping[str, Iterable]`` of accepted values per partition
    column, or ``None`` when nothing constrains the file set.
    Sources are AND'd: a file matches iff its partition value lies
    in *both* the explicit set (when given) and the predicate's
    extracted set (when extractable).

    Predicate extraction routes through
    :func:`yggdrasil.execution.expr.extract_partition_filters`,
    which over-approximates and only reports columns it can pin to
    a finite set — comparisons, ``IN`` lists, ``IS NULL``, and their
    ``AND`` / ``OR`` composition. Ranges, ``NOT``, and arithmetic
    return no constraint for those columns — the row-level filter
    still runs on every surviving file, so the soundness contract
    is preserved.
    """
    if predicate is None or not partition_columns:
        return explicit
    from yggdrasil.execution.expr import extract_partition_filters

    derived = extract_partition_filters(predicate, partition_columns)
    if not derived:
        return explicit
    if not explicit:
        # ``prune_files`` accepts any ``Mapping[str, Iterable]`` —
        # frozensets satisfy that contract directly.
        return derived
    # Intersect column-by-column. Columns constrained on only one
    # side keep that side's set.
    merged: dict = dict(explicit)
    for col_name, derived_set in derived.items():
        if col_name in merged:
            existing = frozenset(merged[col_name])
            merged[col_name] = existing & derived_set
        else:
            merged[col_name] = derived_set
    return merged


def _split_batch(
    batch: pa.RecordBatch,
    partition_columns: "List[str]",
) -> "Iterator[tuple[tuple, pa.RecordBatch]]":
    if not partition_columns:
        yield ((), batch)
        return
    if not all(c in batch.schema.names for c in partition_columns):
        # Mismatch → emit whole batch under all-None keys so the row
        # data isn't silently dropped.
        yield (tuple(None for _ in partition_columns), batch)
        return
    table = pa.Table.from_batches([batch])
    cols = [table.column(c).to_pylist() for c in partition_columns]
    keys: "dict[tuple, list[int]]" = {}
    for row_idx, key in enumerate(zip(*cols)):
        keys.setdefault(key, []).append(row_idx)
    for key, indices in keys.items():
        sub = table.take(pa.array(indices, type=pa.int64())).combine_chunks()
        for sub_batch in sub.to_batches():
            yield key, sub_batch
