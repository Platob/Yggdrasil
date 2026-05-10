"""DeltaIO — :class:`FolderIO` over a Delta Lake table.

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

What changes vs :class:`FolderIO`
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

Every :class:`DeltaIO` carries a :class:`DeltaLog` instance whose
listing + ``_last_checkpoint`` reads are memoized. A read pass that
includes a schema collect, a row count, and a batch scan does
exactly **one** ``_delta_log`` listing and **one** ``_last_checkpoint``
fetch. :meth:`refresh` clears the cache when the caller knows the
table moved underneath them.

Engine bridges
--------------

:class:`DeltaIO` inherits :class:`FolderIO` -> :class:`Tabular`, so
``read_polars_frame`` / ``read_pandas_frame`` / ``read_spark_frame``
work without any per-engine plumbing here. The Arrow batch stream
:meth:`_read_arrow_batches` produces routes through
:mod:`yggdrasil.data.cast` for the engine the caller asked for —
reads go Arrow → engine, writes go engine → Arrow → DeltaIO. That
keeps a single code path for partition pruning, DV masking, and
checkpoint replay regardless of which engine the caller is using.
"""

from __future__ import annotations

import dataclasses
import os
import time
import uuid
from typing import Any, ClassVar, Iterable, Iterator, List, Optional

import pyarrow as pa

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.nested.folder_io import FolderIO, FolderOptions
from yggdrasil.io.primitive.parquet_io import ParquetIO, ParquetOptions
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

__all__ = ["DeltaIO", "DeltaOptions"]


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
    #: When ``True``, :meth:`DeltaIO._delete` marks rows via a deletion
    #: vector on the existing parquet rather than rewriting the file.
    #: Forces the protocol to declare the ``deletionVectors`` feature.
    delete_via_dv: bool = False


# ---------------------------------------------------------------------------
# DeltaIO
# ---------------------------------------------------------------------------


class DeltaIO(FolderIO):
    """:class:`FolderIO` over a Delta Lake table at a :class:`Path`."""

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
        return f"DeltaIO(path={self.path!r})"

    # ==================================================================
    # Cache control
    # ==================================================================

    def refresh(self) -> "DeltaIO":
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
        2. Filter active files via ``options.prune_values`` (partition
           pruning) before any parquet open.
        3. Read each parquet through :class:`ParquetIO` so codec /
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

        prune = options.prune_values
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
        leaf = ParquetIO(holder=file_path, owns_holder=False)

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

        base_offset = 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(leaf_options):
                masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                base_offset += batch.num_rows
                if masked.num_rows == 0:
                    continue
                yield self._stamp_partitions(
                    masked,
                    add.partition_values,
                    partition_columns,
                    target_schema,
                )

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
        - IGNORE — skip when the table already has at least one
          active file.
        - ERROR_IF_EXISTS — raise when the table already has at least
          one active file.

        On a brand-new table (no log directory yet), we write a
        ``protocol`` + ``metaData`` action ahead of the file ``add``s
        so the first commit is self-contained.
        """
        action = self._resolve_action(options.mode)

        snap = self.snapshot(fresh=True)
        is_initial = snap.metadata is None

        if action is Mode.IGNORE and snap.active_files:
            return
        if action is Mode.ERROR_IF_EXISTS and snap.active_files:
            raise FileExistsError(
                f"Delta table at {self.path!s} is non-empty; refusing to "
                f"write under mode={options.mode!r}."
            )

        # Materialize the batches once — we need to peek the first to
        # build the metadata action (schema) before minting parquet
        # parts, and we need to know the partition columns to route
        # each row into the correct directory.
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            # No data and OVERWRITE → still legal: drop the active set.
            if is_initial or action is not Mode.OVERWRITE:
                return

        # Resolve partition columns, target schema.
        if is_initial:
            target_schema = first.schema if first is not None else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        else:
            target_schema = (
                spark_json_to_arrow_schema(snap.schema_string)
                if snap.schema_string
                else (first.schema if first is not None else pa.schema([]))
            )
            partition_columns = snap.partition_columns

        # Write the parquet parts.
        new_adds: list[AddFile] = []
        if first is not None:
            stream = _chain(first, batch_iter)
            new_adds.extend(
                self._write_parts(
                    stream,
                    partition_columns=partition_columns,
                    options=options,
                )
            )

        # Build the action list for this commit.
        actions: list[DeltaAction] = []

        if is_initial:
            min_r, min_w, reader_features, writer_features = self._resolve_protocol(
                options,
                has_dv=False,
            )
            actions.append(
                Protocol(
                    min_reader_version=min_r,
                    min_writer_version=min_w,
                    reader_features=reader_features,
                    writer_features=writer_features,
                )
            )
            # Delta convention: schemaString covers *all* columns,
            # partition columns included; only the parquet payload
            # drops them.
            actions.append(
                Metadata(
                    id=str(uuid.uuid4()),
                    schema_string=arrow_schema_to_spark_json(target_schema),
                    partition_columns=partition_columns,
                    created_time=int(time.time() * 1000),
                )
            )

        if action in (Mode.OVERWRITE, Mode.TRUNCATE):
            actions.extend(self._removes_for_snapshot(snap))

        actions.extend(new_adds)

        if options.txn_app_id is not None and options.txn_version is not None:
            actions.append(
                Txn(app_id=options.txn_app_id, version=int(options.txn_version)),
            )

        actions.append(self._build_commit_info(options=options, mode=action))

        # Commit + maybe checkpoint.
        next_version = (snap.version + 1) if not is_initial else 0
        self._commit(next_version, actions)
        self.refresh()

        # Auto-checkpoint every N commits. ``next_version`` is the
        # version we just minted; checkpoint when the *next* commit
        # would land on a multiple of ``checkpoint_interval``.
        interval = int(options.checkpoint_interval or 0)
        if interval > 0 and (next_version + 1) % interval == 0:
            self._write_checkpoint(next_version, kind=options.checkpoint_kind)

    # ==================================================================
    # Helpers — write path
    # ==================================================================

    def _infer_partition_columns(self, options: DeltaOptions) -> "List[str]":
        """Pull partition columns from ``options.target_field`` tags."""
        target = options.target_field
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

            # Drop partition columns from the parquet payload.
            payload_batches: "list[pa.RecordBatch]" = []
            for sb in sub_batches:
                drop = [c for c in partition_columns if c in sb.schema.names]
                payload_batches.append(sb.drop_columns(drop) if drop else sb)

            leaf = ParquetIO(holder=file_path, owns_holder=False)
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

    def _commit(self, version: int, actions: "Iterable[DeltaAction]") -> None:
        """Write ``_delta_log/<version>.json``.

        Atomic by virtue of the underlying :class:`Path` — local
        writes go through a staging file rename (see
        :class:`LocalPath`), remote writes go through whatever
        atomic-create primitive the backend exposes. We don't try to
        coordinate concurrent writers here; that's a higher-level
        concern (Delta's own ``commit-coordinator`` story).
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

    def _delete(self, predicate: Any, options: DeltaOptions) -> int:
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
            leaf = ParquetIO(holder=file_path, owns_holder=False)
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
        self._commit(next_version, new_actions)
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
        leaf: ParquetIO,
        predicate: Any,
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

    def _read_indexed_batches(
        self,
        *,
        leaf: ParquetIO,
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
    # FolderIO surface — children = active files
    # ==================================================================

    def iter_children(self) -> "Iterator":
        """Yield one :class:`ParquetIO` per active file in the snapshot.

        Override of :meth:`FolderIO.iter_children`: we never list the
        physical folder. The snapshot is the source of truth.
        """
        snap = self.snapshot()
        for add in snap.active_files.values():
            file_path = snap.resolve(add)
            yield self.adopt_child(ParquetIO(holder=file_path, owns_holder=False))


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


def _split_batch(
    batch: pa.RecordBatch,
    partition_columns: "List[str]",
) -> "Iterator[tuple[tuple, pa.RecordBatch]]":
    if not partition_columns:
        yield ((), batch)
        return
    if not all(c in batch.schema.names for c in partition_columns):
        # Mismatch → emit whole batch under all-None keys; same fallback
        # YGGFolderIO uses so the row data isn't silently dropped.
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
