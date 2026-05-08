"""DeltaIO — :class:`Folder` over a Delta Lake table.

The leaf orchestrates three subsystems already documented in this
package:

- :class:`yggdrasil.delta.log.DeltaLog` resolves the snapshot version
  and yields the action stream for a :class:`LogSegment`.
- :class:`yggdrasil.delta.snapshot.Snapshot` reduces the actions into
  the active file set, the metadata, the protocol, the txn map.
- :class:`yggdrasil.delta.deletion_vector` decodes per-file DV blobs
  and masks rows out of each :class:`pyarrow.RecordBatch`.

What changes vs :class:`Folder`
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
- **Checkpoints** fire automatically every
  :attr:`DeltaOptions.checkpoint_interval` commits — V1 by default
  (one parquet covering the full action set), opt-in V2 via
  ``checkpoint_version="v2"``.

Caching
-------

Every :class:`DeltaIO` carries a :class:`DeltaLog` instance whose
listing + ``_last_checkpoint`` reads are memoized. A read pass that
includes a schema collect, a row count, and a batch scan does
exactly **one** ``_delta_log`` listing and **one** ``_last_checkpoint``
fetch. :meth:`refresh` clears the cache when the caller knows the
table moved underneath them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, List, Optional, Tuple

import pyarrow as pa

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.io.bytes_io import BytesIO
from yggdrasil.io.nested.folder_io import Folder, FolderOptions
from yggdrasil.io.primitive.parquet_io import ParquetFile, ParquetOptions

from yggdrasil.delta.deletion_vector import (
    DeletionVector,
    decode_deletion_vector,
    mask_batch_with_dv,
)
from yggdrasil.delta.log import (
    DeltaLog,
    LOG_DIR_NAME,
    format_checkpoint_v1_name,
    format_checkpoint_v2_manifest_name,
    format_commit_name,
)
from yggdrasil.delta.protocol import (
    AddFile,
    CommitInfo,
    DeltaAction,
    Metadata,
    Protocol,
    RemoveFile,
)
from yggdrasil.delta.schema_codec import (
    arrow_schema_to_spark_json,
    spark_json_to_arrow_schema,
)
from yggdrasil.delta.snapshot import Snapshot

if TYPE_CHECKING:
    from yggdrasil.io.path import Path


__all__ = ["DeltaIO", "DeltaOptions"]


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


# ---------------------------------------------------------------------------
# DeltaIO
# ---------------------------------------------------------------------------


class DeltaIO(Folder):
    """:class:`Folder` over a Delta Lake table at a :class:`Path`."""

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
            data, path=path, tabular_parent=tabular_parent, **kwargs,
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
        self, version: "Optional[int]" = None, *, fresh: bool = False,
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
        self, options: DeltaOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow batches from the snapshot's active files.

        Pipeline:

        1. Resolve snapshot at ``options.version`` (HEAD by default).
        2. Filter active files via ``options.prune_values`` (partition
           pruning) before any parquet open.
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

        base_offset = 0
        with leaf as opened:
            for batch in opened._read_arrow_batches(leaf_options):
                masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                base_offset += batch.num_rows
                if masked.num_rows == 0:
                    continue
                yield self._stamp_partitions(
                    masked, add.partition_values, partition_columns, target_schema,
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
                    stream, partition_columns=partition_columns, options=options,
                )
            )

        # Build the action list for this commit.
        actions: list[DeltaAction] = []

        if is_initial:
            # Bump versions when DV / V2 checkpoint features are turned on.
            min_r, min_w, reader_features, writer_features = self._resolve_protocol(
                options, has_dv=False,
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
            from yggdrasil.delta.protocol import Txn
            actions.append(Txn(app_id=options.txn_app_id, version=int(options.txn_version)))

        actions.append(self._build_commit_info(options=options, mode=action))

        # Commit + maybe checkpoint.
        next_version = (snap.version + 1) if not is_initial else 0
        self._commit(next_version, actions)
        self.refresh()

        # Auto-checkpoint every N commits.
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
        self, options: DeltaOptions, *, has_dv: bool,
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
        if has_dv:
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
        self, *, options: DeltaOptions, mode: Mode,
    ) -> CommitInfo:
        return CommitInfo(payload={
            "timestamp": int(time.time() * 1000),
            "operation": str(options.operation or "WRITE"),
            "operationParameters": {"mode": mode.name.lower()},
            "engineInfo": str(options.engine_info or "yggdrasil"),
            "isBlindAppend": mode is Mode.APPEND,
        })

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

            leaf = ParquetFile(holder=file_path, owns_holder=False)
            with leaf as opened:
                opened._write_arrow_batches(
                    payload_batches,
                    ParquetOptions(mode=Mode.OVERWRITE),
                )

            size = int(file_path.size)
            relative = self._table_relative(file_path, partition_columns, kv, stem)
            yield AddFile(
                path=relative,
                partition_values={c: _str_or_none(kv[c]) for c in partition_columns},
                size=size,
                modification_time=int(time.time() * 1000),
                data_change=True,
            )

    def _table_relative(
        self,
        file_path: "Path",
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
            lines.append(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
        body = ("\n".join(lines) + "\n").encode("utf-8")

        with commit_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    def _write_checkpoint(self, version: int, *, kind: str = "v1") -> None:
        """Snapshot the table at *version* into a checkpoint file.

        V1 — a single ``.checkpoint.parquet`` with one row per action.
        V2 — a JSON manifest pointing at one or more sidecar parquets
        under ``_delta_log/_sidecars``. We currently emit a single
        sidecar; the manifest format is forward-compatible with
        multi-sidecar layouts.
        """
        snap = self.snapshot(version, fresh=True)
        actions: list[dict] = []
        if snap.protocol is not None:
            actions.append(snap.protocol.to_action())
        if snap.metadata is not None:
            actions.append(snap.metadata.to_action())
        for app, v in snap.txns.items():
            from yggdrasil.delta.protocol import Txn
            actions.append(Txn(app_id=app, version=v).to_action())
        for add in snap.active_files.values():
            actions.append(add.to_action())

        if not actions:
            return

        if kind == "v2":
            self._write_v2_checkpoint(version, actions)
        else:
            self._write_v1_checkpoint(version, actions)
        self._update_last_checkpoint(version, len(actions), kind=kind)

    def _write_v1_checkpoint(self, version: int, actions: List[dict]) -> None:
        ck_path = self._log.log_path / format_checkpoint_v1_name(version)
        table = _actions_to_arrow_table(actions)

        leaf = ParquetFile(holder=ck_path, owns_holder=False)
        with leaf as opened:
            opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))

    def _write_v2_checkpoint(self, version: int, actions: List[dict]) -> None:
        sidecar_uuid = uuid.uuid4().hex
        sidecar_dir = self._log.log_path / "_sidecars"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        sidecar_name = f"{sidecar_uuid}.parquet"
        sidecar_path = sidecar_dir / sidecar_name

        table = _actions_to_arrow_table(actions)
        leaf = ParquetFile(holder=sidecar_path, owns_holder=False)
        with leaf as opened:
            opened._write_arrow_table(table, ParquetOptions(mode=Mode.OVERWRITE))

        manifest_uuid = uuid.uuid4().hex
        manifest_path = self._log.log_path / format_checkpoint_v2_manifest_name(
            version, manifest_uuid,
        )
        manifest_lines = [
            json.dumps(
                {"sidecar": {
                    "path": sidecar_name,
                    "sizeInBytes": int(sidecar_path.size),
                    "modificationTime": int(time.time() * 1000),
                }},
                separators=(",", ":"),
            ),
            json.dumps(
                {"checkpointMetadata": {"version": int(version), "flavor": "v2"}},
                separators=(",", ":"),
            ),
        ]
        body = ("\n".join(manifest_lines) + "\n").encode("utf-8")
        with manifest_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    def _update_last_checkpoint(
        self, version: int, size: int, *, kind: str,
    ) -> None:
        """Refresh ``_last_checkpoint`` so readers find the new checkpoint
        without scanning the directory.
        """
        from yggdrasil.delta.log import LAST_CHECKPOINT_NAME

        payload: dict = {"version": int(version), "size": int(size)}
        if kind == "v2":
            payload["v2Checkpoint"] = {"version": int(version)}
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        ptr = self._log.log_path / LAST_CHECKPOINT_NAME
        with ptr.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    # ==================================================================
    # Folder surface — children = active files
    # ==================================================================

    def iter_children(self) -> "Iterator":
        """Yield one :class:`ParquetFile` per active file in the snapshot.

        Override of :meth:`Folder.iter_children`: we never list the
        physical folder. The snapshot is the source of truth.
        """
        snap = self.snapshot()
        for add in snap.active_files.values():
            file_path = snap.resolve(add)
            yield self.adopt_child(ParquetFile(holder=file_path, owns_holder=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chain(first: pa.RecordBatch, rest: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
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
    batch: pa.RecordBatch, partition_columns: "List[str]",
) -> "Iterator[tuple[tuple, pa.RecordBatch]]":
    if not partition_columns:
        yield ((), batch)
        return
    if not all(c in batch.schema.names for c in partition_columns):
        # Mismatch → emit whole batch under all-None keys; same fallback
        # YGGFolder uses so the row data isn't silently dropped.
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


def _actions_to_arrow_table(actions: "List[dict]") -> pa.Table:
    """Lay out checkpoint actions as one row per action.

    Each row has exactly one populated column (the action's key);
    the rest are null. Pyarrow infers the union schema from the rows.

    Empty nested dicts get pruned to ``None`` first — pyarrow's
    type-inference can't represent ``struct<>`` (zero-field struct)
    in parquet, so a Metadata action with ``"options": {}`` would
    blow up the writer. Dropping the empty value to ``None`` lets
    the type land as nullable, which parquet *can* write.
    """
    rows: list[dict] = []
    for entry in actions:
        if not entry:
            continue
        key = next(iter(entry))
        rows.append({key: _drop_empties(entry[key])})
    if not rows:
        return pa.table({})
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        k = next(iter(r))
        if k not in seen:
            seen.add(k)
            keys.append(k)
    flattened = [
        {k: (r[k] if k in r else None) for k in keys}
        for r in rows
    ]
    return pa.Table.from_pylist(flattened)


def _drop_empties(value: Any) -> Any:
    """Recursively prune empty ``dict`` / ``list`` values to ``None``.

    Pyarrow refuses to write a ``struct<>`` (zero-field struct) into
    parquet — and that's exactly what an empty ``{}`` infers to. We
    walk the action payload once, dropping empties, before handing it
    to :meth:`pa.Table.from_pylist`. Functional shape: input is JSON-
    safe (dict / list / scalars) and the output is the same shape
    with empty containers replaced by ``None``.
    """
    if isinstance(value, dict):
        cleaned = {k: _drop_empties(v) for k, v in value.items()}
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned or None
    if isinstance(value, list):
        cleaned_list = [_drop_empties(v) for v in value]
        cleaned_list = [v for v in cleaned_list if v is not None]
        return cleaned_list or None
    return value


