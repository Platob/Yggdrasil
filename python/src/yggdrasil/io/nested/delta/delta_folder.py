"""DeltaFolder — :class:`Folder` over a Delta Lake table.

Reworked implementation with full Delta read/write protocol support:

- V1 and V2 checkpoint read/write
- Deletion vectors (inline, UUID sidecar, absolute path)
- Roaring bitmap encode/decode for DVs
- Per-file stats collection (numRecords, minValues, maxValues)
- Concurrent commit with exponential backoff
- APPEND / OVERWRITE / UPSERT / MERGE / IGNORE / ERROR_IF_EXISTS
- Partition pruning from predicates
- Row-level delete via DV or rewrite
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, List, Optional

import pyarrow as pa

from yggdrasil.enums import MimeTypes, Mode
from yggdrasil.path.folder import Folder, FolderOptions
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
    """Raised when retries are exhausted on a Delta commit version race."""


_INLINE_DV_MAX_ROWS = 4096


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DeltaOptions(FolderOptions):
    """:class:`FolderOptions` extended with Delta-specific knobs."""

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
    commit_max_retries: int = 8
    commit_retry_backoff: float = 0.05
    commit_retry_jitter: float = 0.05
    commit_retry_max_delay: float = 1.0
    collect_stats: bool = True
    target_file_size: int = 128 * 1024 * 1024  # 128 MB target per parquet


# ---------------------------------------------------------------------------
# DeltaFolder
# ---------------------------------------------------------------------------


class DeltaFolder(Folder):
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
        if version is not None:
            return Snapshot.from_log(self._log, version)
        if not fresh and self._snapshot is not None:
            return self._snapshot
        self._snapshot = Snapshot.from_log(self._log, None)
        return self._snapshot

    # ==================================================================
    # Schema introspection
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
        snap = self.snapshot(options.version)
        if snap.metadata is None:
            return

        partition_columns = snap.partition_columns
        target_schema = (
            spark_json_to_arrow_schema(snap.schema_string)
            if snap.schema_string
            else None
        )

        sidecar_cache: dict[str, bytes] = {}

        prune = _extract_partition_prune_values(
            options.predicate, partition_columns,
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

        dv: Optional[DeletionVector] = decode_deletion_vector(
            add.deletion_vector,
            table_root=self.path,
            sidecar_cache=sidecar_cache,
        )

        leaf_options = ParquetOptions.check(
            options=None,
            row_size=options.row_size,
            byte_size=options.byte_size,
            use_threads=options.use_threads,
            mode=Mode.READ_ONLY,
        )

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
        if mode is Mode.OVERWRITE or mode is Mode.TRUNCATE:
            return Mode.OVERWRITE
        if mode is Mode.UPSERT or mode is Mode.MERGE:
            return Mode.UPSERT
        if mode is Mode.IGNORE:
            return Mode.IGNORE
        if mode is Mode.ERROR_IF_EXISTS:
            return Mode.ERROR_IF_EXISTS
        return Mode.APPEND

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: DeltaOptions,
    ) -> None:
        action = self._resolve_action(options.mode)

        snap = self.snapshot(fresh=True)

        if action is Mode.IGNORE and snap.active_files:
            return
        if action is Mode.ERROR_IF_EXISTS and snap.active_files:
            raise FileExistsError(
                f"Delta table at {self.path!s} is non-empty; refusing to "
                f"write under mode={options.mode!r}."
            )

        materialized: list[pa.RecordBatch] = list(batches)

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
        is_initial = initial_snap.metadata is None

        if is_initial:
            target_schema = materialized[0].schema if materialized else pa.schema([])
            partition_columns = list(self._infer_partition_columns(options))
        elif action is Mode.OVERWRITE and materialized:
            # OVERWRITE replaces the table contents and may change the
            # schema. Use the incoming data's schema so the new metaData
            # action reflects the actual columns on disk.
            target_schema = materialized[0].schema
            partition_columns = (
                list(self._infer_partition_columns(options))
                or initial_snap.partition_columns
            )
        else:
            target_schema = (
                spark_json_to_arrow_schema(initial_snap.schema_string)
                if initial_snap.schema_string
                else (materialized[0].schema if materialized else pa.schema([]))
            )
            partition_columns = initial_snap.partition_columns

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
            elif action is Mode.OVERWRITE and materialized:
                # OVERWRITE replaces the active set. Re-emit metaData
                # with the incoming schema so the recorded schemaString
                # matches what the new parquet files contain.
                actions.append(
                    Metadata(
                        id=snap.metadata.id,
                        schema_string=arrow_schema_to_spark_json(target_schema),
                        partition_columns=partition_columns,
                        configuration=dict(snap.metadata.configuration),
                        created_time=snap.metadata.created_time,
                    )
                )
            if action is Mode.OVERWRITE:
                actions.extend(self._removes_for_snapshot(snap))
            actions.extend(new_adds)
            self._maybe_append_txn(actions, options)
            actions.append(self._build_commit_info(options=options, mode=action))
            return actions

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

        incoming_adds: list[AddFile] = []
        if materialized:
            incoming_adds = list(
                self._write_parts(
                    iter(materialized),
                    partition_columns=partition_columns,
                    options=options,
                )
            )

        incoming_keys = Folder._collect_keys_from_batches(materialized, match_by)

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
    # Action-list helpers
    # ------------------------------------------------------------------

    def _initial_protocol_metadata(
        self,
        *,
        options: DeltaOptions,
        target_schema: pa.Schema,
        partition_columns: "List[str]",
    ) -> "list[DeltaAction]":
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

    def _collect_file_stats(self, batches: "list[pa.RecordBatch]") -> Optional[str]:
        """Collect per-file stats for AddFile.stats (numRecords, minValues, maxValues)."""
        if not batches:
            return None
        total_rows = sum(b.num_rows for b in batches)
        schema = batches[0].schema

        min_vals: dict[str, Any] = {}
        max_vals: dict[str, Any] = {}
        null_counts: dict[str, int] = {}

        for field in schema:
            col_name = field.name
            if not _is_stats_eligible(field.type):
                continue
            col_min = None
            col_max = None
            col_nulls = 0
            for batch in batches:
                col = batch.column(col_name)
                col_nulls += col.null_count
                if col.null_count == len(col):
                    continue
                import pyarrow.compute as pc
                try:
                    batch_min = pc.min(col).as_py()
                    batch_max = pc.max(col).as_py()
                    if batch_min is not None:
                        if col_min is None or batch_min < col_min:
                            col_min = batch_min
                    if batch_max is not None:
                        if col_max is None or batch_max > col_max:
                            col_max = batch_max
                except Exception:
                    continue
            if col_min is not None:
                min_vals[col_name] = _stats_value(col_min)
            if col_max is not None:
                max_vals[col_name] = _stats_value(col_max)
            null_counts[col_name] = col_nulls

        stats: dict[str, Any] = {"numRecords": total_rows}
        if min_vals:
            stats["minValues"] = min_vals
        if max_vals:
            stats["maxValues"] = max_vals
        if null_counts:
            stats["nullCount"] = null_counts
        return ygg_json.dumps(stats, separators=(",", ":"), to_bytes=False)

    def _write_parts(
        self,
        batches: "Iterator[pa.RecordBatch]",
        *,
        partition_columns: "List[str]",
        options: DeltaOptions,
    ) -> "Iterator[AddFile]":
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

            stats_json = None
            if getattr(options, "collect_stats", True):
                stats_json = self._collect_file_stats(payload_batches)

            size = int(file_path.size)
            relative = self._table_relative(partition_columns, kv, stem)
            yield AddFile(
                path=relative,
                partition_values={c: _str_or_none(kv[c]) for c in partition_columns},
                size=size,
                modification_time=int(time.time() * 1000),
                data_change=True,
                stats=stats_json,
            )

    @staticmethod
    def _table_relative(
        partition_columns: "List[str]",
        kv: "dict[str, Any]",
        stem: str,
    ) -> str:
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
            fd = os.open(full, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
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

        if commit_path.exists():
            raise FileExistsError(
                f"Delta commit at version {version} already exists at "
                f"{commit_path!s}; concurrent writer landed first."
            )
        with commit_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

    def _write_checkpoint(self, version: int, *, kind: str = "v1") -> None:
        snap = self.snapshot(version, fresh=True)
        result = write_checkpoint(snap, log_path=self._log.log_path, kind=kind)
        if result is None:
            return
        size, sidecar_files = result
        update_last_checkpoint(
            log_path=self._log.log_path,
            version=version,
            size=size,
            kind=kind,
            sidecar_files=sidecar_files,
        )

    # ==================================================================
    # Row-level delete — DV-based or rewrite
    # ==================================================================

    def _delete(self, predicate: "Predicate", options: DeltaOptions) -> int:
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
                        stats=add.stats,
                        deletion_vector=dv,
                    )
                )
            else:
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

    _ROW_INDEX_COL = "__yggdrasil_dv_row_index__"

    def _partition_file_rows(
        self,
        *,
        leaf: ParquetFile,
        predicate: "Predicate",
        existing_dv: "Optional[DeletionVector]",
    ) -> "tuple[list[int], list[int]]":
        already_masked = existing_dv.deleted_rows if existing_dv is not None else set()

        kept_indices: list[int] = []
        all_visible: list[int] = []
        total = 0

        with leaf as opened:
            for batch in opened._read_arrow_batches(ParquetOptions()):
                n = batch.num_rows
                if n == 0:
                    continue
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
            stamped = self._stamp_partitions(
                batch,
                partition_values,
                partition_columns,
                target_schema=None,
            )
            out.append(stamped)
        return out

    def _mint_dv(self, deleted_rows: "List[int]") -> DeletionVectorDescriptor:
        if len(deleted_rows) <= _INLINE_DV_MAX_ROWS:
            return encode_inline_deletion_vector(deleted_rows)
        return write_uuid_deletion_vector(deleted_rows, table_root=self.path)

    # ==================================================================
    # Folder surface — children = active files
    # ==================================================================

    def iter_children(self) -> "Iterator":
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


_SIGNED_FOR_UINT_BITS = {
    8: pa.int8,
    16: pa.int16,
    32: pa.int32,
    64: pa.int64,
}


def _reinterpret_unsigned_as_signed(batch: pa.RecordBatch) -> pa.RecordBatch:
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


def _is_stats_eligible(arrow_type: pa.DataType) -> bool:
    """Check if a column type is eligible for min/max stats collection."""
    return (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_date(arrow_type)
        or pa.types.is_timestamp(arrow_type)
        or pa.types.is_decimal(arrow_type)
        or pa.types.is_boolean(arrow_type)
    )


def _stats_value(val: Any) -> Any:
    """Coerce a stats value to a JSON-safe type."""
    import datetime
    import decimal
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.isoformat()
    if isinstance(val, bytes):
        return val.hex()
    if isinstance(val, decimal.Decimal):
        return str(val)
    return val


def _arrow_row_filter_for(
    predicate: "Predicate",
    partition_columns: "List[str]",
    target_schema: "Optional[pa.Schema]",
) -> "Optional[Callable[[pa.RecordBatch], pa.RecordBatch]]":
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
        return None

    try:
        arrow_expr = predicate.to_arrow()
    except Exception:
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
        rebuilt = filtered.combine_chunks().to_batches()
        return rebuilt[0] if rebuilt else pa.RecordBatch.from_pylist(
            [], schema=batch.schema,
        )

    return _filter


def _extract_partition_prune_values(
    predicate: "Predicate",
    partition_columns: "List[str]",
) -> "Optional[dict]":
    if predicate is None or not partition_columns:
        return None
    from yggdrasil.execution.expr import extract_partition_filters

    return extract_partition_filters(predicate, partition_columns) or None


def _split_batch(
    batch: pa.RecordBatch,
    partition_columns: "List[str]",
) -> "Iterator[tuple[tuple, pa.RecordBatch]]":
    if not partition_columns:
        yield ((), batch)
        return
    if not all(c in batch.schema.names for c in partition_columns):
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
