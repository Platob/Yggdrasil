"""DeltaFolder — :class:`Folder` over a Delta Lake table.

Full Delta read/write protocol: V1/V2 checkpoints, deletion vectors,
per-file stats, concurrent commit with exponential backoff, partition
pruning, APPEND / OVERWRITE / UPSERT / MERGE / IGNORE / ERROR_IF_EXISTS,
row-level delete via DV or rewrite, Spark mapInArrow integration.
"""

from __future__ import annotations

import dataclasses
import os
import random
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, Iterator, List, Optional

import pyarrow as pa
import pyarrow.compute as pc

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
_SIGNED_FOR_UINT = {8: pa.int8, 16: pa.int16, 32: pa.int32, 64: pa.int64}


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
    commit_max_retries: int = 8
    commit_retry_backoff: float = 0.05
    commit_retry_jitter: float = 0.05
    commit_retry_max_delay: float = 1.0
    collect_stats: bool = True
    target_file_size: int = 128 * 1024 * 1024


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

    def __repr__(self) -> str:
        return f"DeltaFolder(path={self.path!r})"

    def refresh(self) -> "DeltaFolder":
        self._log.invalidate()
        self._snapshot = None
        return self

    @property
    def log(self) -> DeltaLog:
        return self._log

    def snapshot(self, version: "Optional[int]" = None, *,
                 fresh: bool = False) -> Snapshot:
        if version is not None:
            return Snapshot.from_log(self._log, version)
        if not fresh and self._snapshot is not None:
            return self._snapshot
        self._snapshot = Snapshot.from_log(self._log, None)
        return self._snapshot

    def _collect_schema(self, options: DeltaOptions):
        from yggdrasil.data.schema import Schema
        snap = self.snapshot(options.version)
        if snap.metadata is None or not snap.schema_string:
            return Schema.empty()
        return Schema.from_arrow(spark_json_to_arrow_schema(snap.schema_string))

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(self, options: DeltaOptions) -> Iterator[pa.RecordBatch]:
        snap = self.snapshot(options.version)
        if snap.metadata is None:
            return

        partition_columns = snap.partition_columns
        target_schema = (
            spark_json_to_arrow_schema(snap.schema_string)
            if snap.schema_string else None
        )
        sidecar_cache: dict[str, bytes] = {}
        row_filter = _arrow_row_filter(options.predicate, partition_columns, target_schema)
        prune = _partition_prune_values(options.predicate, partition_columns)

        for add in snap.prune_files(prune_values=prune):
            file_path = snap.resolve(add)
            dv = decode_deletion_vector(
                add.deletion_vector, table_root=self.path,
                sidecar_cache=sidecar_cache,
            )
            leaf = ParquetFile(holder=file_path, owns_holder=False)
            leaf_opts = ParquetOptions.check(
                options=None, row_size=options.row_size,
                byte_size=options.byte_size, use_threads=options.use_threads,
                mode=Mode.READ_ONLY,
            )
            base_offset = 0
            try:
                with leaf as opened:
                    for batch in opened._read_arrow_batches(leaf_opts):
                        masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                        base_offset += batch.num_rows
                        if masked.num_rows == 0:
                            continue
                        stamped = _stamp_partitions(
                            masked, add.partition_values,
                            partition_columns, target_schema,
                        )
                        if row_filter is not None:
                            stamped = row_filter(stamped)
                            if stamped.num_rows == 0:
                                continue
                        yield stamped
            except FileNotFoundError:
                continue

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

    def _write_arrow_batches(self, batches: Iterable[pa.RecordBatch],
                             options: DeltaOptions) -> None:
        action = self._resolve_action(options.mode)
        snap = self.snapshot(fresh=True)

        if action is Mode.IGNORE and snap.active_files:
            return
        if action is Mode.ERROR_IF_EXISTS and snap.active_files:
            raise FileExistsError(
                f"Delta table at {self.path!s} is non-empty; "
                f"mode={options.mode!r}."
            )

        materialized: list[pa.RecordBatch] = list(batches)
        if not materialized and (action is not Mode.OVERWRITE or snap.metadata is None):
            return

        if action is Mode.UPSERT:
            self._commit_upsert(materialized, options=options, initial_snap=snap)
            return

        self._commit_simple(materialized, options=options,
                            initial_snap=snap, action=action)

    def _resolve_schema_and_partitions(
        self, materialized: "list[pa.RecordBatch]", options: DeltaOptions,
        snap: Snapshot, action: Mode,
    ) -> "tuple[pa.Schema, list[str]]":
        """Determine target schema + partition columns for a write."""
        if snap.metadata is None:
            schema = materialized[0].schema if materialized else pa.schema([])
            return schema, list(self._infer_partition_columns(options))
        if action is Mode.OVERWRITE and materialized:
            return (
                materialized[0].schema,
                list(self._infer_partition_columns(options)) or snap.partition_columns,
            )
        schema = (
            spark_json_to_arrow_schema(snap.schema_string)
            if snap.schema_string
            else (materialized[0].schema if materialized else pa.schema([]))
        )
        return schema, snap.partition_columns

    def _commit_simple(self, materialized: "list[pa.RecordBatch]", *,
                       options: DeltaOptions, initial_snap: Snapshot,
                       action: Mode) -> None:
        target_schema, partition_columns = self._resolve_schema_and_partitions(
            materialized, options, initial_snap, action,
        )

        new_adds: list[AddFile] = []
        if materialized:
            new_adds = list(self._write_parts(
                iter(materialized), partition_columns=partition_columns,
                options=options,
            ))

        def build(snap: Snapshot) -> "list[DeltaAction]":
            actions: list[DeltaAction] = []
            if snap.metadata is None:
                actions.extend(self._make_protocol_metadata(
                    options, target_schema, partition_columns,
                ))
            elif action is Mode.OVERWRITE and materialized:
                actions.append(Metadata(
                    id=snap.metadata.id,
                    schema_string=arrow_schema_to_spark_json(target_schema),
                    partition_columns=partition_columns,
                    configuration=dict(snap.metadata.configuration),
                    created_time=snap.metadata.created_time,
                ))
            if action is Mode.OVERWRITE:
                ts = int(time.time() * 1000)
                for path, add in snap.active_files.items():
                    actions.append(RemoveFile(
                        path=path, deletion_timestamp=ts, data_change=True,
                        extended_file_metadata=True,
                        partition_values=dict(add.partition_values),
                        size=int(add.size),
                    ))
            actions.extend(new_adds)
            if options.txn_app_id is not None and options.txn_version is not None:
                actions.append(Txn(app_id=options.txn_app_id,
                                   version=int(options.txn_version)))
            actions.append(CommitInfo(payload={
                "timestamp": int(time.time() * 1000),
                "operation": str(options.operation or "WRITE"),
                "operationParameters": {"mode": action.name.lower()},
                "engineInfo": str(options.engine_info or "yggdrasil"),
                "isBlindAppend": action is Mode.APPEND,
            }))
            return actions

        self._with_commit_retry(build_actions=build, cleanup=None,
                                options=options, initial_snap=initial_snap)

    def _commit_upsert(self, materialized: "list[pa.RecordBatch]", *,
                       options: DeltaOptions, initial_snap: Snapshot) -> None:
        match_by = list(options.match_by_keys or ())
        if not match_by:
            self._commit_simple(materialized, options=options,
                                initial_snap=initial_snap, action=Mode.APPEND)
            return

        target_schema, partition_columns = self._resolve_schema_and_partitions(
            materialized, options, initial_snap, Mode.UPSERT,
        )

        incoming_adds: list[AddFile] = []
        if materialized:
            incoming_adds = list(self._write_parts(
                iter(materialized), partition_columns=partition_columns,
                options=options,
            ))

        incoming_keys = Folder._collect_keys_from_batches(materialized, match_by)
        rewrite_state: dict[str, list[AddFile]] = {"current": []}

        def build(snap: Snapshot) -> "list[DeltaAction]":
            removes: list[RemoveFile] = []
            rewrites: list[AddFile] = []
            ts = int(time.time() * 1000)
            for add in list(snap.active_files.values()):
                file_path = snap.resolve(add)
                matched, survivors = self._partition_file_for_keys(
                    file_path, add=add, match_by=match_by,
                    incoming_keys=incoming_keys,
                )
                if not matched:
                    continue
                if survivors:
                    survivor_batches = self._read_indexed_batches(
                        leaf=ParquetFile(holder=file_path, owns_holder=False),
                        indices=survivors, partition_columns=partition_columns,
                        partition_values=dict(add.partition_values),
                    )
                    rewrites.extend(self._write_parts(
                        iter(survivor_batches),
                        partition_columns=partition_columns, options=options,
                    ))
                removes.append(RemoveFile(
                    path=add.path, deletion_timestamp=ts, data_change=True,
                    extended_file_metadata=True,
                    partition_values=dict(add.partition_values),
                    size=int(add.size), deletion_vector=add.deletion_vector,
                ))

            rewrite_state["current"] = rewrites
            actions: list[DeltaAction] = []
            if snap.metadata is None:
                actions.extend(self._make_protocol_metadata(
                    options, target_schema, partition_columns,
                ))
            actions.extend(removes)
            actions.extend(rewrites)
            actions.extend(incoming_adds)
            if options.txn_app_id is not None and options.txn_version is not None:
                actions.append(Txn(app_id=options.txn_app_id,
                                   version=int(options.txn_version)))
            actions.append(CommitInfo(payload={
                "timestamp": int(time.time() * 1000),
                "operation": str(options.operation or "WRITE"),
                "operationParameters": {"mode": "upsert"},
                "engineInfo": str(options.engine_info or "yggdrasil"),
                "isBlindAppend": False,
            }))
            return actions

        def cleanup() -> None:
            for add in rewrite_state["current"]:
                try:
                    (self.path / add.path).unlink(missing_ok=True)
                except Exception:
                    pass
            rewrite_state["current"] = []

        self._with_commit_retry(build_actions=build, cleanup=cleanup,
                                options=options, initial_snap=initial_snap)

    def _with_commit_retry(self, *, build_actions: "Callable[[Snapshot], list[DeltaAction]]",
                           cleanup: "Optional[Callable[[], None]]",
                           options: DeltaOptions, initial_snap: Snapshot) -> None:
        max_retries = max(0, int(options.commit_max_retries or 0))
        backoff = float(options.commit_retry_backoff or 0.0)
        jitter = float(options.commit_retry_jitter or 0.0)
        max_delay = float(options.commit_retry_max_delay or 0.0)

        for attempt in range(max_retries + 1):
            snap = initial_snap if attempt == 0 else self.snapshot(fresh=True)
            actions = build_actions(snap)
            next_version = (snap.version + 1) if snap.metadata is not None else 0

            try:
                self._commit_atomic(next_version, actions)
            except FileExistsError:
                if cleanup is not None:
                    cleanup()
                self._log.invalidate()
                self._snapshot = None
                if attempt == max_retries:
                    raise ConcurrentDeltaCommitError(
                        f"Failed to commit at {self.path!s} after "
                        f"{attempt + 1} attempts."
                    )
                if attempt > 0 and backoff > 0:
                    delay = min(backoff * (2 ** (attempt - 1)), max_delay) if max_delay > 0 else backoff * (2 ** (attempt - 1))
                    if jitter > 0:
                        delay += random.uniform(0, jitter)
                    time.sleep(delay)
                continue

            self._log.extend_listing(format_commit_name(next_version))
            self._snapshot = None
            interval = int(options.checkpoint_interval or 0)
            if interval > 0 and (next_version + 1) % interval == 0:
                self._write_checkpoint(next_version, kind=options.checkpoint_kind)
            return

    # ==================================================================
    # Helpers
    # ==================================================================

    def _make_protocol_metadata(self, options: DeltaOptions,
                                target_schema: pa.Schema,
                                partition_columns: "List[str]") -> "list[DeltaAction]":
        min_r = max(1, int(options.min_reader_version))
        min_w = max(2, int(options.min_writer_version))
        rf: list[str] = []
        wf: list[str] = []
        if options.checkpoint_kind == "v2":
            min_r, min_w = max(min_r, 3), max(min_w, 7)
            rf.append("v2Checkpoint"); wf.append("v2Checkpoint")
        if options.delete_via_dv:
            min_r, min_w = max(min_r, 3), max(min_w, 7)
            if "deletionVectors" not in rf: rf.append("deletionVectors")
            if "deletionVectors" not in wf: wf.append("deletionVectors")
        return [
            Protocol(min_reader_version=min_r, min_writer_version=min_w,
                     reader_features=rf, writer_features=wf),
            Metadata(id=str(uuid.uuid4()),
                     schema_string=arrow_schema_to_spark_json(target_schema),
                     partition_columns=partition_columns,
                     created_time=int(time.time() * 1000)),
        ]

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

    def _write_parts(self, batches: "Iterator[pa.RecordBatch]", *,
                     partition_columns: "List[str]",
                     options: DeltaOptions) -> "Iterator[AddFile]":
        import urllib.parse

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
                v = kv[col]
                target_dir = target_dir / f"{col}={urllib.parse.quote(str(v), safe='') if v is not None else '__HIVE_DEFAULT_PARTITION__'}"
            target_dir.mkdir(parents=True, exist_ok=True)

            stem = f"part-{int(time.time() * 1000)}-{os.urandom(8).hex()}.parquet"
            file_path = target_dir / stem

            payload_batches: "list[pa.RecordBatch]" = []
            for sb in sub_batches:
                drop = [c for c in partition_columns if c in sb.schema.names]
                stripped = sb.drop_columns(drop) if drop else sb
                if any(pa.types.is_unsigned_integer(f.type) for f in stripped.schema):
                    arrays, fields = [], []
                    for i, field in enumerate(stripped.schema):
                        if pa.types.is_unsigned_integer(field.type):
                            signed = _SIGNED_FOR_UINT[field.type.bit_width]()
                            arrays.append(stripped.column(i).cast(signed, safe=False))
                            fields.append(pa.field(field.name, signed,
                                                   nullable=field.nullable,
                                                   metadata=field.metadata))
                        else:
                            arrays.append(stripped.column(i))
                            fields.append(field)
                    stripped = pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))
                payload_batches.append(stripped)

            leaf = ParquetFile(holder=file_path, owns_holder=False)
            with leaf as opened:
                opened._write_arrow_batches(payload_batches,
                                            ParquetOptions(mode=Mode.OVERWRITE))

            stats_json = None
            if getattr(options, "collect_stats", True):
                stats_json = _collect_stats(payload_batches)

            parts = [f"{col}={urllib.parse.quote(str(kv[col]), safe='') if kv[col] is not None else '__HIVE_DEFAULT_PARTITION__'}"
                     for col in partition_columns]
            parts.append(stem)
            yield AddFile(
                path="/".join(parts),
                partition_values={c: (str(kv[c]) if kv[c] is not None else None)
                                  for c in partition_columns},
                size=int(file_path.size),
                modification_time=int(time.time() * 1000),
                data_change=True,
                stats=stats_json,
            )

    def _commit_atomic(self, version: int,
                       actions: "Iterable[DeltaAction]") -> None:
        self._log.log_path.mkdir(parents=True, exist_ok=True)
        commit_path = self._log.log_path / format_commit_name(version)

        body = ("\n".join(
            ygg_json.dumps(a.to_action(), separators=(",", ":"),
                           ensure_ascii=False, to_bytes=False)
            for a in actions
        ) + "\n").encode("utf-8")

        if getattr(commit_path, "is_local_path", False):
            full = commit_path.full_path()
            fd = os.open(full, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                mv = memoryview(body)
                while mv:
                    written = os.write(fd, mv)
                    if written <= 0:
                        raise OSError(f"os.write returned {written}")
                    mv = mv[written:]
            finally:
                os.close(fd)
            return

        if commit_path.exists():
            raise FileExistsError(
                f"Delta commit v{version} already exists at {commit_path!s}"
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
        update_last_checkpoint(log_path=self._log.log_path, version=version,
                               size=size, kind=kind, sidecar_files=sidecar_files)

    # ==================================================================
    # Row-level delete
    # ==================================================================

    def _delete(self, predicate: "Predicate", options: DeltaOptions) -> int:
        snap = self.snapshot(fresh=True)
        if snap.metadata is None:
            return 0

        sidecar_cache: dict[str, bytes] = {}
        ts = int(time.time() * 1000)
        new_actions: list[DeltaAction] = []
        deleted = 0

        for add_path, add in list(snap.active_files.items()):
            file_path = snap.resolve(add)
            leaf = ParquetFile(holder=file_path, owns_holder=False)
            existing_dv = decode_deletion_vector(
                add.deletion_vector, table_root=self.path,
                sidecar_cache=sidecar_cache,
            )

            survivors, file_deleted_rows = self._partition_file_rows(
                leaf=leaf, predicate=predicate, existing_dv=existing_dv,
            )
            if not file_deleted_rows:
                continue

            deleted += len(file_deleted_rows)
            partition_columns = snap.partition_columns
            partition_values = dict(add.partition_values)

            if options.delete_via_dv:
                prev_masked = existing_dv.deleted_rows if existing_dv is not None else set()
                deleted_rows = sorted(set(file_deleted_rows) | prev_masked)
                dv = (encode_inline_deletion_vector(deleted_rows)
                      if len(deleted_rows) <= _INLINE_DV_MAX_ROWS
                      else write_uuid_deletion_vector(deleted_rows, table_root=self.path))

                new_actions.append(RemoveFile(
                    path=add_path, deletion_timestamp=ts, data_change=True,
                    extended_file_metadata=True, partition_values=partition_values,
                    size=int(add.size), deletion_vector=add.deletion_vector,
                ))
                new_actions.append(AddFile(
                    path=add_path, partition_values=partition_values,
                    size=int(add.size), modification_time=ts,
                    data_change=True, stats=add.stats, deletion_vector=dv,
                ))
            else:
                survivor_batches = self._read_indexed_batches(
                    leaf=leaf, indices=survivors,
                    partition_columns=partition_columns,
                    partition_values=partition_values,
                )
                fresh_adds = list(self._write_parts(
                    iter(survivor_batches),
                    partition_columns=partition_columns, options=options,
                ))
                new_actions.append(RemoveFile(
                    path=add_path, deletion_timestamp=ts, data_change=True,
                    extended_file_metadata=True, partition_values=partition_values,
                    size=int(add.size),
                ))
                new_actions.extend(fresh_adds)

        if not new_actions:
            return 0

        if options.delete_via_dv and snap.protocol is not None:
            if "deletionVectors" not in (snap.protocol.writer_features or []):
                new_actions.insert(0, Protocol(
                    min_reader_version=max(snap.protocol.min_reader_version, 3),
                    min_writer_version=max(snap.protocol.min_writer_version, 7),
                    reader_features=sorted({*(snap.protocol.reader_features or []), "deletionVectors"}),
                    writer_features=sorted({*(snap.protocol.writer_features or []), "deletionVectors"}),
                ))

        new_actions.append(CommitInfo(payload={
            "timestamp": ts, "operation": "DELETE",
            "engineInfo": str(options.engine_info or "yggdrasil"),
            "isBlindAppend": False,
        }))

        next_version = snap.version + 1
        self._commit_atomic(next_version, new_actions)
        self._log.extend_listing(format_commit_name(next_version))
        self._snapshot = None

        interval = int(options.checkpoint_interval or 0)
        if interval > 0 and (next_version + 1) % interval == 0:
            self._write_checkpoint(next_version, kind=options.checkpoint_kind)
        return deleted

    _ROW_INDEX_COL = "__yggdrasil_dv_row_index__"

    def _partition_file_rows(self, *, leaf: ParquetFile,
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
                visible_local = [i for i in range(n) if (total + i) not in already_masked]
                if not visible_local:
                    total += n
                    continue
                visible_table = pa.Table.from_batches([batch]).take(
                    pa.array(visible_local, type=pa.int64()),
                )
                idx_col = pa.array([total + i for i in visible_local], type=pa.int64())
                tagged = visible_table.append_column(self._ROW_INDEX_COL, idx_col)
                all_visible.extend(idx_col.to_pylist())
                matched = predicate.filter_arrow_table(tagged)
                if matched.num_rows:
                    kept_indices.extend(matched.column(self._ROW_INDEX_COL).to_pylist())
                total += n

        deleted_set = set(kept_indices)
        return (
            [i for i in all_visible if i not in deleted_set],
            [i for i in all_visible if i in deleted_set],
        )

    def _partition_file_for_keys(self, file_path: "Any", *, add: AddFile,
                                 match_by: "List[str]",
                                 incoming_keys: "set[tuple]",
                                 ) -> "tuple[bool, list[int]]":
        existing_dv = decode_deletion_vector(
            add.deletion_vector, table_root=self.path,
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

    def _read_indexed_batches(self, *, leaf: ParquetFile,
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
            out.append(_stamp_partitions(batch, partition_values,
                                         partition_columns, None))
        return out

    # ==================================================================
    # Spark integration
    # ==================================================================

    def _read_spark_frame(self, options: DeltaOptions) -> "Any":
        import pickle
        from yggdrasil.environ import PyEnv

        spark = PyEnv.spark_session(options.spark_session, create=True, import_error=True)
        snap = self.snapshot(options.version)
        if snap.metadata is None or not snap.active_files:
            return spark.createDataFrame([], schema=self._collect_schema(options).to_spark_schema())

        target_schema = spark_json_to_arrow_schema(snap.schema_string) if snap.schema_string else None
        if target_schema is None:
            return spark.createDataFrame([], schema=self._collect_schema(options).to_spark_schema())

        spark_schema = self._collect_schema(options).to_spark_schema()
        partition_columns = snap.partition_columns
        prune = _partition_prune_values(options.predicate, partition_columns)
        active_adds = list(snap.prune_files(prune_values=prune))
        if not active_adds:
            return spark.createDataFrame([], schema=spark_schema)

        blobs = [pickle.dumps((self.path, add, partition_columns, target_schema))
                 for add in active_adds]
        leaf_table = pa.table({"_pkl": pa.array(blobs, type=pa.binary())})
        try:
            parallelism = max(spark.sparkContext.defaultParallelism, 1)
        except Exception:
            parallelism = 4
        leaf_df = spark.createDataFrame(leaf_table).coalesce(
            min(len(blobs), parallelism),
        )

        def _read_delta_files(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            import pickle as _pkl
            from yggdrasil.io.primitive.parquet_file import ParquetFile as _PF, ParquetOptions as _PO
            from yggdrasil.io.nested.delta.deletion_vector import decode_deletion_vector as _decode_dv, mask_batch_with_dv as _mask
            from yggdrasil.enums import Mode as _Mode
            import pyarrow as _pa

            for batch in batches:
                for blob in batch.column("_pkl").to_pylist():
                    table_root, add, part_cols, t_schema = _pkl.loads(blob)
                    dv = _decode_dv(add.deletion_vector, table_root=table_root)
                    leaf = _PF(holder=table_root / add.path, owns_holder=False)
                    base_offset = 0
                    with leaf as opened:
                        for rb in opened._read_arrow_batches(_PO(mode=_Mode.READ_ONLY)):
                            masked = _mask(rb, dv, base_offset=base_offset)
                            base_offset += rb.num_rows
                            if masked.num_rows == 0:
                                continue
                            yield _stamp_partitions(masked, add.partition_values,
                                                    part_cols, t_schema)

        return leaf_df.mapInArrow(_read_delta_files, schema=spark_schema)

    def _write_spark_frame(self, frame: "Any", options: DeltaOptions) -> None:
        for method in ("toArrow", "toArrowBatchIterator"):
            fn = getattr(frame, method, None)
            if not callable(fn):
                continue
            try:
                result = fn()
                if isinstance(result, pa.Table):
                    self._write_arrow_batches(result.to_batches(), options)
                else:
                    self._write_arrow_batches(list(result), options)
                return
            except Exception:
                continue
        self._write_arrow_batches(
            pa.Table.from_pandas(frame.toPandas()).to_batches(), options,
        )

    # ==================================================================
    # Folder surface
    # ==================================================================

    def iter_children(self) -> "Iterator":
        snap = self.snapshot()
        for add in snap.active_files.values():
            yield self.adopt_child(
                ParquetFile(holder=snap.resolve(add), owns_holder=False),
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _stamp_partitions(batch: pa.RecordBatch,
                      values: "dict[str, Optional[str]]",
                      columns: "List[str]",
                      target_schema: "Optional[pa.Schema]") -> pa.RecordBatch:
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
        batch = batch.append_column(
            target_field if target_field is not None
            else pa.field(col, arrow_type, nullable=True),
            arr,
        )
    return batch


def _coerce_partition(raw: "Optional[str]", arrow_type: pa.DataType) -> Any:
    if raw is None or raw == "":
        return None
    try:
        return pa.scalar(raw).cast(arrow_type).as_py()
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError):
        return raw


def _collect_stats(batches: "list[pa.RecordBatch]") -> "Optional[str]":
    if not batches:
        return None
    import datetime
    import decimal

    total_rows = sum(b.num_rows for b in batches)
    schema = batches[0].schema
    min_vals: dict[str, Any] = {}
    max_vals: dict[str, Any] = {}
    null_counts: dict[str, int] = {}

    for field in schema:
        t = field.type
        if not (pa.types.is_integer(t) or pa.types.is_floating(t)
                or pa.types.is_string(t) or pa.types.is_large_string(t)
                or pa.types.is_date(t) or pa.types.is_timestamp(t)
                or pa.types.is_decimal(t) or pa.types.is_boolean(t)):
            continue
        col_name = field.name
        col_min = col_max = None
        col_nulls = 0
        for batch in batches:
            col = batch.column(col_name)
            col_nulls += col.null_count
            if col.null_count == len(col):
                continue
            try:
                v_min = pc.min(col).as_py()
                v_max = pc.max(col).as_py()
                if v_min is not None and (col_min is None or v_min < col_min):
                    col_min = v_min
                if v_max is not None and (col_max is None or v_max > col_max):
                    col_max = v_max
            except Exception:
                continue
        def _sv(val: Any) -> Any:
            if isinstance(val, (datetime.date, datetime.datetime)):
                return val.isoformat()
            if isinstance(val, bytes):
                return val.hex()
            if isinstance(val, decimal.Decimal):
                return str(val)
            return val
        if col_min is not None:
            min_vals[col_name] = _sv(col_min)
        if col_max is not None:
            max_vals[col_name] = _sv(col_max)
        null_counts[col_name] = col_nulls

    stats: dict[str, Any] = {"numRecords": total_rows}
    if min_vals:
        stats["minValues"] = min_vals
    if max_vals:
        stats["maxValues"] = max_vals
    if null_counts:
        stats["nullCount"] = null_counts
    return ygg_json.dumps(stats, separators=(",", ":"), to_bytes=False)


def _arrow_row_filter(
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

    available = set(partition_columns)
    if target_schema is not None:
        available |= set(target_schema.names)
    if not set(free_columns(predicate)).issubset(available):
        return None

    try:
        arrow_expr = predicate.to_arrow()
    except Exception:
        return None

    import pyarrow.dataset as pds

    def _filter(batch: pa.RecordBatch) -> pa.RecordBatch:
        if batch.num_rows == 0:
            return batch
        filtered = pds.dataset(pa.Table.from_batches([batch])).to_table(filter=arrow_expr)
        if filtered.num_rows == 0:
            return pa.RecordBatch.from_pylist([], schema=batch.schema)
        rebuilt = filtered.combine_chunks().to_batches()
        return rebuilt[0] if rebuilt else pa.RecordBatch.from_pylist([], schema=batch.schema)

    return _filter


def _partition_prune_values(predicate: "Predicate",
                            partition_columns: "List[str]") -> "Optional[dict]":
    if predicate is None or not partition_columns:
        return None
    from yggdrasil.execution.expr import extract_partition_filters
    return extract_partition_filters(predicate, partition_columns) or None


def _split_batch(batch: pa.RecordBatch,
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
