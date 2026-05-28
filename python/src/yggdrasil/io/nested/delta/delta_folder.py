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
from yggdrasil.io.primitive.parquet_file import ParquetFile, ParquetOptions
from yggdrasil.pickle import json as ygg_json

from yggdrasil.io.nested.delta._names import format_commit_name
from yggdrasil.io.nested.delta.checkpoint import update_last_checkpoint, write_checkpoint
from yggdrasil.io.nested.delta.deletion_vector import (
    DeletionVector, decode_deletion_vector, encode_inline_deletion_vector,
    mask_batch_with_dv, write_uuid_deletion_vector,
)
from yggdrasil.io.nested.delta.log import DeltaLog
from yggdrasil.io.nested.delta.protocol import (
    AddFile, CommitInfo, DeltaAction, DeletionVectorDescriptor,
    Metadata, Protocol, RemoveFile, Txn,
)
from yggdrasil.io.nested.delta.schema_codec import (
    arrow_schema_to_spark_json, spark_json_to_arrow_schema,
    schema_to_spark_json, spark_json_to_schema,
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
        if snap.metadata is None:
            return

        partition_columns = snap.partition_columns
        target_schema = (spark_json_to_arrow_schema(snap.schema_string)
                         if snap.schema_string else None)
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
        leaf_opts = ParquetOptions.check(
            options=None, row_size=options.row_size,
            byte_size=options.byte_size, use_threads=options.use_threads,
            mode=Mode.READ_ONLY,
        )

        for add in snap.prune_files(prune_values=prune):
            dv = decode_deletion_vector(add.deletion_vector, table_root=self.path,
                                        sidecar_cache=sidecar_cache)
            leaf = ParquetFile(holder=snap.resolve(add), owns_holder=False)
            base_offset = 0
            try:
                with leaf as opened:
                    for batch in opened._read_arrow_batches(leaf_opts):
                        masked = mask_batch_with_dv(batch, dv, base_offset=base_offset)
                        base_offset += batch.num_rows
                        if masked.num_rows == 0:
                            continue
                        stamped = _stamp_partitions(masked, add.partition_values,
                                                    partition_columns, target_schema)
                        if row_filter is not None:
                            stamped = row_filter(stamped)
                            if stamped.num_rows == 0:
                                continue
                        yield stamped
            except FileNotFoundError:
                continue

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
                actions.append(Protocol(min_reader_version=min_r, min_writer_version=min_w,
                                        reader_features=rf, writer_features=wf))
                actions.append(Metadata(id=str(uuid.uuid4()),
                                        schema_string=arrow_schema_to_spark_json(target_schema),
                                        partition_columns=partition_columns,
                                        created_time=int(time.time() * 1000)))
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

        self._with_commit_retry(build_actions=build, cleanup=None,
                                options=options, initial_snap=snap)

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
            next_version = (snap.version + 1) if snap.metadata is not None else 0
            try:
                self._commit_atomic(next_version, build_actions(snap))
            except FileExistsError:
                if cleanup is not None:
                    cleanup()
                self._log.invalidate()
                self._snapshot = None
                if attempt == max_retries:
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

            # Strip partition columns + reinterpret unsigned ints
            payload_batches: "list[pa.RecordBatch]" = []
            for sb in sub_batches:
                drop = [c for c in partition_columns if c in sb.schema.names]
                sb = sb.drop_columns(drop) if drop else sb
                if any(pa.types.is_unsigned_integer(f.type) for f in sb.schema):
                    arrays = [sb.column(i).cast(_SIGNED_FOR_UINT[f.type.bit_width](), safe=False)
                              if pa.types.is_unsigned_integer(f.type) else sb.column(i)
                              for i, f in enumerate(sb.schema)]
                    fields = [pa.field(f.name, _SIGNED_FOR_UINT[f.type.bit_width](), nullable=f.nullable, metadata=f.metadata)
                              if pa.types.is_unsigned_integer(f.type) else f
                              for f in sb.schema]
                    sb = pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))
                payload_batches.append(sb)

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
                stats=_collect_stats(payload_batches) if getattr(options, "collect_stats", True) else None,
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

        if commit_path.exists():
            raise FileExistsError(f"Delta commit v{version} already exists at {commit_path!s}")
        with commit_path.open("wb") as bio:
            bio.truncate(0)
            bio.write_bytes(body)

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

    def _partition_file_rows(self, *, leaf: ParquetFile, predicate: "Predicate",
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
                matched = predicate.filter_arrow_table(tagged)
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

        blobs = [pickle.dumps((self.path, add, partition_columns, target_schema)) for add in active_adds]
        try: parallelism = max(spark.sparkContext.defaultParallelism, 1)
        except Exception: parallelism = 4
        leaf_df = spark.createDataFrame(
            pa.table({"_pkl": pa.array(blobs, type=pa.binary())}),
        ).coalesce(min(len(blobs), parallelism))

        def _read_delta_files(batches: "Iterator[pa.RecordBatch]") -> "Iterator[pa.RecordBatch]":
            import pickle as _pkl
            from yggdrasil.io.primitive.parquet_file import ParquetFile as _PF, ParquetOptions as _PO
            from yggdrasil.io.nested.delta.deletion_vector import decode_deletion_vector as _dv, mask_batch_with_dv as _mask
            from yggdrasil.enums import Mode as _M
            for batch in batches:
                for blob in batch.column("_pkl").to_pylist():
                    root, add, pcols, ts = _pkl.loads(blob)
                    dv = _dv(add.deletion_vector, table_root=root)
                    base = 0
                    with _PF(holder=root / add.path, owns_holder=False) as f:
                        for rb in f._read_arrow_batches(_PO(mode=_M.READ_ONLY)):
                            m = _mask(rb, dv, base_offset=base); base += rb.num_rows
                            if m.num_rows > 0:
                                yield _stamp_partitions(m, add.partition_values, pcols, ts)
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
    return batch


def _collect_stats(batches: "list[pa.RecordBatch]") -> "Optional[str]":
    if not batches: return None
    total_rows = sum(b.num_rows for b in batches)
    schema = batches[0].schema
    min_vals, max_vals, null_counts = {}, {}, {}

    def _sv(val: Any) -> Any:
        if isinstance(val, (datetime.date, datetime.datetime)): return val.isoformat()
        if isinstance(val, bytes): return val.hex()
        if isinstance(val, decimal.Decimal): return str(val)
        return val

    for field in schema:
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
        if col_min is not None: min_vals[field.name] = _sv(col_min)
        if col_max is not None: max_vals[field.name] = _sv(col_max)
        null_counts[field.name] = col_nulls

    stats: dict[str, Any] = {"numRecords": total_rows}
    if min_vals: stats["minValues"] = min_vals
    if max_vals: stats["maxValues"] = max_vals
    if null_counts: stats["nullCount"] = null_counts
    return ygg_json.dumps(stats, separators=(",", ":"), to_bytes=False)


def _partition_prune_values(predicate: "Predicate", partition_columns: "List[str]") -> "Optional[dict]":
    if predicate is None or not partition_columns: return None
    from yggdrasil.execution.expr import extract_partition_filters
    return extract_partition_filters(predicate, partition_columns) or None
