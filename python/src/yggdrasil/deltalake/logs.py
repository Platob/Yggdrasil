"""
logs.py — Delta transaction log reader, selector, and writer.

Parses the Delta log from any PyArrow-supported filesystem (S3, GCS, Azure
Blob, local) into a structured, queryable in-memory index, and provides
methods to write new commits back to the log.

Reading
-------
::

    tbl = DeltaTable(fs=fs, storage_location="s3://bucket/path/")
    tbl = DeltaTable(fs=fs, storage_location="s3://bucket/path/", version=42)

    tbl.version                                      # int
    tbl.partition_columns                            # list[str]
    tbl.column_names                                 # list[str]
    tbl.configuration                                # dict[str, str]
    tbl.has_deletion_vectors                         # bool
    tbl.stats()                                      # DeltaStats

    files = tbl.select()
    files = tbl.select(partition_filter={"commodity": "crude_oil"})
    files = tbl.select(
        partition_filter={"commodity": "crude_oil", "date": "2024-01-15"},
        stats_filters=[("price", 70.0, 90.0), ("volume", 1000, None)],
    )

    # Dataset — deletion vectors applied automatically when present
    dataset = tbl.to_arrow_dataset(files, schema=arrow_schema)
    dataset = tbl.to_arrow_dataset(files, schema=arrow_schema, apply_deletion_vectors=False)

Writing
-------
::

    # Write Arrow data and register in one call
    version = tbl.write_arrow_dataset(
        data,
        schema=arrow_schema,
        mode="append",
        partition_by=["trade_date", "commodity"],
    )

    # Register pre-written Parquet files
    version = tbl.add_files(delta_files)

    # Tombstone files (physical files stay until VACUUM)
    version = tbl.remove_files(old_delta_files)

    # Atomic compaction / Z-order rewrite
    version = tbl.replace_files(remove=old_files, add=new_files)

    # Update table properties
    version = tbl.commit_metadata(
        description="Settlement prices — crude oil",
        configuration={"delta.autoOptimize.optimizeWrite": "true"},
    )

    # Schema evolution
    version = tbl.commit_schema(new_arrow_schema)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional, Union, TYPE_CHECKING

import pyarrow as pa
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
from ._uri import strip_uri_scheme, has_uri_scheme
from .models import DeltaProtocol, DeltaMetadata, DeltaFile, DeltaStats
from .schema import arrow_schema_to_schema_string
from ..dataclasses.expiring import Expiring

if TYPE_CHECKING:
    import pyarrow.dataset as ds


__all__ = ["DeltaTable", "DeltaLog"]


class DeltaTable:
    """Parsed representation of a Delta table's transaction log.

    Reads ``_delta_log/*.checkpoint.parquet`` and ``*.json`` commit files
    using any PyArrow ``FileSystem`` and builds an in-memory index of the
    active file set plus table metadata.

    Read capabilities
    ~~~~~~~~~~~~~~~~~
    - Version time-travel.
    - Partition equality pruning via :meth:`select`.
    - Column min/max stats pruning via :meth:`select`.
    - Aggregate file statistics via :meth:`stats`.
    - Deletion vector application — rows deleted by DV are filtered from
      :meth:`to_arrow_dataset` automatically.

    Write capabilities
    ~~~~~~~~~~~~~~~~~~
    - Write Arrow data directly via :meth:`write_arrow_dataset`.
    - Append new Parquet files via :meth:`add_files`.
    - Logical deletion via :meth:`remove_files`.
    - Atomic compaction / rewrite via :meth:`replace_files`.
    - Table property updates via :meth:`commit_metadata`.
    - Schema evolution via :meth:`commit_schema`.
    - Initialise a brand-new table via :meth:`DeltaTable.init`.

    Limitations
    ~~~~~~~~~~~
    **Optimistic concurrency** is not implemented.  Concurrent writers to
    the same table version will silently clobber each other.  Use
    Databricks server-side locking for multi-writer workloads.

    **v2Checkpoint** is handled by skipping the checkpoint and replaying
    the full JSON commit history instead.
    """

    # Reader features this implementation handles correctly.
    SUPPORTED_READER_FEATURES: frozenset[str] = frozenset({
        "columnMapping",    # log structure unchanged; only path encoding varies
        "v2Checkpoint",     # handled by skipping checkpoint → JSON replay
        "timestampNtz",     # irrelevant to file selection
        "domainMetadata",   # irrelevant to file selection
        "deletionVectors",  # parsed and applied in to_arrow_dataset
    })

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        fs: pa_fs.FileSystem | Expiring[pa_fs.FileSystem],
        storage_location: str,
        version: Optional[int] = None,
    ) -> None:
        self._fs = fs
        self._storage_location = storage_location.rstrip("/")
        self._base = strip_uri_scheme(self._storage_location)
        self._log_dir = f"{self._base}/_delta_log"

        self.protocol: Optional[DeltaProtocol] = None
        self.metadata: Optional[DeltaMetadata] = None

        # Relative or absolute path → DeltaFile
        self._active: dict[str, DeltaFile] = {}
        self._resolved_version: int = -1

        self._load(version)

    # ------------------------------------------------------------------
    # Factory — initialise a new Delta table
    # ------------------------------------------------------------------

    @classmethod
    def init(
        cls,
        *,
        fs: pa_fs.FileSystem,
        storage_location: str,
        schema: pa.Schema,
        partition_columns: Optional[list[str]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        configuration: Optional[dict[str, str]] = None,
        min_reader_version: int = 1,
        min_writer_version: int = 2,
    ) -> "DeltaTable":
        """Initialise a brand-new Delta table.

        Writes commit ``00000000000000000000.json`` containing a ``protocol``
        action, a ``metaData`` action, and a ``commitInfo`` action.  No data
        files are written — use :meth:`write_arrow_dataset` or
        :meth:`add_files` for that.

        Args:
            fs:                  PyArrow ``FileSystem`` for the table location.
            storage_location:    Root URI, e.g. ``"s3://bucket/trading/crude_oil/"``.
            schema:              PyArrow schema for the table.  ``b"comment"``
                                 or ``b"description"`` in schema metadata is
                                 used as *description* when not supplied explicitly.
            partition_columns:   Ordered partition column names.
            name:                Human-readable table name stored in the log.
            description:         Table description / comment.
            configuration:       Delta table properties dict.
            min_reader_version:  Delta protocol reader version (default 1).
            min_writer_version:  Delta protocol writer version (default 2).

        Returns:
            A :class:`DeltaTable` loaded from the freshly written commit.

        Example::

            tbl = DeltaTable.init(
                fs=s3fs,
                storage_location="s3://my-bucket/trading/raw/crude_oil/",
                schema=pa.schema([
                    pa.field("trade_date", pa.date32(),        nullable=False),
                    pa.field("commodity",  pa.string(),         nullable=False),
                    pa.field("price",      pa.float64()),
                    pa.field("volume",     pa.int64()),
                    pa.field("notional",   pa.decimal128(18, 6)),
                ]),
                partition_columns=["trade_date", "commodity"],
                description="Raw crude oil trades",
                configuration={"delta.autoOptimize.optimizeWrite": "true"},
            )
        """
        tbl = cls.__new__(cls)
        tbl._fs = fs
        tbl._storage_location = storage_location.rstrip("/")
        tbl._base = strip_uri_scheme(tbl._storage_location)
        tbl._log_dir = f"{tbl._base}/_delta_log"
        tbl.protocol = None
        tbl.metadata = None
        tbl._active = {}
        tbl._resolved_version = -1

        schema_string = arrow_schema_to_schema_string(schema)

        if description is None and schema.metadata:
            raw = schema.metadata.get(b"comment") or schema.metadata.get(b"description")
            description = raw.decode("utf-8") if isinstance(raw, bytes) else raw

        protocol_action = DeltaProtocol(
            min_reader_version=min_reader_version,
            min_writer_version=min_writer_version,
        ).to_action()

        metadata_action: dict[str, Any] = {
            "metaData": {
                "id":               str(uuid.uuid4()),
                "name":             name,
                "description":      description,
                "format":           {"provider": "parquet", "options": {}},
                "schemaString":     schema_string,
                "partitionColumns": partition_columns or [],
                "configuration":    configuration or {},
                "createdTime":      int(time.time() * 1000),
            }
        }

        tbl._write_commit([protocol_action, metadata_action])
        tbl._load(None)
        return tbl

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fs(self) -> pa_fs.FileSystem:
        """The underlying PyArrow ``FileSystem``."""
        if isinstance(self._fs, Expiring):
            return self._fs.value
        return self._fs

    @property
    def storage_location(self) -> str:
        """Root URI or path of the table."""
        return self._storage_location

    @property
    def version(self) -> int:
        """Resolved Delta commit version (latest or as requested at init)."""
        return self._resolved_version

    @property
    def schema(self) -> dict:
        """Raw Delta schema dict (lazily parsed from ``schemaString``)."""
        return self.metadata.schema if self.metadata else {}

    @property
    def partition_columns(self) -> list[str]:
        """Ordered list of partition column names."""
        return self.metadata.partition_columns if self.metadata else []

    @property
    def column_names(self) -> list[str]:
        """Column names in schema field order."""
        return self.metadata.column_names if self.metadata else []

    @property
    def configuration(self) -> dict[str, str]:
        """Delta table properties (``TBLPROPERTIES``)."""
        return self.metadata.configuration if self.metadata else {}

    @property
    def has_deletion_vectors(self) -> bool:
        """``True`` if any active file carries a deletion vector."""
        return any(f.has_deletion_vector for f in self._active.values())

    def stats(self) -> DeltaStats:
        """Aggregate statistics over the current active file set.

        Note:
            ``total_records`` reflects the count stored in file-level stats
            and does **not** subtract rows masked by deletion vectors.
        """
        s = DeltaStats()
        for f in self._active.values():
            s.num_files += 1
            s.total_bytes += f.size
            if f.num_records is not None:
                s.total_records += f.num_records
        return s

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def all_files(self) -> list[DeltaFile]:
        """Return the full active file list with no filtering."""
        return list(self._active.values())

    def select(
        self,
        *,
        partition_filter: Optional[dict[str, str]] = None,
        stats_filters: Optional[list[tuple[str, Any, Any]]] = None,
    ) -> list[DeltaFile]:
        """Return active :class:`DeltaFile` objects matching the given filters.

        Both filter types are conservative — a file is excluded only when it
        is *provably* outside the requested range.

        Args:
            partition_filter:
                ``{column: value}`` equality filters on partition columns.
                All conditions are ANDed.
                E.g. ``{"commodity": "crude_oil", "date": "2024-01-15"}``.

            stats_filters:
                ``(column, lo, hi)`` tuples for min/max range pruning.
                Pass ``None`` for an open bound.
                E.g. ``[("price", 70.0, 90.0), ("volume", 1000, None)]``.

        Returns:
            Filtered ``list[DeltaFile]``.
        """
        files: list[DeltaFile] = list(self._active.values())

        if partition_filter:
            files = [f for f in files if f.matches_partition(partition_filter)]

        if stats_filters:
            for column, lo, hi in stats_filters:
                files = [f for f in files if f.matches_stats(column, lo=lo, hi=hi)]

        return files

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def uri_paths(self, files: Optional[list[DeltaFile]] = None) -> list[str]:
        """Absolute URI paths for *files* (or all active files if omitted).

        Relative paths are resolved against ``storage_location``.  Absolute
        paths (those with a URI scheme) are returned unchanged.
        """
        target = files if files is not None else list(self._active.values())
        base = self._storage_location
        return [
            f.path if has_uri_scheme(f.path) else f"{base}/{f.path}"
            for f in target
        ]

    def bare_paths(self, files: Optional[list[DeltaFile]] = None) -> list[str]:
        """URI-scheme-stripped paths for *files*.

        PyArrow filesystems expect bare paths without a URI scheme.  Pass
        the result directly to ``fs.open_input_file()``, ``pq.read_table()``,
        or ``pyarrow.dataset.dataset()``.
        """
        return [strip_uri_scheme(p) for p in self.uri_paths(files)]

    # ------------------------------------------------------------------
    # Deletion vector application
    # ------------------------------------------------------------------

    def _read_deletion_mask(self, file: DeltaFile) -> Optional[set[int]]:
        """Load and deserialise the deletion vector for *file*.

        Returns:
            ``None`` when the file has no DV.
            ``set[int]`` of deleted row indices when a DV is present.
            Empty set on read failure — the file is consumed without masking
            rather than being silently skipped.

        Raises:
            ImportError: Propagated when ``pyroaring`` is not installed.
        """
        if not file.has_deletion_vector:
            return None
        try:
            return file.deletion_vector.read_deleted_rows(
                self.fs, self._storage_location
            )
        except ImportError:
            raise
        except Exception:
            return set()

    def _apply_deletion_vectors(
        self,
        files: list[DeltaFile],
        batches: "list[pa.RecordBatch]",
    ) -> "list[pa.RecordBatch]":
        """Filter *batches* by removing rows whose indices appear in any DV.

        Assumes *batches* were produced by scanning *files* in order and that
        each file maps to a contiguous block of rows across the batch list.

        Args:
            files:   :class:`DeltaFile` objects in scan order.
            batches: ``RecordBatch`` list from the dataset scan.

        Returns:
            A new ``list[pa.RecordBatch]`` with deleted rows removed.
        """
        global_offset = 0
        rows_to_drop: set[int] = set()
        for f in files:
            deleted = self._read_deletion_mask(f)
            if deleted:
                rows_to_drop.update(global_offset + r for r in deleted)
            global_offset += f.num_records or 0

        if not rows_to_drop:
            return batches

        result: list[pa.RecordBatch] = []
        global_row = 0
        for batch in batches:
            n = len(batch)
            keep = [i for i in range(n) if (global_row + i) not in rows_to_drop]
            if len(keep) == n:
                result.append(batch)
            elif keep:
                result.append(batch.take(pa.array(keep, type=pa.int64())))
            global_row += n
        return result

    # ------------------------------------------------------------------
    # Dataset I/O
    # ------------------------------------------------------------------

    def to_arrow_dataset(
        self,
        files: Optional[list[DeltaFile]] = None,
        *,
        schema: Optional[pa.Schema] = None,
        partitioning: Any = "hive",
        apply_deletion_vectors: bool = True,
    ) -> "ds.Dataset":
        """Build a ``pyarrow.dataset.Dataset`` from *files*.

        When any selected file has a deletion vector and
        *apply_deletion_vectors* is ``True``, the dataset is materialised in
        memory, deleted rows are removed, and an in-memory dataset is returned.
        Files without DVs always yield a lazy dataset with zero overhead.

        Args:
            files:                   Files to read.  Defaults to all active files.
            schema:                  Arrow schema.  Pass ``table.arrow_schema``
                                     to enforce the Unity Catalog schema.
            partitioning:            Partitioning scheme for ``ds.dataset()``.
                                     Defaults to ``"hive"``.
            apply_deletion_vectors:  ``True`` (default) — filter DV-deleted rows.
                                     ``False`` — return raw files; useful for
                                     inspection or external DV handling.

        Returns:
            A lazy ``ds.Dataset`` when no DVs are applied; an in-memory
            ``ds.Dataset`` backed by a ``pa.Table`` when DVs are applied.
        """
        import pyarrow.dataset as ds

        selected = files if files is not None else list(self._active.values())
        paths = self.bare_paths(selected)

        if not paths:
            return ds.dataset([], schema=schema, format="parquet")

        lazy_ds = ds.dataset(
            paths,
            filesystem=self.fs,
            format="parquet",
            schema=schema,
            partitioning=partitioning,
        )

        dv_files = [f for f in selected if f.has_deletion_vector]
        if not dv_files or not apply_deletion_vectors:
            return lazy_ds

        batches = lazy_ds.to_batches()
        filtered = self._apply_deletion_vectors(selected, batches)

        if not filtered:
            empty_schema = schema or lazy_ds.schema
            return ds.dataset(pa.table({c.name: [] for c in empty_schema}))

        return ds.dataset(
            pa.Table.from_batches(filtered, schema=schema or lazy_ds.schema)
        )

    def write_arrow(
        self,
        data: Union["ds.Dataset", pa.Table, pa.RecordBatch],
        *,
        schema: Optional[pa.Schema] = None,
        mode: str = "append",
        partition_by: Optional[list[str]] = None,
        basename_template: Optional[str] = None,
        max_rows_per_file: Optional[int] = None,
        min_rows_per_group: Optional[int] = None,
        max_rows_per_group: Optional[int] = None,
        existing_data_behavior: str = "overwrite_or_ignore",
        commit: bool = True,
    ) -> int:
        """Write Arrow data to this table's storage and commit to the Delta log.

        Writes Parquet files via the underlying ``FileSystem``, then (when
        *commit* is ``True``) registers each new file in the Delta log.

        Args:
            data:                   Source data — ``ds.Dataset``, ``pa.Table``,
                                    or ``pa.RecordBatch``.
            schema:                 Write schema.  Falls back to ``data.schema``.
            mode:                   ``"append"`` *(default)* — add alongside
                                    existing data.  ``"overwrite"`` — tombstone
                                    all current files atomically with the new adds.
            partition_by:           Partition column names.  Falls back to
                                    ``self.partition_columns``.
            basename_template:      Parquet filename template, e.g.
                                    ``"part-{i}.parquet"``.  Defaults to
                                    ``"part-{i}-<uuid8>.parquet"`` to avoid
                                    collisions across concurrent writers.
            max_rows_per_file:      Row cap per Parquet file.
            min_rows_per_group:     Minimum rows per Parquet row group.
            max_rows_per_group:     Maximum rows per Parquet row group.
            existing_data_behavior: PyArrow ``write_dataset`` flag.  Forced to
                                    ``"delete_matching"`` for ``mode="overwrite"``.
            commit:                 ``True`` (default) — write a Delta commit.
                                    ``False`` — write files only; caller handles
                                    the commit via :meth:`add_files`.

        Returns:
            The Delta version number of the new commit, or :attr:`version`
            when *commit* is ``False``.

        Raises:
            ValueError: If *mode* is not ``"append"`` or ``"overwrite"``.

        Example::

            # Append a batch
            tbl.write_arrow_dataset(pa_table, mode="append")

            # Overwrite-by-partition
            tbl.write_arrow_dataset(
                pa_table,
                mode="overwrite",
                partition_by=["trade_date", "commodity"],
            )
        """
        import pyarrow.dataset as ds

        if mode not in ("append", "overwrite"):
            raise ValueError(f"mode must be 'append' or 'overwrite', got {mode!r}")

        if isinstance(data, (pa.Table, pa.RecordBatch)):
            data = ds.dataset(data)

        write_schema = schema or data.schema
        effective_partition_by = partition_by or self.partition_columns or []

        if mode == "overwrite":
            existing_data_behavior = "delete_matching"

        write_id = uuid.uuid4().hex[:8]
        template = basename_template or f"part-{{i}}-{write_id}.parquet"

        write_kwargs: dict[str, Any] = {
            "data":                   data,
            "base_dir":               self._base,
            "basename_template":      template,
            "format":                 ds.ParquetFileFormat(),
            "filesystem":             self.fs,
            "schema":                 write_schema,
            "existing_data_behavior": existing_data_behavior,
        }
        if effective_partition_by:
            write_kwargs["partitioning"]        = effective_partition_by
            write_kwargs["partitioning_flavor"] = "hive"
        if max_rows_per_file is not None:
            write_kwargs["max_rows_per_file"] = max_rows_per_file
        if min_rows_per_group is not None:
            write_kwargs["min_rows_per_group"] = min_rows_per_group
        if max_rows_per_group is not None:
            write_kwargs["max_rows_per_group"] = max_rows_per_group

        ds.write_dataset(**write_kwargs)

        if not commit:
            return self._resolved_version

        new_files = self._discover_written_files(partition_by=effective_partition_by)

        if mode == "overwrite":
            return self.replace_files(
                remove=list(self._active.values()),
                add=new_files,
                data_change=True,
            )
        return self.add_files(new_files, data_change=True)

    def _discover_written_files(
        self,
        *,
        partition_by: list[str],
    ) -> list[DeltaFile]:
        """Scan storage for Parquet files not yet in ``_active``.

        Called immediately after ``ds.write_dataset`` to build the list of
        newly written files for the Delta commit.  Skips ``_delta_log/`` and
        any path already tracked in ``_active``.

        Args:
            partition_by: Column names used for partitioning, used to parse
                          hive-style directory components.

        Returns:
            List of :class:`DeltaFile` objects ready for :meth:`add_files`.
        """
        try:
            infos = self.fs.get_file_info(
                pa_fs.FileSelector(self._base, recursive=True)
            )
        except Exception:
            return []

        new_files: list[DeltaFile] = []
        for info in infos:
            if info.type != pa_fs.FileType.File:
                continue
            if not info.path.endswith(".parquet"):
                continue
            if "/_delta_log/" in info.path:
                continue

            rel_path = info.path[len(self._base):].lstrip("/")
            if rel_path in self._active:
                continue

            partition_values: dict[str, str] = {}
            if partition_by:
                for part in rel_path.split("/")[:-1]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if k in partition_by:
                            partition_values[k] = v

            num_records = 0
            try:
                meta = pq.read_metadata(info.path, filesystem=self.fs)
                num_records = meta.num_rows
            except Exception:
                pass

            new_files.append(DeltaFile(
                path=rel_path,
                size=info.size or 0,
                partition_values=partition_values,
                modification_time=int(time.time() * 1000),
                data_change=True,
                stats={"numRecords": num_records} if num_records else {},
                tags={},
                deletion_vector=None,
            ))

        return new_files

    # ------------------------------------------------------------------
    # Log writing — internal commit machinery
    # ------------------------------------------------------------------

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _next_version(self) -> int:
        return max(self._resolved_version + 1, 0)

    def _commit_path(self, version: int) -> str:
        return f"{self._log_dir}/{version:020d}.json"

    def _write_commit(
        self,
        actions: list[dict[str, Any]],
        *,
        operation: str = "WRITE",
        operation_parameters: Optional[dict[str, Any]] = None,
        is_blind_append: bool = True,
    ) -> int:
        """Write a single JSON commit file to ``_delta_log``.

        A ``commitInfo`` action is prepended automatically.

        Args:
            actions:              Ordered Delta log action dicts.
            operation:            Operation label in ``commitInfo``.
            operation_parameters: Optional metadata for ``commitInfo``.
            is_blind_append:      ``True`` for pure-append commits.

        Returns:
            The version number of the newly written commit.

        Raises:
            RuntimeError: If the write fails.
        """
        version = self._next_version()
        path = self._commit_path(version)

        commit_info: dict[str, Any] = {
            "commitInfo": {
                "timestamp":           self._now_ms(),
                "operation":           operation,
                "operationParameters": operation_parameters or {},
                "isBlindAppend":       is_blind_append,
                "version":             version,
            }
        }

        lines = "\n".join(
            json.dumps(a, separators=(",", ":"))
            for a in [commit_info, *actions]
        ) + "\n"

        try:
            with self.fs.open_output_stream(path) as fh:
                fh.write(lines.encode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"Failed to write Delta commit v{version} to {path}: {e}"
            ) from e

        self._resolved_version = version
        return version

    # ------------------------------------------------------------------
    # Log writing — public API
    # ------------------------------------------------------------------

    def add_files(
        self,
        files: list[DeltaFile],
        *,
        data_change: bool = True,
    ) -> int:
        """Register *files* as active in the Delta log.

        Appends ``add`` actions and updates the in-memory ``_active`` index.

        Args:
            files:       :class:`DeltaFile` objects to register.
            data_change: ``True`` for normal writes; ``False`` for compaction.

        Returns:
            The version number of the new commit.
        """
        ts = self._now_ms()
        actions: list[dict[str, Any]] = []
        for f in files:
            add: dict[str, Any] = {
                "path":             f.path,
                "size":             f.size,
                "partitionValues":  f.partition_values,
                "modificationTime": ts,
                "dataChange":       data_change,
                "stats":            json.dumps(f.stats) if f.stats else "{}",
                "tags":             f.tags or {},
            }
            if f.deletion_vector is not None:
                add["deletionVector"] = f.deletion_vector.to_dict()
            actions.append({"add": add})

        version = self._write_commit(
            actions, operation="WRITE", is_blind_append=data_change
        )
        for f in files:
            self._active[f.path] = f
        return version

    def remove_files(
        self,
        files: list[DeltaFile],
        *,
        data_change: bool = True,
    ) -> int:
        """Tombstone *files* in the Delta log (logical delete).

        Appends ``remove`` actions and evicts files from ``_active``.  The
        underlying Parquet files are **not** deleted — Delta's ``VACUUM``
        operation handles physical deletion after the retention window.

        Args:
            files:       :class:`DeltaFile` objects to tombstone.
            data_change: ``True`` for data-change removals; ``False`` for
                         compaction tombstones.

        Returns:
            The version number of the new commit.
        """
        ts = self._now_ms()
        actions = [
            {
                "remove": {
                    "path":              f.path,
                    "deletionTimestamp": ts,
                    "dataChange":        data_change,
                    "partitionValues":   f.partition_values,
                    "size":              f.size,
                }
            }
            for f in files
        ]
        version = self._write_commit(actions, operation="DELETE", is_blind_append=False)
        for f in files:
            self._active.pop(f.path, None)
        return version

    def replace_files(
        self,
        remove: list[DeltaFile],
        add: list[DeltaFile],
        *,
        data_change: bool = False,
    ) -> int:
        """Atomically replace *remove* files with *add* files.

        Both tombstones and new ``add`` actions land in a single commit,
        making the swap appear instantaneous to concurrent readers.  This is
        the correct primitive for compaction and Z-ordering.

        Args:
            remove:      Files to tombstone.
            add:         Files to register as active.
            data_change: ``False`` (default) for compaction; ``True`` for DML.

        Returns:
            The version number of the new commit.
        """
        ts = self._now_ms()

        remove_actions = [
            {
                "remove": {
                    "path":              f.path,
                    "deletionTimestamp": ts,
                    "dataChange":        data_change,
                    "partitionValues":   f.partition_values,
                    "size":              f.size,
                }
            }
            for f in remove
        ]

        add_actions: list[dict[str, Any]] = []
        for f in add:
            entry: dict[str, Any] = {
                "path":             f.path,
                "size":             f.size,
                "partitionValues":  f.partition_values,
                "modificationTime": ts,
                "dataChange":       data_change,
                "stats":            json.dumps(f.stats) if f.stats else "{}",
                "tags":             f.tags or {},
            }
            if f.deletion_vector is not None:
                entry["deletionVector"] = f.deletion_vector.to_dict()
            add_actions.append({"add": entry})

        version = self._write_commit(
            [*remove_actions, *add_actions],
            operation="OPTIMIZE",
            is_blind_append=False,
        )
        for f in remove:
            self._active.pop(f.path, None)
        for f in add:
            self._active[f.path] = f
        return version

    def commit_metadata(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        configuration: Optional[dict[str, str]] = None,
        partition_columns: Optional[list[str]] = None,
    ) -> int:
        """Write a ``metaData`` action to update table-level properties.

        Merges with existing metadata — only supply the fields you want to
        change.  ``configuration`` is merged (not replaced) so existing
        properties survive unless explicitly overridden.

        Args:
            name:              Human-readable table name.
            description:       Table description / comment.
            configuration:     Properties to merge, e.g.
                               ``{"delta.autoOptimize.optimizeWrite": "true"}``.
            partition_columns: Override partition columns (use carefully on
                               non-empty tables).

        Returns:
            The version number of the new commit.

        Raises:
            ValueError: If no fields are provided, or if no metadata exists.
        """
        if not any([name, description, configuration, partition_columns is not None]):
            raise ValueError("At least one metadata field must be provided.")
        if self.metadata is None:
            raise ValueError(
                "No existing metadata — use DeltaTable.init() to create a new table."
            )

        new_metadata = self.metadata.with_updates(
            name=name,
            description=description,
            configuration=configuration,
            partition_columns=partition_columns,
        )
        version = self._write_commit(
            [new_metadata.to_action()], operation="SET_TABLE_PROPERTIES"
        )
        self.metadata = new_metadata
        return version

    def commit_schema(
        self,
        schema: pa.Schema,
        *,
        description: Optional[str] = None,
    ) -> int:
        """Evolve the table schema by writing a ``metaData`` action.

        All other metadata fields (configuration, partition columns, table
        name) are preserved.

        Note:
            Delta schema evolution rules (adding nullable columns, widening
            types) are **not** enforced here — the caller is responsible for
            backward compatibility.

        Args:
            schema:      New PyArrow schema.
            description: Optional updated description.

        Returns:
            The version number of the new commit.

        Raises:
            ValueError: If no metadata exists to evolve from.
        """
        if self.metadata is None:
            raise ValueError(
                "No existing metadata — use DeltaTable.init() to create a new table first."
            )
        new_metadata = self.metadata.with_updates(
            schema_string=arrow_schema_to_schema_string(schema),
            description=description,
        )
        version = self._write_commit(
            [new_metadata.to_action()], operation="CHANGE_COLUMN"
        )
        self.metadata = new_metadata
        return version

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._active)

    def __repr__(self) -> str:
        return (
            f"DeltaTable("
            f"version={self._resolved_version}, "
            f"{self.stats()}, "
            f"partitions={self.partition_columns})"
        )

    # ------------------------------------------------------------------
    # Log loading — internals
    # ------------------------------------------------------------------

    def _load(self, version: Optional[int]) -> None:
        target = version if version is not None else 2 ** 63
        cp_version, cp_ok = self._load_checkpoint(target)
        self._load_json_commits(
            start=max(0, cp_version + 1) if cp_ok else 0,
            end=target,
        )
        if self._resolved_version < 0 <= cp_version:
            self._resolved_version = cp_version

    def _read_last_checkpoint(self) -> dict:
        try:
            with self.fs.open_input_file(f"{self._log_dir}/_last_checkpoint") as fh:
                return json.loads(fh.read())
        except Exception:
            return {}

    def _load_checkpoint(self, target: int) -> tuple[int, bool]:
        """Load the latest checkpoint at or before *target*.

        Returns ``(checkpoint_version, loaded_successfully)``.  Falls back
        to JSON replay on failure (corrupt file, v2Checkpoint, beyond target).
        """
        last = self._read_last_checkpoint()
        cp_version: int = last.get("version", -1)

        if cp_version < 0 or cp_version > target:
            return -1, False
        if last.get("v2Checkpoint"):
            # v2Checkpoint format not parsed; fall back to JSON replay.
            return cp_version, False

        cp_path = f"{self._log_dir}/{cp_version:020d}.checkpoint.parquet"
        try:
            tbl = pq.read_table(cp_path, filesystem=self.fs)
            self._replay_checkpoint_table(tbl)
            self._resolved_version = cp_version
            return cp_version, True
        except Exception:
            self._active.clear()
            self.protocol = None
            self.metadata = None
            return cp_version, False

    def _replay_checkpoint_table(self, tbl: pa.Table) -> None:
        names = set(tbl.schema.names)
        for batch in tbl.to_batches():
            if "protocol" in names:
                for entry in batch.column("protocol"):
                    if entry.is_valid and (d := entry.as_py()):
                        self.protocol = DeltaProtocol.from_action(d)
            if "metaData" in names:
                for entry in batch.column("metaData"):
                    if entry.is_valid and (d := entry.as_py()):
                        self.metadata = DeltaMetadata.from_action(d)
            if "add" in names:
                for entry in batch.column("add"):
                    if entry.is_valid and (d := entry.as_py()) and d.get("path"):
                        self._active[d["path"]] = DeltaFile.from_add_action(d)
            if "remove" in names:
                for entry in batch.column("remove"):
                    if entry.is_valid and (d := entry.as_py()) and d.get("path"):
                        self._active.pop(d["path"], None)

    def _load_json_commits(self, start: int, end: int) -> None:
        try:
            infos = self.fs.get_file_info(
                pa_fs.FileSelector(self._log_dir, recursive=False)
            )
        except Exception:
            return

        commits = sorted(
            fi.path for fi in infos
            if fi.path.endswith(".json") and not fi.path.endswith("_last_checkpoint")
        )

        for path in commits:
            fname = path.rsplit("/", 1)[-1]
            try:
                ver = int(fname.removesuffix(".json"))
            except ValueError:
                continue
            if ver < start:
                continue
            if ver > end:
                break
            self._replay_json_commit(path)
            self._resolved_version = ver

    def _replay_json_commit(self, path: str) -> None:
        try:
            with self.fs.open_input_file(path) as fh:
                content = fh.read().decode()
        except Exception:
            return

        for raw in content.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                action = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if p := action.get("protocol"):
                self.protocol = DeltaProtocol.from_action(p)
            elif m := action.get("metaData"):
                self.metadata = DeltaMetadata.from_action(m)
            elif a := action.get("add"):
                if a.get("path"):
                    self._active[a["path"]] = DeltaFile.from_add_action(a)
            elif r := action.get("remove"):
                if r.get("path"):
                    self._active.pop(r["path"], None)


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

#: Alias kept for any code that imported ``DeltaLog`` before the rename.
DeltaLog = DeltaTable