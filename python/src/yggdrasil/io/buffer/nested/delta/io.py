"""DeltaIO: the partition-capable :class:`FolderIO` that ties Delta together.

Reads
-----

Replay the log, walk live AddFiles, yield one fragment per file.
On read, apply the file's deletion vector (if any) as a row
filter on the parquet batch stream. The DV decode is keyed on
the AddFile's :attr:`deletion_vector` descriptor.

Writes — OVERWRITE / APPEND
----------------------------

Stage parquet children via the inherited
:class:`FolderIO` machinery (which routes by partition
key when partitions are declared); commit one transaction with
Adds for new files and Removes for previously-live files
(OVERWRITE only). Failures during the parquet write are caught
in the inherited path; failures during the commit write are
caught here and the freshly-written parquet files are cleaned
up before re-raising.

Writes — UPSERT with deletion vectors
--------------------------------------

The interesting one. Given a match-by key and incoming rows:

1. Read every live AddFile, projecting the source file path and
   per-file row ordinal alongside the data.
2. Anti-semi-join existing rows against incoming on
   ``match_by_names`` to identify matching ordinals per source
   file.
3. For each source file with at least one match: build an updated
   roaring bitmap (existing DV ∪ matched ordinals), encode as a
   new DV blob, write to a fresh ``.bin`` file, emit
   ``Remove(old_addfile) + Add(same_addfile_with_new_DV)``. If
   the new DV's cardinality equals the file's row count, emit
   only ``Remove`` (the file becomes fully dead).
4. Incoming rows are then written as one or more new AddFiles
   under the right partition prefixes.

This implementation makes one strong assumption: ``match_by_names``
is unique on the existing side. If existing has duplicate keys for
a value present in incoming, *all* duplicates are deleted and the
incoming row replaces the lot. This matches Delta MERGE-INTO's
default behavior on duplicate-source matches.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Iterable as IterableABC
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Sequence

import pyarrow as pa

from yggdrasil.arrow.cast import any_to_arrow_table
from yggdrasil.data.schema import Field, Schema
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaTypes, Mode, MimeType, MimeTypes
from yggdrasil.io.buffer.base import TabularIO
from .actions import (
    AddFile,
    CommitInfo,
    Metadata,
    Protocol,
    RemoveFile,
)
from .commit import write_commit
from .constants import (
    DEFAULT_ENGINE_INFO,
    DEFAULT_FRESH_WRITER_VERSION,
    DV_DIR_NAME,
    MAX_LEGACY_WRITER_VERSION,
    READER_VERSION_FEATURES,
    SUPPORTED_READER_VERSION_LEGACY,
    SUPPORTED_WRITER_FEATURES,
    WRITER_VERSION_FEATURES,
)
from .deletion_vector import (
    MAX_INLINE_DV_BYTES,
    DeletionVectorDescriptor,
    decode_dv_blob,
    decode_inline_descriptor,
    empty_bitmap,
    encode_dv_blob,
    make_inline_descriptor,
)
from .replay import ReplayResult, replay_log
from ..folder_io import (
    FolderIO,
    FolderOptions,
)

if TYPE_CHECKING:
    from pyroaring import BitMap64  # type: ignore[import-untyped]
    from yggdrasil.io.fs import Path

__all__ = ["DeltaIO", "DeltaOptions"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class DeltaOptions(FolderOptions):
    """Partitioned-folder options + Delta knobs.

    :param parquet_compression: child parquet compression. Default
        ``snappy`` matches Delta's typical convention.
    :param commit_info_engine: written into ``CommitInfo.engineInfo``.
    :param require_existing_table: refuse OVERWRITE on a non-existent
        table when True. Default False.
    :param dv_inline_threshold: maximum DV blob size (bytes) for
        inline storage. Above this, externalize to a ``.bin`` file
        under ``deletion_vectors/``. Default
        :data:`MAX_INLINE_DV_BYTES`.
    :param checkpoint_v2: emit v2-style checkpoints when checkpointing.
        We don't auto-checkpoint on write in this version, but the
        flag is plumbed through for future :meth:`DeltaIO.checkpoint`.
    """

    parquet_compression: str = "snappy"
    commit_info_engine: str = DEFAULT_ENGINE_INFO
    require_existing_table: bool = False
    dv_inline_threshold: int = MAX_INLINE_DV_BYTES
    checkpoint_v2: bool = False


# ---------------------------------------------------------------------------
# DeltaIO
# ---------------------------------------------------------------------------


class DeltaIO(FolderIO):
    """Delta Lake table as a partitioned folder.

    Construction:

        >>> io = DeltaIO(path="/tables/trades/")
        >>> for child in io.iter_children():
        ...     print(child.path, child.static_values)

    Reads replay ``_delta_log/`` and yield one child IO per live
    AddFile, applying any DV at batch time. Writes go through
    ``_write_arrow_batches`` and produce a single transaction.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_dir: Path = self.path / "_delta_log"
        self._replay_cache: ReplayResult | None = None
        self._replay_cache_version: int = -1
        self._column_rename_map_cache: dict[str, str] | None | bool = False  # False = not computed

    @classmethod
    def options_class(cls):
        return DeltaOptions

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.DELTA_FOLDER

    def _default_child_media_type(self) -> Any:
        return MediaTypes.PARQUET

    # ==================================================================
    # Hide the log directory from generic enumeration
    # ==================================================================

    def _is_ignored_path(self, child: Path) -> bool:
        if child.name == "_delta_log":
            return True
        if child.name.startswith(("_", ".")):
            return True
        return False

    # ==================================================================
    # Replay caching
    # ==================================================================

    def _replay(self) -> ReplayResult:
        """Replay the log, cached per-instance.

        Cache key is the latest commit version. ``replay_log`` is
        called only when a new commit appears since our last sample,
        so iter_fragments / collect_schema in tight loops don't
        re-walk every time.
        """
        from .replay import latest_commit_version

        latest = latest_commit_version(self._log_dir)
        if latest == -1:
            return ReplayResult.empty()
        if (
            self._replay_cache is not None
            and self._replay_cache_version == latest
        ):
            return self._replay_cache

        result = replay_log(self._log_dir)
        self._replay_cache = result
        self._replay_cache_version = result.version
        return result

    def _invalidate_replay_cache(self) -> None:
        self._replay_cache = None
        self._replay_cache_version = -1
        self._column_rename_map_cache = False

    # ==================================================================
    # Partition column resolution — from Metadata
    # ==================================================================

    def _resolve_partition_columns(
        self,
        options: "FolderOptions | None" = None,
    ) -> "tuple[Field, ...]":
        if options is not None and options.partition_columns is not None:
            return tuple(Field.from_any(c) for c in options.partition_columns)
        if self._partition_columns is not None:
            return self._partition_columns

        try:
            replay = self._replay()
        except (FileNotFoundError, ValueError):
            return super()._infer_partition_columns()

        if replay.metadata is None:
            return ()

        out: list[Field] = []
        for name in replay.metadata.partition_columns:
            f = replay.metadata.schema.get(name)
            if f is None:
                out.append(Field.from_any(name))
            else:
                out.append(f)
        return tuple(out)

    # ==================================================================
    # Schema collection — from Metadata, not file footers
    # ==================================================================

    def _collect_schema(self, options: DeltaOptions) -> Schema:
        try:
            replay = self._replay()
        except FileNotFoundError:
            return Schema.empty()
        if replay.metadata is None:
            return Schema.empty()
        return replay.metadata.schema

    def is_empty(self) -> bool:
        try:
            replay = self._replay()
        except FileNotFoundError:
            return True
        return len(replay.live_files) == 0

    # ==================================================================
    # Children enumeration — driven by the replay
    # ==================================================================

    def _iter_children(
        self,
        options: "DeltaOptions",
    ) -> "Iterator[TabularIO | BytesIO]":
        """Yield one child :class:`BytesIO` (concrete tabular leaf) per live AddFile.

        Each child has:

        - ``parent`` set to ``self``;
        - ``partition_values`` populated from the AddFile's
          declared partition mapping;
        - ``deletion_vector`` set to the AddFile's DV descriptor
          (or ``None``) so :meth:`_read_child_batches` can apply
          the row filter at read time.
        """
        try:
            replay = self._replay()
        except FileNotFoundError:
            return
        if replay.metadata is None:
            return

        for add in replay.live_files:
            child_path = self.path.joinpath(*add.path.split("/"))
            # Skip the exists() check here — the replay says the file is
            # live, and we'll get a clear FileNotFoundError on read if it's
            # actually gone. Saves one HeadObject per AddFile on S3-backed
            # tables which adds up fast on wide tables.

            child_io = self._open_file_child(child_path)
            if child_io is None:
                continue

            self._attach(child_io)
            # Stamp partition values onto the leaf's
            # ``static_values`` slot (Arrow-typed via the
            # FolderIO helper) so the inherited
            # :meth:`TabularIO.read_arrow_batches` injection
            # picks them up — no per-batch wrapping at the
            # Delta level. ``deletion_vector`` / ``delta_add_file``
            # remain free attributes the DV-aware read loop
            # below consults directly.
            if add.partition_values:
                child_io.static_values = self._typed_partition_static(
                    add.partition_values,
                )
            child_io.deletion_vector = add.deletion_vector
            child_io.delta_add_file = add
            yield child_io

    # ==================================================================
    # Read derivation — chain children, apply DV + predicate per file
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: DeltaOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream batches over live AddFiles, with three filters:

        1. **Partition pruning** — when ``options.predicate`` references
           only partition columns, evaluate the predicate against each
           AddFile's partition values and skip files whose values
           can't satisfy it. One round-trip avoided per pruned file.
        2. **Deletion vector** — decode the AddFile's DV (if any) and
           drop rows whose ordinal falls inside the bitmap. The DV
           decode is per-file; ordinals reset per AddFile.
        3. **Row predicate** — applied at the
           :class:`TabularIO` layer in :meth:`_iter_public_batches`,
           so we don't repeat it here. Predicates that reference
           data columns flow through naturally; partition-only
           predicates are caught by step (1).

        Partition columns are injected by the leaf's own
        :meth:`TabularIO.read_arrow_batches` via the
        ``static_values`` stamp set in :meth:`_iter_children`, so the
        DV mask shrinks data and partition columns in lockstep.
        """
        predicate = getattr(options, "predicate", None)
        prune_fn = self._build_partition_prune_fn(predicate)

        # Pre-compute column rename map once for the whole read, not per-file.
        rename_map = self._build_column_rename_map()

        for child_io in self._iter_children(options):
            if not isinstance(child_io, TabularIO):
                continue

            add = getattr(child_io, "delta_add_file", None)
            if (
                prune_fn is not None
                and add is not None
                and not prune_fn(add.partition_values or {})
            ):
                continue

            yield from self._read_addfile_batches(child_io, options, rename_map=rename_map)

    def _read_addfile_batches(
        self,
        child_io: TabularIO,
        options: DeltaOptions,
        *,
        rename_map: dict[str, str] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Drain one AddFile's batches with its DV applied row-wise."""
        dv = getattr(child_io, "deletion_vector", None)
        dv_bitmap = (
            self._load_dv_bitmap_from_descriptor(dv)
            if dv is not None else None
        )
        has_dv = dv_bitmap is not None and len(dv_bitmap) > 0


        with child_io:
            row_offset = 0
            for batch in child_io.read_arrow_batches(options=options):
                n = batch.num_rows
                if has_dv:
                    mask_values = [
                        (row_offset + i) not in dv_bitmap
                        for i in range(n)
                    ]
                    row_offset += n
                    if not any(mask_values):
                        continue
                    if not all(mask_values):
                        mask = pa.array(mask_values, type=pa.bool_())
                        batch = batch.filter(mask)
                else:
                    row_offset += n

                if batch.num_rows == 0:
                    continue

                if rename_map:
                    batch = _rename_batch(batch, rename_map)

                yield batch

    def _build_column_rename_map(self) -> dict[str, str] | None:
        """Build physical→logical column rename map for columnMapping tables.

        Returns ``None`` when column mapping is off (mode ``"none"`` or
        absent) — the common case for tables we wrote ourselves. When
        the table uses ``name`` or ``id`` mapping, returns a dict from
        physical parquet column name to logical Delta column name,
        extracted from each field's ``delta.columnMapping.physicalName``
        metadata entry.

        Cached per-instance alongside the replay cache.
        """
        if self._column_rename_map_cache is not False:
            return self._column_rename_map_cache  # type: ignore[return-value]

        try:
            replay = self._replay()
        except (FileNotFoundError, ValueError):
            self._column_rename_map_cache = None
            return None
        if replay.metadata is None:
            self._column_rename_map_cache = None
            return None

        mode = replay.metadata.configuration.get("delta.columnMapping.mode", "none")
        if mode == "none":
            self._column_rename_map_cache = None
            return None

        rename: dict[str, str] = {}
        _collect_physical_to_logical(replay.metadata.schema, rename)
        result = rename or None
        self._column_rename_map_cache = result
        return result

    def _build_partition_prune_fn(self, predicate: Any):

        """Build a callable that decides whether to scan a partition.

        Returns ``None`` when there's no predicate, when the
        predicate references non-partition columns, or when
        compilation fails. The callable takes an AddFile's
        ``partition_values`` mapping and returns ``False`` to skip
        the file. Three-valued logic: UNKNOWN keeps the file (we
        can't prove the predicate fails on this partition).
        """
        if predicate is None:
            return None

        try:
            from yggdrasil.data.expr.nodes import free_columns
        except Exception:
            return None
        try:
            cols = set(free_columns(predicate))
        except Exception:
            return None
        if not cols:
            return None

        partition_names = {f.name for f in self._resolve_partition_columns()}
        if not cols.issubset(partition_names):
            return None

        try:
            fn = predicate.to_python()
        except Exception:
            return None

        def keep(partition_values: dict) -> bool:
            try:
                verdict = fn(partition_values or {})
            except Exception:
                return True  # Can't decide → don't prune.
            return verdict is not False

        return keep

    # ==================================================================
    # DV loading — handles all three storage types
    # ==================================================================

    def _load_dv_bitmap(self, add: AddFile) -> "BitMap64 | None":
        """Decode the AddFile's DV to a bitmap, or return None for no DV."""
        return self._load_dv_bitmap_from_descriptor(add.deletion_vector)

    def _load_dv_bitmap_from_descriptor(
        self,
        descriptor: "DeletionVectorDescriptor | None",
    ) -> "BitMap64 | None":
        """Decode a DV descriptor to a bitmap, or return None for no DV."""
        if descriptor is None:
            return None
        if descriptor.is_empty:
            return empty_bitmap()

        if descriptor.storage_type == "i":
            return decode_inline_descriptor(descriptor)

        if descriptor.storage_type == "p":
            bin_path = self._resolve_relative_path(descriptor.path_or_inline)
            return self._load_dv_bitmap_from_file(bin_path, descriptor)

        if descriptor.storage_type == "u":
            bin_path = self._resolve_uuid_dv_path(descriptor.path_or_inline)
            return self._load_dv_bitmap_from_file(bin_path, descriptor)

        raise ValueError(
            f"Unhandled DV storageType {descriptor.storage_type!r}."
        )

    def _resolve_relative_path(self, rel: str) -> Path:
        return self.path.joinpath(*rel.split("/"))

    def _resolve_uuid_dv_path(self, encoded: str) -> Path:
        """Resolve a ``storageType="u"`` DV path.

        Wire form: ``<z85-uuid>[suffix]`` where the optional suffix
        starts with ``/`` and gives the path relative to the table
        root. Without a suffix, the file lives at
        ``<table>/deletion_vectors/<uuid-hex>.bin`` per the
        reference convention.

        We accept both forms on read: the suffix takes precedence;
        the bare-UUID form falls through to the default location.
        """
        # The Z85 UUID is 20 ASCII chars (encoding a 16-byte UUID).
        if len(encoded) >= 20 and len(encoded) > 20 and encoded[20] == "/":
            suffix = encoded[21:]
            return self._resolve_relative_path(suffix)
        # Bare UUID — convert to hex form and look in the default dir.
        if len(encoded) == 20:
            from .deletion_vector import _z85_decode

            uid_bytes = _z85_decode(encoded)
            uid_hex = uid_bytes.hex()
            return self.path / DV_DIR_NAME / f"{uid_hex}.bin"
        raise ValueError(
            f"Malformed 'u' DV path {encoded!r}: expected 20-char Z85 "
            "UUID, optionally followed by '/<relative-path>'."
        )

    def _load_dv_bitmap_from_file(
        self,
        bin_path: Path,
        descriptor: DeletionVectorDescriptor,
    ) -> "BitMap64":
        """Read the DV blob from *bin_path*, decode using *descriptor*."""
        if not bin_path.exists():
            raise FileNotFoundError(
                f"DV file {bin_path!r} referenced by descriptor "
                f"{descriptor!r} not found."
            )

        offset = descriptor.offset or 0
        size = descriptor.size_in_bytes
        # Read just our slice — DV files can hold many DVs.
        blob = bin_path.pread(n=size, pos=offset)
        if len(blob) != size:
            raise ValueError(
                f"DV pread short: requested {size} bytes from "
                f"{bin_path!r}@{offset}, got {len(blob)}."
            )

        # CRC presence: total size > 1 + 4 + payload_size signals
        # a trailing CRC.
        import struct
        (payload_size,) = struct.unpack_from("<I", blob, 1)
        has_crc = size == 5 + payload_size + 4

        return decode_dv_blob(
            blob,
            expected_cardinality=descriptor.cardinality,
            has_crc=has_crc,
        )

    # ==================================================================
    # Write paths
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: "IterableABC[pa.RecordBatch]",
        options: DeltaOptions,
    ) -> None:
        """Single-transaction write dispatch.

        Resolution rules (applied to the resolved
        :class:`Mode` from :meth:`_resolve_save_mode`):

        - ``IGNORE`` → no-op, no commit.
        - ``UPSERT`` → DV-emitting merge-on-read (see
          :meth:`_delta_upsert`); requires
          ``options.match_by_names``.
        - ``OVERWRITE`` / ``APPEND`` → stage parquet via
          :class:`FolderIO`'s child machinery, then a single
          commit with Add (and Remove for OVERWRITE) actions.
        - Anything else raises — Delta's commit semantics are
          enumerable, no point inventing modes that don't map to
          a transaction.

        Failures during the parquet write are caught by the
        inherited path; failures during the commit roll back the
        new parquet files before re-raising so the table is left
        consistent with whatever's already been committed.
        """
        mode = self._resolve_save_mode(options.mode)
        if mode is Mode.IGNORE:
            return
        if mode is Mode.UPSERT:
            self._delta_upsert(batches, options)
            return
        if mode in (Mode.OVERWRITE, Mode.APPEND):
            self._delta_overwrite_or_append(batches, options, mode)
            return
        raise NotImplementedError(
            f"DeltaIO supports OVERWRITE / APPEND / UPSERT; got "
            f"resolved action {mode!r}."
        )

    # ------------------------------------------------------------------
    # OVERWRITE / APPEND
    # ------------------------------------------------------------------

    def _delta_overwrite_or_append(
        self,
        batches: "IterableABC[pa.RecordBatch]",
        options: DeltaOptions,
        mode: Mode,
    ) -> None:
        """Write parquet, then commit Add/Remove actions."""
        try:
            existing = self._replay()
        except FileNotFoundError:
            existing = ReplayResult.empty()

        if (
            options.require_existing_table
            and existing.metadata is None
        ):
            raise FileNotFoundError(
                f"DeltaIO write with require_existing_table=True but "
                f"{self.path!r} is not an existing Delta table."
            )

        partition_cols = self._resolve_partition_columns(options)
        partition_names = tuple(c.name for c in partition_cols)

        before_files = self._scan_data_files()
        write_options = options.copy(
            mode=Mode.OVERWRITE if mode is Mode.OVERWRITE else Mode.APPEND,
        )

        # Probe iterator emptiness so empty writes still produce a
        # valid no-op commit when the table is new.
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            self._handle_empty_write(mode, existing, partition_names, options)
            return

        try:
            super()._write_arrow_batches(
                chain([first], batch_iter), write_options,
            )
        except Exception:
            self._cleanup_new_files(before_files)
            raise

        after_files = self._scan_data_files()
        new_relpaths = sorted(after_files - before_files)

        new_adds = self._build_addfiles_from_relpaths(
            new_relpaths, partition_names,
        )

        if mode is Mode.OVERWRITE:
            target_schema = self._schema_from_first_file(new_relpaths)
        else:
            target_schema = (
                existing.metadata.schema
                if existing.metadata is not None
                else self._schema_from_first_file(new_relpaths)
            )

        try:
            if mode is Mode.OVERWRITE:
                self._commit_overwrite(
                    new_adds, existing, partition_names,
                    target_schema, options,
                )
            else:
                self._commit_append(
                    new_adds, existing, partition_names,
                    target_schema, options,
                )
        except Exception:
            self._cleanup_new_files(before_files)
            raise

        self._invalidate_replay_cache()

    def _handle_empty_write(
        self,
        mode: Mode,
        existing: ReplayResult,
        partition_names: Sequence[str],
        options: DeltaOptions,
    ) -> None:
        """Empty-input write paths.

        APPEND of nothing is a no-op (no commit emitted — Delta
        commits are not free, and a no-op append doesn't change
        the table).

        OVERWRITE of nothing on a non-empty table is meaningful:
        emit Removes for every previously-live file plus a new
        Metadata if needed. OVERWRITE of nothing on an empty
        table initializes the log (commit version 0 with
        Protocol + Metadata) so subsequent reads see an existing
        table.
        """
        if mode is Mode.APPEND:
            return

        if existing.live_files:
            self._commit_overwrite(
                new_files=[],
                previous=existing,
                partition_names=partition_names,
                schema=Schema.empty(),
                options=options,
            )
        else:
            # Initialize an empty table.
            self._commit_append(
                new_files=[],
                previous=existing,
                partition_names=partition_names,
                schema=Schema.empty(),
                options=options,
            )
        self._invalidate_replay_cache()

    # ------------------------------------------------------------------
    # UPSERT — DV-emitting merge-on-read
    # ------------------------------------------------------------------

    def _delta_upsert(
        self,
        batches: "IterableABC[pa.RecordBatch]",
        options: DeltaOptions,
    ) -> None:
        """UPSERT via DV-emitting merge-on-read.

        Read existing rows with their (source_file, ordinal) tags;
        identify which existing rows match incoming on
        ``options.match_by_names``; write new DVs for the affected
        files; write the incoming rows as new AddFiles. One
        transaction.
        """
        match_by = options.match_by_names
        if not match_by:
            raise ValueError(
                "DeltaIO UPSERT requires options.match_by_names. "
                "For 'replace everything,' use Mode.OVERWRITE instead."
            )
        match_by = tuple(match_by)

        try:
            existing = self._replay()
        except FileNotFoundError:
            existing = ReplayResult.empty()

        if existing.metadata is None:
            # Fresh table — UPSERT degenerates to a regular append.
            self._delta_overwrite_or_append(batches, options, Mode.APPEND)
            return

        partition_cols = self._resolve_partition_columns(options)
        partition_names = tuple(c.name for c in partition_cols)

        # Materialize incoming. The merge needs random access
        # to the keys; streaming would not help here.
        incoming_table = any_to_arrow_table(batches, options)
        if incoming_table.num_rows == 0:
            return

        # Build the set of incoming match keys for fast membership
        # test on the existing side.
        incoming_keys = self._extract_key_set(incoming_table, match_by)
        if not incoming_keys:
            return

        # When ``update_column_names`` is set, columns outside the update
        # list must keep their existing values on a key match. Read the
        # current full table and let :meth:`_restrict_update_columns`
        # rebuild the incoming side with preserved column values pulled
        # from the existing rows that share each key. Fresh rows (no
        # existing match) keep null in the preserved columns.
        if options.update_column_names is not None:
            existing_table = any_to_arrow_table(
                self._read_arrow_batches(options.copy(read_seek=0)),
                options,
            )
            incoming_table = self._restrict_update_columns(
                existing_table, incoming_table,
                match_by=match_by,
                update_column_names=tuple(options.update_column_names),
            )

        # For each live AddFile that has at least one row matching
        # an incoming key: build the new DV.
        before_files = self._scan_data_files()
        dv_writes: list[_DVWritePlan] = []

        for add in existing.live_files:
            plan = self._plan_dv_for_addfile(
                add, match_by, incoming_keys,
            )
            if plan is not None:
                dv_writes.append(plan)

        # Write new DVs to disk. Each plan has its own descriptor
        # produced once finalized.
        new_dv_descriptors: dict[str, DeletionVectorDescriptor] = {}
        try:
            for plan in dv_writes:
                new_dv_descriptors[plan.add.path] = self._materialize_dv(
                    plan, options=options,
                )
        except Exception:
            # Clean up any DV files we wrote so far.
            for plan in dv_writes:
                if plan.materialized_path is not None:
                    try:
                        plan.materialized_path.remove(allow_not_found=True)
                    except Exception:
                        pass
            raise

        # Write the incoming rows as new parquet files.
        write_options = options.copy(
            mode=Mode.APPEND,
        )
        try:
            super()._write_arrow_batches(
                incoming_table.to_batches(
                    max_chunksize=options.row_size or None,
                ),
                write_options,
            )
        except Exception:
            self._cleanup_new_files(before_files)
            for path in (p.materialized_path for p in dv_writes):
                if path is not None:
                    try:
                        path.remove(allow_not_found=True)
                    except Exception:
                        pass
            raise

        after_files = self._scan_data_files()
        new_relpaths = sorted(after_files - before_files)

        # Filter out DV .bin files from the data set (they'd be
        # under deletion_vectors/, but our _scan_data_files already
        # excludes non-parquet files; defense in depth).
        new_relpaths = [p for p in new_relpaths if p.endswith(".parquet")]
        new_adds = self._build_addfiles_from_relpaths(
            new_relpaths, partition_names,
        )

        # Build the action list.
        actions: list[Any] = [
            CommitInfo(
                timestamp=int(time.time() * 1000),
                operation="MERGE",
                operation_parameters={"matchBy": list(match_by)},
                engine_info=options.commit_info_engine,
            ),
        ]
        # Promote protocol if needed (DVs require feature support).
        new_protocol = self._protocol_for_dv_write(existing.protocol)
        if new_protocol is not None:
            actions.append(new_protocol)

        # For each DV plan: emit Remove(old) + Add(updated).
        # If the new DV covers every row in the file (cardinality ==
        # original row count), skip the Add — the file is dead.
        dead_files: set[str] = set()
        now_ms = int(time.time() * 1000)
        for plan in dv_writes:
            new_dv = new_dv_descriptors[plan.add.path]
            file_dead = (
                plan.original_row_count is not None
                and new_dv.cardinality >= plan.original_row_count
            )
            actions.append(self._build_remove(plan.add, now_ms, with_dv=True))
            if file_dead:
                dead_files.add(plan.add.path)
            else:
                actions.append(
                    dataclasses.replace(plan.add, deletion_vector=new_dv)
                )

        for add in new_adds:
            actions.append(add)

        try:
            write_commit(
                self._log_dir,
                existing.version + 1,
                actions,
            )
        except Exception:
            # Clean up the parquet AND DV files we wrote this commit.
            self._cleanup_new_files(before_files)
            for plan in dv_writes:
                if plan.materialized_path is not None:
                    try:
                        plan.materialized_path.remove(allow_not_found=True)
                    except Exception:
                        pass
            raise

        self._invalidate_replay_cache()

    # ==================================================================
    # UPSERT helpers
    # ==================================================================

    @staticmethod
    def _extract_key_set(
        table: pa.Table,
        match_by: Sequence[str],
    ) -> set[tuple]:
        """Build a Python set of key tuples from the incoming table."""
        missing = [c for c in match_by if c not in table.column_names]
        if missing:
            raise ValueError(
                f"UPSERT match_by columns missing from incoming: "
                f"{missing!r}. Available: {list(table.column_names)!r}."
            )
        cols = [table[c].to_pylist() for c in match_by]
        return set(zip(*cols))

    def _plan_dv_for_addfile(
        self,
        add: AddFile,
        match_by: Sequence[str],
        incoming_keys: set[tuple],
    ) -> "_DVWritePlan | None":
        """Build a DV write plan for *add* if any rows match.

        Reads the parquet file once, collects ordinals of matching
        rows, OR's with the existing DV, and stages an updated
        bitmap. Returns ``None`` if no rows match (no DV write
        needed for this file).
        """
        child_path = self.path.joinpath(*add.path.split("/"))
        if not child_path.exists():
            return None

        existing_dv = self._load_dv_bitmap(add)
        new_bitmap = empty_bitmap()
        if existing_dv is not None:
            new_bitmap |= existing_dv

        child_io = self._open_file_child(child_path)
        if not isinstance(child_io, TabularIO):
            return None

        # We need the *full* file row count for the dead-file check
        # later (skip the Add if the new DV covers everything).
        row_count = 0
        match_count_added = 0

        with child_io:
            row_offset = 0
            for batch in child_io.read_arrow_batches():
                n = batch.num_rows
                if n == 0:
                    continue

                missing = [c for c in match_by if c not in batch.schema.names]
                if missing:
                    raise ValueError(
                        f"UPSERT match_by column(s) {missing!r} not in "
                        f"file {add.path!r}. The match-by columns must "
                        "exist in every data file's schema."
                    )

                cols = [batch[c].to_pylist() for c in match_by]
                for i in range(n):
                    key = tuple(c[i] for c in cols)
                    if key in incoming_keys:
                        ordinal = row_offset + i
                        # Only count newly-added ordinals — existing
                        # DV may have already covered them (but
                        # they wouldn't have appeared in the read
                        # unless the existing DV is wrong; defense).
                        if (
                            existing_dv is None
                            or ordinal not in existing_dv
                        ):
                            new_bitmap.add(ordinal)
                            match_count_added += 1
                row_offset += n
                row_count += n

        if match_count_added == 0:
            return None

        return _DVWritePlan(
            add=add,
            new_bitmap=new_bitmap,
            original_row_count=row_count + (
                len(existing_dv) if existing_dv is not None else 0
            ),
            materialized_path=None,
        )

    def _materialize_dv(
        self,
        plan: "_DVWritePlan",
        *,
        options: DeltaOptions,
    ) -> DeletionVectorDescriptor:
        """Encode and persist *plan*'s bitmap; return a descriptor.

        Inline if the framed blob is below
        ``options.dv_inline_threshold``, else externalize to
        ``deletion_vectors/<uuid>.bin``. Mutates ``plan.materialized_path``
        for cleanup.
        """
        framed = encode_dv_blob(plan.new_bitmap, include_crc=True)
        if len(framed) <= options.dv_inline_threshold:
            return make_inline_descriptor(plan.new_bitmap)

        # Externalize. One DV per .bin file is the simplest scheme
        # — sharing files saves space but complicates cleanup. We
        # use one-per-file here.
        dv_dir = self.path / DV_DIR_NAME
        dv_dir.mkdir(parents=True, exist_ok=True)

        bin_name = f"{uuid.uuid4().hex}.bin"
        bin_path = dv_dir / bin_name
        bin_path.write_bytes(framed)
        plan.materialized_path = bin_path

        return DeletionVectorDescriptor(
            storage_type="p",
            path_or_inline=f"{DV_DIR_NAME}/{bin_name}",
            size_in_bytes=len(framed),
            cardinality=len(plan.new_bitmap),
            offset=0,
        )

    # ==================================================================
    # Commit construction
    # ==================================================================

    def _commit_append(
        self,
        new_files: Sequence[AddFile],
        previous: ReplayResult,
        partition_names: Sequence[str],
        schema: Schema,
        options: DeltaOptions,
    ) -> None:
        actions: list[Any] = [
            CommitInfo(
                timestamp=int(time.time() * 1000),
                operation="WRITE",
                operation_parameters={"mode": "Append"},
                engine_info=options.commit_info_engine,
                is_blind_append=True,
            ),
        ]

        if previous.metadata is None:
            protocol = self._fresh_protocol(write_dv_capable=False)
            metadata = self._fresh_metadata(schema, partition_names)
            actions.append(protocol)
            actions.append(metadata)
            target_version = 0
        else:
            self._verify_writer_known(previous.protocol)
            target_version = previous.version + 1

        for add in new_files:
            actions.append(add)

        write_commit(self._log_dir, target_version, actions)

    def _commit_overwrite(
        self,
        new_files: Sequence[AddFile],
        previous: ReplayResult,
        partition_names: Sequence[str],
        schema: Schema,
        options: DeltaOptions,
    ) -> None:
        actions: list[Any] = [
            CommitInfo(
                timestamp=int(time.time() * 1000),
                operation="WRITE",
                operation_parameters={"mode": "Overwrite"},
                engine_info=options.commit_info_engine,
                is_blind_append=False,
            ),
        ]

        if previous.metadata is None:
            protocol = self._fresh_protocol(write_dv_capable=False)
            metadata = self._fresh_metadata(schema, partition_names)
            actions.append(protocol)
            actions.append(metadata)
            target_version = 0
        else:
            self._verify_writer_known(previous.protocol)
            new_metadata = dataclasses.replace(
                previous.metadata,
                schema=schema,
                partition_columns=tuple(partition_names),
            )
            actions.append(new_metadata)
            target_version = previous.version + 1

        now_ms = int(time.time() * 1000)
        for old in previous.live_files:
            actions.append(self._build_remove(old, now_ms, with_dv=True))
        for add in new_files:
            actions.append(add)

        write_commit(self._log_dir, target_version, actions)

    @staticmethod
    def _build_remove(
        add: AddFile, deletion_ts: int, *, with_dv: bool,
    ) -> RemoveFile:
        return RemoveFile(
            path=add.path,
            deletion_timestamp=deletion_ts,
            data_change=True,
            extended_file_metadata=True,
            partition_values=dict(add.partition_values),
            size=add.size,
            tags=dict(add.tags) if add.tags is not None else None,
            deletion_vector=add.deletion_vector if with_dv else None,
            base_row_id=add.base_row_id,
            default_row_commit_version=add.default_row_commit_version,
        )

    # ==================================================================
    # Protocol management
    # ==================================================================

    def _fresh_protocol(self, *, write_dv_capable: bool) -> Protocol:
        """Build the Protocol action for a fresh table.

        Default fresh tables use the legacy integer protocol —
        smaller commits, broader reader compatibility. If
        ``write_dv_capable`` is True (i.e. this write emits DVs),
        we promote to the table-features model.
        """
        if write_dv_capable:
            return Protocol(
                min_reader_version=READER_VERSION_FEATURES,
                min_writer_version=WRITER_VERSION_FEATURES,
                reader_features=("deletionVectors",),
                writer_features=("deletionVectors",),
            )
        return Protocol(
            min_reader_version=SUPPORTED_READER_VERSION_LEGACY,
            min_writer_version=DEFAULT_FRESH_WRITER_VERSION,
        )

    def _protocol_for_dv_write(
        self, current: Protocol | None,
    ) -> Protocol | None:
        """Return a Protocol action to emit, or None if no change needed.

        UPSERT may be the first operation that emits a DV against
        a previously DV-free table. In that case we need to bump
        the protocol so other writers respect the new feature.
        """
        if current is None:
            return self._fresh_protocol(write_dv_capable=True)

        already_capable = (
            current.min_reader_version >= READER_VERSION_FEATURES
            and "deletionVectors" in current.reader_features
        )
        if already_capable:
            return None

        # Upgrade. Preserve any features the previous protocol
        # already had.
        new_reader_features = tuple(
            sorted(set(current.reader_features) | {"deletionVectors"})
        )
        new_writer_features = tuple(
            sorted(set(current.writer_features) | {"deletionVectors"})
        )
        return Protocol(
            min_reader_version=max(
                current.min_reader_version, READER_VERSION_FEATURES,
            ),
            min_writer_version=max(
                current.min_writer_version, WRITER_VERSION_FEATURES,
            ),
            reader_features=new_reader_features,
            writer_features=new_writer_features,
        )

    def _verify_writer_known(self, protocol: Protocol | None) -> None:
        if protocol is None:
            return
        if protocol.min_writer_version >= WRITER_VERSION_FEATURES:
            # Table-features model. Any writer feature outside our
            # supported set means we can't safely write.
            unsupported = [
                f for f in protocol.writer_features
                if f not in SUPPORTED_WRITER_FEATURES
            ]
            if unsupported:
                raise ValueError(
                    f"Delta table at {self.path!r} requires writer "
                    f"feature(s) {unsupported!r} that yggdrasil DeltaIO "
                    "does not implement. Refusing to write."
                )
            return

        if protocol.min_writer_version > MAX_LEGACY_WRITER_VERSION:
            raise ValueError(
                f"Delta table at {self.path!r} declares legacy "
                f"minWriterVersion={protocol.min_writer_version} > "
                f"{MAX_LEGACY_WRITER_VERSION}; refusing to write."
            )

    # ==================================================================
    # Metadata construction
    # ==================================================================

    @staticmethod
    def _fresh_metadata(
        schema: Schema,
        partition_names: Sequence[str],
    ) -> Metadata:
        return Metadata(
            id=str(uuid.uuid4()),
            schema=schema,
            partition_columns=tuple(partition_names),
            configuration={},
            created_time=int(time.time() * 1000),
        )

    # ==================================================================
    # File scanning + path manipulation
    # ==================================================================

    def _scan_data_files(self) -> set[str]:
        """List all parquet files under the table root, relative paths."""
        from ..folder_io import _walk_leaves

        out: set[str] = set()
        if not self.path.exists():
            return out
        for leaf in _walk_leaves(self.path, self._is_ignored_path):
            if not leaf.name.lower().endswith(".parquet"):
                continue
            try:
                rel_parts = leaf.relative_to(self.path).parts
            except Exception:
                continue
            out.add("/".join(rel_parts))
        return out

    def _cleanup_new_files(self, before_files: set[str]) -> None:
        try:
            after = self._scan_data_files()
        except Exception:
            return
        for rel in after - before_files:
            target = self.path.joinpath(*rel.split("/"))
            try:
                target.remove(allow_not_found=True)
            except Exception:
                pass

    def _build_addfiles_from_relpaths(
        self,
        relpaths: Sequence[str],
        partition_names: Sequence[str],
    ) -> list[AddFile]:
        out: list[AddFile] = []
        now_ms = int(time.time() * 1000)
        for rel in relpaths:
            target = self.path.joinpath(*rel.split("/"))
            try:
                size = int(target.size)
            except Exception:
                size = 0
            partition_values = self._partition_values_from_relpath(
                rel, partition_names,
            )
            out.append(AddFile(
                path=rel,
                partition_values=partition_values,
                size=size,
                modification_time=now_ms,
                data_change=True,
            ))
        return out

    def _partition_values_from_relpath(
        self, rel_path: str, partition_names: Sequence[str],
    ) -> dict[str, str | None]:
        from ..folder_io import _parse_kv_segment

        if not partition_names:
            return {}
        parts = rel_path.split("/")
        kv_segments = parts[:-1]
        if len(kv_segments) != len(partition_names):
            raise ValueError(
                f"Path {rel_path!r} has {len(kv_segments)} k=v "
                f"segment(s); expected {len(partition_names)}."
            )
        out: dict[str, str | None] = {}
        for segment, expected in zip(kv_segments, partition_names):
            kv = _parse_kv_segment(segment)
            if kv is None or kv[0] != expected:
                raise ValueError(
                    f"Path segment {segment!r} doesn't match expected "
                    f"partition column {expected!r}."
                )
            out[expected] = kv[1]
        return out

    def _schema_from_first_file(self, new_files: Sequence[str]) -> Schema:
        if not new_files:
            return Schema.empty()
        first_path = self.path.joinpath(*new_files[0].split("/"))
        first_io = TabularIO.from_path(first_path, media_type=MediaTypes.PARQUET)
        with first_io:
            return first_io.collect_schema()


# ---------------------------------------------------------------------------
# Internal: DV write plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class _DVWritePlan:
    """Internal: a planned DV update for one AddFile.

    Mutable: ``materialized_path`` is set after
    :meth:`DeltaIO._materialize_dv` writes the .bin file, so
    failure-path cleanup can find it.
    """
    add: AddFile
    new_bitmap: "BitMap64"
    original_row_count: int | None
    materialized_path: Path | None


# ---------------------------------------------------------------------------
# Column mapping helpers
# ---------------------------------------------------------------------------


def _collect_physical_to_logical(
    schema: Schema,
    out: dict[str, str],
    *,
    _prefix: str = "",
) -> None:
    """Walk *schema* fields and populate *out* with physical→logical mappings.

    Handles nested structs recursively. The physical name lives in
    each field's metadata under ``delta.columnMapping.physicalName``.
    """
    for field in schema.fields:
        meta = getattr(field, "metadata", None) or {}
        # Metadata keys can be bytes (Arrow convention) or strings.
        physical = None
        for key in (b"delta.columnMapping.physicalName", "delta.columnMapping.physicalName"):
            val = meta.get(key)
            if val is not None:
                physical = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
                break

        if physical and physical != field.name:
            out[_prefix + physical] = _prefix + field.name

        # Recurse into struct children so nested physical names get mapped.
        sub_fields = getattr(field, "fields", None)
        if sub_fields:
            child_prefix = (_prefix + field.name + ".") if _prefix else ""
            for sub in sub_fields:
                sub_meta = getattr(sub, "metadata", None) or {}
                sub_physical = None
                for key in (b"delta.columnMapping.physicalName", "delta.columnMapping.physicalName"):
                    val = sub_meta.get(key)
                    if val is not None:
                        sub_physical = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
                        break
                if sub_physical and sub_physical != sub.name:
                    out[sub_physical] = sub.name


def _rename_batch(
    batch: pa.RecordBatch,
    rename_map: dict[str, str],
) -> pa.RecordBatch:
    """Rename columns in *batch* using *rename_map* (physical→logical).

    Columns not in the map keep their existing name. Handles nested
    struct fields by recursively renaming child fields.
    """
    new_names = [rename_map.get(name, name) for name in batch.schema.names]

    # Recursively rename struct child fields.
    new_fields: list[pa.Field] = []
    for i, field in enumerate(batch.schema):
        logical_name = new_names[i]
        new_field = _rename_field(field, logical_name, rename_map)
        new_fields.append(new_field)

    new_schema = pa.schema(new_fields, metadata=batch.schema.metadata)
    return pa.RecordBatch.from_arrays(batch.columns, schema=new_schema)


def _rename_field(
    field: pa.Field,
    logical_name: str,
    rename_map: dict[str, str],
) -> pa.Field:
    """Rename a single Arrow field, recursing into struct children."""
    typ = field.type
    if pa.types.is_struct(typ):
        new_children: list[pa.Field] = []
        for child in typ:
            child_logical = rename_map.get(child.name, child.name)
            new_children.append(_rename_field(child, child_logical, rename_map))
        new_type = pa.struct(new_children)
        return pa.field(logical_name, new_type, nullable=field.nullable, metadata=field.metadata)
    return field.with_name(logical_name)

