"""YggIO: the :class:`PartitionedFolderIO` that ties the ygg protocol
together.

A ygg table is a :class:`FolderIO`-shaped tree with a small ``_ygg/``
metadata side-folder. The metadata side-folder holds exactly one
Arrow IPC manifest at any given time. Reads are O(1) (one manifest
open + footer parse), writes are atomic (stage-and-rename). There
is no version history; ``OVERWRITE`` is a hard delete of every
file referenced by the previous manifest.

Reads
-----

#. Read ``_ygg/manifest.arrow`` (cheap — one mmap-friendly
   Arrow IPC open).
#. Walk :attr:`Manifest.entries`. If the caller passed
   ``options.predicate``, evaluate it against each entry's
   :class:`ColumnStats` + ``partition_values`` and skip files the
   predicate can rule out.
#. For each surviving entry, mint a :class:`PrimitiveIO` for the
   data file with its ``partition_values`` populated, and yield
   it from :meth:`iter_children`.
#. Inside :meth:`_read_arrow_batches`: if a predicate is active,
   load the file's Arrow table, derive an ``int64`` row-index
   array via :func:`row_indices`, and yield ``table.take(indices)``
   — i.e. the predicate's row-level work happens once per file
   rather than once per batch.

Writes — OVERWRITE / APPEND
---------------------------

Stage Arrow IPC children via the inherited
:class:`PartitionedFolderIO` machinery (which routes by partition
key when partitions are declared); compute per-file stats for the
declared primary key columns; rewrite the manifest. On
``OVERWRITE`` the previous manifest's data files are unlinked
*after* the new manifest commits — readers always see one
consistent state.

Failure modes:

- New parquet/IPC write fails: data files just produced are
  cleaned up; the previous manifest is still pointed-at and
  still authoritative. Old data files are untouched.
- Manifest write fails: same as above — new data files cleaned
  up, previous manifest authoritative. The "we wrote files but
  the manifest didn't land" case never produces a partially-
  promoted commit.
- Old-file unlink fails on ``OVERWRITE``: best effort. The new
  manifest is already live and doesn't reference the orphaned
  files; a vacuum sweep can collect them.

Why this is faster than Delta on read
-------------------------------------

Delta reads are O(N_commits) — log replay touches every JSON
commit since the last checkpoint. Ygg reads are O(1): one Arrow
IPC open. The trade-off: every commit rewrites the full live file
list. For tables with a few million live files (a manifest entry
is ~100 bytes), that fits in a single Arrow IPC file and the
rewrite cost is amortized by the read win.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Iterable as IterableABC
from itertools import chain
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.schema import Field, Schema
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.enums import MediaTypes, MimeType, MimeTypes, Mode
from yggdrasil.io.fs import Path
from yggdrasil.io.tabular import TabularIO

from ..folder_io import (
    _coerce_partition_column,
    _inject_partition_columns,
    _parse_kv_segment,
)
from ..partitioned_io import PartitionedFolderIO, PartitionedOptions
from .commit import manifest_path, read_manifest, write_manifest
from .constants import (
    DEFAULT_ENGINE_INFO,
    META_DIR_NAME,
)
from .manifest import (
    ColumnStats,
    Manifest,
    ManifestEntry,
)
from .predicate import Predicate, row_indices


__all__ = ["YggIO", "YggOptions"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class YggOptions(PartitionedOptions):
    """Partitioned-folder options + ygg knobs.

    :param child_media_type: data file format. Defaults to Arrow IPC
        for the ygg fast path; pass :data:`MediaTypes.PARQUET` for a
        ygg-shaped folder whose data lives in parquet.
    :param primary_key_columns: column names whose per-file
        ``[min, max, null_count]`` stats are computed at write
        time and recorded on every manifest entry. Used by the
        :class:`Predicate` pruner at read time. ``None`` (default)
        defers to the IO's constructor-declared key list, which
        defers to the live manifest's. Pass an empty tuple to
        explicitly disable stats computation.
    :param predicate: read-time pre-filter. When set,
        :meth:`iter_children` skips entries the predicate can rule
        out from stats / partition values, and
        :meth:`_read_arrow_batches` resolves the predicate to an
        ``int64`` row-index array per file before yielding batches.
        ``None`` (default) is "no filter, fastest path."
    :param engine_info: written into every new manifest's metadata.
        Identifies the writer in commit history.
    :param require_existing_table: refuse OVERWRITE on a
        non-existent table when True. Default False.
    """

    primary_key_columns: Any = None
    predicate: Any = None
    engine_info: str = DEFAULT_ENGINE_INFO
    require_existing_table: bool = False


# ---------------------------------------------------------------------------
# YggIO
# ---------------------------------------------------------------------------


class YggIO(PartitionedFolderIO):
    """Ygg table as a partitioned folder.

    Construction:

        >>> io = YggIO(path="/tables/trades/", primary_key_columns=["id"])
        >>> for child in io.iter_children():
        ...     print(child.path, child.partition_values)

    Predicate read:

        >>> from yggdrasil.io.buffer.nested.ygg import eq
        >>> with YggIO(path="/tables/trades/") as io:
        ...     out = io.read_arrow_table(predicate=eq("id", 42))

    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(
        self,
        *args: Any,
        primary_key_columns: "Sequence[str] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._meta_dir: Path = self.path / META_DIR_NAME
        self._primary_key_columns: tuple[str, ...] | None = (
            tuple(primary_key_columns) if primary_key_columns is not None else None
        )
        self._manifest_cache: Manifest | None = None
        self._manifest_cache_mtime: int = -1

    @classmethod
    def options_class(cls):
        return YggOptions

    @classmethod
    def default_mime_type(cls) -> MimeType:
        return MimeTypes.YGG_FOLDER

    def _default_child_media_type(self) -> Any:
        """Arrow IPC is the canonical ygg data payload — fast,
        zero-decode, mmap-friendly. Callers wanting parquet data
        files inside a ygg-managed table pass
        ``options.child_media_type=MediaTypes.PARQUET``.
        """
        return MediaTypes.ARROW_IPC

    # ==================================================================
    # Hide the metadata directory from generic enumeration
    # ==================================================================

    def _is_ignored_path(self, child: Path) -> bool:
        if child.name == META_DIR_NAME:
            return True
        if child.name.startswith(("_", ".")):
            return True
        return False

    # ==================================================================
    # Manifest cache — keyed by manifest file mtime
    # ==================================================================

    def _load_manifest(self) -> Manifest | None:
        """Read the live manifest, cached per-instance.

        Cache key is the manifest file's mtime. We re-stat the file
        on every call to detect a fresh commit; the stat is cheap
        compared to the IPC parse.
        """
        target = manifest_path(self.path)
        if not target.exists():
            self._manifest_cache = None
            self._manifest_cache_mtime = -1
            return None

        try:
            mtime = int(target.stat().mtime)
        except Exception:
            mtime = -1

        if (
            self._manifest_cache is not None
            and self._manifest_cache_mtime == mtime
        ):
            return self._manifest_cache

        manifest = read_manifest(self.path)
        self._manifest_cache = manifest
        self._manifest_cache_mtime = mtime
        return manifest

    def _invalidate_manifest_cache(self) -> None:
        self._manifest_cache = None
        self._manifest_cache_mtime = -1

    # ==================================================================
    # Partition / primary-key column resolution
    # ==================================================================

    def _resolve_partition_columns(
        self,
        options: "PartitionedOptions | None" = None,
    ) -> "tuple[Field, ...]":
        if options is not None and options.partition_columns is not None:
            return tuple(
                _coerce_partition_column(c) for c in options.partition_columns
            )
        if self._partition_columns is not None:
            return self._partition_columns

        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return super()._infer_partition_columns()

        if manifest is None:
            return super()._infer_partition_columns()

        out: list[Field] = []
        for name in manifest.partition_columns:
            f = manifest.data_schema.get(name) if manifest.data_schema else None
            if f is None:
                out.append(_coerce_partition_column(name))
            else:
                out.append(f)
        return tuple(out)

    def _resolve_primary_key_columns(
        self, options: "YggOptions | None" = None,
    ) -> tuple[str, ...]:
        """Resolve which columns get stats. Order of precedence:

        1. ``options.primary_key_columns`` if explicitly set.
        2. The constructor-declared list.
        3. The live manifest's, when reading an existing table.
        4. Empty tuple — no stats, no pruning.
        """
        if options is not None and options.primary_key_columns is not None:
            return tuple(options.primary_key_columns)
        if self._primary_key_columns is not None:
            return self._primary_key_columns
        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return ()
        if manifest is None:
            return ()
        return manifest.primary_key_columns

    # ==================================================================
    # Schema collection — from manifest, not file footers
    # ==================================================================

    def _collect_schema(self, options: YggOptions) -> Schema:
        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return Schema.empty()
        if manifest is None:
            return Schema.empty()
        return manifest.data_schema

    def is_empty(self) -> bool:
        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return True
        return manifest is None or len(manifest.entries) == 0

    # ==================================================================
    # Children enumeration — driven by the manifest, predicate-pruned
    # ==================================================================

    def iter_children(
        self,
        options: "YggOptions | None" = None,
        **kwargs: Any,
    ) -> "Iterator[TabularIO | BytesIO]":
        """Yield one child :class:`PrimitiveIO` per surviving entry.

        Each child has:

        - ``parent`` set to ``self``;
        - ``partition_values`` populated from the entry's mapping;
        - ``ygg_manifest_entry`` set to the source entry so callers
          can reach the per-file stats / size / mtime;
        - the file's media type inferred from its extension.

        When ``options.predicate`` is set, entries the predicate
        can rule out from their partition values + per-column
        stats are silently skipped — they never produce a child
        IO. Entries whose backing file is missing on disk are also
        skipped (typically a partial restore).
        """
        self.check_options(options, overrides=locals())

        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return
        if manifest is None:
            return

        predicate: Predicate | None = (
            options.predicate if options is not None else None
        )

        for entry in manifest.entries:
            if predicate is not None and not predicate.matches_entry(entry):
                continue

            child_path = self.path.joinpath(*entry.path.split("/"))
            if not child_path.exists():
                continue

            child_io = self._open_file_child(child_path)
            if child_io is None:
                continue

            self._attach(child_io)
            child_io.partition_values = dict(entry.partition_values)
            child_io.ygg_manifest_entry = entry
            yield child_io

    # ==================================================================
    # Read derivation — predicate-aware
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: YggOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream batches. With a predicate, resolve int64 row indices
        per file and ``take(...)`` the matching slice.
        """
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        partition_cols = self._resolve_partition_columns(options)
        partition_by_name: Mapping[str, Field] = {
            c.name: c for c in partition_cols
        }
        predicate: Predicate | None = options.predicate

        for child_io in self.iter_children(options):
            if not isinstance(child_io, TabularIO):
                continue

            partition_values = getattr(child_io, "partition_values", {}) or {}

            with child_io:
                if predicate is None:
                    for batch in child_io.read_arrow_batches(options=options):
                        if batch.num_rows == 0:
                            continue
                        if partition_by_name and partition_values:
                            batch = _inject_partition_columns(
                                batch, partition_values, partition_by_name,
                            )
                        yield batch
                    continue

                # Predicate path: read the whole file as a table,
                # derive int64 row indices, take, and emit batches.
                # Reading once per file keeps the Arrow compute
                # vectorized; predicate evaluation per-batch would
                # work but loses some of the bitmap-merge wins.
                table = child_io.read_arrow_table(options=options)
                if table.num_rows == 0:
                    continue
                if partition_by_name and partition_values:
                    table = _inject_partition_into_table(
                        table, partition_values, partition_by_name,
                    )
                indices = row_indices(predicate, table)
                if len(indices) == 0:
                    continue
                taken = table.take(indices)
                for batch in taken.to_batches():
                    if batch.num_rows > 0:
                        yield batch

    # ==================================================================
    # Predicate-driven row-index API (caller-controlled)
    # ==================================================================

    def iter_matching_indices(
        self,
        predicate: Predicate,
        options: "YggOptions | None" = None,
    ) -> "Iterator[tuple[TabularIO, pa.Int64Array]]":
        """Yield ``(child_io, int64_indices)`` for each surviving file.

        The child IO is *closed* when yielded — caller opens it
        inside a ``with`` block. The int64 array contains the
        positions, within that file's flattened table, of every
        matching row (i.e. exactly what :func:`pa.Table.take`
        wants).

        This is the low-level surface for callers that want to
        decide what to do with the matches themselves (count rows,
        load only specific columns, hand off to a downstream
        consumer that wants row coordinates). Entries the
        predicate prunes from stats / partition values are not
        yielded; surviving entries with an empty match set yield
        an empty array.
        """
        opts = self.check_options(options)
        # Stamp the predicate onto the options so the iter_children
        # pruner sees it. Callers that already passed it in via
        # ``options.predicate`` get a no-op overwrite.
        opts = opts.copy(predicate=predicate)

        for child_io in self.iter_children(opts):
            if not isinstance(child_io, TabularIO):
                continue
            with child_io:
                table = child_io.read_arrow_table(options=opts)
                if table.num_rows == 0:
                    continue
                # Inject partition columns so a predicate on a
                # partition column can be re-checked at the row
                # level for safety (the file pruner uses partition
                # values, but a Hive-stripped data file doesn't
                # have the column on its own).
                pv = getattr(child_io, "partition_values", {}) or {}
                pcols = self._resolve_partition_columns(opts)
                if pv and pcols:
                    by_name = {c.name: c for c in pcols}
                    table = _inject_partition_into_table(table, pv, by_name)
                indices = row_indices(predicate, table)
            yield child_io, indices

    # ==================================================================
    # Write paths
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: "IterableABC[pa.RecordBatch]",
        options: YggOptions,
    ) -> None:
        mode = self._resolve_save_mode(options.mode)
        if mode is Mode.IGNORE:
            return

        if mode is Mode.UPSERT:
            self._arrow_upsert_via_rewrite(batches, options)
            return

        if mode not in (Mode.OVERWRITE, Mode.APPEND):
            raise NotImplementedError(
                f"YggIO supports OVERWRITE / APPEND / UPSERT; got "
                f"resolved action {mode!r}."
            )

        self._ygg_overwrite_or_append(batches, options, mode)

    # ------------------------------------------------------------------
    # OVERWRITE / APPEND
    # ------------------------------------------------------------------

    def _ygg_overwrite_or_append(
        self,
        batches: "IterableABC[pa.RecordBatch]",
        options: YggOptions,
        mode: Mode,
    ) -> None:
        """Write data, then commit a fresh manifest. Hard-delete on overwrite."""
        try:
            existing = self._load_manifest()
        except FileNotFoundError:
            existing = None

        if (
            options.require_existing_table
            and existing is None
        ):
            raise FileNotFoundError(
                f"YggIO write with require_existing_table=True but "
                f"{self.path!r} is not an existing ygg table."
            )

        partition_cols = self._resolve_partition_columns(options)
        partition_names = tuple(c.name for c in partition_cols)
        primary_key_columns = self._resolve_primary_key_columns(options)

        before_files = self._scan_data_files()
        write_options = options.copy(
            mode=Mode.OVERWRITE if mode is Mode.OVERWRITE else Mode.APPEND,
            child_media_type=options.child_media_type or MediaTypes.ARROW_IPC,
        )

        # Probe iterator emptiness so empty writes still produce a
        # valid no-op commit when the table is new.
        batch_iter = iter(batches)
        first = next(batch_iter, None)

        if first is None:
            self._handle_empty_write(
                mode, existing, partition_names, primary_key_columns, options,
            )
            return

        try:
            # We never let the inherited writer clear children — we
            # do hard-deletion ourselves *after* the new manifest
            # commits, so a mid-write failure leaves the previous
            # state intact.
            super()._write_arrow_batches(
                chain([first], batch_iter),
                write_options.copy(mode=Mode.APPEND),
            )
        except Exception:
            self._cleanup_new_files(before_files)
            raise

        after_files = self._scan_data_files()
        new_relpaths = sorted(after_files - before_files)

        new_entries = self._build_entries_from_relpaths(
            new_relpaths, partition_names, primary_key_columns,
        )

        if mode is Mode.OVERWRITE:
            target_schema = self._schema_from_first_file(new_relpaths)
            target_table_id = (
                existing.table_id if existing is not None else str(uuid.uuid4())
            )
            entries: list[ManifestEntry] = list(new_entries)
            old_relpaths_to_delete = (
                [e.path for e in existing.entries] if existing is not None else []
            )
        else:
            # APPEND: keep existing entries, add new.
            target_schema = (
                existing.data_schema
                if existing is not None
                else self._schema_from_first_file(new_relpaths)
            )
            target_table_id = (
                existing.table_id if existing is not None else str(uuid.uuid4())
            )
            entries = list(existing.entries) if existing is not None else []
            entries.extend(new_entries)
            old_relpaths_to_delete = []

        manifest = Manifest(
            timestamp=int(time.time() * 1000),
            table_id=target_table_id,
            partition_columns=tuple(partition_names),
            primary_key_columns=tuple(primary_key_columns),
            data_schema=target_schema,
            engine_info=options.engine_info,
            entries=tuple(entries),
        )

        try:
            write_manifest(self.path, manifest)
        except Exception:
            self._cleanup_new_files(before_files)
            raise

        # Manifest committed — the new state is live. Now hard-delete
        # the previous data files referenced only by the *old*
        # manifest. Best effort: a failure here leaves orphans on
        # disk, harmless until vacuumed.
        for rel in old_relpaths_to_delete:
            target = self.path.joinpath(*rel.split("/"))
            try:
                target.remove(allow_not_found=True)
            except Exception:
                pass

        self._invalidate_manifest_cache()

    def _handle_empty_write(
        self,
        mode: Mode,
        existing: Manifest | None,
        partition_names: Sequence[str],
        primary_key_columns: Sequence[str],
        options: YggOptions,
    ) -> None:
        """Empty-input write paths.

        APPEND of nothing on an existing table is a no-op.

        OVERWRITE of nothing on a non-empty table emits a manifest
        with no entries and hard-deletes the previously-live files.

        OVERWRITE of nothing on a fresh table initializes the
        manifest so subsequent reads see an existing-but-empty table.
        """
        if mode is Mode.APPEND and existing is not None:
            return

        target_table_id = (
            existing.table_id if existing is not None else str(uuid.uuid4())
        )
        target_schema = (
            existing.data_schema if existing is not None else Schema.empty()
        )

        manifest = Manifest(
            timestamp=int(time.time() * 1000),
            table_id=target_table_id,
            partition_columns=tuple(partition_names),
            primary_key_columns=tuple(primary_key_columns),
            data_schema=target_schema,
            engine_info=options.engine_info,
            entries=(),
        )

        write_manifest(self.path, manifest)
        if mode is Mode.OVERWRITE and existing is not None:
            for entry in existing.entries:
                target = self.path.joinpath(*entry.path.split("/"))
                try:
                    target.remove(allow_not_found=True)
                except Exception:
                    pass
        self._invalidate_manifest_cache()

    # ==================================================================
    # Per-file stats computation
    # ==================================================================

    def _compute_stats(
        self,
        file_path: Path,
        primary_key_columns: Sequence[str],
    ) -> tuple[Mapping[str, ColumnStats], int | None]:
        """Compute stats + row count for a freshly-written file.

        Reads the file once; uses pyarrow.compute for the
        min/max/null_count reductions (vectorized C++ kernels).
        Returns ``({}, None)`` when no key columns are declared —
        the file is unread in that case, saving the I/O.
        """
        if not primary_key_columns:
            # Even without stats we still want a row count for
            # downstream tooling, but reading the file just for
            # that is wasteful. Return None so the manifest
            # records "row count not computed."
            return {}, None

        try:
            io = TabularIO.from_path(file_path)
        except Exception:
            return {}, None

        try:
            with io:
                table = io.read_arrow_table()
        except Exception:
            # File unreadable: emit no stats. The pruner will
            # fail-open on this entry and the read path will skip
            # it gracefully.
            return {}, None

        out: dict[str, ColumnStats] = {}
        for col in primary_key_columns:
            if col not in table.column_names:
                # Column not in the file (Hive-stripped partition
                # column, schema mismatch). Skip.
                continue
            arr = table.column(col)
            if len(arr) == 0:
                out[col] = ColumnStats(min=None, max=None, null_count=0)
                continue
            try:
                null_count = int(arr.null_count)
            except Exception:
                null_count = -1
            try:
                min_v = pc.min(arr).as_py()
                max_v = pc.max(arr).as_py()
            except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
                # Some types (nested, decimal-with-NaN, ...) don't
                # support min/max. Stats unknown — pruner fails open.
                min_v = None
                max_v = None
            out[col] = ColumnStats(
                min=_serialize_stat_scalar(min_v),
                max=_serialize_stat_scalar(max_v),
                null_count=null_count,
            )

        return out, table.num_rows

    # ==================================================================
    # File / manifest helpers
    # ==================================================================

    def _scan_data_files(self) -> set[str]:
        """List all data files under the table root, relative paths.

        Walks every leaf, excluding the ``_ygg/`` metadata side-folder
        and other ignored entries. Recognised data extensions are
        IPC (``.arrow`` / ``.feather`` / ``.ipc``) and parquet —
        anything else is skipped so a stray text file doesn't end
        up in the manifest.
        """
        out: set[str] = set()
        if not self.path.exists():
            return out

        for leaf in self._walk_data_leaves(self.path):
            name = leaf.name.lower()
            if not (
                name.endswith(".arrow")
                or name.endswith(".feather")
                or name.endswith(".parquet")
                or name.endswith(".ipc")
            ):
                continue
            try:
                rel_parts = leaf.relative_to(self.path).parts
            except Exception:
                continue
            out.add("/".join(rel_parts))
        return out

    def _walk_data_leaves(self, root: Path) -> Iterator[Path]:
        """Yield leaf files under *root*, skipping the metadata dir."""
        if not root.exists():
            return

        stack: list[Path] = [root]
        while stack:
            current = stack.pop()
            try:
                children = list(current.iterdir())
            except FileNotFoundError:
                continue

            for child in children:
                if current is root and child.name == META_DIR_NAME:
                    continue
                if (
                    child.name.startswith((".", "_"))
                    and "=" not in child.name
                ):
                    continue
                try:
                    is_dir = child.is_dir()
                except Exception:
                    continue
                if is_dir:
                    stack.append(child)
                else:
                    yield child

    def _cleanup_new_files(self, before_files: set[str]) -> None:
        """Best-effort: unlink data files that appeared since *before_files*."""
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

    def _build_entries_from_relpaths(
        self,
        relpaths: Sequence[str],
        partition_names: Sequence[str],
        primary_key_columns: Sequence[str],
    ) -> list[ManifestEntry]:
        out: list[ManifestEntry] = []
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
            stats, num_rows = self._compute_stats(target, primary_key_columns)
            out.append(ManifestEntry(
                path=rel,
                size=size,
                modification_time=now_ms,
                num_rows=num_rows,
                partition_values=partition_values,
                stats=stats,
            ))
        return out

    def _partition_values_from_relpath(
        self,
        rel_path: str,
        partition_names: Sequence[str],
    ) -> dict[str, str | None]:
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
        first_io = TabularIO.from_path(first_path)
        with first_io:
            return first_io.collect_schema()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_stat_scalar(v: Any) -> Any:
    """Make a Python scalar safe for JSON encoding.

    Stats are stored as JSON in the manifest body. Datetime-shaped
    values become ISO 8601 strings; everything else passes through.
    The predicate evaluator does string-vs-string comparisons for
    ISO timestamps which is order-preserving, so the prefilter
    keeps working without round-tripping into the original dtype.
    """
    if v is None:
        return None
    from datetime import date, datetime, time as _time
    if isinstance(v, (datetime, date, _time)):
        return v.isoformat()
    return v


def _inject_partition_into_table(
    table: pa.Table,
    partition_values: Mapping[str, str | None],
    partition_by_name: Mapping[str, Field],
) -> pa.Table:
    """Append partition columns to *table* — table-level analogue of
    :func:`_inject_partition_columns`.

    The folder/io read shim works batch-by-batch; the predicate
    path collects the file as a single table before evaluating, so
    we need a table-level helper. Rather than re-implementing the
    casting logic we materialize via the per-batch helper and
    reassemble — fewer bugs at the cost of one extra concat for a
    multi-batch file.
    """
    if not partition_values or not partition_by_name:
        return table

    out_batches: list[pa.RecordBatch] = []
    for batch in table.to_batches():
        out_batches.append(
            _inject_partition_columns(batch, partition_values, partition_by_name)
        )
    if not out_batches:
        return table
    return pa.Table.from_batches(out_batches)
