"""YggIO: the :class:`PartitionedFolderIO` that ties the ygg protocol
together.

A ygg table is a :class:`FolderIO`-shaped tree with a small ``_ygg/``
metadata side-folder. The metadata side-folder holds one Arrow IPC
manifest per committed version plus a tiny ``_LATEST`` pointer to
the live one.

Reads
-----

#. Read ``_LATEST`` for the live version (cheap — a few bytes).
#. Memory-map ``_ygg/versions/v<N>.arrow`` and parse its footer
   (cheap — Arrow IPC indexes batches in the footer).
#. For each :class:`ManifestEntry`, mint a :class:`PrimitiveIO` for
   the data file with its ``partition_values`` populated, and yield
   it from :meth:`iter_children`.

The base :class:`FolderIO._read_arrow_batches` chains those
children into a single batch stream and injects partition columns
just like a Hive folder. No log replay, no checkpoint dance — the
manifest *is* the snapshot.

Writes — OVERWRITE / APPEND
---------------------------

Stage Arrow IPC children via the inherited
:class:`PartitionedFolderIO` machinery (which routes by partition
key when partitions are declared); compute the new manifest from
the post-write directory state; write the manifest with version
N+1 and flip ``_LATEST``. On any failure, freshly-written data
files are unlinked before re-raising — the previous manifest is
still pointed-at and still authoritative.

Why this is faster than Delta on read
-------------------------------------

Delta reads are O(N_commits) — the log replay touches every JSON
commit since the last checkpoint, deserializing each. Ygg reads
are O(1): one Arrow IPC open. The trade-off is that every commit
rewrites the full live file list; for tables with up to a few
million live files that fits comfortably in a single Arrow IPC
file (a manifest entry is ~100 bytes), and the rewrite cost is
amortized by the read win every time.
"""

from __future__ import annotations

import dataclasses
import time
import uuid
from collections.abc import Iterable as IterableABC
from itertools import chain
from typing import Any, ClassVar, Iterator, Mapping, Sequence

import pyarrow as pa

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
from .commit import (
    latest_pointer_path,
    manifest_path_for_version,
    read_latest_version,
    versions_dir,
    write_manifest,
)
from .constants import (
    DEFAULT_ENGINE_INFO,
    META_DIR_NAME,
)
from .manifest import Manifest, ManifestEntry, decode_manifest


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
    :param engine_info: written into every new manifest's metadata.
        Identifies the writer in commit history.
    :param require_existing_table: refuse OVERWRITE on a
        non-existent table when True. Default False.
    :param keep_old_manifests: retain previous version manifests for
        time-travel reads. Default True. When False, the writer
        unlinks the previous manifest after a successful pointer
        flip. Old data files are never auto-removed by the writer
        — that's a vacuum's job.
    """

    engine_info: str = DEFAULT_ENGINE_INFO
    require_existing_table: bool = False
    keep_old_manifests: bool = True


# ---------------------------------------------------------------------------
# YggIO
# ---------------------------------------------------------------------------


class YggIO(PartitionedFolderIO):
    """Ygg table as a partitioned folder.

    Construction:

        >>> io = YggIO(path="/tables/trades/")
        >>> for child in io.iter_children():
        ...     print(child.path, child.partition_values)

    Reads parse the latest manifest and yield one child IO per
    live data file. Writes go through ``_write_arrow_batches`` and
    produce a single manifest commit.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._meta_dir: Path = self.path / META_DIR_NAME
        self._manifest_cache: Manifest | None = None
        self._manifest_cache_version: int = -1

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
    # Manifest cache
    # ==================================================================

    def _load_manifest(self) -> Manifest | None:
        """Read the live manifest, cached per-instance.

        Cache key is the live version. We re-stat ``_LATEST`` on
        every call to detect a fresh commit; if the version hasn't
        moved we return the cached manifest. The stat is cheap
        compared to the IPC parse.
        """
        latest = read_latest_version(self.path)
        if latest < 0:
            self._manifest_cache = None
            self._manifest_cache_version = -1
            return None

        if (
            self._manifest_cache is not None
            and self._manifest_cache_version == latest
        ):
            return self._manifest_cache

        manifest_path = manifest_path_for_version(self.path, latest)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"ygg pointer at {latest_pointer_path(self.path)!r} "
                f"advertises version {latest}, but the manifest file "
                f"{manifest_path!r} is missing. The table is corrupt; "
                "rerun the last commit or restore from backup."
            )

        blob = manifest_path.read_bytes()
        manifest = decode_manifest(blob)
        if manifest.version != latest:
            raise ValueError(
                f"ygg manifest at {manifest_path!r} declares version "
                f"{manifest.version}, but its filename / pointer "
                f"says {latest}. Refusing to use a mislabeled manifest."
            )

        self._manifest_cache = manifest
        self._manifest_cache_version = manifest.version
        return manifest

    def _invalidate_manifest_cache(self) -> None:
        self._manifest_cache = None
        self._manifest_cache_version = -1

    # ==================================================================
    # Partition column resolution — from manifest
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
            # Hive partition columns are stripped from the data files
            # on write, so they typically don't appear in
            # ``manifest.data_schema``. Fall back to a string-typed
            # field by name in that case (Hive convention).
            f = manifest.data_schema.get(name) if manifest.data_schema else None
            if f is None:
                out.append(_coerce_partition_column(name))
            else:
                out.append(f)
        return tuple(out)

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
    # Children enumeration — driven by the manifest
    # ==================================================================

    def iter_children(
        self,
        options: "YggOptions | None" = None,
        **kwargs: Any,
    ) -> "Iterator[TabularIO | BytesIO]":
        """Yield one child :class:`PrimitiveIO` per :class:`ManifestEntry`.

        Each child has:

        - ``parent`` set to ``self``;
        - ``partition_values`` populated from the entry's mapping;
        - the file's media type inferred from its extension.

        Entries whose backing file is missing on disk are silently
        skipped — the typical reason is a partial restore. Callers
        that need strict consistency can compare ``len(entries)``
        with the yielded count themselves.
        """
        self.check_options(options, overrides=locals())

        try:
            manifest = self._load_manifest()
        except FileNotFoundError:
            return

        if manifest is None:
            return

        for entry in manifest.entries:
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
    # Read derivation — chain children, inject partition columns
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: YggOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Stream batches from each manifest entry in order.

        Per child:

        - drain its batches via the leaf IO's
          :meth:`read_arrow_batches`;
        - inject partition columns row-for-row using the per-child
          ``partition_values`` mapping populated by
          :meth:`iter_children`.
        """
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        partition_cols = self._resolve_partition_columns(options)
        partition_by_name: Mapping[str, Field] = {
            c.name: c for c in partition_cols
        }

        for child_io in self.iter_children(options):
            if not isinstance(child_io, TabularIO):
                continue

            partition_values = getattr(child_io, "partition_values", {}) or {}

            with child_io:
                for batch in child_io.read_arrow_batches(options=options):
                    if batch.num_rows == 0:
                        continue
                    if partition_by_name and partition_values:
                        batch = _inject_partition_columns(
                            batch, partition_values, partition_by_name,
                        )
                    yield batch

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
            # No DV-style merge-on-read in ygg yet; fall back to
            # the generic read-merge-overwrite helper inherited from
            # NestedIO. That handles match-by-names correctness; the
            # resulting OVERWRITE re-enters this method.
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
        """Write data, then commit a new manifest."""
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

        before_files = self._scan_data_files()
        write_options = options.copy(
            mode=Mode.OVERWRITE if mode is Mode.OVERWRITE else Mode.APPEND,
            child_media_type=options.child_media_type or MediaTypes.ARROW_IPC,
        )

        # Probe iterator emptiness so empty writes still produce a
        # valid no-op commit when the table is new.
        batch_iter = iter(batches)
        first = next(batch_iter, None)

        # OVERWRITE on a non-empty table clears the data files
        # before the new write; the new manifest then has only the
        # newly-written entries. The base class' _clear_children
        # respects _is_ignored_path, so ``_ygg/`` survives.
        if mode is Mode.OVERWRITE and not self.is_empty():
            self._clear_children()
            before_files = set()  # everything we wrote is "new"

        if first is None:
            self._handle_empty_write(
                mode, existing, partition_names, options,
            )
            return

        try:
            # Force the FolderIO writer to APPEND-style child minting
            # (no clear_children re-fire) — we already cleared above
            # for OVERWRITE. APPEND-style ensures it just adds new
            # part files alongside whatever exists.
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
            new_relpaths, partition_names,
        )

        if mode is Mode.OVERWRITE:
            target_schema = self._schema_from_first_file(new_relpaths)
            target_table_id = (
                existing.table_id if existing is not None else str(uuid.uuid4())
            )
            entries: list[ManifestEntry] = list(new_entries)
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

        next_version = (existing.version + 1) if existing is not None else 0
        manifest = Manifest(
            version=next_version,
            timestamp=int(time.time() * 1000),
            table_id=target_table_id,
            partition_columns=tuple(partition_names),
            data_schema=target_schema,
            engine_info=options.engine_info,
            entries=tuple(entries),
        )

        try:
            write_manifest(self.path, manifest)
        except Exception:
            self._cleanup_new_files(before_files)
            raise

        if not options.keep_old_manifests and existing is not None:
            self._unlink_manifest(existing.version)

        self._invalidate_manifest_cache()

    def _handle_empty_write(
        self,
        mode: Mode,
        existing: Manifest | None,
        partition_names: Sequence[str],
        options: YggOptions,
    ) -> None:
        """Empty-input write paths.

        APPEND of nothing on an existing table is a no-op (no
        commit emitted — committing a manifest that's identical to
        the previous one is wasted I/O).

        OVERWRITE of nothing on a non-empty table emits a manifest
        with no entries (the table becomes logically empty; old
        data files are still on disk for vacuum).

        OVERWRITE of nothing on a fresh table initializes the log
        — write version 0 with an empty entry list so subsequent
        reads see an existing-but-empty table.
        """
        if mode is Mode.APPEND and existing is not None:
            return

        target_table_id = (
            existing.table_id if existing is not None else str(uuid.uuid4())
        )
        target_schema = (
            existing.data_schema if existing is not None else Schema.empty()
        )
        next_version = (existing.version + 1) if existing is not None else 0

        manifest = Manifest(
            version=next_version,
            timestamp=int(time.time() * 1000),
            table_id=target_table_id,
            partition_columns=tuple(partition_names),
            data_schema=target_schema,
            engine_info=options.engine_info,
            entries=(),
        )

        write_manifest(self.path, manifest)
        if not options.keep_old_manifests and existing is not None:
            self._unlink_manifest(existing.version)
        self._invalidate_manifest_cache()

    # ==================================================================
    # File / manifest helpers
    # ==================================================================

    def _scan_data_files(self) -> set[str]:
        """List all data files under the table root, relative paths.

        Walks every leaf, excluding the ``_ygg/`` metadata side-folder
        and other ignored entries. Recognised data extensions are
        IPC (``.arrow`` / ``.feather``) and parquet — anything else
        is skipped so a stray text file doesn't end up in a manifest.
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
                # Hide the metadata side-folder regardless of depth.
                if current is root and child.name == META_DIR_NAME:
                    continue
                # Generic hide: don't descend hidden / underscore
                # entries, but keep ``key=value`` partition dirs.
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
        """Best-effort: unlink data files that appeared since *before_files*.

        Used on the failure path of a write — if the manifest
        commit didn't land, the files we wrote are unreferenced
        garbage. Removing them keeps the table tidy; failing to
        remove them is non-fatal (a vacuum will catch them).
        """
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
            out.append(ManifestEntry(
                path=rel,
                size=size,
                modification_time=now_ms,
                num_rows=None,
                partition_values=partition_values,
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

    def _unlink_manifest(self, version: int) -> None:
        """Best-effort removal of an old manifest file."""
        try:
            target = manifest_path_for_version(self.path, version)
            target.remove(allow_not_found=True)
        except Exception:
            pass

    # ==================================================================
    # Diagnostics
    # ==================================================================

    @property
    def current_version(self) -> int:
        """Return the latest committed version, or ``-1`` if none."""
        try:
            return read_latest_version(self.path)
        except Exception:
            return -1

    def list_versions(self) -> list[int]:
        """List all committed manifest versions on disk, sorted ascending."""
        from .constants import MANIFEST_VERSION_RE

        out: list[int] = []
        vdir = versions_dir(self.path)
        if not vdir.exists():
            return out
        for entry in vdir.iterdir():
            m = MANIFEST_VERSION_RE.match(entry.name)
            if m is not None:
                out.append(int(m.group(1)))
        out.sort()
        return out

    def read_manifest_at(self, version: int) -> Manifest:
        """Read a specific manifest version (for time travel / diagnostics)."""
        target = manifest_path_for_version(self.path, version)
        if not target.exists():
            raise FileNotFoundError(
                f"ygg manifest version {version} does not exist at "
                f"{target!r}."
            )
        return decode_manifest(target.read_bytes())
