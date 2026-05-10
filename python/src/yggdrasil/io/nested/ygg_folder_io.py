"""Hive-partitioned :class:`FolderIO` with metadata caching.

:class:`YGGFolderIO` is the local-cache backend for the response
batch layer. The shape — ``<root>/<col>=<val>/<col=val>/...`` — is
the standard Hive layout, so any external reader (Spark,
``pyarrow.dataset``, polars) can scan the same tree without going
through this class.

What it adds on top of :class:`FolderIO`
----------------------------------------

1. **Schema-driven partitioning** — the bound :class:`Schema`
   declares which fields carry ``partition_by`` tags; writes group
   rows by those columns and route each group into its
   ``col=val/...`` subdirectory.
2. **Partition pruning** — :meth:`_read_arrow_batches` only walks
   directories whose ``col=val`` segments survive the predicate.
   Two ways to drive the prune:

   - ``options.prune_values = {col: (v1, v2, ...)}`` — explicit
     ``IN`` set; the most common shape (the response cache uses
     this with the request batch's partition values).
   - ``options.predicate`` — a :class:`Predicate` over partition
     columns, evaluated per directory. Falls back to the leaf-level
     predicate filter for non-partition column references.
3. **Listing cache** — a short-TTL :class:`ExpiringDict` over the
   directory walk. Repeated reads (a Delta replay, a tight test
   loop) collapse to one ``os.scandir`` per partition window.
4. **Optimize** — :meth:`optimize` compacts small parquet parts
   per partition into a single file. Idempotent; cheap when
   partitions already hold a single part.

The :data:`MimeTypes.YGG_FOLDER` mime type registers the class so
``Tabular.for_holder`` dispatches a folder path with that hint to
this leaf.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import threading
import time
import urllib.parse
from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator

import pyarrow as pa

from yggdrasil.dataclasses.expiring import ExpiringDict
from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.schema import Schema
from yggdrasil.io.nested.folder_io import FolderIO, FolderOptions
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.tabular.execution.expr.nodes import (
    Logical,
    LogicalOp,
    free_columns,
)

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field
    from yggdrasil.io.path import Path


__all__ = ["YGGFolderIO"]


#: Default listing-cache TTL. Short enough that a stale directory
#: doesn't linger across a parallel-write race; long enough to
#: collapse the dozen scandir calls a Delta replay typically makes.
_LISTING_TTL_SECONDS: float = 15.0
_LISTING_MAX: int = 1024

#: Glob to match part files inside a partition directory. ``*.*``
#: keeps anything with an extension; ``.schema`` and other dot-
#: prefixed sidecars stay invisible (FolderIO already filters those).
_PART_PATTERN: str = "part-*"

#: Sidecar directory holding YGGFolderIO-managed metadata. Dot-
#: prefixed so :meth:`FolderIO.iter_children` skips it on the data
#: walk; subfolders inside it are free to use any naming scheme
#: without being mistaken for partition directories.
_METADATA_DIR_NAME: str = ".ygg"

#: Schema sidecar filename inside :data:`_METADATA_DIR_NAME`. Stores
#: the bound :class:`Schema` as Arrow IPC bytes so future readers
#: can reconstruct the partition tags / nested types without
#: inferring from a part file's footer.
_SCHEMA_FILENAME: str = ".schema"

#: Sentinel filename inside :data:`_METADATA_DIR_NAME` recording the
#: epoch of the last successful :meth:`YGGFolderIO.cleanup_stale`
#: sweep. Its mtime is the cross-process throttle for
#: :meth:`YGGFolderIO.cleanup_stale_once` — younger than the TTL
#: means a sibling process already swept; skip the walk.
_CLEANUP_SENTINEL_FILENAME: str = ".last_cleanup"

#: Default TTL (seconds) for :meth:`YGGFolderIO.cleanup_stale`.
#: Files whose ``part-{epoch_ms}-...`` prefix is older than this
#: are unlinked. Mirrors ``_STAGING_TTL_SECONDS`` in
#: :mod:`yggdrasil.io.path.local_path`.
_DEFAULT_CLEANUP_TTL_SECONDS: float = 86_400.0  # 1 day

#: In-process throttle for :meth:`YGGFolderIO.cleanup_stale_once`:
#: each ``str(path)`` is swept at most once per Python run. The
#: sentinel file handles longer-lived (cross-process) deduplication;
#: this set just keeps the per-call overhead at one set lookup
#: after the first sweep.
_CLEANUP_LOCK = threading.Lock()
_CLEANUP_DONE: set[str] = set()


def _quote(value: Any) -> str:
    """URL-quote a partition value the same way Spark / Hive do."""
    if value is None:
        return "__HIVE_DEFAULT_PARTITION__"
    return urllib.parse.quote(str(value), safe="")


def _unquote(value: str) -> str:
    if value == "__HIVE_DEFAULT_PARTITION__":
        return ""
    return urllib.parse.unquote(value)


def _coerce_partition_value(field: "Field", raw: str) -> Any:
    """Turn a directory-name fragment back into the field's dtype.

    Hive layout is text-only (``col=42`` lives on disk as the
    string ``"42"``); the predicate evaluator wants the typed
    value back. The cheap pyarrow ``cast`` round-trip is enough
    for the integer / string / temporal cases the response cache
    actually uses.
    """
    if not raw or raw == "__HIVE_DEFAULT_PARTITION__":
        return None
    arrow_type = field.dtype.to_arrow()
    try:
        return pa.scalar(raw).cast(arrow_type).as_py()
    except Exception:
        return raw


class YGGFolderIO(FolderIO):
    """:class:`FolderIO` with Hive-partitioned writes + read pruning."""

    mime_type: ClassVar[MimeTypes] = MimeTypes.YGG_FOLDER

    __slots__ = ("_schema", "_listing_cache")

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        schema: "Schema | None" = None,
        listing_ttl: float = _LISTING_TTL_SECONDS,
        listing_max: int = _LISTING_MAX,
        tabular_parent: "Any | None" = None,
    ) -> None:
        super().__init__(data, path=path, tabular_parent=tabular_parent)
        # Load the schema from the .ygg/.schema sidecar when the
        # caller didn't pass one — lets a "just point me at this
        # folder" call still pick up the partition layout that
        # produced the data.
        if schema is None:
            schema = self._load_schema_sidecar()
        self._schema: "Schema | None" = schema
        # Per-instance cache: maps (partition tuple-key) → list of
        # part files. Short TTL covers dirs that grow under
        # concurrent writes without making cold reads slow.
        self._listing_cache: "ExpiringDict[str, tuple[str, ...]]" = ExpiringDict(
            default_ttl=listing_ttl,
            max_size=listing_max,
        )

    # ==================================================================
    # Metadata sidecar — .ygg/.schema persists the bound Schema
    # ==================================================================

    @property
    def _metadata_dir(self) -> "Path":
        return self.path / _METADATA_DIR_NAME

    @property
    def _schema_sidecar_path(self) -> "Path":
        return self._metadata_dir / _SCHEMA_FILENAME

    def _persist_schema_sidecar(self) -> None:
        """Write the bound :class:`Schema` to ``.ygg/.schema``.

        No-op when no schema is bound or when the sidecar already
        holds the same bytes — keeps the call cheap on repeated
        writes. The sidecar is created lazily on the first write
        rather than at construction so a read-only :class:`YGGFolderIO`
        doesn't materialize the ``.ygg/`` directory.
        """
        if self._schema is None:
            return
        try:
            payload = self._schema.to_arrow_schema().serialize().to_pybytes()
        except Exception:
            return

        target = self._schema_sidecar_path
        try:
            if target.exists() and target.size == len(payload):
                with target.open(mode="rb") as bio:
                    if bio.to_bytes() == payload:
                        return
        except Exception:
            pass

        try:
            self._metadata_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        try:
            with target.open(mode="wb") as bio:
                bio.write(payload)
        except Exception:
            pass

    def _load_schema_sidecar(self) -> "Schema | None":
        """Read ``.ygg/.schema`` back into a :class:`Schema`, or ``None``.

        Robust to missing folders / sidecars / partial writes — the
        sidecar is a best-effort hint; data files still carry their
        own footer schema, so a corrupted side-car shouldn't break
        the read path.
        """
        target = self._schema_sidecar_path
        try:
            if not target.exists():
                return None
            with target.open(mode="rb") as bio:
                payload = bio.to_bytes()
        except Exception:
            return None
        if not payload:
            return None
        try:
            arrow_schema = pa.ipc.read_schema(pa.BufferReader(payload))
        except Exception:
            return None
        try:
            return Schema.from_arrow(arrow_schema)
        except Exception:
            return None

    # ==================================================================
    # Schema-driven partition discovery
    # ==================================================================

    def _resolve_partition_columns(self) -> "list[Field]":
        """Fields with the ``partition_by`` tag, in declaration order.

        Used by both the read and the write paths to build /
        decompose the directory tree. Returns an empty list when no
        schema is bound — :meth:`_read_arrow_batches` and
        :meth:`_write_arrow_batches` short-circuit through the
        plain :class:`FolderIO` flow in that case.
        """
        if self._schema is None:
            return []
        return [
            f for f in self._schema.fields
            if f._tag_flag(b"partition_by")
        ]

    @property
    def partition_columns(self) -> "list[str]":
        """Public view of :meth:`_resolve_partition_columns`'s names."""
        return [f.name for f in self._resolve_partition_columns()]

    # ==================================================================
    # Read path
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        # Self prune first — when this YGGFolderIO is itself nested
        # under another partition tree (or was minted with a
        # ``static_values`` seed), the predicate may be provably
        # false against the inherited KV before we even list
        # partitions. Same helper as :class:`FolderIO`, single
        # consistent gate at every read level.
        if self._should_prune_by_predicate(options):
            return

        parts = self._resolve_partition_columns()
        if not parts:
            yield from super()._read_arrow_batches(options)
            return

        prune = options.prune_values or {}
        for part_path, part_kv in self._iter_partitions(parts, prune):
            # Stamp the leaf folder with the partition KV as
            # ``static_values`` so the predicate-prune helper can
            # short-circuit subtrees whose KV makes the predicate
            # provably false. ``options.prune_values`` already does
            # the explicit-IN form; ``_should_prune_by_predicate``
            # covers arbitrary :class:`Predicate` shapes the session
            # passes through ``options.predicate`` (e.g. the cache
            # flow's ``partition_key.is_in([...]) & received_at >= t``).
            sub = FolderIO(
                path=part_path,
                tabular_parent=self,
                static_values=part_kv,
            )
            if sub._should_prune_by_predicate(options):
                continue
            for batch in sub._read_arrow_batches(options):
                yield self._stamp_partitions(batch, part_kv, parts)

    def _iter_partitions(
        self,
        parts: "list[Field]",
        prune: "dict[str, Any] | Any",
    ) -> "Iterator[tuple[Path, dict[str, Any]]]":
        """Yield ``(leaf_path, partition_values)`` for every partition.

        Walks the partition tree level-by-level; at each level,
        filters the candidate values against ``prune`` (when the
        column appears in the prune map). Cache key is the path of
        the parent directory at each level.
        """
        if not self.path.exists():
            return
        yield from self._walk(self.path, parts, 0, {}, prune)

    def _walk(
        self,
        current: "Path",
        parts: "list[Field]",
        depth: int,
        kv: "dict[str, Any]",
        prune: "dict[str, Any] | Any",
    ) -> "Iterator[tuple[Path, dict[str, Any]]]":
        if depth == len(parts):
            # Leaf — yield only when there's actual data on disk.
            yield current, dict(kv)
            return

        field = parts[depth]
        col = field.name
        prefix = f"{col}="

        children = self._cached_listing(current)
        allowed = self._allowed_values(prune, col)

        for name in children:
            if not name.startswith(prefix):
                continue
            raw = name[len(prefix):]
            value = _coerce_partition_value(field, _unquote(raw))
            if allowed is not None and value not in allowed:
                continue
            child_path = current / name
            sub_kv = dict(kv)
            sub_kv[col] = value
            yield from self._walk(child_path, parts, depth + 1, sub_kv, prune)

    @staticmethod
    def _allowed_values(prune: Any, col: str) -> "set[Any] | None":
        """Pull the ``IN`` set for *col* out of an ``options.prune_values``
        mapping.

        Accepts tuple / list / set / single-scalar shapes and
        normalizes to a set for O(1) membership. ``None`` means
        "no prune for this column," and the walk admits everything.
        """
        if not prune:
            return None
        try:
            values = prune.get(col)
        except AttributeError:
            return None
        if values is None:
            return None
        if isinstance(values, (str, bytes, int, float, bool)):
            return {values}
        try:
            return set(values)
        except TypeError:
            return {values}

    def _cached_listing(self, directory: "Path") -> "tuple[str, ...]":
        """Cached scandir of *directory* — names only.

        Names are returned without the partition prefix or trailing
        slash. Cold-path cost is one ``os.scandir``; warm path is a
        dict lookup.
        """
        key = str(directory)
        hit = self._listing_cache.get(key)
        if hit is not None:
            return hit
        try:
            with os.scandir(str(directory)) as it:
                names = tuple(
                    sorted(
                        e.name for e in it
                        if not e.name.startswith(".")
                    )
                )
        except (FileNotFoundError, NotADirectoryError):
            names = ()
        self._listing_cache.set(key, names)
        return names

    def invalidate_listing(self, directory: "Path | None" = None) -> None:
        """Drop cached listings.

        With no argument, clears the whole cache. With *directory*,
        invalidates that path and all of its ancestors up to the
        folder root — the entry that materialized them no longer
        reflects on-disk truth.
        """
        if directory is None:
            self._listing_cache.clear()
            return
        target = str(directory)
        root = str(self.path)
        # Drop the directory itself and every parent up to root.
        cur = target
        while True:
            self._listing_cache.pop(cur, None)
            if cur == root or len(cur) <= len(root):
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

    @staticmethod
    def _stamp_partitions(
        batch: pa.RecordBatch,
        kv: "dict[str, Any]",
        parts: "list[Field]",
    ) -> pa.RecordBatch:
        """Re-attach partition columns to a batch read from a leaf.

        Per Hive convention the partition columns aren't stored
        inside the parquet files — only the directory names carry
        them. We rebuild the column from the directory KV so the
        batch's schema lines up with the original :class:`Schema`.
        """
        schema = batch.schema
        for field in parts:
            if field.name in schema.names:
                continue
            arrow_type = field.dtype.to_arrow()
            value = kv.get(field.name)
            arr = pa.array([value] * batch.num_rows, type=arrow_type)
            batch = batch.append_column(field.to_arrow_field(), arr)
        return batch

    # ==================================================================
    # Write path
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: FolderOptions,
    ) -> None:
        # Drop the schema sidecar before any data writes — first call
        # creates ``.ygg/.schema`` so future readers (or a fresh
        # :class:`YGGFolderIO` over the same path) inherit the
        # partition layout without touching a part file's footer.
        self._persist_schema_sidecar()

        parts = self._resolve_partition_columns()
        if not parts:
            super()._write_arrow_batches(batches, options)
            return

        action = self._resolve_action(options.mode)

        # OVERWRITE wipes the whole partition tree before writing —
        # the same shape as plain ``FolderIO``'s overwrite.
        if action is Mode.OVERWRITE and self.path.exists():
            self._clear_partition_tree()

        # Group every batch's rows by their partition value tuple
        # and route each group into its own ``col=val/...`` subdir.
        # Multiple batches that share a partition stack into one
        # part file per write call (one MakeChild per partition).
        groups: "dict[tuple, list[pa.RecordBatch]]" = defaultdict(list)
        partition_names = [f.name for f in parts]

        for batch in batches:
            if batch.num_rows == 0:
                continue
            for key, sub_batch in self._partition_groups(batch, partition_names):
                # Drop partition columns from the stored payload —
                # Hive readers reconstruct them from the directory
                # name, and storing them again wastes bytes.
                inner = sub_batch.drop_columns(
                    [c for c in partition_names if c in sub_batch.schema.names]
                )
                if inner.num_rows > 0:
                    groups[key].append(inner)

        if not groups:
            return

        # Partition columns are constant within a partition (encoded
        # in the directory name and dropped from the per-leaf payload),
        # so they can't drive a per-leaf merge. Strip them out of the
        # match-by set before delegating; if nothing useful remains,
        # null out the field so the leaf write skips the merge path.
        leaf_options = options
        if options.match_by:
            # Filter out partition columns — they're encoded in the
            # folder layout and aren't visible at the leaf, so they
            # can't drive a per-leaf merge.
            sub_match = [
                f for f in options.match_by if f.name not in partition_names
            ]
            if len(sub_match) != len(options.match_by):
                leaf_options = dataclasses.replace(
                    options, match_by=sub_match or None,
                )

        for key_tuple, sub_batches in groups.items():
            kv = dict(zip(partition_names, key_tuple))
            target = self._ensure_partition_dir(parts, kv)
            # Seed the per-partition folder with the kv as
            # ``static_values`` — every leaf minted underneath
            # inherits ``{partition_col: value}`` via the parent
            # chain, so future reads / matches against this leaf
            # don't have to re-derive the kv from the directory
            # name.
            sub = FolderIO(
                path=target, tabular_parent=self, static_values=kv,
            )
            sub._write_arrow_batches(
                sub_batches,
                # The leaf write picks the right Tabular leaf via
                # ``child_extension`` (defaults to arrow IPC).
                leaf_options,
            )
            self.invalidate_listing(target)

    @staticmethod
    def _partition_groups(
        batch: pa.RecordBatch,
        partition_names: "list[str]",
    ) -> "Iterator[tuple[tuple, pa.RecordBatch]]":
        """Group ``batch`` rows by the values in ``partition_names``.

        Uses pyarrow Table.group_by under the hood for the actual
        split — fast for large batches, still correct on small ones.
        """
        if not partition_names:
            yield ((), batch)
            return

        if not all(c in batch.schema.names for c in partition_names):
            # Schema mismatch: yield the whole batch under None keys
            # so the caller still emits something rather than
            # silently dropping rows.
            yield (tuple(None for _ in partition_names), batch)
            return

        table = pa.Table.from_batches([batch])
        # ``group_by`` is the public surface; build the unique key
        # tuples manually to keep the per-key payload as a batch
        # (group_by aggregates by default; we want partition slicing).
        cols = [table.column(c).to_pylist() for c in partition_names]
        keys: "dict[tuple, list[int]]" = {}
        for row_idx, key in enumerate(zip(*cols)):
            keys.setdefault(key, []).append(row_idx)
        for key, indices in keys.items():
            yield key, table.take(indices).combine_chunks().to_batches()[0]

    def _ensure_partition_dir(
        self,
        parts: "list[Field]",
        kv: "dict[str, Any]",
    ) -> "Path":
        target = self.path
        for f in parts:
            target = target / f"{f.name}={_quote(kv.get(f.name))}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _clear_partition_tree(self) -> None:
        """Remove the entire partition tree under :attr:`path`."""
        try:
            shutil.rmtree(str(self.path))
        except FileNotFoundError:
            pass
        self._listing_cache.clear()

    # ==================================================================
    # Row-level delete with partition pruning
    # ==================================================================

    def _delete(self, predicate: Any, options: FolderOptions) -> int:
        """Delete rows matching *predicate*, pruning partitions whose
        directory KV makes the predicate trivially false.

        Strategy:

        1. Flatten the predicate's top-level ``AND`` into conjuncts.
        2. For each partition leaf:

           - Conjuncts that reference only partition columns are
             evaluated against the partition's KV. If any such conjunct
             evaluates to ``False`` / ``NULL``, no row in this partition
             can match — skip the whole subtree (no IO).
           - Conjuncts that mix partition and non-partition columns,
             or reference only non-partition columns, are kept as
             "residual" — they need a row-level filter inside the
             partition.

        3. If every conjunct is partition-only and all evaluate to
           ``True``, the partition matches wholesale: count rows then
           ``rmtree`` the directory (one stat-and-unlink per file
           rather than reading + rewriting).
        4. Otherwise hand the residual predicate to a plain
           :class:`FolderIO` rooted at the partition leaf, which
           applies the per-leaf filter shape from
           :meth:`FolderIO._delete`.

        Top-level ``OR`` / ``NOT`` predicates that aren't decomposable
        into partition-only conjuncts fall through to step 4 unchanged
        — correct, just no pruning win.
        """
        parts = self._resolve_partition_columns()
        if not parts:
            return super()._delete(predicate, options)

        partition_names = {f.name for f in parts}
        conjuncts = list(_iter_and_conjuncts(predicate))

        deleted = 0
        for leaf_path, kv in self._iter_partitions(parts, prune={}):
            # One-row table over the partition KV — every partition-only
            # conjunct compiles to a pyarrow expression and runs against
            # this in C++. Conjuncts that mix partition and non-partition
            # columns can't be evaluated here (the non-partition columns
            # aren't in this row's schema), so they fall through to the
            # row-level filter inside the partition.
            #
            # Inlined rather than routed through
            # :meth:`Tabular.matches_static` because delete needs a
            # tristate (definite-True → drop conj from residual,
            # definite-False → prune subtree, undecidable → prune
            # subtree to avoid over-deleting on eval bugs).
            # ``matches_static`` is read-shaped (undecidable → True,
            # i.e. "include the leaf"), and would risk wholesale
            # deletes if reused here.
            kv_table = pa.Table.from_pydict({k: [v] for k, v in kv.items()})

            residual: "list[Any]" = []
            prune_partition = False
            for conj in conjuncts:
                free = set(free_columns(conj))
                if free and free.issubset(partition_names):
                    try:
                        matches = conj.filter_arrow_table(kv_table).num_rows
                    except Exception:
                        matches = 0
                    if matches == 1:
                        continue
                    # 0 (predicate False) or eval failure → conservatively
                    # treat as "no row in this partition can match" so we
                    # never delete rows that might survive at row level.
                    prune_partition = True
                    break
                residual.append(conj)
            if prune_partition:
                continue

            if not residual:
                deleted += self._wholesale_delete_partition(leaf_path)
                continue

            residual_pred = _and_combine(residual)
            # Seed the per-partition folder with the kv so the
            # row-level delete (and any future static-values
            # consumer) sees the partition constants without
            # re-deriving them from the directory name.
            sub = FolderIO(
                path=leaf_path, tabular_parent=self, static_values=kv,
            )
            deleted += sub._delete(residual_pred, FolderOptions())
            self.invalidate_listing(leaf_path)
        return deleted

    def _wholesale_delete_partition(self, leaf_path: "Path") -> int:
        """Count rows in *leaf_path* then ``rmtree`` it."""
        sub = FolderIO(path=leaf_path)
        count = 0
        try:
            for batch in sub._read_arrow_batches(FolderOptions()):
                count += batch.num_rows
        except Exception:
            count = 0
        try:
            shutil.rmtree(str(leaf_path))
        except FileNotFoundError:
            pass
        except Exception:
            return 0
        self.invalidate_listing(leaf_path)
        return count

    # ==================================================================
    # Optimize — compact small parts per partition
    # ==================================================================

    def optimize(
        self,
        byte_size: "int | None" = None,
        *,
        target_media_type: "Any" = None,
        tolerance: float = FolderIO.OPTIMIZE_TOLERANCE,
        partitions: "dict[str, Any] | None" = None,
        **kwargs: Any,
    ) -> int:
        """Compact each partition leaf's small parts.

        Walks the partition tree (one branch per ``col=val`` segment)
        and dispatches each leaf folder to :meth:`FolderIO.optimize`,
        which does the actual bin-pack:

        - ``byte_size=None`` collapses every leaf with more than one
          part into a single file — the legacy shape the local-cache
          compaction loop in :class:`Session` calls with.
        - ``byte_size=N`` packs small parts into bundles near *N*
          bytes; parts within ``±tolerance`` of *N* (or already
          larger) are left untouched.
        - ``partitions={col: (v1, v2, ...)}`` (or a single scalar /
          set) restricts the walk to the matching ``col=val`` leaves
          only. Same shape the read path consumes via
          ``options.prune_values``, so the session's per-batch
          compaction can compact just the ``partition_key=…`` leaves
          a write actually touched without paying for a full-tree
          walk. ``None`` (default) keeps the legacy "every leaf"
          behaviour.

        Returns the total number of new part files written across
        every leaf. A no-schema :class:`YGGFolderIO` falls through to
        :meth:`FolderIO.optimize`, so the operation still works on
        an unpartitioned tree.
        """
        if not self.path.exists():
            return 0

        from yggdrasil.data.enums import MediaType, MediaTypes
        media = MediaType.from_(target_media_type, default=MediaTypes.ARROW_IPC)

        parts = self._resolve_partition_columns()
        if not parts:
            return super().optimize(
                byte_size=byte_size,
                target_media_type=media,
                tolerance=tolerance,
            )

        prune = partitions or {}
        compacted = 0
        for leaf_path, _kv in self._iter_partitions(parts, prune=prune):
            if not leaf_path.exists():
                continue
            leaf = FolderIO(path=leaf_path)
            compacted += leaf._optimize_walk(
                leaf_path,
                byte_size=byte_size,
                target_media_type=media,
                tolerance=tolerance,
            )
            self.invalidate_listing(leaf_path)
        return compacted

    # ==================================================================
    # Cleanup — unlink stale part files older than TTL
    # ==================================================================

    CLEANUP_TTL_SECONDS: ClassVar[float] = _DEFAULT_CLEANUP_TTL_SECONDS

    @property
    def _cleanup_sentinel_path(self) -> "Path":
        return self._metadata_dir / _CLEANUP_SENTINEL_FILENAME

    def cleanup_stale(self, ttl_seconds: "float | None" = None) -> int:
        """Unlink ``part-*`` files older than *ttl_seconds*. Returns the count.

        File-age comparison uses the ``part-{epoch_ms}-{seed}.{ext}``
        filename convention :meth:`FolderIO.make_child` mints — the
        epoch_ms prefix is what we compare against the cutoff. This
        is robust to filesystem mtime quirks (rsync'd caches, bind-
        mounted images, ``cp -p`` round-trips) where ``stat`` mtime
        doesn't reflect when the entry was actually written.

        Walks the partition tree via :meth:`_iter_partitions` when a
        schema is bound, otherwise walks :attr:`path` directly. The
        ``.ygg/`` sidecar (schema, sentinel, future metadata) is
        always skipped.

        Best-effort: every OS error encountered while listing or
        unlinking is swallowed so a permission denial / race with a
        concurrent writer never breaks the surrounding call. The
        listing cache is invalidated for every directory we touched.

        :param ttl_seconds: minimum age (in seconds) for a file to
            be unlinked. ``None`` uses :attr:`CLEANUP_TTL_SECONDS`.
        :returns: number of files unlinked.
        """
        ttl = self.CLEANUP_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
        try:
            if not self.path.exists():
                return 0
        except OSError:
            return 0

        cutoff_ms = int((time.time() - ttl) * 1000)
        deleted = 0
        touched: set[str] = set()

        for leaf in self._iter_cleanup_leaves():
            try:
                children = list(leaf.iterdir())
            except OSError:
                continue
            for child in children:
                name = child.name
                if not name.startswith("part-"):
                    continue
                try:
                    if not child.is_file():
                        continue
                    # ``part-{epoch_ms}-{seed}.{ext}`` — split on the
                    # first two dashes; seed/ext tail is irrelevant.
                    _, epoch_str, _tail = name.split("-", 2)
                    if int(epoch_str) >= cutoff_ms:
                        continue
                except (OSError, ValueError, IndexError):
                    continue
                try:
                    child.unlink()
                except OSError:
                    continue
                deleted += 1
                touched.add(str(leaf))

        for directory_key in touched:
            # Don't construct a fresh Path for invalidate; the cache
            # is keyed by the string form, and ``invalidate_listing``
            # walks back up through parents until ``self.path``.
            self._listing_cache.pop(directory_key, None)
        if touched:
            self._listing_cache.pop(str(self.path), None)
        return deleted

    def cleanup_stale_once(self, ttl_seconds: "float | None" = None) -> int:
        """:meth:`cleanup_stale`, throttled to once per *ttl_seconds*.

        Two-layer throttle, same shape as the staging-directory
        sweep in :mod:`yggdrasil.io.path.local_path`:

        * **In-process** — :data:`_CLEANUP_DONE` records every path
          already swept by this Python run. Subsequent calls return
          immediately; a send_many burst that calls
          :meth:`CacheConfig.local_cache` repeatedly only pays for
          one sentinel stat plus one set lookup.
        * **Cross-process** — ``{path}/.ygg/.last_cleanup`` is
          written after each sweep. A sentinel younger than *ttl*
          short-circuits the walk: a parallel process / a quickly-
          restarted run shares the throttle.

        Sentinel is touched *after* the sweep, so a partial sweep
        (process killed mid-walk, permission denial halfway through)
        gets retried by the next caller after the TTL elapses
        rather than being recorded as a successful sweep.

        Returns the number of files unlinked, or ``0`` when the
        sweep was throttled. Never raises.
        """
        ttl = self.CLEANUP_TTL_SECONDS if ttl_seconds is None else float(ttl_seconds)
        key = str(self.path)
        if key in _CLEANUP_DONE:
            return 0
        with _CLEANUP_LOCK:
            if key in _CLEANUP_DONE:
                return 0
            _CLEANUP_DONE.add(key)

        sentinel = self._cleanup_sentinel_path
        now_s = time.time()
        try:
            last_mtime = sentinel.mtime
        except OSError:
            last_mtime = 0.0
        if last_mtime and last_mtime >= now_s - ttl:
            return 0

        try:
            deleted = self.cleanup_stale(ttl)
        except Exception:
            # Defensive: ``cleanup_stale`` already swallows OSErrors,
            # but a bug in a Path subclass shouldn't poison the cache
            # call piggy-backing on the sweep.
            return 0

        # Touch the sentinel last so a partial sweep retries on
        # the next call after TTL, rather than being marked done.
        try:
            self._metadata_dir.mkdir(parents=True, exist_ok=True)
            with sentinel.open(mode="wb") as bio:
                bio.write(str(int(now_s)).encode("ascii"))
        except OSError:
            pass
        return deleted

    def _iter_cleanup_leaves(self) -> "Iterator[Path]":
        """Directories that may contain ``part-*`` files.

        Partitioned tree → every ``col=val/.../`` leaf via
        :meth:`_iter_partitions`. Unpartitioned (or schema-less)
        tree → just :attr:`path`. The ``.ygg/`` sidecar is never
        a partition leaf (its name doesn't match the ``col=val``
        pattern), so the walker skips it for free.
        """
        parts = self._resolve_partition_columns()
        if not parts:
            yield self.path
            return
        try:
            for leaf_path, _kv in self._iter_partitions(parts, prune={}):
                yield leaf_path
        except OSError:
            return

    # ==================================================================
    # Repr
    # ==================================================================

    def __repr__(self) -> str:
        cols = ", ".join(self.partition_columns)
        return f"YGGFolderIO(path={self.path!r}, partition_by=[{cols}])"


def _iter_and_conjuncts(predicate: Any) -> "Iterator[Any]":
    """Flatten nested top-level ``AND`` nodes into individual conjuncts."""
    if isinstance(predicate, Logical) and predicate.op is LogicalOp.AND:
        for op in predicate.operands:
            yield from _iter_and_conjuncts(op)
        return
    yield predicate


def _and_combine(conjs: "list[Any]") -> Any:
    """Re-assemble *conjs* into a single ``AND`` (or pass through one)."""
    if len(conjs) == 1:
        return conjs[0]
    return Logical(LogicalOp.AND, tuple(conjs))
