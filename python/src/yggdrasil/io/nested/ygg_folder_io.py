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

#: Subdirectory inside :data:`_METADATA_DIR_NAME` that holds one log
#: file per ``checkpoint_id``. Each log stores the relative path of
#: the most recent part file fully drained by a checkpointed read,
#: so the next read with the same id resumes after it.
_CHECKPOINTS_DIR_NAME: str = "checkpoints"


def _relpath(root: str, entry: "Path") -> str:
    """Path of *entry* relative to *root*, normalized with ``/``.

    Used as the checkpoint sort key so log files stay portable
    across platforms (Windows writers / Linux readers and vice
    versa).
    """
    rel = os.path.relpath(str(entry), root)
    return rel.replace(os.sep, "/")


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
        if options.checkpoint_id:
            yield from self._read_checkpointed(options)
            return

        parts = self._resolve_partition_columns()
        if not parts:
            yield from super()._read_arrow_batches(options)
            return

        prune = options.prune_values or {}
        for part_path, part_kv in self._iter_partitions(parts, prune):
            sub = FolderIO(path=part_path)
            for batch in sub._read_arrow_batches(options):
                yield self._stamp_partitions(batch, part_kv, parts)

    # ==================================================================
    # Checkpoint streaming — resume reads after the last drained file
    # ==================================================================

    def _read_checkpointed(
        self,
        options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Yield batches from part files strictly newer than the log.

        "Newer" is lexicographic over the file's path relative to
        :attr:`path` — :meth:`make_child` mints
        ``part-{epoch_ms}-{seed}.{ext}``, so lex order over the
        relative path matches creation order across both the
        partitioned and the flat layout. After each part file is
        fully drained, the log is rewritten with that file's relative
        path so a crash mid-stream resumes from the last *complete*
        file rather than re-yielding it.

        ``options.prune_values`` still applies on the partitioned
        path; the checkpoint just narrows the file set further.
        """
        checkpoint_id = options.checkpoint_id
        last_seen = self._read_checkpoint(checkpoint_id)
        log_path = self._checkpoint_log_path(checkpoint_id)

        parts = self._resolve_partition_columns()
        for rel_path, file_path, kv in self._iter_checkpoint_files(
            parts, options.prune_values or {},
        ):
            if last_seen is not None and rel_path <= last_seen:
                continue
            # Checkpoint file lives under the same tree we're scanning;
            # never feed it back through the data reader.
            if file_path == log_path:
                continue
            leaf = self._leaf_for(file_path)
            if leaf is None:
                continue
            for batch in leaf._read_arrow_batches(leaf.options_class()()):
                if parts and kv:
                    batch = self._stamp_partitions(batch, kv, parts)
                yield batch
            self._write_checkpoint(checkpoint_id, rel_path)

    def _iter_checkpoint_files(
        self,
        parts: "list[Field]",
        prune: "dict[str, Any] | Any",
    ) -> "Iterator[tuple[str, Path, dict[str, Any]]]":
        """Yield ``(relative_path, file_path, partition_kv)`` for every
        part file under :attr:`path`, sorted by relative path.

        Relative paths use ``/`` as the separator regardless of OS so
        the log is portable across platforms.
        """
        if not self.path.exists():
            return
        root = str(self.path)
        collected: "list[tuple[str, Path, dict[str, Any]]]" = []

        if parts:
            for leaf_path, kv in self._iter_partitions(parts, prune):
                for entry in self._iter_part_entries(leaf_path):
                    collected.append((_relpath(root, entry), entry, kv))
        else:
            for entry in self._iter_part_entries_recursive(self.path):
                collected.append((_relpath(root, entry), entry, {}))

        collected.sort(key=lambda t: t[0])
        yield from collected

    def _iter_part_entries(self, directory: "Path") -> "Iterator[Path]":
        """Direct, non-recursive list of part files in *directory*."""
        if not directory.exists():
            return
        try:
            entries = list(directory.iterdir())
        except FileNotFoundError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir():
                    continue
            except Exception:
                continue
            yield entry

    def _iter_part_entries_recursive(
        self, directory: "Path",
    ) -> "Iterator[Path]":
        """Recursive walk for the no-schema (flat / nested-folder) case."""
        try:
            entries = list(directory.iterdir())
        except FileNotFoundError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                is_dir = entry.is_dir()
            except Exception:
                continue
            if is_dir:
                yield from self._iter_part_entries_recursive(entry)
            else:
                yield entry

    def _checkpoint_log_path(self, checkpoint_id: str) -> "Path":
        return self._metadata_dir / _CHECKPOINTS_DIR_NAME / checkpoint_id

    def _read_checkpoint(self, checkpoint_id: str) -> "str | None":
        target = self._checkpoint_log_path(checkpoint_id)
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
            return payload.decode("utf-8").strip() or None
        except Exception:
            return None

    def _write_checkpoint(self, checkpoint_id: str, rel_path: str) -> None:
        """Persist *rel_path* as the latest fully-drained file.

        Best-effort: a write failure shouldn't take down the read
        stream, so disk errors are swallowed. The next read just
        re-yields any files that came after the previous successful
        write.
        """
        target = self._checkpoint_log_path(checkpoint_id)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = rel_path.encode("utf-8")
        try:
            with target.open(mode="wb") as bio:
                bio.write(payload)
        except Exception:
            pass

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
        if options.match_by_names:
            sub_match = [
                m for m in options.match_by_names if m not in partition_names
            ]
            if sub_match != list(options.match_by_names):
                leaf_options = dataclasses.replace(
                    options, match_by_names=sub_match or None,
                )

        for key_tuple, sub_batches in groups.items():
            kv = dict(zip(partition_names, key_tuple))
            target = self._ensure_partition_dir(parts, kv)
            sub = FolderIO(path=target)
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
            sub = FolderIO(path=leaf_path)
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

        compacted = 0
        for leaf_path, _kv in self._iter_partitions(parts, prune={}):
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
