"""Generic folder-of-tabular-files :class:`NestedIO` leaf.

:class:`FolderIO` is the canonical concrete :class:`NestedIO`: a
directory whose contents are tabular files (Parquet, IPC, CSV,
JSON, ...), a tree of sub-folders containing such files, or a
Hive-style ``key=value/`` partition layout — all transparently
handled by the same class.

What it does
------------

- :meth:`iter_children` walks :attr:`path` and yields one IO per
  direct entry: file children become :class:`PrimitiveIO` (format
  inferred from extension), sub-directory children become a fresh
  :class:`FolderIO` bound to that sub-directory. Reading a folder
  through :meth:`read_arrow_batches` therefore recurses
  transparently — a tree of sub-folders flattens into one batch
  stream without the caller doing anything special.

- :meth:`make_child` is the inverse: build a fresh write target
  under :attr:`path`. Forward-slash separators in ``name`` are
  honoured (the parent directories are auto-created), so the
  partitioned write path can pass ``"year=2025/part-00000.parquet"``
  in one call.

- Partition discovery: when ``partition_columns`` is supplied (or
  inferred from the directory layout), each leaf carries a
  ``partition_values`` mapping derived from its position in the
  tree, and the read path injects those columns into every batch.
  Writes group rows by partition tuple and route each group to a
  child file under the matching ``key=value/`` prefix.

What it doesn't do
------------------

- Schema homogeneity enforcement. Heterogeneous-children folders
  work but :meth:`collect_schema` returns the union (via
  :meth:`Schema.merge_with`); writers assume the caller knows what
  they're doing.
- Transaction semantics. APPEND adds a sibling, OVERWRITE clears
  and writes; there's no atomicity beyond the staging-rename
  pattern of a single child file.

Subclasses
----------

:class:`DeltaIO` (in ``yggdrasil.io.buffer.nested.delta``) inherits
from :class:`FolderIO` and overrides:

- :meth:`_is_ignored_path` to hide ``_delta_log/``;
- :meth:`iter_children` to read the commit log, not the directory;
- :meth:`_finalize_child` to write a commit-log entry instead of a
  plain rename.
"""

from __future__ import annotations

import dataclasses
import urllib.parse
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.schema import Field, Schema
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.primitive import PrimitiveIO
from yggdrasil.io.enums import MimeType, MimeTypes, Mode
from yggdrasil.io.fs import Path
from yggdrasil.io.tabular import TabularIO
from .base import NestedIO, NestedOptions

if TYPE_CHECKING:
    pass


__all__ = ["FolderIO", "FolderOptions"]


@dataclasses.dataclass(frozen=True, slots=True)
class FolderOptions(NestedOptions):
    """:class:`NestedOptions` extended with partition-routing knobs.

    :param partition_columns: explicit list of partition columns as
        :class:`Field` instances (or anything :meth:`Field.from_any`
        accepts). When set, this is authoritative. When ``None``
        (default), the IO falls back to its constructor-declared
        ``partition_columns`` and finally to inference from the
        directory layout's first leaf.
    :param sort_partitions: on write, sort the input batches by
        partition columns before grouping. Default ``True``: produces
        one child file per distinct partition combination per write
        call. ``False`` skips the sort — cheaper memory, but emits
        one child per partition combination *per batch*.
    :param partition_strict: when reading a partitioned tree, raise
        on a leaf whose path doesn't match the declared partition
        column shape. Default ``True``.
    """

    partition_columns: Any = None
    sort_partitions: bool = True
    partition_strict: bool = True


class FolderIO(NestedIO[FolderOptions]):
    """A directory of tabular files, transparently recursive and
    partition-aware.

    Construction:

        >>> io = FolderIO(path="/tmp/store/")
        >>> for child in io.iter_children():
        ...     # child is a PrimitiveIO for a file or a FolderIO
        ...     # for a sub-folder; either way it can be opened and
        ...     # drained on its own.
        ...     print(child.path, child.parent is io)

    With Hive partitions:

        >>> io = FolderIO(
        ...     path="/tables/trades/",
        ...     partition_columns=["year", "month"],
        ... )
        >>> table = io.read_arrow_table()  # year/month columns injected
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        return MimeTypes.FOLDER

    @classmethod
    def options_class(cls) -> type[FolderOptions]:
        return FolderOptions

    def _default_child_media_type(self) -> Any:
        """Parquet is the canonical folder-of-tables payload.

        Subclasses with a fixed format (Delta = parquet, IcebergV1
        = avro) override; callers that want IPC/CSV/... pass
        ``options.child_media_type`` explicitly.
        """
        return MimeTypes.PARQUET

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        partition_columns: "Sequence[Any] | None" = None,
        recursive: bool = True,
        partition_values: "Mapping[str, str | None] | None" = None,
        **kwargs: Any,
    ) -> None:
        """Construct a folder IO.

        :param partition_columns: declared at construction. Order
            matters — it determines the on-disk directory layering.
            Each entry is a :class:`Field` (or anything coercible).
            Default ``None`` defers to the option / inference
            fallbacks described in :class:`FolderOptions`.
        :param recursive: walk sub-folders during iteration. Default
            True; pass False to clip enumeration at one level.
        :param partition_values: partition values accumulated from
            an enclosing tree. Set by :meth:`iter_children` when
            handing back a sub-:class:`FolderIO`; users normally
            don't pass this themselves.
        """
        super().__init__(data, path=path, **kwargs)
        self._partition_columns = (
            tuple(_coerce_partition_column(c) for c in partition_columns)
            if partition_columns is not None
            else None
        )
        self._recursive = recursive
        self.partition_values: "dict[str, str | None] | None" = (
            dict(partition_values) if partition_values else None
        )

    # ==================================================================
    # Children — direct enumeration with transparent recursion
    # ==================================================================

    def _iter_children(
        self,
        options: FolderOptions,
    ) -> "Iterator[TabularIO | BytesIO]":
        """Yield one IO per direct child of :attr:`path`.

        Sub-directories are returned as fresh :class:`FolderIO`
        instances (sharing this folder's partition declaration and
        accumulating any ``key=value`` segment from the directory
        name into ``partition_values``). Files are returned as
        :class:`PrimitiveIO` when their extension maps to a
        registered tabular format; opaque files are returned as
        :class:`BytesIO` so callers can still pull bytes from them.
        Unknown / un-readable entries are silently skipped.

        Each child's ``parent`` attribute is stamped to ``self``.
        Children are returned closed (un-acquired); caller opens
        them inside a ``with`` block.

        Missing folder is treated as empty: no children yielded,
        no error raised. Hidden entries (name starting with ``.``)
        are filtered out by :meth:`_is_ignored_path`.
        """
        if not self.path.exists():
            return

        recursive = self._recursive

        for entry in self.path.iterdir():
            if self._is_ignored_path(entry):
                continue

            try:
                is_dir = entry.is_dir()
            except Exception:
                # Stat failure on a child mid-listing — skip rather
                # than abort. Listings on remote stores can race
                # with deletes.
                continue

            if is_dir:
                if not recursive:
                    continue
                child = self._open_subfolder(entry)
            else:
                child = self._open_file_child(entry)

            if child is None:
                continue

            self._attach(child)
            yield child

    def _open_subfolder(self, sub_path: Path) -> "FolderIO":
        """Build a sub-:class:`FolderIO` rooted at *sub_path*.

        Carries forward this folder's partition declaration and the
        accumulated partition values, augmenting them with any
        ``key=value`` segment parsed from the sub-folder's name.
        """
        kv = _parse_kv_segment(sub_path.name)
        sub_pv = dict(self.partition_values) if self.partition_values else {}
        if kv is not None:
            sub_pv[kv[0]] = kv[1]

        return type(self)(
            path=sub_path,
            partition_columns=self._partition_columns,
            recursive=self._recursive,
            partition_values=sub_pv if sub_pv else None,
            auto_open=False,
        )

    def _open_file_child(
        self,
        file_path: Path,
    ) -> "TabularIO | BytesIO | None":
        """Build an IO for a file leaf.

        Tries :meth:`TabularIO.from_path` first; falls back to
        :class:`BytesIO` for files with no registered tabular
        format. Returns ``None`` if neither can bind (rare —
        usually means the entry vanished mid-listing).
        """
        try:
            io = TabularIO.from_path(file_path)
        except Exception:
            io = None

        if isinstance(io, PrimitiveIO):
            # Stamp partition values for downstream injection.
            if self.partition_values:
                io.partition_values = dict(self.partition_values)
            return io

        # No tabular registration — fall back to a raw BytesIO so
        # callers can still reach the bytes via iter_children.
        try:
            return BytesIO(path=file_path, auto_open=False)
        except Exception:
            return None

    # ==================================================================
    # Cheap is_empty — avoid full iteration
    # ==================================================================

    def is_empty(self) -> bool:
        """True if no non-ignored entries exist anywhere in the tree."""
        if not self.path.exists():
            return True
        return self._is_subtree_empty(self.path)

    def _is_subtree_empty(self, root: Path) -> bool:
        for entry in root.iterdir():
            if self._is_ignored_path(entry):
                continue
            try:
                is_dir = entry.is_dir()
            except Exception:
                continue
            if not is_dir:
                return False
            if self._recursive and not self._is_subtree_empty(entry):
                return False
        return True

    # ==================================================================
    # Child IO factory — write side
    # ==================================================================

    def make_child(
        self,
        name: str,
        *,
        media_type: Any = None,
    ) -> PrimitiveIO:
        """Build a fresh write target under :attr:`path`.

        ``name`` may include forward-slash separators for nested
        layouts (Hive partitions, sub-folder writes); the parent
        directories are auto-created. Backslashes are rejected
        (URL semantics) and ``..`` segments are rejected (path
        traversal).

        Returns a closed (un-acquired) :class:`PrimitiveIO` with
        ``parent`` set to ``self``.
        """
        if "\\" in name:
            raise ValueError(
                f"Child name must not contain backslashes; got {name!r}. "
                "Use forward slashes for nested paths."
            )
        if name.startswith("/"):
            raise ValueError(
                f"Child name must be relative; got {name!r}."
            )
        segments = name.split("/")
        if any(s == ".." for s in segments):
            raise ValueError(
                f"Child name must not contain '..' segments; got {name!r}."
            )

        child_path = self.path.joinpath(*segments)
        child_path.parent.mkdir(parents=True, exist_ok=True)

        io = TabularIO.from_path(child_path, media_type=media_type)

        if not isinstance(io, PrimitiveIO):
            raise TypeError(
                f"FolderIO child factory expected a PrimitiveIO for "
                f"name={name!r}, media_type={media_type!r}; got "
                f"{type(io).__name__}."
            )
        return self._attach(io)

    # ==================================================================
    # Partition column resolution
    # ==================================================================

    def _resolve_partition_columns(
        self,
        options: "FolderOptions | None" = None,
    ) -> "tuple[Field, ...]":
        """Resolve the active partition column list.

        Order of precedence:

        1. ``options.partition_columns`` if explicitly set.
        2. The constructor-declared ``self._partition_columns``.
        3. Inferred from the first leaf's directory layout (string
           dtype for each, since the path can't tell us more).

        Returns a tuple of :class:`Field` instances. Empty tuple
        means "not partitioned" — partition-aware code paths
        degrade to flat-folder behavior in that case.
        """
        if options is not None and options.partition_columns is not None:
            return tuple(
                _coerce_partition_column(c) for c in options.partition_columns
            )

        if self._partition_columns is not None:
            return self._partition_columns

        return self._infer_partition_columns()

    def _infer_partition_columns(self) -> "tuple[Field, ...]":
        """Inference fallback: walk to the first leaf, parse its path."""
        if not self.path.exists():
            return ()

        for leaf in _walk_leaves(self.path, self._is_ignored_path):
            try:
                rel_parts = leaf.relative_to(self.path).parts
            except Exception:
                continue
            kv_parts = rel_parts[:-1]
            keys: list[Field] = []
            for seg in kv_parts:
                kv = _parse_kv_segment(seg)
                if kv is None:
                    return ()
                key, _ = kv
                keys.append(_coerce_partition_column(key))
            return tuple(keys)

        return ()

    # ==================================================================
    # Read derivation — chain children with optional partition injection
    # ==================================================================

    def _read_arrow_batches(
        self,
        options: FolderOptions,
    ) -> Iterator[pa.RecordBatch]:
        """Chain children; inject partition columns when declared."""
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        partition_cols = self._resolve_partition_columns(options)
        partition_by_name = (
            {c.name: c for c in partition_cols} if partition_cols else None
        )
        strict = bool(options.partition_strict)

        for child in self._iter_children(options):
            if isinstance(child, NestedIO):
                with child:
                    yield from child._read_arrow_batches(options)
                continue

            if not isinstance(child, TabularIO):
                continue

            pv = self._partition_values_for(
                child, partition_cols, strict=strict,
            )
            with child:
                for batch in child.read_arrow_batches(options=options):
                    if pv and partition_by_name:
                        batch = _inject_partition_columns(
                            batch, pv, partition_by_name,
                        )
                    yield batch

    def _partition_values_for(
        self,
        child: Any,
        partition_cols: "Sequence[Field]",
        *,
        strict: bool,
    ) -> "Mapping[str, str | None] | None":
        """Resolve a leaf's partition values.

        Prefer the leaf's own ``partition_values`` (set by the
        recursive ``iter_children`` walk). When unset, parse from
        the child's path relative to root.
        """
        if not partition_cols:
            return None

        pv = getattr(child, "partition_values", None)
        if pv:
            return pv

        leaf_path: Path | None = getattr(child, "path", None)
        if leaf_path is None:
            return None

        try:
            rel_parts = leaf_path.relative_to(self.path).parts
        except Exception:
            return None

        kv_parts = rel_parts[:-1]
        expected_keys = tuple(c.name for c in partition_cols)
        if len(kv_parts) != len(expected_keys):
            if strict:
                raise ValueError(
                    f"Partition depth mismatch for leaf {leaf_path!r}: "
                    f"path has {len(kv_parts)} k=v segment(s), expected "
                    f"{len(expected_keys)} ({list(expected_keys)!r})."
                )
            return None

        out: dict[str, str | None] = {}
        for segment, expected_key in zip(kv_parts, expected_keys):
            kv = _parse_kv_segment(segment)
            if kv is None:
                if strict:
                    raise ValueError(
                        f"Path segment {segment!r} for leaf "
                        f"{leaf_path!r} is not in 'key=value' form."
                    )
                return None
            key, value = kv
            if key != expected_key:
                if strict:
                    raise ValueError(
                        f"Partition key mismatch for leaf "
                        f"{leaf_path!r} at segment {segment!r}: got "
                        f"key {key!r}, expected {expected_key!r}."
                    )
                return None
            out[key] = value
        return out

    # ==================================================================
    # Schema collection — children union, plus partition columns
    # ==================================================================

    def _collect_schema(self, options: FolderOptions) -> Schema:
        children_schema = super()._collect_schema(options)
        partition_cols = self._resolve_partition_columns(options)
        if not partition_cols:
            return children_schema

        out = children_schema.copy()
        for f in partition_cols:
            out[f.name] = f
        return out

    # ==================================================================
    # Write derivation — partition-aware routing
    # ==================================================================

    def _write_arrow_batches(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: FolderOptions,
    ) -> None:
        """Route batches to children, with partition grouping when needed."""
        partition_cols = self._resolve_partition_columns(options)
        if not partition_cols:
            super()._write_arrow_batches(batches, options)
            return

        partition_names = [c.name for c in partition_cols]

        mode = self._resolve_save_mode(options.mode)
        if mode is Mode.IGNORE:
            return
        if mode is Mode.UPSERT:
            self._arrow_upsert_via_rewrite(batches, options)
            return
        if mode is Mode.OVERWRITE:
            self._clear_children()

        if options.sort_partitions:
            self._write_partitioned_sorted(batches, partition_names, options)
        else:
            self._write_partitioned_streaming(batches, partition_names, options)

    def _write_partitioned_sorted(
        self,
        batches: "Iterable[pa.RecordBatch]",
        partition_names: "Sequence[str]",
        options: FolderOptions,
    ) -> None:
        """Materialize, sort, group → one child per partition combination."""
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        all_batches = [first]
        all_batches.extend(batch_iter)
        table = pa.Table.from_batches(all_batches)

        missing = [n for n in partition_names if n not in table.column_names]
        if missing:
            raise ValueError(
                f"Partitioned write missing partition columns "
                f"{missing!r} in input. Available: "
                f"{list(table.column_names)!r}."
            )

        sort_keys = [(n, "ascending") for n in partition_names]
        table = table.sort_by(sort_keys)

        partition_table = table.select(partition_names)
        partition_rows = partition_table.to_pylist()
        n = table.num_rows
        if n == 0:
            return

        run_start = 0
        run_key: tuple = tuple(
            partition_rows[0].get(name) for name in partition_names
        )
        for i in range(1, n + 1):
            if i == n:
                next_key = None
            else:
                next_key = tuple(
                    partition_rows[i].get(name) for name in partition_names
                )
            if i == n or next_key != run_key:
                slice_table = table.slice(run_start, i - run_start)
                self._write_partition_chunk(
                    slice_table, partition_names, run_key, options,
                )
                run_start = i
                if next_key is not None:
                    run_key = next_key

    def _write_partitioned_streaming(
        self,
        batches: "Iterable[pa.RecordBatch]",
        partition_names: "Sequence[str]",
        options: FolderOptions,
    ) -> None:
        """No-sort routing: one child per (batch, partition) pair."""
        for batch in batches:
            if batch.num_rows == 0:
                continue
            missing = [
                n for n in partition_names if n not in batch.schema.names
            ]
            if missing:
                raise ValueError(
                    f"Partitioned write missing partition columns "
                    f"{missing!r} in input batch. Available: "
                    f"{list(batch.schema.names)!r}."
                )

            sub_table = pa.Table.from_batches([batch])
            partition_table = sub_table.select(partition_names)
            partition_rows = partition_table.to_pylist()

            buckets: dict[tuple, list[int]] = {}
            for i, row in enumerate(partition_rows):
                key = tuple(row.get(name) for name in partition_names)
                buckets.setdefault(key, []).append(i)

            for key, indices in buckets.items():
                slice_table = sub_table.take(pa.array(indices))
                self._write_partition_chunk(
                    slice_table, partition_names, key, options,
                )

    def _write_partition_chunk(
        self,
        chunk: pa.Table,
        partition_names: "Sequence[str]",
        partition_key: tuple,
        options: FolderOptions,
    ) -> None:
        """Write *chunk* (single partition tuple) to a child file.

        Strips the partition columns before writing — Hive convention.
        """
        if chunk.num_rows == 0:
            return

        keep = [c for c in chunk.column_names if c not in partition_names]
        data_chunk = chunk.select(keep)

        partition_values = {
            name: (None if val is None else str(val))
            for name, val in zip(partition_names, partition_key)
        }
        relative_dir = _partition_path_segment(partition_values)

        media_type = options.child_media_type or self._default_child_media_type()

        partition_root = (
            self.path.joinpath(*relative_dir.split("/"))
            if relative_dir else self.path
        )
        partition_root.mkdir(parents=True, exist_ok=True)

        staging_path = partition_root.make_staging(media_type=media_type)
        staging_name = staging_path.name

        child_relative = (
            f"{relative_dir}/{staging_name}" if relative_dir else staging_name
        )
        child = self.make_child(child_relative, media_type=media_type)

        try:
            with child:
                child.write_arrow_table(data_chunk, options=options)
        except Exception:
            try:
                staging_path.remove(allow_not_found=True)
            except Exception:
                pass
            raise

        final_name = self._next_child_name_in(
            partition_root, media_type=media_type,
        )
        final_path = partition_root / final_name
        staging_path.rename(final_path)

    def _next_child_name_in(self, parent: Path, *, media_type: Any) -> str:
        """Like :meth:`_next_child_name` but under an arbitrary parent."""
        ext = self._extension_for(media_type)
        prefix = "part-"
        max_idx = -1

        if parent.exists():
            for child in parent.iterdir():
                name = child.name
                if not name.startswith(prefix):
                    continue
                stem = name[len(prefix):]
                idx_str = stem.split(".", 1)[0]
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx

        next_idx = max_idx + 1
        suffix = f".{ext}" if ext else ""
        return f"{prefix}{next_idx:05d}{suffix}"


# ---------------------------------------------------------------------------
# Module-private helpers — partition path / leaf walk / column injection
# ---------------------------------------------------------------------------


def _coerce_partition_column(value: Any) -> Field:
    """Build a :class:`Field` from a partition-column hint.

    Accepts a raw column name (``"year"``) as well as anything
    :meth:`Field.from_any` knows how to coerce. Default dtype for
    a name-only entry is string (Hive partition values are strings
    on the wire; the read path casts to a richer dtype if the
    caller declared one explicitly).
    """
    if isinstance(value, Field):
        return value
    if isinstance(value, str):
        return Field(name=value, dtype="string")
    return Field.from_any(value)


def _parse_kv_segment(segment: str) -> "tuple[str, str] | None":
    """Parse a ``key=value`` directory segment.

    Returns ``(key, value)`` with the value URL-unquoted (Hive
    convention — values containing path-unsafe characters get
    percent-encoded). Returns ``None`` if the segment doesn't
    contain ``=`` at all.
    """
    if "=" not in segment:
        return None
    key, _, value = segment.partition("=")
    if not key:
        return None
    return key, urllib.parse.unquote(value)


def _walk_leaves(
    root: Path,
    is_ignored: "Any",
) -> Iterator[Path]:
    """Yield leaf files under *root* depth-first, skipping ignored entries."""
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
            if is_ignored(child):
                continue
            try:
                is_dir = child.is_dir()
            except Exception:
                continue
            if is_dir:
                stack.append(child)
            else:
                yield child


def _partition_path_segment(values: "Mapping[str, str | None]") -> str:
    """Build the relative ``key=value/key=value`` path segment.

    Hive escapes path-unsafe characters with percent encoding; we
    mirror that on write so reads round-trip.
    """
    parts = []
    for key, value in values.items():
        if value is None:
            raise ValueError(
                f"Cannot serialize None partition value for key "
                f"{key!r}. Replace nulls in partition columns "
                "before writing, or filter the rows out."
            )
        quoted = urllib.parse.quote(str(value), safe="")
        parts.append(f"{key}={quoted}")
    return "/".join(parts)


def _inject_partition_columns(
    batch: pa.RecordBatch,
    partition_values: "Mapping[str, str | None]",
    partition_by_name: "Mapping[str, Field]",
) -> pa.RecordBatch:
    """Append partition columns to *batch*, casting per declared dtype."""
    if not partition_values:
        return batch

    n_rows = batch.num_rows
    for key, value in partition_values.items():
        if key in batch.schema.names:
            continue

        field = partition_by_name.get(key)
        if field is None:
            arrow_type = pa.string()
        else:
            arrow_type = field.dtype.to_arrow_dtype() if hasattr(
                field.dtype, "to_arrow_dtype"
            ) else pa.string()

        scalar = (
            pa.scalar(value, type=pa.string())
            if value is not None
            else pa.scalar(None, type=pa.string())
        )
        try:
            typed_scalar = (
                pc.cast(scalar, arrow_type)
                if scalar.type != arrow_type
                else scalar
            )
        except Exception:
            typed_scalar = scalar
            arrow_type = pa.string()

        column = pa.array(
            [typed_scalar.as_py()] * n_rows,
            type=arrow_type,
        )
        batch = batch.append_column(
            pa.field(key, arrow_type), column,
        )
    return batch
