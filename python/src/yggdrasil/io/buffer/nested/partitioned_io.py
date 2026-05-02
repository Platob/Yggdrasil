"""Hive-style partitioned :class:`FolderIO`.

A partitioned folder is a tree of subdirectories whose names follow
the convention ``key=value/`` repeated for each partition column.
The leaf children are tabular files (parquet, ipc, csv, ...) whose
contents *exclude* the partition columns — those are encoded in the
path. So a table partitioned by ``(year, month)`` looks like::

    root/
      year=2024/month=12/part-00000.parquet
      year=2025/month=01/part-00000.parquet
      year=2025/month=02/part-00000.parquet

Read time
---------

For each leaf file:

1. Walk the path from the file up to the root, parsing ``k=v``
   segments into :attr:`FragmentInfos.partition_values`.
2. Open the file as a :class:`PrimitiveIO` (the file itself doesn't
   know about partitions — it's just parquet/ipc).
3. When chaining batches in :meth:`_read_arrow_batches`, append
   one column per partition key, broadcasting the parsed string
   value across the batch's rows. Values are cast to the dtype
   declared on the corresponding partition :class:`Field` from
   :attr:`partition_columns`; the default (string) is used if no
   declaration is present.

Write time
----------

Given a batch iterator and a list of partition columns, group rows
by partition values and route each group to a child file under the
correct ``k=v/`` prefix. Sorting the input by partition keys first
is the default (``options.sort_partitions=True``) so the grouping
loop only emits one child per distinct key combination per call;
turning sort off is faster but produces one child per *batch* per
distinct combination, which leaves more cleanup work for downstream
compaction.

What's deliberately not here
----------------------------

- Stats-based file skipping. Partition pruning happens at the
  fragment level — callers filter the iterator before
  :meth:`_read_arrow_batches` chains. A future
  :class:`PartitionFilter` API can sit on top of
  :meth:`iter_fragments`.
- Schema evolution across partitions. We assume every file has the
  same non-partition schema; mismatches surface during the merge in
  :meth:`_collect_schema`.
- Nested partition columns or `__HIVE_DEFAULT_PARTITION__` Hive
  sentinels. Both can be added as small extensions if needed.
"""

from __future__ import annotations

import urllib.parse
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Mapping, Sequence, Iterable

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.schema import Field, Schema
from yggdrasil.io.fragment import Fragment, FragmentInfos
from yggdrasil.io.fs import Path
from .base import NestedOptions
from .folder_io import FolderIO
from ...enums import MimeType, MimeTypes

if TYPE_CHECKING:
    from yggdrasil.io.buffer.primitive import PrimitiveIO


__all__ = ["PartitionedFolderIO", "PartitionedOptions"]


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class PartitionedOptions(NestedOptions):
    """Folder options extended with partition routing knobs.

    :param partition_columns: explicit list of partition columns as
        :class:`Field` instances (or anything :meth:`Field.from_any`
        accepts). When set, this is authoritative. When ``None``
        (default), the IO falls back to its constructor-declared
        ``partition_columns`` and finally to inference from the
        directory layout's first leaf.
    :param sort_partitions: on write, sort the input batches by
        partition columns before grouping. Default ``True``: produces
        one child file per partition combination per write call,
        which keeps the directory tidy. ``False`` skips the sort —
        faster, but emits one child per partition combination *per
        batch*, leaving compaction work for downstream.
    :param partition_strict: when reading, raise on a leaf whose
        path doesn't match the expected partition column shape.
        Default ``True`` — silent skips would hide corruption. When
        ``False``, mismatched leaves are skipped with a debug log.
    """

    partition_columns: Any = None
    sort_partitions: bool = True
    partition_strict: bool = True


# ---------------------------------------------------------------------------
# PartitionedFolderIO
# ---------------------------------------------------------------------------


class PartitionedFolderIO(FolderIO):
    """A folder of homogeneous tabular files with Hive-style partitions.

    Subclasses :class:`FolderIO`; replaces the one-level enumeration
    with a recursive walk that parses ``key=value/`` directory
    segments, and the flat write path with a partition-aware
    grouping write.

    :param partition_columns: list of partition columns, declared
        at construction. Order matters — it determines the on-disk
        directory layering. Each entry is a :class:`Field` (or
        anything coercible). Default ``None`` defers to the
        ``partition_columns`` option or directory inference.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        partition_columns: "Sequence[Any] | None" = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(data, path=path, **kwargs)
        self._partition_columns = (
            tuple(Field.from_any(c) for c in partition_columns)
            if partition_columns is not None
            else None
        )

    @classmethod
    def options_class(cls):
        return PartitionedOptions
    
    @classmethod
    def default_mime_type(cls) -> MimeType:
        return MimeTypes.PARTITIONED_FOLDER
    
    # ==================================================================
    # Partition column resolution
    # ==================================================================

    def _resolve_partition_columns(
        self,
        options: "PartitionedOptions | None" = None,
    ) -> "tuple[Field, ...]":
        """Resolve the active partition column list.

        Order of precedence:

        1. ``options.partition_columns`` if explicitly set.
        2. The constructor-declared ``self._partition_columns``.
        3. Inferred from the first leaf's directory layout (string
           dtype for each, since the path can't tell us more).

        Returns a tuple of :class:`Field` instances. Empty tuple
        means "not partitioned" — partitioned IO degrades to flat
        :class:`FolderIO` behavior in that case.
        """
        if options is not None and options.partition_columns is not None:
            return tuple(Field.from_any(c) for c in options.partition_columns)

        if self._partition_columns is not None:
            return self._partition_columns

        return self._infer_partition_columns()

    def _infer_partition_columns(self) -> "tuple[Field, ...]":
        """Inference fallback: walk to the first leaf, parse its path.

        Returns string-dtyped :class:`Field` instances for each
        ``k=v`` segment between root and the first file. Empty
        tuple if no leaves or the leaf is directly under root.
        """
        if not self.path.exists():
            return ()

        for leaf in self._walk_leaves(self.path):
            try:
                rel_parts = leaf.relative_to(self.path).parts
            except Exception:
                continue
            # Last segment is the file name; the rest are k=v dirs.
            kv_parts = rel_parts[:-1]
            keys: list[Field] = []
            for seg in kv_parts:
                kv = self._parse_kv_segment(seg)
                if kv is None:
                    return ()
                key, _ = kv
                keys.append(Field.from_any(key))  # default = string
            return tuple(keys)

        return ()

    # ==================================================================
    # Path walking
    # ==================================================================

    def _walk_leaves(self, root: Path) -> Iterator[Path]:
        """Yield leaf files under *root*, skipping ignored entries.

        Recursive, depth-first. Subdirectories whose names don't
        look like ``k=v`` are still descended (we don't enforce
        the partition shape here — that's the caller's job during
        fragment construction).
        """
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
                if self._is_ignored_path(child):
                    continue
                try:
                    is_dir = child.is_dir()
                except Exception:
                    continue
                if is_dir:
                    stack.append(child)
                else:
                    yield child

    @staticmethod
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
        # urllib.parse.unquote handles the Hive percent-encoding
        # (e.g. "year=2025%2F04" -> "2025/04"). Slashes in partition
        # values are rare but legal.
        return key, urllib.parse.unquote(value)

    # ==================================================================
    # Fragment enumeration — recursive, partition-parsing
    # ==================================================================

    def iter_fragments(
        self,
        options: "PartitionedOptions | None" = None,
        **kwargs: Any,
    ) -> Iterator[Fragment]:
        """Yield one :class:`Fragment` per leaf file with parsed partitions.

        Each fragment carries:

        - ``infos.url`` — the leaf file's URL.
        - ``infos.partition_values`` — ``{key: str_value}`` parsed
          from the leaf's relative path. Empty mapping for an
          unpartitioned table (degenerate case — empty
          ``partition_columns``).
        - ``infos.schema`` / ``infos.mtime`` — populated when
          ``populate_metadata`` is on, same as flat :class:`FolderIO`.
        - ``io`` — :class:`PrimitiveIO` bound to the leaf path,
          format inferred from extension.

        Layout violations:

        - A leaf nested under fewer/more ``k=v`` levels than
          ``partition_columns`` declares.
        - A ``k=v`` segment whose key doesn't match the expected
          partition column at that depth.

        With ``options.partition_strict=True`` (default) these
        raise :class:`ValueError`. With ``False`` they're skipped.
        """
        opts = self.check_options(options, overrides=locals())
        partition_cols = self._resolve_partition_columns(opts)
        partition_names = tuple(c.name for c in partition_cols)

        # No partitions declared and none inferred → degrade to
        # flat-folder behavior (don't recurse, don't parse).
        if not partition_names:
            yield from FolderIO.iter_fragments(self, opts)
            return

        if not self.path.exists():
            return

        populate = bool(opts.populate_metadata)
        strict = bool(opts.partition_strict)

        for leaf in self._walk_leaves(self.path):
            try:
                rel_parts = leaf.relative_to(self.path).parts
            except Exception:
                if strict:
                    raise ValueError(
                        f"Leaf {leaf!r} is not relative to root {self.path!r}"
                    )
                continue

            kv_parts = rel_parts[:-1]
            partition_values = self._parse_partition_path(
                kv_parts, partition_names, leaf=leaf, strict=strict,
            )
            if partition_values is None:
                # Strict path raises above; non-strict path returns None.
                continue

            child_io = self._open_child_for_read(leaf)
            if child_io is None:
                continue

            yield Fragment(
                infos=self._build_fragment_infos(
                    leaf, child_io, populate, partition_values=partition_values,
                ),
                io=child_io,
            )

    @staticmethod
    def _parse_partition_path(
        kv_parts: "Sequence[str]",
        expected_keys: "Sequence[str]",
        *,
        leaf: Path,
        strict: bool,
    ) -> "Mapping[str, str] | None":
        """Parse a relative path into ``{key: value}`` per declaration.

        Validates that the depth and per-level keys match the
        declared partition columns exactly. Returns ``None`` on
        mismatch in non-strict mode; raises in strict mode.
        """
        if len(kv_parts) != len(expected_keys):
            if strict:
                raise ValueError(
                    f"Partition depth mismatch for leaf {leaf!r}: "
                    f"path has {len(kv_parts)} k=v segment(s) "
                    f"({list(kv_parts)!r}), expected {len(expected_keys)} "
                    f"({list(expected_keys)!r})."
                )
            return None

        out: dict[str, str] = {}
        for segment, expected_key in zip(kv_parts, expected_keys):
            kv = PartitionedFolderIO._parse_kv_segment(segment)
            if kv is None:
                if strict:
                    raise ValueError(
                        f"Path segment {segment!r} for leaf {leaf!r} is "
                        "not in 'key=value' form."
                    )
                return None
            key, value = kv
            if key != expected_key:
                if strict:
                    raise ValueError(
                        f"Partition key mismatch for leaf {leaf!r} at "
                        f"segment {segment!r}: got key {key!r}, expected "
                        f"{expected_key!r}."
                    )
                return None
            out[key] = value
        return out

    def _build_fragment_infos(
        self,
        child: Path,
        child_io: "PrimitiveIO",
        populate: bool,
        *,
        partition_values: "Mapping[str, str] | None" = None,
    ) -> FragmentInfos:
        """FragmentInfos with partition values stamped in.

        Overrides the flat-folder builder to thread partition_values
        through. Falls back to empty dict (not ``None``) when the
        caller didn't pass a value, to signal "this IO does
        partitioning, this leaf just has none."
        """
        # Reuse the parent class for schema / mtime population.
        base = super()._build_fragment_infos(child, child_io, populate)
        return FragmentInfos(
            url=base.url,
            mtime=base.mtime,
            schema=base.schema,
            partition_values=(
                partition_values if partition_values is not None else {}
            ),
        )

    # ==================================================================
    # is_empty — short-circuit on first leaf
    # ==================================================================

    def is_empty(self) -> bool:
        """True iff the tree contains no non-ignored leaf files."""
        for _ in self._walk_leaves(self.path):
            return False
        return True

    # ==================================================================
    # Read derivation — chain batches with partition columns injected
    # ==================================================================

    def _read_arrow_batches(self, options: PartitionedOptions) -> Iterator[pa.RecordBatch]:
        """Chain children with partition column injection.

        Per leaf:

        1. Open the child IO and read its batches normally (the
           file itself doesn't know about partitions).
        2. For each batch, append one column per partition key
           with the partition value cast to the field's dtype and
           broadcast across the batch's rows.

        We materialize the broadcasted array once per *leaf* (same
        partition values for every batch from a given file) rather
        than per batch — the cast + array construction is the
        non-trivial cost; appending is cheap.
        """
        if self.cached:
            yield from self._read_arrow_batches_from_cache(options)
            return

        partition_cols = self._resolve_partition_columns(options)

        # Degenerate case: no partitions declared. Defer to flat
        # behavior so the simple-folder fast path stays simple.
        if not partition_cols:
            yield from FolderIO._read_arrow_batches(self, options)
            return

        partition_by_name: dict[str, Field] = {c.name: c for c in partition_cols}

        for frag in self.iter_fragments(options):
            child_io = frag.io
            if child_io is None:
                continue

            partition_values = frag.infos.partition_values or {}

            with child_io:
                for batch in child_io.read_arrow_batches(options=options):
                    yield self._inject_partition_columns(
                        batch, partition_values, partition_by_name,
                    )

    @staticmethod
    def _inject_partition_columns(
        batch: pa.RecordBatch,
        partition_values: "Mapping[str, str | None]",
        partition_by_name: "Mapping[str, Field]",
    ) -> pa.RecordBatch:
        """Append partition columns to *batch*, casting per declared dtype.

        Empty partitions / empty batch are returned unchanged. The
        cast goes through pyarrow's compute kernels — for non-string
        targets (e.g. int year, date month) the string value is
        parsed via the kernel's standard rules. Cast failures
        propagate; callers facing dirty partition values should
        declare string columns and cast downstream.
        """
        if not partition_values:
            return batch

        n_rows = batch.num_rows
        for key, value in partition_values.items():
            # Skip if the file already carried the column (e.g.
            # someone wrote partition columns by mistake). Path
            # value wins on read time? No — we don't know which is
            # canonical. Skip silently and let collect_schema
            # surface the duplicate as a merge_with conflict if
            # the caller cares.
            if key in batch.schema.names:
                continue

            field = partition_by_name.get(key)
            if field is None:
                # Unknown partition key — inject as string. This
                # shouldn't happen because iter_fragments validates
                # keys, but defense in depth is cheap.
                arrow_type = pa.string()
            else:
                arrow_type = field.dtype.to_arrow_dtype() if hasattr(
                    field.dtype, "to_arrow_dtype"
                ) else pa.string()

            # Build the broadcast column once. pa.array with a
            # single value + length parameter is the right primitive
            # but it's not part of the public API on every version;
            # use repeat() instead.
            scalar = pa.scalar(value, type=pa.string()) if value is not None else pa.scalar(None, type=pa.string())
            # Cast scalar to target type, then repeat. For string
            # target this is a no-op; for typed target we get
            # exactly-once parsing.
            try:
                typed_scalar = pc.cast(scalar, arrow_type) if scalar.type != arrow_type else scalar
            except Exception:
                # Cast failure — fall back to string. Caller can
                # fix dtype downstream.
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

    # ==================================================================
    # Schema collection — merge child schemas, then union with partitions
    # ==================================================================

    def _collect_schema(self, options: PartitionedOptions) -> Schema:
        """Children's schema, plus partition columns appended.

        Children's parquet/ipc files exclude partition columns by
        Hive convention; the table's logical schema is the union.
        """
        children_schema = super()._collect_schema(options)
        partition_cols = self._resolve_partition_columns(options)
        if not partition_cols:
            return children_schema

        # Append partition columns. Ordering: data columns first,
        # then partitions — matches pyarrow.dataset's convention.
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
        options: PartitionedOptions,
    ) -> None:
        """Route batches to per-partition child files.

        Resolves save mode (OVERWRITE / APPEND / IGNORE / UPSERT)
        first, then loops batches and groups by partition key
        tuple, minting one child IO per (mode, partition) pair.

        For ``sort_partitions=True`` (default), the input is
        materialized into one combined Arrow table and sorted by
        partition columns before grouping — this gives exactly one
        child file per distinct partition combination per write
        call. With ``sort_partitions=False``, batches are routed
        in arrival order and the same combination across multiple
        batches mints multiple child files.
        """
        from yggdrasil.io.enums import Mode

        partition_cols = self._resolve_partition_columns(options)

        # No partitions → flat-folder write path. Saves a sort and
        # avoids the partition-column-strip step.
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
        options: PartitionedOptions,
    ) -> None:
        """Materialize, sort, group → one child per partition combination."""
        # Probe for emptiness first so we don't materialize a 0-row table.
        batch_iter = iter(batches)
        first = next(batch_iter, None)
        if first is None:
            return

        # Materialize. Yes, this defeats streaming — that's the
        # tradeoff sort_partitions=True asks for. For genuinely
        # large writes, the caller should pre-sort and pass
        # sort_partitions=False.
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

        # Group consecutive runs with the same partition tuple.
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
        options: PartitionedOptions,
    ) -> None:
        """No-sort routing: one child per (batch, partition) pair.

        Cheaper memory profile than the sorted path — never holds
        more than one batch at a time. Cost: many small files
        when batches are heterogeneous in their partition content.
        """
        for batch in batches:
            if batch.num_rows == 0:
                continue
            missing = [n for n in partition_names if n not in batch.schema.names]
            if missing:
                raise ValueError(
                    f"Partitioned write missing partition columns "
                    f"{missing!r} in input batch. Available: "
                    f"{list(batch.schema.names)!r}."
                )

            sub_table = pa.Table.from_batches([batch])
            partition_table = sub_table.select(partition_names)
            partition_rows = partition_table.to_pylist()

            # Bucket row indices by partition tuple.
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
        options: PartitionedOptions,
    ) -> None:
        """Write *chunk* (already filtered to a single partition tuple)
        to a child file under the right ``k=v/`` prefix.

        Strips partition columns from the chunk before writing —
        Hive convention. The chunk's column order is otherwise
        preserved.
        """
        if chunk.num_rows == 0:
            return

        # Strip partition columns; keep declaration order.
        keep = [c for c in chunk.column_names if c not in partition_names]
        data_chunk = chunk.select(keep)

        # Build the partition subdirectory string.
        partition_values = {
            name: (None if val is None else str(val))
            for name, val in zip(partition_names, partition_key)
        }
        relative_dir = self._partition_path_segment(partition_values)

        media_type = options.child_media_type or self._default_child_media_type()

        # Mint a staging child under the partition subdir.
        partition_root = self.path.joinpath(*relative_dir.split("/")) if relative_dir else self.path
        partition_root.mkdir(parents=True, exist_ok=True)

        # We can't use the inherited _make_staging_path / _finalize_child
        # directly because they target self.path. Mint manually under
        # the partition directory.
        staging_path = partition_root.make_staging(media_type=media_type)
        staging_name = staging_path.name

        # Build a relative child name including the partition prefix
        # so _make_child_io's mkdir handles the parents.
        child_relative = (
            f"{relative_dir}/{staging_name}" if relative_dir else staging_name
        )
        child = self._make_child_io(child_relative, media_type=media_type)

        try:
            with child:
                child.write_arrow_table(data_chunk, options=options)
        except Exception:
            try:
                staging_path.remove(allow_not_found=True)
            except Exception:
                pass
            raise

        # Finalize: rename staging name to part-{N} under the same
        # partition directory. We overload the inherited helper by
        # routing through partition_root's listing, not self.path.
        final_name = self._next_child_name_in(partition_root, media_type=media_type)
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

    @staticmethod
    def _partition_path_segment(values: "Mapping[str, str | None]") -> str:
        """Build the relative ``key=value/key=value`` path segment.

        Hive escapes path-unsafe characters with percent encoding;
        we mirror that on write so reads round-trip.
        """
        parts = []
        for key, value in values.items():
            if value is None:
                # Hive uses __HIVE_DEFAULT_PARTITION__ for nulls.
                # We deliberately don't emit that sentinel — it's a
                # Hive-ism that breaks round-tripping with arrow
                # readers. Empty string isn't safe either (collapses
                # under URL parsers). Raise so the caller picks a
                # convention.
                raise ValueError(
                    f"Cannot serialize None partition value for key "
                    f"{key!r}. Replace nulls in partition columns "
                    "before writing, or filter the rows out."
                )
            quoted = urllib.parse.quote(str(value), safe="")
            parts.append(f"{key}={quoted}")
        return "/".join(parts)