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
import uuid
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
from yggdrasil.io.enums import MimeType, MimeTypes, Mode
from yggdrasil.io.fs import Path
from yggdrasil.io.buffer.base import TabularIO
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

    def __new__(cls, data: Any = None, *args: Any, **kwargs: Any):
        """Construct a folder IO, auto-upgrading to :class:`YGGFolderIO`
        when the target already carries a ``.ygg/`` sidecar.

        Calls placed against :class:`FolderIO` directly opt in to the
        upgrade; callers that explicitly construct :class:`YGGFolderIO`
        or another subclass keep their requested type. This keeps
        ``FolderIO(path="/tmp/store/")`` ergonomic for plain folders
        and one-call-correct for folders that already have sidecar
        state.
        """
        if cls is FolderIO:
            raw = kwargs.get("path", data)
            if raw is not None and _has_ygg_sidecar(raw):
                # Local import to avoid a circular import on module
                # load (ygg_folder_io imports back from this module).
                from .ygg_folder_io import YGGFolderIO
                return NestedIO.__new__(YGGFolderIO, data, *args, **kwargs)
        return NestedIO.__new__(cls, data, *args, **kwargs)

    def _default_child_media_type(self) -> Any:
        """Parquet is the canonical folder-of-tables payload.

        Subclasses with a fixed format (Delta = parquet, IcebergV1
        = avro) override; callers that want IPC/CSV/... pass
        ``options.child_media_type`` explicitly.
        """
        return MimeTypes.PARQUET

    #: Sidecar filename used by :meth:`_persist_schema` /
    #: :meth:`_load_persisted_schema`. Hidden (``.``-prefix) so the
    #: default :meth:`_is_ignored_path` skips it during enumeration —
    #: the schema is metadata, not data.
    SCHEMA_FILE_NAME: ClassVar[str] = ".schema"


    def __init__(
        self,
        data: Any = None,
        *,
        path: Any = None,
        partition_columns: "Sequence[Any] | None" = None,
        schema: "Schema | pa.Schema | None" = None,
        recursive: bool = True,
        partition_values: "Mapping[str, str | None] | None" = None,
        mirror_leaves: bool = False,
        mirror_ttl: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Construct a folder IO.

        :param partition_columns: declared at construction. Order
            matters — it determines the on-disk directory layering.
            Each entry is a :class:`Field` (or anything coercible).
            Default ``None`` defers to ``schema``-derived partitions,
            then to the option / inference fallbacks described in
            :class:`FolderOptions`.
        :param schema: declared full :class:`Schema` for the folder
            (a :class:`pyarrow.Schema` is auto-coerced). When set,
            partition columns auto-derive from
            :attr:`Schema.partition_fields` — fields tagged with
            ``partition_by=True`` — unless ``partition_columns`` is
            given explicitly. The schema is persisted to
            ``<root>/<SCHEMA_FILE_NAME>`` on first write so future
            reads pick it up without inferring from the leaves.
        :param recursive: walk sub-folders during iteration. Default
            True; pass False to clip enumeration at one level.
        :param partition_values: partition values accumulated from
            an enclosing tree. Set by :meth:`iter_children` when
            handing back a sub-:class:`FolderIO`; users normally
            don't pass this themselves.
        :param mirror_leaves: when ``True`` and a leaf path is
            non-local, wrap it in a :class:`MirrorPath` so reads go
            through the local mirror under
            ``~/.yggdrasil/mirror``. Default ``False`` — opt in
            because not every folder layout has immutable leaves
            (only Delta-style append-only ones do safely). Hot
            metadata (the ``.schema`` sidecar, the Delta
            ``_delta_log/``, the folder listing itself) is
            intentionally NOT mirrored — those change on every
            write and need authoritative remote reads.
        :param mirror_ttl: freshness window passed to
            :meth:`MirrorPath.local_mirror`. Default 60s; a longer
            value is fine for truly immutable leaves (Delta
            AddFiles), shorter for plain folders where leaves
            could be replaced in place.
        """
        super().__init__(data, path=path, **kwargs)
        self._mirror_leaves = bool(mirror_leaves)
        self._mirror_ttl = float(mirror_ttl)
        self._declared_schema: "Schema | None" = (
            schema if isinstance(schema, Schema)
            else Schema.from_arrow(schema) if schema is not None
            else None
        )
        # Explicit ``partition_columns`` win; otherwise fall back to
        # ``schema.partition_fields`` so a single tagged Schema can
        # carry the whole layout. The deferred (None) state preserves
        # the legacy precedence — read-time options or layout
        # inference take over later.
        if partition_columns is not None:
            self._partition_columns = tuple(
                _coerce_partition_column(c) for c in partition_columns
            )
        elif self._declared_schema is not None:
            schema_parts = tuple(self._declared_schema.partition_fields)
            self._partition_columns = schema_parts or None
        else:
            self._partition_columns = None
        self._recursive = recursive
        self.partition_values: "dict[str, str | None] | None" = (
            dict(partition_values) if partition_values else None
        )

    # ==================================================================
    # Schema sidecar — persist on first write, load on demand
    # ==================================================================

    def declared_schema(self) -> "Schema | None":
        """Schema declared at construction or loaded from the sidecar.

        Resolution order, with one-time caching:

        1. Schema passed to the constructor.
        2. ``<root>/<SCHEMA_FILE_NAME>`` if it exists.
        3. ``None`` (caller falls back to inference / collect).

        Cached on first hit so repeat calls are free; the cache is
        only populated, never invalidated — folders are immutable
        with respect to their declared schema once a write has
        landed.
        """
        if self._declared_schema is not None:
            return self._declared_schema
        loaded = self._load_persisted_schema()
        if loaded is not None:
            self._declared_schema = loaded
        return self._declared_schema

    def _load_persisted_schema(self) -> "Schema | None":
        """Read the sidecar schema, returning ``None`` on miss."""
        try:
            sidecar = self.path / self.SCHEMA_FILE_NAME
        except Exception:
            return None
        if not sidecar.exists():
            return None
        try:
            data = sidecar.read_bytes()
            arrow_schema = pa.ipc.read_schema(pa.BufferReader(data))
            return Schema.from_arrow(arrow_schema)
        except Exception:
            # A corrupt sidecar shouldn't tank reads — drop back to
            # inference. The next write will re-persist.
            return None

    def _persist_schema_if_declared(self) -> None:
        """Drop the sidecar on first write, idempotent thereafter.

        Safe to call from concurrent writers — uses a stage+rename
        through :meth:`Path.make_staging` so the final ``.schema``
        file appears atomically. Re-runs are idempotent: the
        existence check up front skips the write entirely when the
        sidecar is already in place.
        """
        schema = self._declared_schema
        if schema is None:
            return
        sidecar = self.path / self.SCHEMA_FILE_NAME
        if sidecar.exists():
            return
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            arrow_schema = (
                schema.to_arrow_schema() if isinstance(schema, Schema)
                else schema
            )
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, arrow_schema):
                # Empty stream — schema-only payload.
                pass
            sidecar_bytes = sink.getvalue().to_pybytes()
            try:
                staging = self.path.make_staging()
                staging.write_bytes(sidecar_bytes)
                staging.rename(sidecar)
            except Exception:
                # Fall back to a direct write if the staging dance
                # isn't supported by this filesystem.
                sidecar.write_bytes(sidecar_bytes)
        except Exception:
            # Sidecar persistence is best-effort — never fail a
            # write because we couldn't drop the metadata.
            pass

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

        ``options.children_predicate`` is honoured via the shared
        :func:`yggdrasil.io.buffer.base.matches_children_predicate`
        helper. Combined with backend-specific ignores (the default
        :meth:`_is_ignored_path` hides dot-prefixed entries; Delta
        adds ``_delta_log/``), this is the canonical filter point —
        no glob-pattern fallbacks. To exclude files by extension
        write the predicate explicitly:
        ``children_predicate=~col("name").like("%.tmp")``.

        Each child's ``parent`` attribute is stamped to ``self``.
        Children are returned closed (un-acquired); caller opens
        them inside a ``with`` block.

        Missing folder is treated as empty: no children yielded,
        no error raised.
        """
        from yggdrasil.io.buffer.base import matches_children_predicate

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

            if not matches_children_predicate(
                options, entry.name,
                path=str(entry), is_dir=is_dir,
            ):
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

        Carries forward this folder's partition declaration, the
        declared schema, and the accumulated partition values,
        augmenting them with any ``key=value`` segment parsed from
        the sub-folder's name. The accumulated map also becomes
        :attr:`TabularIO.static_values` on the sub-folder so a
        direct :meth:`read_arrow_table` against the sub-folder
        injects the partition columns for free.
        """
        kv = _parse_kv_segment(sub_path.name)
        sub_pv = dict(self.partition_values) if self.partition_values else {}
        if kv is not None:
            sub_pv[kv[0]] = kv[1]

        sub = type(self)(
            path=sub_path,
            partition_columns=self._partition_columns,
            schema=self._declared_schema,
            recursive=self._recursive,
            partition_values=sub_pv if sub_pv else None,
            auto_open=False,
        )
        # Mirror the partition-values dict into ``static_values`` so
        # the inherited :meth:`TabularIO.read_arrow_batches` /
        # :meth:`TabularIO.read_arrow_table` injection covers
        # direct reads against the sub-folder. Leaves carry the
        # same accumulation independently — the additive injection
        # contract makes the duplicate harmless.
        if sub_pv:
            sub.static_values = self._typed_partition_static(sub_pv)
        return sub

    def _open_file_child(
        self,
        file_path: Path,
    ) -> "TabularIO | BytesIO | None":
        """Build an IO for a file leaf.

        Tries :meth:`TabularIO.from_path` first; falls back to
        :class:`BytesIO` for files with no registered tabular
        format. Returns ``None`` if neither can bind (rare —
        usually means the entry vanished mid-listing).

        When the enclosing folder carries
        :attr:`partition_values`, the leaf's
        :attr:`TabularIO.static_values` is stamped with the
        Arrow-typed equivalent so its read path automatically
        injects the partition columns — no per-batch wrapping
        needed at the folder level.

        When ``mirror_leaves=True`` was passed at construction and
        ``file_path`` is non-local, the path is wrapped in a
        :class:`MirrorPath` so repeat reads serve from the local
        mirror. Local paths bypass the wrap (no benefit, just
        overhead).
        """
        if (
            self._mirror_leaves
            and isinstance(file_path, Path)
            and not file_path.is_local
        ):
            from yggdrasil.io.fs.mirror_path import MirrorPath
            file_path = MirrorPath(file_path, ttl=self._mirror_ttl)

        try:
            io = TabularIO.from_path(file_path)
        except Exception:
            io = None

        # A registered tabular leaf is a :class:`BytesIO` subclass
        # with a format-specific class (so ``type(io) is BytesIO``
        # would be false). Stamp the typed partition values onto
        # ``static_values`` so the inherited
        # :meth:`TabularIO.read_arrow_batches` injection picks
        # them up — no folder-side per-batch wrapping needed.
        if isinstance(io, BytesIO) and type(io) is not BytesIO:
            if self.partition_values:
                io.static_values = self._typed_partition_static(
                    self.partition_values,
                )
            return io

        # No tabular registration — fall back to a raw BytesIO so
        # callers can still reach the bytes via iter_children.
        try:
            return BytesIO(path=file_path, auto_open=False)
        except Exception:
            return None

    def _typed_partition_static(
        self,
        values: "Mapping[str, str | None]",
    ) -> "dict[str, Any]":
        """Build an Arrow-typed ``static_values`` map for ``values``.

        Each value is wrapped in a :class:`pyarrow.Scalar` typed
        to its declared partition column dtype so the
        :func:`yggdrasil.io.buffer.base.inject_static_values_into_batch`
        helper emits a column with the right schema-side type.
        Falls back to a string scalar when the column isn't
        declared or the dtype can't translate to Arrow.
        """
        cols = self._resolve_partition_columns()
        by_name = {c.name: c for c in cols} if cols else {}
        out: dict[str, Any] = {}
        for key, value in values.items():
            field = by_name.get(key)
            arrow_type: "pa.DataType | None" = None
            if field is not None:
                for attr in ("to_arrow_dtype", "to_arrow"):
                    fn = getattr(field.dtype, attr, None)
                    if callable(fn):
                        try:
                            arrow_type = fn()
                            break
                        except Exception:
                            continue
            if arrow_type is None:
                arrow_type = pa.string()
            try:
                if value is None:
                    out[key] = pa.scalar(None, type=arrow_type)
                else:
                    base = pa.scalar(value, type=pa.string())
                    out[key] = (
                        pc.cast(base, arrow_type)
                        if base.type != arrow_type
                        else base
                    )
            except Exception:
                # Type coercion failed (e.g. ``"foo"`` → int) —
                # keep the raw string scalar so the column at
                # least carries the value verbatim.
                out[key] = pa.scalar(
                    None if value is None else str(value),
                    type=pa.string(),
                )
        return out

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
    ) -> BytesIO:
        """Build a fresh write target under :attr:`path`.

        ``name`` may include forward-slash separators for nested
        layouts (Hive partitions, sub-folder writes); the parent
        directories are auto-created. Backslashes are rejected
        (URL semantics) and ``..`` segments are rejected (path
        traversal).

        Returns a closed (un-acquired) registered tabular leaf —
        a :class:`BytesIO` subclass for the resolved mime type
        (ParquetIO, CsvIO, …) — with ``parent`` set to ``self``.
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

        if not isinstance(io, BytesIO) or type(io) is BytesIO:
            raise TypeError(
                f"FolderIO child factory expected a registered tabular "
                f"leaf for name={name!r}, media_type={media_type!r}; got "
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
        3. The declared :class:`Schema`'s ``partition_fields`` (the
           ``partition_by``-tagged fields), whether passed at
           construction or loaded from the ``.schema`` sidecar.
        4. Inferred from the first leaf's directory layout (string
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

        schema = self.declared_schema()
        if schema is not None:
            schema_parts = tuple(schema.partition_fields)
            if schema_parts:
                # Cache the resolution so subsequent calls don't
                # re-walk the schema on every read/write.
                self._partition_columns = schema_parts
                return schema_parts

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

    # ``_read_arrow_batches`` is intentionally NOT overridden:
    # the inherited :meth:`NestedIO._read_arrow_batches` already
    # iterates children and dispatches to each. Leaves stamped
    # with :attr:`TabularIO.static_values` (see
    # :meth:`_open_file_child`) auto-inject their partition
    # columns when their public read runs, so no per-batch
    # wrapping at this level is needed.

    def _partition_values_for(
        self,
        child: Any,
        partition_cols: "Sequence[Field]",
        *,
        strict: bool,
    ) -> "Mapping[str, str | None] | None":
        """Resolve a leaf's partition values from its directory path.

        Used during partition inference and by subclasses (Delta)
        that handle their own per-leaf metadata. The folder's
        normal read path doesn't call this — it relies on
        :attr:`TabularIO.static_values` stamped at child-open
        time. Kept here for backward-compatible introspection on
        leaves that weren't opened through this folder.
        """
        if not partition_cols:
            return None

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
        # Persist the declared schema sidecar on every write entry —
        # idempotent: the existence check skips the IO when it's
        # already there. Keeps a freshly-empty folder's first write
        # from racing with a concurrent reader that wants the
        # schema before any data has landed.
        self._persist_schema_if_declared()

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

    # ==================================================================
    # Upsert — partition-aware with auto-pruned incoming
    # ==================================================================

    def _arrow_upsert_via_rewrite(
        self,
        batches: Any,
        options: FolderOptions,
    ) -> None:
        """UPSERT with auto-pruned partition rewrites.

        For a partitioned folder, the canonical upsert is "merge
        only the partitions the incoming side actually touches" —
        anything else would mean reading the entire tree just to
        merge a handful of distinct partition tuples. We derive the
        prune set from the incoming table's distinct partition
        values (or honor ``options.prune_values`` when the caller
        already pinned them), read each touched partition once,
        merge with the matching incoming slice, and overwrite
        in-place by clearing that partition's leaves before the
        rewrite.

        Non-partitioned folders fall through to the base
        :meth:`NestedIO._arrow_upsert_via_rewrite` (read-all,
        merge, overwrite) — no partition tree to prune by.
        """
        from yggdrasil.arrow.cast import any_to_arrow_table

        partition_cols = self._resolve_partition_columns(options)
        if not partition_cols:
            super()._arrow_upsert_via_rewrite(batches, options)
            return

        match_by = options.match_by_names
        if not match_by:
            raise ValueError(
                f"{type(self).__name__} UPSERT requires "
                "options.match_by_names to be a non-empty sequence "
                "of column names. For 'replace everything,' use "
                "Mode.OVERWRITE instead."
            )

        incoming_table = any_to_arrow_table(batches, options)
        if incoming_table.num_rows == 0:
            return

        partition_names = [c.name for c in partition_cols]
        missing = [
            n for n in partition_names if n not in incoming_table.column_names
        ]
        if missing:
            raise ValueError(
                f"Partitioned UPSERT missing partition columns "
                f"{missing!r} in incoming. Available: "
                f"{list(incoming_table.column_names)!r}."
            )

        explicit_pruned = self._explicit_prune_tuples(
            options, partition_names,
        )
        sort_keys = [(n, "ascending") for n in partition_names]
        sorted_incoming = incoming_table.sort_by(sort_keys)
        partition_view = sorted_incoming.select(partition_names).to_pylist()
        n = sorted_incoming.num_rows

        run_start = 0
        run_key = tuple(partition_view[0].get(c) for c in partition_names)
        for i in range(1, n + 1):
            next_key = (
                None if i == n
                else tuple(partition_view[i].get(c) for c in partition_names)
            )
            if i == n or next_key != run_key:
                if explicit_pruned is None or run_key in explicit_pruned:
                    self._upsert_partition(
                        partition_names, run_key,
                        sorted_incoming.slice(run_start, i - run_start),
                        options,
                    )
                run_start = i
                if next_key is not None:
                    run_key = next_key

    def _explicit_prune_tuples(
        self,
        options: FolderOptions,
        partition_names: "Sequence[str]",
    ) -> "set[tuple] | None":
        """Resolve ``options.prune_values`` into a set of partition tuples.

        Returns ``None`` when no explicit prune set is configured —
        callers fall back to "every distinct partition tuple in the
        incoming side." Honors only the partition columns we know
        about; extra keys in ``prune_values`` are ignored.
        """
        prune = getattr(options, "prune_values", None)
        if not prune:
            return None
        try:
            value_lists = [
                tuple(prune[name]) if name in prune else None
                for name in partition_names
            ]
        except Exception:
            return None
        if any(v is None for v in value_lists):
            return None
        out: set[tuple] = set()
        for combo in zip(*value_lists):
            out.add(tuple(combo))
        return out

    def _upsert_partition(
        self,
        partition_names: "Sequence[str]",
        partition_key: tuple,
        incoming_slice: pa.Table,
        options: FolderOptions,
    ) -> None:
        """Merge one partition: read existing → merge → overwrite that dir.

        Limits the read/write blast radius to a single ``key=value``
        directory per call — the whole point of the auto-prune
        pass. Existing leaves in the partition are removed before
        the merged rows are re-written, so the partition's content
        is exactly the union of "existing rows whose key is not in
        ``incoming``" + "every incoming row" — Hive-MERGE
        semantics.
        """
        partition_values = {
            name: (None if val is None else str(val))
            for name, val in zip(partition_names, partition_key)
        }
        relative_dir = _partition_path_segment(partition_values)
        partition_root = (
            self.path.joinpath(*relative_dir.split("/"))
            if relative_dir else self.path
        )

        existing = self._read_partition_table(
            partition_root, partition_names, partition_key, options,
        )
        merged = self.merge_upsert_tables(
            existing, incoming_slice,
            match_by=options.match_by_names,
            update_column_names=options.update_column_names,
        )

        if partition_root.exists():
            self._clear_partition_leaves(partition_root)

        self._write_partition_chunk(
            merged, partition_names, partition_key,
            options.copy(mode=Mode.APPEND),
        )

    def _read_partition_table(
        self,
        partition_root: Path,
        partition_names: "Sequence[str]",
        partition_key: tuple,
        options: FolderOptions,
    ) -> pa.Table:
        """Read all leaves under ``partition_root`` into one table.

        Skips the FolderIO discovery walk — we already know the
        partition values, so we just read direct leaf files. Empty
        / missing partition returns an empty table with the merged
        schema (so :meth:`merge_upsert_tables` has typed columns
        to work against). Partition columns are injected back in so
        the merge sees the same shape as the incoming side.
        """
        leaf_tables: list[pa.Table] = []
        if partition_root.exists():
            for entry in partition_root.iterdir():
                if entry.is_dir():
                    continue
                if self._is_ignored_path(entry):
                    continue
                try:
                    leaf_io = TabularIO.from_path(entry)
                except Exception:
                    continue
                with leaf_io:
                    try:
                        leaf_tables.append(leaf_io.read_arrow_table())
                    except Exception:
                        continue

        if not leaf_tables:
            schema = self.declared_schema()
            if schema is not None:
                return schema.to_arrow_schema().empty_table()
            return pa.table({n: pa.array([], type=pa.string())
                             for n in partition_names})

        existing = pa.concat_tables(leaf_tables, promote_options="default")
        # Inject partition columns so the merge keys / values line
        # up across the existing and incoming sides.
        for name, val in zip(partition_names, partition_key):
            if name in existing.column_names:
                continue
            existing = existing.append_column(
                name,
                pa.array([val] * existing.num_rows),
            )
        return existing

    def _clear_partition_leaves(self, partition_root: Path) -> None:
        """Delete leaf files under ``partition_root`` (skip sub-dirs).

        Called from :meth:`_upsert_partition` between read and
        write. Sub-directories aren't touched — Hive partitioning
        keys siblings inside the partition, not under it, so any
        nested directory is foreign and should stay.
        """
        for entry in partition_root.iterdir():
            if entry.is_dir():
                continue
            if self._is_ignored_path(entry):
                continue
            try:
                entry.remove(allow_not_found=True)
            except Exception:
                pass

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
        """Mint a unique leaf name under ``parent``.

        Uses a UUID-suffixed ``part-<hex>.<ext>`` form so concurrent
        writers can't collide on the same final name — the legacy
        sequential ``part-NNNNN`` scheme silently lost data when two
        workers raced through ``_finalize_child`` against the same
        partition (both saw the same ``max_idx``, both renamed to
        ``part-(max_idx+1)``, the second clobbering the first).

        Parent existence isn't checked here — callers create the
        directory tree before they ask for a name. ``parent`` is
        retained on the signature for parity with
        :meth:`NestedIO._next_child_name` and so subclasses keying
        off the parent path (Delta, Iceberg) can still override.
        """
        ext = self._extension_for(media_type)
        suffix = f".{ext}" if ext else ""
        return f"part-{uuid.uuid4().hex}{suffix}"


# ---------------------------------------------------------------------------
# Module-private helpers — partition path / leaf walk / column injection
# ---------------------------------------------------------------------------



def _has_ygg_sidecar(path_like: Any) -> bool:
    """One-round-trip probe: does ``path_like`` carry a ``.ygg/`` folder?

    Used by :meth:`FolderIO.__new__` to auto-upgrade plain
    :class:`FolderIO` constructions to :class:`YGGFolderIO` when the
    target already has sidecar state. Failures (path doesn't parse,
    backend transient, permission denied) collapse to ``False`` so
    we never accidentally fail a plain construction.
    """
    try:
        if isinstance(path_like, Path):
            probe = path_like
        else:
            probe = Path.from_(path_like, default=None)
        if probe is None:
            return False
        return (probe / ".ygg").exists()
    except Exception:
        return False


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
