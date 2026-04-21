"""Filesystem-backed I/O for single files and directories.

Architecture
------------
:class:`PathIO` reads from a :class:`pathlib.Path` — a single file or a
directory tree of files. Unlike the BytesIO-backed MediaIO subclasses
(Parquet, CSV, IPC, …), PathIO doesn't own a buffer; it delegates to
those classes for individual file parsing or to
:mod:`pyarrow.dataset` for multi-file scans.

Two execution paths, chosen by the file's media type:

* **Dataset-capable formats** (Parquet, IPC, ORC, CSV, TSV, JSON,
  NDJSON) — scan via :class:`pyarrow.dataset.Dataset`. Filters and
  column projection push down to the scanner, which is orders of
  magnitude faster than post-read filtering for large files.
* **Fallback** (everything else: XLSX, XML, ZIP, or any format
  registered outside :mod:`pyarrow.dataset`) — iterate files, delegate
  each to its inner MediaIO via a path-backed :class:`BytesIO` (no
  full payload materialization), apply filter + projection in Python.

Cast is applied symmetrically at the batch iterator level
(:meth:`options.cast.cast_iterator`), so ``read_arrow_table`` and
``read_arrow_batches`` produce identically-cast output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import InitVar, dataclass
from pathlib import Path as _PathlibPath
from typing import TYPE_CHECKING, Any, Optional

from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.io.enums import MediaType, MimeType, MimeTypes
from yggdrasil.io.fs import Path
from .bytes_io import BytesIO
from .media_io import MediaIO
from .media_options import MediaOptions

if TYPE_CHECKING:
    import pyarrow.dataset as ds

__all__ = ["PathOptions", "PathIO"]


_DEFAULT_IGNORE_PREFIXES = (".", "_")

# Formats that pyarrow.dataset can scan natively. For any other mime
# type (XLSX, XML, ZIP, …) we fall back to per-file MediaIO delegation.
_DATASET_CAPABLE_MIMES: frozenset[MimeType] = frozenset(
    {
        MimeTypes.PARQUET,
        MimeTypes.ARROW_IPC,
        MimeTypes.ORC,
        MimeTypes.CSV,
        MimeTypes.TSV,
        MimeTypes.JSON,
        MimeTypes.NDJSON,
    }
)

# Everything PathIO is willing to consider, dataset or not. Used by
# `iter_files(supported_only=True)`.
_SUPPORTED_MIME_TYPES: tuple[MimeType, ...] = tuple(_DATASET_CAPABLE_MIMES)


# =====================================================================
# Options
# =====================================================================


@dataclass
class PathOptions(MediaOptions):
    """Options for path-backed reads.

    Parameters
    ----------
    filter:
        Filter spec applied to rows.

        * :class:`pyarrow.dataset.Expression` — used verbatim, pushed
          down when the dataset path is taken.
        * ``dict`` — ``{"col": value, ...}``; value can be a scalar,
          ``None`` (IS NULL), or a sequence (IN).
        * sequence of tuples — ``[(col, value), (col, op, value), ...]``
          where ``op`` is one of ``=``, ``!=``, ``>``, ``>=``, ``<``,
          ``<=``, ``in``, ``not in``, ``is``, ``is not``.
        * a single tuple of the same shape — treated as a length-1 list.

        All entries are AND-ed. ``None`` means no filter.
    recursive, include_hidden, supported_only, ignore_prefixes:
        Directory-walk controls.
    format:
        Dataset format hint: ``"parquet"``, ``"ipc"``, ``"csv"``,
        ``"tsv"``, ``"json"``, ``"ndjson"``, ``"orc"``, or a
        :class:`pyarrow.dataset.FileFormat` instance. ``None`` → infer
        from the first file's extension.
    partitioning, partition_base_dir:
        Hive or directory-name partitioning. ``partition_base_dir``
        defaults to the directory being scanned.
    exclude_invalid_files:
        Forwarded to :func:`pyarrow.dataset.dataset`.
    batch_readahead, fragment_readahead:
        Scanner performance knobs (dataset path only).
    """

    filter: Any = None
    recursive: bool = True
    include_hidden: bool = False
    supported_only: bool = True
    format: Any = None
    partitioning: str | Sequence[str] | None = "hive"
    partition_base_dir: str | Path | _PathlibPath | None = None
    exclude_invalid_files: bool | None = None
    ignore_prefixes: Sequence[str] | None = _DEFAULT_IGNORE_PREFIXES
    batch_readahead: int = 16
    fragment_readahead: int = 4

    def __post_init__(self) -> None:
        super().__post_init__()

        if not isinstance(self.recursive, bool):
            raise TypeError(
                f"recursive must be bool, got {type(self.recursive).__name__}"
            )
        if not isinstance(self.include_hidden, bool):
            raise TypeError(
                f"include_hidden must be bool, got {type(self.include_hidden).__name__}"
            )
        if not isinstance(self.supported_only, bool):
            raise TypeError(
                f"supported_only must be bool, got {type(self.supported_only).__name__}"
            )

        if self.partition_base_dir is not None:
            # ``partition_base_dir`` is compared to file paths via
            # ``_path_parts`` which works on anything with ``.parts`` or a
            # string rendering. Keep the value as-is when it's already a
            # yggdrasil :class:`Path`; otherwise coerce through the safe
            # entry point so str / pathlib.Path inputs land on the right
            # backend.
            if not isinstance(self.partition_base_dir, Path):
                self.partition_base_dir = Path.from_any(self.partition_base_dir)

        if self.ignore_prefixes is not None:
            if isinstance(self.ignore_prefixes, (str, bytes)):
                raise TypeError(
                    "ignore_prefixes must be a sequence of strings, not a single string"
                )
            prefixes = list(self.ignore_prefixes)
            if not all(isinstance(item, str) for item in prefixes):
                raise TypeError("ignore_prefixes must contain only str values")
            self.ignore_prefixes = tuple(prefixes)

        self.batch_readahead = self._normalize_non_negative_int(
            "batch_readahead", self.batch_readahead
        )
        self.fragment_readahead = self._normalize_non_negative_int(
            "fragment_readahead", self.fragment_readahead
        )

    @staticmethod
    def _normalize_non_negative_int(name: str, value: int | None) -> int:
        if value is None:
            return 0
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be int|None, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
        return value

    @classmethod
    def resolve(
        cls, *, options: "PathOptions | None" = None, **overrides: Any
    ) -> "PathOptions":
        return cls.check_parameters(options=options, **overrides)


# =====================================================================
# PathIO
# =====================================================================


@dataclass(slots=True)
class PathIO(MediaIO[PathOptions], ABC):
    """PathIO backs reads/writes with a :class:`Path`, stored on the holder.

    The filesystem path lives on :attr:`holder.path` (``self.holder._path``),
    the same slot :class:`BytesIO` uses for a spilled backing file. That
    keeps a single source of truth — anyone inspecting ``io.holder.path``
    or ``io.path`` sees the same object — and lets the base :class:`MediaIO`
    machinery manage buffer lifecycle alongside the path.

    :attr:`path` is exposed as a property for backward compatibility.
    Setting it rewrites the holder's backing path.
    """

    # ``path`` is declared as an :class:`InitVar` so the dataclass-generated
    # ``__init__`` (on this class and on subclasses that use
    # ``@dataclass``) accepts ``path=`` as a kwarg. The value is routed
    # to :meth:`__post_init__` and stashed on ``self.holder._path`` —
    # that's the single source of truth.
    #
    # An :class:`InitVar` default is read from the class attribute at
    # decoration time; defining a ``@property path`` in this class body
    # would overwrite that default with the property descriptor. To
    # avoid the collision the ``path`` property is attached **after**
    # the class is built, below.
    path: InitVar[Any] = None

    def __post_init__(self, path: Any = None) -> None:
        # Prefer an explicit ``path=`` kwarg; fall back to whatever the
        # holder is already carrying (some callers build a BytesIO from
        # the path first and hand it in as the holder).
        resolved = path if path is not None else self.holder.path
        if resolved is None:
            raise ValueError(
                f"{type(self).__name__} requires a non-None path. "
                "Pass a str, pathlib.Path, yggdrasil.io.fs.Path, or any "
                "object with read_bytes/is_file/is_dir."
            )

        # Safe-parse stringy / pathlib / PathLike inputs via the typed
        # entry point so the correct backend (LocalPath, DBFSPath, …)
        # gets picked. Anything that already looks like a Path — either
        # one of ours or a duck-typed remote (e.g. DatabricksPath) — is
        # left alone so subclasses can keep their own concrete types.
        self.holder._path = self._coerce_path(resolved)

        if self.media_type is None:
            self.media_type = MediaType.parse(
                self.infer_mime_type(),
                default=MediaType(MimeTypes.PARQUET),
            )

    @staticmethod
    def _coerce_path(value: Any) -> Any:
        """Coerce *value* to a path-like object without forcing a backend.

        Routes plain strings, ``pathlib.Path``, ``bytes``, and
        ``os.PathLike`` through :meth:`Path.from_any` so the right
        :class:`yggdrasil.io.fs.Path` subclass is picked. Anything
        already implementing the path protocol (our :class:`Path`, or a
        duck-typed external type such as :class:`DatabricksPath`) passes
        through unchanged.
        """
        if isinstance(value, Path):
            return value
        # Duck-type: if it walks like a path, don't force a re-parse.
        if hasattr(value, "read_bytes") and hasattr(value, "is_file"):
            return value
        return Path.from_any(value)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @classmethod
    def check_options(
        cls,
        options: Optional[PathOptions],
        *args,
        **kwargs,
    ) -> PathOptions:
        del args
        return PathOptions.check_parameters(options=options, **kwargs)

    @classmethod
    @abstractmethod
    def make(
        cls,
        path: str | Path | _PathlibPath,
        media: MediaType | MimeType | str | None = None,
    ) -> "PathIO":
        raise TypeError(
            f"{cls.__name__} is abstract and cannot be instantiated directly"
        )

    @abstractmethod
    def iter_files(
        self,
        recursive: bool = True,
        *,
        include_hidden: bool = False,
        supported_only: bool = True,
        mime_type: MimeType | str | None = None,
    ) -> Iterator["PathIO"]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Path probes
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def is_file(self) -> bool:
        return self.path.is_file()

    @property
    def is_dir(self) -> bool:
        return self.path.is_dir()

    def infer_mime_type(
        self,
        *,
        recursive: bool = True,
        include_hidden: bool = False,
        supported_only: bool = False,
    ) -> MimeType:
        """Return the mime type of this path (or its first file).

        For a file, parses the extension. For a directory, inspects
        the first matching file. Default on no match: ``PARQUET``.
        """
        if self.media_type is not None:
            return self.media_type.mime_type

        if self.path.is_file():
            return MimeType.parse(self.path, default=MimeTypes.PARQUET)

        first_file = next(
            self.iter_files(
                recursive=recursive,
                include_hidden=include_hidden,
                supported_only=supported_only,
            ),
            None,
        )
        if first_file is None:
            return MimeTypes.PARQUET
        return MimeType.parse(first_file.path, default=MimeTypes.PARQUET)

    def _is_dataset_capable(self) -> bool:
        """True if this path's mime type can be scanned via pyarrow.dataset."""
        return self.infer_mime_type() in _DATASET_CAPABLE_MIMES

    @staticmethod
    def _resolve_cast_target_schema(cast: Any) -> "pa.Schema | None":
        """Extract an Arrow schema from a CastOptions target, if set.

        Returns ``None`` when the cast has no target (identity cast) or
        when the target can't be converted to an Arrow schema — in
        which case the dataset reader will infer the schema from the
        first fragment, matching the default pyarrow.dataset behavior.
        """
        target_field = getattr(cast, "target_field", None)
        if target_field is None:
            return None

        # Try known schema-producing conversions in order of specificity.
        # Different yggdrasil Field versions expose different methods —
        # probe defensively rather than locking to one spelling.
        for attr in ("to_arrow_schema", "arrow_schema"):
            converter = getattr(target_field, attr, None)
            if converter is None:
                continue
            try:
                schema = converter() if callable(converter) else converter
            except Exception:
                continue
            if isinstance(schema, pa.Schema):
                return schema

        # Last resort: if target_field itself is already a pa.Schema.
        if isinstance(target_field, pa.Schema):
            return target_field

        return None

    # ------------------------------------------------------------------
    # Public dataset/scanner API
    # ------------------------------------------------------------------

    def read_dataset(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        """Return a :class:`pyarrow.dataset.Dataset` over this path.

        Raises :class:`NotImplementedError` when the path's format is
        not dataset-capable (e.g. XLSX, XML, ZIP) — use
        :meth:`read_arrow_table` for those.
        """
        import pyarrow.dataset as ds

        resolved = self.check_options(options=options, **option_kwargs)
        if not self._is_dataset_capable():
            mime = self.infer_mime_type()
            raise NotImplementedError(
                f"read_dataset not supported for {mime!r}; "
                f"use read_arrow_table/read_arrow_batches instead"
            )

        dataset_format = self._resolve_dataset_format(
            format=resolved.format,
            recursive=resolved.recursive,
            include_hidden=resolved.include_hidden,
            supported_only=resolved.supported_only,
        )
        arrow_schema = self._resolve_cast_target_schema(resolved.cast)

        if self.path.is_dir():
            files = [
                str(file_io.path)
                for file_io in self.iter_files(
                    recursive=resolved.recursive,
                    include_hidden=resolved.include_hidden,
                    supported_only=resolved.supported_only,
                )
            ]
            if not files:
                return ds.dataset(
                    pa.Table.from_batches([], schema=arrow_schema or pa.schema([]))
                )
            return ds.dataset(
                files,
                schema=arrow_schema,
                format=dataset_format,
                partitioning=resolved.partitioning,
                partition_base_dir=str(resolved.partition_base_dir or self.path),
                exclude_invalid_files=resolved.exclude_invalid_files,
                ignore_prefixes=(
                    list(resolved.ignore_prefixes)
                    if resolved.ignore_prefixes is not None
                    else None
                ),
            )

        return ds.dataset(
            str(self.path),
            schema=arrow_schema,
            format=dataset_format,
            partitioning=resolved.partitioning,
            partition_base_dir=str(resolved.partition_base_dir or self.path.parent),
            exclude_invalid_files=resolved.exclude_invalid_files,
            ignore_prefixes=(
                list(resolved.ignore_prefixes)
                if resolved.ignore_prefixes is not None
                else None
            ),
        )

    def to_arrow_dataset(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Dataset":
        return self.read_dataset(options=options, **option_kwargs)

    def scanner(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "ds.Scanner":
        resolved = self.check_options(options=options, **option_kwargs)
        dataset = self.read_dataset(options=resolved)
        return dataset.scanner(
            columns=list(resolved.columns) if resolved.columns is not None else None,
            filter=self._normalize_filter_to_expression(resolved.filter),
            batch_size=resolved.batch_size or 131_072,
            batch_readahead=resolved.batch_readahead or 16,
            fragment_readahead=resolved.fragment_readahead or 4,
            use_threads=resolved.use_threads,
        )

    # ------------------------------------------------------------------
    # Core read protocol — signatures match base contract (positional)
    # ------------------------------------------------------------------

    def _read_arrow_batches(
        self,
        options: PathOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Yield Arrow batches with filter, projection, and cast applied.

        Dispatches to the dataset scanner for dataset-capable formats
        (fast path, pushdown) or to per-file MediaIO delegation
        otherwise (fallback, Python-level filter).

        The cast is applied last, at the batch level, via
        :meth:`options.cast.cast_iterator` — identical to how
        ParquetIO/IPCIO handle cast. This guarantees ``read_arrow_table``
        and ``read_arrow_batches`` produce identically-typed output.
        """
        if self._is_dataset_capable():
            raw_batches = self._iter_batches_via_dataset(options)
        else:
            raw_batches = self._iter_batches_via_fallback(options)

        yield from options.cast.cast_iterator(raw_batches)

    def _write_arrow_batches(
        self,
        batches: Iterator["pa.RecordBatch"],
        options: PathOptions,
    ) -> None:
        """Write support is not implemented yet."""
        del batches, options
        raise NotImplementedError(f"{type(self).__name__} does not support writes yet")

    def _collect_arrow_schema(self, full: bool = False) -> "pa.Schema":
        """Return the Arrow schema of this path.

        For a single file, reads that file's schema (header-only when
        the format allows). For a directory, inspects the first file
        by default, or every file when *full* is ``True`` (returning
        the unified schema).
        """
        options = self.check_options(options=None)
        mime_type = self.media_type.mime_type if self.is_file else None

        collected: list[pa.Schema] = []
        for file_io in self.iter_files(
            recursive=options.recursive,
            include_hidden=options.include_hidden,
            supported_only=options.supported_only,
            mime_type=mime_type,
        ):
            collected.append(file_io._file_schema(full=full))
            if not full:
                break

        if not collected:
            return pa.schema([])
        if len(collected) == 1:
            return collected[0]
        return pa.unify_schemas(collected, promote_options="default")

    def _file_schema(self, *, full: bool) -> "pa.Schema":
        """Read a single file's schema without loading its body."""
        media = MediaType.parse(str(self.path), default=self.media_type)
        # Path-backed BytesIO memory-maps; it doesn't read the file into
        # RAM. Inner media IOs (Parquet, IPC) read only the footer/header
        # to resolve schema, so peak memory stays tiny even for
        # multi-GB files.
        with BytesIO(self.path, media_type=media) as buffer:
            return buffer.media_io(media)._collect_arrow_schema(full=full)

    # ------------------------------------------------------------------
    # Public read helpers
    # ------------------------------------------------------------------

    def count_rows(
        self,
        *,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> int:
        """Return the number of rows after filter + projection."""
        resolved = self.check_options(options=options, **option_kwargs)
        # Dataset path has an O(metadata) count for Parquet. For fallback
        # we must materialize — still cheaper than read_arrow_table since
        # we can skip column payload via projecting to filter columns only.
        if self._is_dataset_capable() and resolved.filter is None:
            return self.read_dataset(options=resolved).count_rows()
        return self.read_arrow_table(options=resolved).num_rows

    def select_columns(
        self,
        *columns: str,
        options: PathOptions | None = None,
        **option_kwargs: Any,
    ) -> "pa.Table":
        resolved = self.check_options(
            options=options, columns=list(columns), **option_kwargs
        )
        return self.read_arrow_table(options=resolved)

    # ------------------------------------------------------------------
    # read_arrow_table — delegated to base class
    # ------------------------------------------------------------------
    #
    # The base MediaIO.read_arrow_table calls _read_arrow_batches and
    # wraps with pa.Table.from_batches. Our _read_arrow_batches already
    # applies filter + projection + cast, so no override is needed.
    # The return type is always pa.Table — the old version's "maybe
    # return an iterator when batch_size is set" was a bug; batched
    # reads are what read_arrow_batches is for.

    # ------------------------------------------------------------------
    # Dataset path — fast: filter + projection pushdown
    # ------------------------------------------------------------------

    def _iter_batches_via_dataset(
        self,
        options: PathOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Stream batches from a pyarrow.dataset scanner."""
        scanner = self.scanner(options=options)
        yield from scanner.to_batches()

    # ------------------------------------------------------------------
    # Fallback path — per-file delegation with post-read filter/project
    # ------------------------------------------------------------------

    def _iter_batches_via_fallback(
        self,
        options: PathOptions,
    ) -> Iterator["pa.RecordBatch"]:
        """Stream batches from per-file MediaIO delegation.

        Used for formats pyarrow.dataset doesn't handle (XLSX, XML,
        ZIP, …). We must emulate the dataset scanner's semantics in
        Python: partition-value injection, filter, column projection.
        """
        partition_base_dir = options.partition_base_dir or (
            self.path if self.path.is_dir() else getattr(self.path, "parent", self.path)
        )

        # Inner-file reads pull ALL filter columns even if the caller
        # projected them away, since we need them for the mask. We
        # re-project to `options.columns` after filtering.
        filter_columns = self._filter_columns(options.filter)
        inner_columns = self._augment_columns_with_filter(
            requested=options.columns, needed=filter_columns
        )

        # Inner MediaIO should NOT apply cast — we do it at the PathIO
        # level in _read_arrow_batches via cast_iterator. Prevents double
        # casting.
        inner_options = options.with_cast(None) if options.cast is not None else options

        mime_type = self.media_type.mime_type if self.is_file else None

        for file_io in self.iter_files(
            recursive=options.recursive,
            include_hidden=options.include_hidden,
            supported_only=options.supported_only,
            mime_type=mime_type,
        ):
            partition_values = self._partition_values(
                file_path=file_io.path,
                partitioning=options.partitioning,
                partition_base_dir=partition_base_dir,
            )
            for batch in file_io._iter_file_batches(
                columns=inner_columns,
                batch_size=inner_options.batch_size,
                use_threads=inner_options.use_threads,
            ):
                batch = self._inject_partition_values(batch, partition_values)
                if batch.num_rows == 0:
                    if options.ignore_empty:
                        continue
                    yield self._final_project(batch, options.columns)
                    continue

                if options.filter is not None:
                    import pyarrow.compute as pc

                    mask = self._build_filter_mask(
                        table_or_batch=batch, filter_spec=options.filter, pc=pc
                    )
                    batch = batch.filter(mask)
                    if batch.num_rows == 0 and options.ignore_empty:
                        continue

                yield self._final_project(batch, options.columns)

    def _iter_file_batches(
        self,
        *,
        columns: list[str] | None,
        batch_size: int | None,
        use_threads: bool,
    ) -> Iterator["pa.RecordBatch"]:
        """Parse a single file via its inner MediaIO as a batch iterator.

        Uses a path-backed :class:`BytesIO` so multi-GB files don't get
        slurped into RAM — the inner MediaIO reads via memory-mapped
        I/O where its format supports it.
        """
        media = MediaType.parse(str(self.path), default=self.media_type)
        with BytesIO(self.path, media_type=media) as buffer:
            inner_io = buffer.media_io(media)
            # Build inner options with only what the inner reader cares
            # about — avoid passing PathOptions-specific fields.
            yield from inner_io.read_arrow_batches(
                columns=columns,
                batch_size=batch_size or 0,
                use_threads=use_threads,
            )

    @staticmethod
    def _augment_columns_with_filter(
        *,
        requested: Sequence[str] | None,
        needed: Sequence[str],
    ) -> list[str] | None:
        """Ensure filter-referenced columns are loaded even when projected out."""
        if requested is None:
            return None  # loading all columns → filter cols are already in
        out = list(requested)
        for col in needed:
            if col not in out:
                out.append(col)
        return out

    @staticmethod
    def _final_project(
        batch: "pa.RecordBatch",
        columns: Sequence[str] | None,
    ) -> "pa.RecordBatch":
        """Re-project to the caller's requested columns (drop filter-only ones)."""
        if columns is None:
            return batch
        existing = [c for c in columns if c in batch.schema.names]
        return batch.select(existing)

    @staticmethod
    def _inject_partition_values(
        batch: "pa.RecordBatch",
        partition_values: dict[str, str],
    ) -> "pa.RecordBatch":
        """Append Hive-style partition key columns to a batch."""
        if not partition_values:
            return batch

        arrays = list(batch.columns)
        names = list(batch.schema.names)
        existing = set(names)
        n = batch.num_rows

        for key, value in partition_values.items():
            if key in existing:
                continue
            arrays.append(pa.array([value] * n, type=pa.string()))
            names.append(key)

        return pa.RecordBatch.from_arrays(arrays, names=names)

    # ------------------------------------------------------------------
    # Filter normalization — two target forms
    # ------------------------------------------------------------------
    #
    # Two masks produced from the same input shape:
    #
    #   _normalize_filter_to_expression → ds.Expression (pushdown path)
    #   _build_filter_mask              → pa.BooleanArray (fallback path)
    #
    # The code below deliberately keeps them parallel so bugs fixed in
    # one implementation can be mirrored to the other.

    @staticmethod
    def _normalize_filter_to_expression(filter: Any) -> Any:
        """Turn a filter spec into a :class:`pyarrow.dataset.Expression`."""
        if filter is None:
            return None

        import pyarrow.dataset as ds

        if isinstance(filter, ds.Expression):
            return filter

        if isinstance(filter, dict):
            expr = None
            for key, value in filter.items():
                current = PathIO._expression_from_value(key, value)
                expr = current if expr is None else expr & current
            return expr

        if isinstance(filter, Sequence) and not isinstance(filter, (str, bytes)):
            # Single-tuple shorthand: (col, value) or (col, op, value).
            if len(filter) in {2, 3} and isinstance(filter[0], str):
                return PathIO._expression_from_tuple(filter)

            expr = None
            for item in filter:
                if (
                    not isinstance(item, Sequence)
                    or isinstance(item, (str, bytes))
                    or len(item) not in {2, 3}
                ):
                    raise TypeError(
                        "filter sequences must contain (column, value) or "
                        "(column, operator, value) items"
                    )
                current = PathIO._expression_from_tuple(item)
                expr = current if expr is None else expr & current
            return expr

        raise TypeError(
            "filter must be a pyarrow.dataset.Expression, mapping, or a "
            "sequence of filter tuples"
        )

    @staticmethod
    def _expression_from_value(column: str, value: Any):
        """Build a dataset expression for ``column == value`` (or IN/IS NULL)."""
        import pyarrow.dataset as ds

        field_expr = ds.field(column)

        if isinstance(value, range):
            value = list(value)
        if isinstance(value, set):
            value = list(value)

        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            values = list(value)
            if not values:
                # Empty IN — match nothing.
                return field_expr.is_null() & ~field_expr.is_null()
            expr = None
            for item in values:
                current = field_expr == item
                expr = current if expr is None else expr | current
            return expr

        if value is None:
            return field_expr.is_null()

        return field_expr == value

    @staticmethod
    def _expression_from_tuple(item: Sequence[Any]):
        """Build a dataset expression for a 2- or 3-tuple filter entry."""
        import pyarrow.dataset as ds

        if len(item) == 2:
            column, value = item
            return PathIO._expression_from_value(column, value)

        column, operator, value = item
        field_expr = ds.field(column)
        op = str(operator).strip().lower()

        if op in {"=", "==", "eq"}:
            return PathIO._expression_from_value(column, value)
        if op in {"!=", "<>", "ne"}:
            return field_expr != value
        if op in {">", "gt"}:
            return field_expr > value
        if op in {">=", "gte", "ge"}:
            return field_expr >= value
        if op in {"<", "lt"}:
            return field_expr < value
        if op in {"<=", "lte", "le"}:
            return field_expr <= value
        if op == "in":
            return PathIO._expression_from_value(column, value)
        if op == "not in":
            return ~PathIO._expression_from_value(column, value)
        if op == "is":
            return field_expr.is_null() if value is None else field_expr == value
        if op == "is not":
            return field_expr.is_valid() if value is None else field_expr != value

        raise ValueError(f"Unsupported filter operator: {operator!r}")

    @classmethod
    def _build_filter_mask(cls, *, table_or_batch, filter_spec: Any, pc):
        """Build a boolean mask (post-read path) from a filter spec."""
        if isinstance(filter_spec, dict):
            mask = None
            for key, value in filter_spec.items():
                current = cls._mask_from_value(
                    table_or_batch=table_or_batch, column=key, value=value, pc=pc
                )
                mask = current if mask is None else pc.and_kleene(mask, current)
            return mask

        if isinstance(filter_spec, (list, tuple)) and not isinstance(
            filter_spec, (str, bytes)
        ):
            if len(filter_spec) in {2, 3} and isinstance(filter_spec[0], str):
                return cls._mask_from_tuple(
                    table_or_batch=table_or_batch, item=filter_spec, pc=pc
                )

            mask = None
            for item in filter_spec:
                if not isinstance(item, (list, tuple)) or len(item) not in {2, 3}:
                    raise TypeError(
                        "filter sequences must contain (column, value) or "
                        "(column, operator, value) items"
                    )
                current = cls._mask_from_tuple(
                    table_or_batch=table_or_batch, item=item, pc=pc
                )
                mask = current if mask is None else pc.and_kleene(mask, current)
            return mask

        raise TypeError("filter must be a mapping or a sequence of filter tuples")

    @staticmethod
    def _mask_from_value(*, table_or_batch, column: str, value: Any, pc):
        """Build a boolean mask for a single column == value check."""
        array = table_or_batch.column(column)

        if isinstance(value, range):
            value = list(value)
        if isinstance(value, set):
            value = list(value)

        if isinstance(value, (list, tuple)) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            if not value:
                # Empty IN → never matches. pa.BooleanArray doesn't
                # overload `&`, so build an all-false array directly.
                return pa.array([False] * len(array), type=pa.bool_())
            mask = None
            for item in value:
                current = pc.equal(array, item)
                mask = current if mask is None else pc.or_kleene(mask, current)
            return mask

        if value is None:
            return pc.is_null(array)

        return pc.equal(array, value)

    @classmethod
    def _mask_from_tuple(cls, *, table_or_batch, item, pc):
        """Build a boolean mask for a 2- or 3-tuple filter entry."""
        if len(item) == 2:
            column, value = item
            return cls._mask_from_value(
                table_or_batch=table_or_batch, column=column, value=value, pc=pc
            )

        column, operator, value = item
        array = table_or_batch.column(column)
        op = str(operator).strip().lower()

        if op in {"=", "==", "eq"}:
            return cls._mask_from_value(
                table_or_batch=table_or_batch, column=column, value=value, pc=pc
            )
        if op in {"!=", "<>", "ne"}:
            return pc.not_equal(array, value)
        if op in {">", "gt"}:
            return pc.greater(array, value)
        if op in {">=", "gte", "ge"}:
            return pc.greater_equal(array, value)
        if op in {"<", "lt"}:
            return pc.less(array, value)
        if op in {"<=", "lte", "le"}:
            return pc.less_equal(array, value)
        if op == "in":
            return cls._mask_from_value(
                table_or_batch=table_or_batch, column=column, value=value, pc=pc
            )
        if op == "not in":
            return pc.invert(
                cls._mask_from_value(
                    table_or_batch=table_or_batch, column=column, value=value, pc=pc
                )
            )
        if op == "is":
            return pc.is_null(array) if value is None else pc.equal(array, value)
        if op == "is not":
            return (
                pc.invert(pc.is_null(array))
                if value is None
                else pc.not_equal(array, value)
            )

        raise ValueError(f"Unsupported filter operator: {operator!r}")

    @staticmethod
    def _filter_columns(filter_spec: Any) -> list[str]:
        """Extract the column names referenced by a filter spec.

        Returned unordered. Used to augment inner-reader projection so
        columns referenced by the filter are still loaded even when
        the caller projected them away at the outer level.
        """
        if filter_spec is None:
            return []
        if isinstance(filter_spec, dict):
            return list(filter_spec.keys())
        if isinstance(filter_spec, Sequence) and not isinstance(
            filter_spec, (str, bytes)
        ):
            if len(filter_spec) in {2, 3} and isinstance(filter_spec[0], str):
                return [filter_spec[0]]

            out: list[str] = []
            for item in filter_spec:
                if (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                    and item
                ):
                    column = item[0]
                    if isinstance(column, str):
                        out.append(column)
            return out
        return []

    # ------------------------------------------------------------------
    # Path + partitioning helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _path_parts(path: Any) -> tuple[str, ...]:
        parts = getattr(path, "parts", None)
        if parts is not None:
            return tuple(parts)
        return tuple(str(path).replace("\\", "/").split("/"))

    @classmethod
    def _relative_parts(
        cls,
        *,
        file_path: Any,
        partition_base_dir: Any,
    ) -> tuple[str, ...]:
        file_parts = cls._path_parts(file_path)
        base_parts = cls._path_parts(partition_base_dir)
        if base_parts and file_parts[: len(base_parts)] == base_parts:
            return file_parts[len(base_parts) :]
        return file_parts

    @classmethod
    def _partition_values(
        cls,
        *,
        file_path: Any,
        partitioning: str | Sequence[str] | None,
        partition_base_dir: Any,
    ) -> dict[str, str]:
        """Extract Hive or directory-name partition values from a file path."""
        if partitioning is None:
            return {}

        relative_parts = cls._relative_parts(
            file_path=file_path,
            partition_base_dir=partition_base_dir,
        )
        directory_parts = relative_parts[:-1]  # drop the filename
        if not directory_parts:
            return {}

        if isinstance(partitioning, str):
            if partitioning.lower() != "hive":
                return {}
            out: dict[str, str] = {}
            for segment in directory_parts:
                if "=" not in segment:
                    continue
                key, value = segment.split("=", 1)
                if key:
                    out[key] = value
            return out

        names = list(partitioning)
        return {name: value for name, value in zip(names, directory_parts) if name}

    # ------------------------------------------------------------------
    # Dataset format resolution
    # ------------------------------------------------------------------

    def _resolve_dataset_format(
        self,
        *,
        format: Any = None,
        recursive: bool = True,
        include_hidden: bool = False,
        supported_only: bool = False,
    ):
        """Resolve a dataset format from an explicit hint or the path's mime."""
        import pyarrow.csv as pa_csv
        import pyarrow.dataset as ds

        if format is not None:
            if hasattr(format, "make_fragment"):
                # Already a FileFormat instance.
                return format
            normalized = str(format).strip().lower()
            if normalized in {"parquet", "pq"}:
                return ds.ParquetFileFormat()
            if normalized in {"ipc", "arrow", "feather"}:
                return ds.IpcFileFormat()
            if normalized == "csv":
                return ds.CsvFileFormat()
            if normalized == "tsv":
                return ds.CsvFileFormat(
                    parse_options=pa_csv.ParseOptions(delimiter="\t")
                )
            if normalized in {"json", "ndjson"}:
                return ds.JsonFileFormat()
            if normalized == "orc":
                return ds.OrcFileFormat()
            raise ValueError(f"Unsupported dataset format: {format!r}")

        mime_type = self.infer_mime_type(
            recursive=recursive,
            include_hidden=include_hidden,
            supported_only=supported_only,
        )

        if mime_type is MimeTypes.PARQUET:
            return ds.ParquetFileFormat()
        if mime_type is MimeTypes.ARROW_IPC:
            return ds.IpcFileFormat()
        if mime_type is MimeTypes.CSV:
            return ds.CsvFileFormat()
        if mime_type is MimeTypes.TSV:
            return ds.CsvFileFormat(parse_options=pa_csv.ParseOptions(delimiter="\t"))
        if mime_type in {MimeTypes.JSON, MimeTypes.NDJSON}:
            return ds.JsonFileFormat()
        if mime_type is MimeTypes.ORC:
            return ds.OrcFileFormat()

        raise NotImplementedError(f"Unsupported dataset format for {mime_type!r}")


# --------------------------------------------------------------------------
# ``PathIO.path`` — property assigned after the @dataclass decorator
# --------------------------------------------------------------------------
#
# Attaching the property here (instead of in the class body) avoids the
# dataclass trap where a class-level ``path`` descriptor shadows the
# :class:`InitVar` default. Behaviorally this is a normal property:
# ``io.path`` reads ``io.holder.path`` and ``io.path = X`` rewrites
# the holder's backing after invalidating any cached mmap / read handle.


def _pathio_get_path(self: PathIO) -> Any:
    """Backing filesystem path. Stored on :attr:`PathIO.holder.path`."""
    return self.holder.path


def _pathio_set_path(self: PathIO, value: Any) -> None:
    self.holder._invalidate_mmap()
    self.holder._close_read_fh()
    self.holder._path = PathIO._coerce_path(value)


PathIO.path = property(_pathio_get_path, _pathio_set_path)  # type: ignore[assignment]
