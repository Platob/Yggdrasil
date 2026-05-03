"""Abstract tabular I/O contract over Apache Arrow record batches.

:class:`TabularIO` is the pure protocol layer: it knows nothing about
byte buffers, codecs, save modes, or path bindings. It defines what
it means to be a *tabular reader/writer* in yggdrasil — two abstract
hooks (``_read_arrow_batches`` / ``_write_arrow_batches``), an
options-class declaration, and the engine-agnostic surface
(Arrow / Polars / Pandas / Spark / Python-native, plus a
``write_table`` dispatcher) that derives from those hooks.

Storage concerns (``BytesIO`` inheritance, path acquire/release,
codec round-tripping), save-mode resolution, and append/upsert via
rewrite live in the concrete :class:`DataIO` subclass. Anything that
needs only "give me Arrow batches in / out" can target
:class:`TabularIO` and stay decoupled from the byte-buffer stack.

The contract
------------

A subclass implements three things:

- :meth:`options_class`        — what :class:`CastOptions` subtype it
                                  consumes. Default: :class:`CastOptions`
                                  itself.
- :meth:`_read_arrow_batches`  — yield :class:`pa.RecordBatch`.
- :meth:`_write_arrow_batches` — consume an iterable of batches.

Everything else (``read_arrow_table``, ``read_polars_frame``,
``write_pandas_frame``, ``write_table``, dataset views, schema
collection, …) funnels through :meth:`check_options` to land on a
single resolved options instance, then dispatches to one of the two
hooks.

Why no :class:`BytesIO` here
----------------------------

Plenty of tabular sources aren't byte buffers: a Spark catalog table,
a JDBC cursor, an in-memory Arrow Table wrapper, a remote dataset
service. They all satisfy "yield Arrow batches / consume Arrow
batches" but they don't have a meaningful byte representation. By
keeping :class:`TabularIO` storage-agnostic, those backends inherit
the full engine-conversion surface for free without pretending to be
byte buffers.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator as AbcIterator, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterator,
    Literal,
    TypeVar, ClassVar, Sequence,
)

import pyarrow as pa
import pyarrow.compute as pc
from yggdrasil.arrow.cast import any_to_arrow_table, any_to_arrow_batch_iterator
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.disposable import Disposable
from yggdrasil.environ import PyEnv
from yggdrasil.io.enums import MediaType, MimeTypes, MimeType, MediaTypes, Mode
from yggdrasil.lazy_imports import (
    bytes_io_class,
    polars_module,
    pyarrow_dataset_module, path_class,
)

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame
    from yggdrasil.io.buffer.bytes_io import BytesIO
    from yggdrasil.io.fs import Path


__all__ = ["TabularIO", "ChildrenOptions"]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

O = TypeVar("O", bound=CastOptions)


# ---------------------------------------------------------------------------
# Children discovery options
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ChildrenOptions(CastOptions):
    """Options carried through :meth:`TabularIO.iter_children`.

    Children discovery is conceptually distinct from cast/IO options:
    it filters which sub-IOs an aggregator (folder, archive, partitioned
    table) hands back, before any of them is opened or its batches are
    drained. Discovery knobs collected here:

    :param include_patterns: glob patterns (``fnmatch``-style, matched
        against the child's *name*). When set, only children matching at
        least one pattern are yielded. ``None`` (default) means "no
        include filter — accept everything not excluded".
    :param exclude_patterns: glob patterns; matching children are
        dropped. ``None`` (default) means "no exclude filter".
    :param exclude_private: skip children whose name starts with ``.``
        (the conventional hidden/private prefix). Default ``True`` —
        matches the existing :meth:`NestedIO._is_ignored_path` rule.
    :param max_depth: clip recursive walks at this depth (``0`` =
        only this level, ``1`` = one nested level, …). ``None`` (default)
        means "no depth cap" — recurse as far as :attr:`recursive`
        allows.
    :param follow_symlinks: whether to descend into symlinked directories
        on backends that distinguish them. Default ``False`` — matches
        the safe default to avoid cycles on local filesystems.

    :attr:`recursive` (inherited from :class:`CastOptions`) decides
    whether sub-folders are walked at all. ``ChildrenOptions`` does not
    redefine its default, so the base ``False`` still applies; folder
    IOs that recurse by default carry that intent on the instance.
    """

    include_patterns: tuple[str, ...] | None = None
    exclude_patterns: tuple[str, ...] | None = None
    exclude_private: bool = True
    max_depth: int | None = None
    follow_symlinks: bool = False

    def matches_name(self, name: str) -> bool:
        """Apply this options' filters to a candidate child *name*.

        Returns ``True`` if the child should be yielded, ``False`` if
        the include/exclude/private rules drop it. Pure-string match —
        no filesystem stat, no IO open. Subclasses with backend-specific
        introspection are free to layer richer checks on top.
        """
        if self.exclude_private and name.startswith("."):
            return False
        if self.exclude_patterns:
            for pattern in self.exclude_patterns:
                if fnmatch.fnmatchcase(name, pattern):
                    return False
        if self.include_patterns:
            for pattern in self.include_patterns:
                if fnmatch.fnmatchcase(name, pattern):
                    return True
            return False
        return True

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_DATAIO_REGISTRY: dict[str, type] = {}


def _normalize_media_key(obj: Any) -> "str | None":
    """Reduce a media-type-like value to a lowercase registry key."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj.strip().lower() or None
    name = getattr(obj, "name", None)
    if name:
        return str(name).lower()
    mime_type = getattr(obj, "mime_type", None)
    if mime_type is not None:
        name = getattr(mime_type, "name", None)
        if name:
            return str(name).lower()
    return None


# ---------------------------------------------------------------------------
# TabularIO
# ---------------------------------------------------------------------------


class TabularIO(Disposable, ABC, Generic[O]):
    """Abstract tabular I/O with Arrow record batches as the primitive.

    Pure protocol — no storage, no codec, no mode. Subclasses
    declare an options class and implement two hooks; everything
    else (engine conversions, schema inspection, ``write_table``
    dispatch) is derived.

    See the module docstring for the full contract.
    """

    _FINAL_TABULAR_IO: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Construction / persistence cache
    # ------------------------------------------------------------------

    @classmethod
    def default_mime_type(cls) -> "MimeType | None":
        """Canonical :class:`MimeType` this subclass handles.

        Concrete subclasses override; intermediates (``PrimitiveIO``,
        ``NestedIO``) leave it as ``None`` so they don't register
        and accidentally claim factory dispatch.
        """
        return MimeTypes.OCTET_STREAM

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses against their mime type."""
        super().__init_subclass__(**kwargs)
        try:
            mime = cls.default_mime_type()
        except Exception:
            return
        key = _normalize_media_key(mime)
        if key is None:
            return
        existing = _DATAIO_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            warnings.warn(
                f"DataIO registry: {cls.__name__} overrides "
                f"{existing.__name__} for mime_type {key!r}",
                RuntimeWarning,
                stacklevel=2,
            )

        _DATAIO_REGISTRY[key] = cls

    def __new__(cls, data=None, *args, copy=False, media_type=None, path=None,
                spill_bytes=128 * 1024 * 1024, spill_ttl=86400, auto_open=True, **kwargs):
        if cls._FINAL_TABULAR_IO:
            return object.__new__(cls)

        if data is not None:
            if isinstance(data, bytes_io_class()):
                media_type = media_type or data._media_type
                path = path or data.path

        if media_type is not None:
            media_type = MediaType.from_(media_type)
        elif path is not None:
            path = path_class().from_(path)
            media_type = path.url.infer_media_type(default=None)

        if media_type is None:
            return object.__new__(cls)

        target = cls.media_type_class(media_type, default=cls)
        if not issubclass(target, cls) and target is not cls:
            target = cls
        return object.__new__(target)

    def __init__(
        self,
        *args: Any,
        media_type: Any = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the state every :class:`TabularIO` carries.

        Sets up:

        * Disposable lifecycle (``opened`` / ``closed`` flags).
        * ``_media_type`` — the format identity. ``None`` for opaque
          buffers; concrete leaves (ParquetIO, CsvIO, …) fill it from
          their ``default_mime_type`` on construction.
        * Persist cache slots — ``_arrow_table`` / ``_spark_frame``
          set by :meth:`persist`, cleared by :meth:`unpersist`.
        * Spill-path slots — ``_spill_path`` / ``_owns_spill_path``
          used by :class:`BytesIO` (and any byte-buffer-backed
          subclass) to track durable storage. They live on
          :class:`TabularIO` so non-buffer subclasses inherit
          consistent ``None`` defaults without each writing the
          same boilerplate.

        Subclasses with extra state extend via ``super().__init__()``
        rather than re-zeroing the same slots; ``__init_subclass__``
        is reserved for the registry hook.
        """
        Disposable.__init__(self)
        # Format identity
        if media_type is None:
            cls_mime = type(self).default_mime_type()
            if cls_mime is not None and not getattr(cls_mime, "is_any_bytes", False):
                self._media_type = MediaType(cls_mime)
            else:
                self._media_type = None
        else:
            self._media_type = MediaType.from_(media_type)
        # Persist cache slots
        self._arrow_table = None
        self._spark_frame = None
        # Buffer-backing slots — concrete byte buffers fill these in;
        # non-byte TabularIOs (StatementResult, NestedIO trees, …)
        # leave them at None.
        self._spill_path = None
        self._owns_spill_path = True

    # ==================================================================
    # Tabular view shortcut — at TabularIO so every subclass shares it
    # ==================================================================

    def as_media(self, media_type: "Any | None" = None) -> "TabularIO":
        """Return a tabular view of self for the requested *media_type*.

        For a final-leaf instance (ParquetIO, CsvIO, ZipIO, …)
        carrying its own format, ``as_media()`` returns ``self`` —
        the leaf is already the cheapest view. Any other call builds
        a fresh registered leaf wrapping this object's underlying
        bytes / path / state.
        """
        if media_type is not None:
            media_type = MediaType.from_(media_type)

        if self._FINAL_TABULAR_IO:
            if media_type is None:
                return self
            if media_type.is_octet or media_type.mime_type == self.default_mime_type():
                return self
            target = self.media_type_class(media_type=media_type)
            return target(self, media_type=media_type)

        target = self.media_type_class(media_type=media_type, default=type(self))
        if target is type(self):
            return self
        return target(self, media_type=media_type)

    @classmethod
    def media_type_class(
        cls,
        media_type: "MediaType | None",
        *,
        default: "type[TabularIO[CastOptions]] | EllipsisType" = ...,
    ) -> "type[TabularIO[CastOptions]]":
        """Resolve a media-type-like value to the IO class that handles it.

        Lookup order:

        1. Direct registry hit. Concrete leaves register themselves
           in :data:`_DATAIO_REGISTRY` via :meth:`__init_subclass__`
           when their module is imported.
        2. Force-load the leaf packages that hold the most common
           concrete formats — :mod:`yggdrasil.io.buffer.primitive`
           (Parquet, Arrow IPC, CSV, JSON, NDJSON, XLSX) and
           :mod:`yggdrasil.io.buffer.nested` (Folder, Zip, Delta).
           If the caller started at a leaf module the registry is
           already populated; this is the safety net for callers
           that started elsewhere.
        3. Re-check the registry after the import sweep.
        4. Fall back to *default* (or raise when ``default`` is the
           ``...`` sentinel).

        Concrete tabular IO is always a :class:`BytesIO` subclass
        (single-buffer leaf) or a :class:`NestedIO` subclass
        (folder / archive aggregation). The registry contains both
        kinds keyed by mime, so a single lookup suffices for any
        registered format.
        """
        if media_type is None:
            if default is ...:
                raise RuntimeError("media_type_class called with None and no default")
            return default

        if not isinstance(media_type, MediaType):
            media_type = MediaType.from_(media_type, default=MediaTypes.OCTET_STREAM)

        key = _normalize_media_key(media_type)
        if key is None:
            if default is ...:
                raise RuntimeError(
                    f"No tabular IO registered for media_type {media_type!r}"
                )
            return default

        hit = _DATAIO_REGISTRY.get(key)
        if hit is not None:
            return hit

        # Safety net — force-load the canonical leaf packages so
        # their __init_subclass__ registry-population runs. Tolerate
        # missing optional packages (the import itself may pull in
        # extras like openpyxl); a partial load still populates the
        # leaves that did import.
        for module_path in (
            "yggdrasil.io.buffer.primitive",
            "yggdrasil.io.buffer.nested",
        ):
            try:
                __import__(module_path)
            except ImportError:
                continue

        hit = _DATAIO_REGISTRY.get(key)
        if hit is not None:
            return hit

        if default is ...:
            raise RuntimeError(
                f"No tabular IO registered for media_type {media_type!r}"
            )
        return default

    @property
    @abstractmethod
    def cached(self) -> bool:
        return self._arrow_table is not None or self._spark_frame is not None

    def _release(self) -> None:
        self.unpersist()

    @abstractmethod
    def unpersist(self) -> None:
        self._arrow_table = None
        self._spark_frame = None

    @abstractmethod
    def persist(
        self,
        engine: Literal["arrow", "polars", "spark", "auto"] = "auto",
        *,
        data: Any | None = None,
    ) -> "TabularIO":
        if self.cached:
            return self

        if not engine or engine == "auto":
            engine = "spark" if PyEnv.in_databricks() else "arrow"

        if data is None:
            if engine == "spark":
                self._spark_frame = self.read_spark_frame()
            elif engine == "arrow":
                self._arrow_table = self.read_arrow_table()
            else:
                raise ValueError(f"Unsupported engine: {engine}")
        else:
            if engine == "spark":
                from yggdrasil.spark.cast import any_to_spark_dataframe

                self._spark_frame = any_to_spark_dataframe(data)
            elif engine == "arrow":
                self._arrow_table = any_to_arrow_table(data)
            else:
                raise ValueError(f"Unsupported engine: {engine}")

        return self

    # ==================================================================
    # Abstract protocol — subclasses implement these
    # ==================================================================

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        media_type: MediaType | None = None,
        default: Any = ...,
        **kwargs: Any
    ):
        if cls._FINAL_TABULAR_IO and isinstance(obj, cls):
            return obj

        buffer = bytes_io_class().from_(obj, media_type=media_type, default=None)
        
        if buffer is None:
            if default is ...:
                raise RuntimeError(
                    f"No tabular IO registered for media_type {media_type!r}"
                )
            return default

        return cls.from_bytes_io(buffer, media_type=media_type, default=default)

    @classmethod
    def from_bytes_io(
        cls,
        buffer: "BytesIO",
        media_type: MediaType | None = None,
        default: Any = ...
    ):
        buffer = bytes_io_class().from_(buffer)
        if media_type is None:
            media_type = buffer.media_type
        else:
            media_type = MediaType.from_(media_type)
            
        target = cls.media_type_class(media_type, default=cls)
        
        if target._FINAL_TABULAR_IO:
            return cls(buffer, media_type=media_type)
        
        if default is ...:
            raise RuntimeError(
                f"No tabular IO registered for media_type {media_type!r}"
            )
        return default

    @classmethod
    def from_path(
        cls,
        path: "Path",
        media_type: MediaType | None = None
    ) -> "TabularIO[O]":
        path = path_class().from_(path)
        return cls(path=path, media_type=media_type)

    @classmethod
    def options_class(cls) -> type[O]:
        """The :class:`CastOptions` subclass this IO consumes."""
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def children_options_class(cls) -> type[ChildrenOptions]:
        """The :class:`ChildrenOptions` subclass :meth:`iter_children` consumes.

        Subclasses with extra discovery knobs (e.g. partition-aware
        filters) override to return their richer subclass.
        """
        return ChildrenOptions

    @classmethod
    def check_options(
        cls,
        options: "O | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> O:
        """Validate and merge caller kwargs into a resolved options instance."""
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))

        return cls.options_class().check(options, **kwargs)

    @classmethod
    def check_children_options(
        cls,
        options: "ChildrenOptions | Mapping[str, Any] | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> ChildrenOptions:
        """Resolve a :class:`ChildrenOptions` from any accepted input shape.

        Mirrors :meth:`check_options` but routes through
        :meth:`children_options_class` so callers (and subclasses)
        can pass discovery-specific kwargs without polluting
        :class:`CastOptions`.
        """
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))

        return cls.children_options_class().check(options, **kwargs)

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the underlying source."""

    def _read_arrow_batches_from_cache(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the cache."""
        if not self.cached:
            raise RuntimeError("No cache available")

        if self._arrow_table is None:
            if self._spark_frame is not None:
                arrow_table = self._spark_frame.toArrow()
            else:
                raise RuntimeError("No cache available")
        else:
            arrow_table = self._arrow_table

        yield from options.cast_arrow_tabular(arrow_table).to_batches(
            max_chunksize=options.row_size
        )

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Consume Arrow record batches and persist them."""

    def _iter_children(self, options: ChildrenOptions) -> "Iterator[TabularIO]":
        """Yield this IO's direct children, each as a :class:`TabularIO`.

        Single-buffer leaves (``PrimitiveIO``, ``BytesIO`` without a
        tabular media type, ``StatementResult``, …) yield nothing —
        they ARE the leaf and have no sub-IO surface. Folder/archive
        aggregations (:class:`NestedIO` subclasses, :class:`ZipIO`)
        override to yield one IO per data unit (file, entry,
        partition leaf), each itself a :class:`TabularIO` openable
        on its own.

        :class:`Fragment` indirection is intentionally absent —
        children are real IOs with their own ``parent`` back-pointer,
        so callers can walk the tree and drain Arrow batches
        uniformly.

        ``options`` carries discovery knobs (:attr:`recursive`,
        include/exclude patterns, hidden filtering, depth cap).
        Subclasses are expected to honor them where meaningful;
        :meth:`ChildrenOptions.matches_name` is the canonical
        name-based filter.

        Default: yields nothing — the right answer for any
        single-source TabularIO. :class:`NestedIO` re-declares this
        abstract so folder-shaped subclasses can't accidentally
        inherit the empty default.
        """
        return iter(())

    def iter_children_or_self(
        self,
        options: "ChildrenOptions | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> "Iterator[TabularIO]":
        if self.has_children():
            yield from self.iter_children(options, **kwargs)
        else:
            yield self

    def iter_children(
        self,
        options: "ChildrenOptions | Mapping[str, Any] | None" = None,
        **kwargs: Any,
    ) -> "Iterator[TabularIO]":
        """Public wrapper around :meth:`_iter_children`.

        Accepts a :class:`ChildrenOptions`, a mapping of discovery
        kwargs, or ``None`` (for "use the defaults"). Discovery kwargs
        passed positionally as ``**kwargs`` are merged into the
        resolved options.
        """
        return self._iter_children(
            self.check_children_options(options, overrides=locals())
        )

    def has_children(self) -> bool:
        """Return ``True`` iff this IO exposes at least one child.

        Default probes :meth:`_iter_children` with default options and
        peeks the first element. Subclasses with cheaper introspection
        (a directory-listing call, a central-directory scan, an
        already-cached child set) should override.

        Single-buffer leaves with no children surface return ``False``
        via the inherited empty :meth:`_iter_children`. Folder-shaped
        IOs (:class:`NestedIO`, :class:`ZipIO`,
        :class:`ZipEntryFolderIO`) override or rely on the default
        peek to answer "is this a leaf or a container?".
        """
        try:
            return next(iter(self._iter_children(self._has_children_options())), None) is not None
        except (FileNotFoundError, StopIteration):
            return False

    def _has_children_options(self) -> ChildrenOptions:
        """Cheap default options for :meth:`has_children`'s peek.

        Split out so subclasses with mandatory option fields can
        provide a probe-friendly default without rebuilding
        :meth:`has_children`.
        """
        return self.children_options_class()()

    # ==================================================================
    # Static helpers
    # ==================================================================

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> list[dict]:
        """Backfill every row to the union of keys seen across all rows."""
        rows = list(data) if not isinstance(data, list) else data
        if not rows:
            return []

        all_keys: dict[str, None] = {}
        needs_backfill = False
        reference: "tuple[str, ...] | None" = None

        for row in rows:
            if row is None:
                needs_backfill = True
                continue
            keys = tuple(row.keys())
            if reference is None:
                reference = keys
            elif keys != reference:
                needs_backfill = True
            for k in keys:
                if k not in all_keys:
                    all_keys[k] = None

        if not needs_backfill:
            return rows

        key_tuple = tuple(all_keys)
        return [
            {k: (row.get(k) if row is not None else None) for k in key_tuple}
            for row in rows
        ]

    # ==================================================================
    # Schema inspection
    # ==================================================================

    def collect_schema(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Schema:
        """Return the yggdrasil :class:`Schema` of the source."""
        return self._collect_schema(self.check_options(options, overrides=locals()))

    def _collect_schema(self, options: O) -> Schema:
        batches = self._read_arrow_batches(options)
        first = next(iter(batches), None)
        if first is None:
            return Schema.empty()

        schema = Schema.from_arrow(first.schema)
        if not getattr(options, "safe", False):
            return schema

        for batch in batches:
            schema = schema.merge_with(
                Schema.from_arrow(batch.schema),
                inplace=True,
            )
        return schema

    # ==================================================================
    # Arrow surface
    # ==================================================================

    def to_arrow_batches(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_arrow_batches(options=options, **kwargs)

    def read_arrow_batches(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches. Primary streaming read path."""
        yield from self._read_arrow_batches(
            self.check_options(options, overrides=locals())
        )

    def to_arrow_table(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_arrow_table(options=options, **kwargs)

    def read_arrow_table(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> pa.Table:
        """Read everything into a single :class:`pyarrow.Table`."""
        return self._read_arrow_table(self.check_options(options, overrides=locals()))

    def _read_arrow_table(self, options: O) -> pa.Table:
        batches = list(self._read_arrow_batches(options))
        if not batches:
            schema = (
                getattr(options, "target_schema", None)
                or getattr(options, "source_schema", None)
                or Schema.empty()
            )
            return schema.to_arrow_schema().empty_table()
        return pa.Table.from_batches(batches)

    def write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write an iterable of Arrow record batches."""
        self._write_arrow_batches(
            batches, self.check_options(options, overrides=locals())
        )

    def write_arrow_table(
        self,
        table: pa.Table,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a :class:`pyarrow.Table`."""
        self._write_arrow_table(table, self.check_options(options, overrides=locals()))

    def _write_arrow_table(self, table: pa.Table, options: O) -> None:
        row_size = getattr(options, "row_size", None) or None
        self._write_arrow_batches(
            table.to_batches(max_chunksize=row_size),
            options.check_source(table).copy(row_size=None) if row_size else options,
        )

    def to_arrow_batch_reader(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_arrow_batch_reader(options=options, **kwargs)

    def read_arrow_batch_reader(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "pa.RecordBatchReader":
        """Return a :class:`pyarrow.RecordBatchReader` view."""
        return self._read_arrow_batch_reader(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_batch_reader(self, options: O) -> "pa.RecordBatchReader":
        schema = options.check_target(obj=self.collect_schema).merged_schema
        arrow_schema = schema.to_arrow_schema()
        return pa.RecordBatchReader.from_batches(
            arrow_schema,
            self._read_arrow_batches(options),
        )

    def to_arrow_dataset(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_arrow_dataset(options=options, **kwargs)

    def read_arrow_dataset(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "pds.Dataset":
        """Return a :class:`pyarrow.dataset.Dataset` view."""
        return self._read_arrow_dataset(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_dataset(self, options: O) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        reader = self._read_arrow_batch_reader(options)
        return pds.dataset(reader, schema=reader.schema)

    # ==================================================================
    # Polars
    # ==================================================================

    def to_polars(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_polars_frame(options=options, **kwargs)

    def read_polars_frame(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "pl.DataFrame":
        """Read as a Polars :class:`DataFrame`. Eager."""
        return self._read_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frame(self, options: O) -> "pl.DataFrame":
        pl = polars_module()
        return pl.from_arrow(self._read_arrow_table(options))  # type: ignore[return-value]

    def read_polars_frames(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator["pl.DataFrame"]:
        """Yield one Polars frame per Arrow record batch. Streaming."""
        yield from self._read_polars_frames(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frames(self, options: O) -> Iterator["pl.DataFrame"]:
        pl = polars_module()
        for batch in self._read_arrow_batches(options):
            yield pl.from_arrow(batch, rechunk=False)  # type: ignore[misc]

    def scan_polars(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.scan_polars_frame(options=options, **kwargs)

    def scan_polars_frame(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "pl.LazyFrame":
        """Return a Polars :class:`LazyFrame` wrapping the Arrow dataset."""
        return self._scan_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _scan_polars_frame(self, options: O) -> "pl.LazyFrame":
        pl = polars_module()
        return pl.scan_pyarrow_dataset(self._read_arrow_dataset(options))

    def write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a Polars :class:`DataFrame` or :class:`LazyFrame`."""
        self._write_polars_frame(frame, self.check_options(options, overrides=locals()))

    def _write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: O,
    ) -> None:
        pl = polars_module()

        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()

        if frame.height == 0:
            return

        row_size = getattr(options, "row_size", None) or 0
        byte_size = getattr(options, "byte_size", None) or 0

        if row_size > 0:
            chunks: Iterator["pl.DataFrame"] = frame.iter_slices(n_rows=row_size)
        elif byte_size > 0:
            total = frame.estimated_size(unit="b")
            if total == 0:
                chunks = iter((frame,))
            else:
                rows_per_chunk = max(1, int(frame.height * byte_size / total))
                chunks = frame.iter_slices(n_rows=rows_per_chunk)
        else:
            chunks = iter((frame,))

        self._write_polars_frames(chunks, options)

    def _write_polars_frames(
        self,
        frames: Iterator["pl.DataFrame"],
        options: O,
    ) -> None:
        def gen() -> Iterator[pa.RecordBatch]:
            for f in frames:
                yield from f.to_arrow().to_batches()

        self._write_arrow_batches(gen(), options)

    # ==================================================================
    # Pandas
    # ==================================================================

    def to_pandas(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_pandas_frame(options=options, **kwargs)

    def read_pandas_frame(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "pandas.DataFrame":
        """Read as a Pandas :class:`DataFrame`."""
        return self._read_pandas_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_pandas_frame(self, options: O) -> "pandas.DataFrame":
        return self._read_arrow_table(options).to_pandas()

    def write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a Pandas :class:`DataFrame`."""
        self._write_pandas_frame(frame, self.check_options(options, overrides=locals()))

    def _write_pandas_frame(
        self,
        frame: "pandas.DataFrame",
        options: O,
    ) -> None:
        import pandas as pd

        is_default_range = (
            isinstance(frame.index, pd.RangeIndex) and frame.index.name is None
        )
        table = pa.Table.from_pandas(frame, preserve_index=not is_default_range)
        self._write_arrow_table(table, options)

    # ==================================================================
    # Spark
    # ==================================================================

    def to_spark(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "SparkDataFrame":
        """Materialize on the driver and build a Spark DataFrame."""
        return self.read_spark_frame(options, **kwargs)

    def read_spark_frame(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "SparkDataFrame":
        """Materialize on the driver and build a Spark DataFrame."""
        if self.cached:
            return self._read_spark_frame_from_cache(options)
        return self._read_spark_frame(self.check_options(options, overrides=locals()))

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        spark = PyEnv.spark_session(create=True)
        arrow_table = self._read_arrow_table(options)
        return options.cast_spark(spark.createDataFrame(arrow_table))

    def _read_spark_frame_from_cache(self, options: O) -> "SparkDataFrame":
        if self._spark_frame is None:
            from yggdrasil.spark.cast import any_to_spark_dataframe

            return any_to_spark_dataframe(self._arrow_table, options=options)
        else:
            df = self._spark_frame

        return options.cast_spark_tabular(df)

    def write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Collect the Spark frame to Arrow and write."""
        self._write_spark_frame(frame, self.check_options(options, overrides=locals()))

    def _write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: O,
    ) -> None:
        to_arrow = getattr(frame, "toArrow", None)
        if to_arrow is not None:
            self._write_arrow_table(to_arrow(), options)
            return
        self._write_pandas_frame(frame.toPandas(), options)

    # ==================================================================
    # Python-native
    # ==================================================================

    def to_pylist(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_pylist(options=options, **kwargs)

    def read_pylist(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "list[dict]":
        """Read as a list of row dicts."""
        return self._read_pylist(self.check_options(options, overrides=locals()))

    def _read_pylist(self, options: O) -> "list[dict]":
        return self._read_arrow_table(options).to_pylist()

    def write_pylist(
        self,
        data: Iterable[dict],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a list (or iterable) of row dicts."""
        self._write_pylist(data, self.check_options(options, overrides=locals()))

    def _write_pylist(self, data: Iterable[dict], options: O) -> None:
        rows = self._normalize_records(data)
        if not rows:
            return
        self._write_arrow_table(pa.Table.from_pylist(rows), options)

    def to_record_iterator(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator[Mapping[str, Any]]:
        """Alias for :meth:`read_record_iterator`."""
        return self.read_record_iterator(options=options, **kwargs)

    def read_record_iterator(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator[Mapping[str, Any]]:
        """Stream rows as ``dict`` records, one per row.

        True streaming: row N is yielded as soon as its batch is
        decoded; the full table is never materialized. Useful for
        large sources where :meth:`read_pylist` would blow memory.

        Each yielded row is a fresh ``dict`` (no shared state with
        the underlying batch — the caller is free to mutate it).
        Nested types come out as ``list`` / ``dict`` per pyarrow's
        :meth:`pyarrow.RecordBatch.to_pylist` conventions, preserving
        type fidelity that hand-rolled per-column reconstruction
        would lose.
        """
        return self._read_record_iterator(
            self.check_options(options, overrides=locals())
        )

    def _read_record_iterator(self, options: O) -> Iterator[Mapping[str, Any]]:
        for batch in self._read_arrow_batches(options):
            # batch.to_pylist() does the column-major → row-major
            # rotation in pyarrow C++ once per batch; per-row yield
            # then costs only the dict reference. Cheaper than
            # reconstructing rows in Python and keeps nested type
            # fidelity for free.
            yield from batch.to_pylist()

    # ==================================================================
    # Record streaming — typed rows sharing a singleton Schema
    # ==================================================================

    def to_records(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "Iterator[Any]":
        """Alias for :meth:`read_records`."""
        return self.read_records(options=options, **kwargs)

    def read_records(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "Iterator[Any]":
        """Stream rows as :class:`yggdrasil.data.record.Record` objects.

        Each :class:`Record` is a Mapping over one row's values keyed
        by field name; the underlying :class:`Schema` is materialized
        once per stream and shared by reference across every yielded
        record. Lower per-row allocation than :meth:`read_pylist` for
        sources where the schema is stable across batches (the common
        case).
        """
        return self._read_records(
            self.check_options(options, overrides=locals())
        )

    def _read_records(self, options: O) -> "Iterator[Any]":
        # Default impl: derive Records from `_read_arrow_batches`,
        # locking the Schema to the first batch's schema. Subclasses
        # with a row-native source (a SQL cursor, a Spark Row stream)
        # should override to skip the column→Python materialisation.
        from yggdrasil.data.record import Record

        yield from Record.from_arrow_batches(
            self._read_arrow_batches(options),
        )

    def write_records(
        self,
        records: "Iterable[Any]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a stream of :class:`Record` (or row-mapping) objects."""
        self._write_records(
            records, self.check_options(options, overrides=locals()),
        )

    def _write_records(
        self,
        records: "Iterable[Any]",
        options: O,
    ) -> None:
        """Default impl: bucket records by `options.row_size` and
        delegate to :meth:`_write_arrow_batches`. Subclasses with a
        row-native sink (a SQL bulk-insert, a Spark createDataFrame)
        should override to skip the row→Arrow round-trip.

        The first record's schema becomes the writer's target schema —
        a record with a different schema in the same stream is
        silently re-aligned via dict-order iteration, so callers that
        care about strictness should pre-validate.
        """
        from yggdrasil.data.record import Record

        chunk_size = max(1, options.row_size or 1024)
        chunk_rows: list[dict] = []
        chunk_schema: "pa.Schema | None" = None

        def _flush() -> None:
            if not chunk_rows:
                return
            batch = pa.RecordBatch.from_pylist(chunk_rows, schema=chunk_schema)
            self._write_arrow_batches([batch], options)
            chunk_rows.clear()

        for rec in records:
            if isinstance(rec, Record):
                if chunk_schema is None:
                    chunk_schema = rec.schema.to_arrow_schema()
                chunk_rows.append(rec.to_dict())
            elif isinstance(rec, Mapping):
                chunk_rows.append(dict(rec))
            else:
                raise TypeError(
                    f"_write_records expected Record or Mapping rows; "
                    f"got {type(rec).__name__}: {rec!r}"
                )
            if len(chunk_rows) >= chunk_size:
                _flush()
        _flush()

    def to_pydict(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ):
        return self.read_pydict(options=options, **kwargs)

    def read_pydict(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "dict[str, list]":
        """Read as a column-oriented dict."""
        return self._read_pydict(self.check_options(options, overrides=locals()))

    def _read_pydict(self, options: O) -> "dict[str, list]":
        return self._read_arrow_table(options).to_pydict()

    def write_pydict(
        self,
        data: "dict[str, list]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write a column-oriented dict."""
        self._write_pydict(data, self.check_options(options, overrides=locals()))

    def _write_pydict(self, data: "dict[str, list]", options: O) -> None:
        self._write_arrow_table(pa.Table.from_pydict(data), options)

    # ==================================================================
    # Generic write dispatcher
    # ==================================================================

    def write_io(
        self,
        io: "TabularIO",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Drain another :class:`TabularIO` into self."""
        self._write_io(io, self.check_options(options, overrides=locals()))

    def _write_io(self, io: "TabularIO", options: O) -> None:
        self._write_arrow_batches(
            io.read_arrow_batches(options=options),
            options,
        )

    def write_table(
        self,
        obj: Any,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Write any supported tabular object."""
        self._write_table(obj, self.check_options(options, overrides=locals()))

    def _write_table(self, obj: Any, options: O) -> None:
        if isinstance(obj, pa.Table):
            self._write_arrow_table(obj, options)
            return
        if isinstance(obj, pa.RecordBatch):
            self._write_arrow_table(pa.Table.from_batches([obj]), options)
            return

        module = (type(obj).__module__ or "").split(".", 1)[0]
        if module == "polars":
            self._write_polars_frame(obj, options)
            return
        if module == "pandas":
            self._write_pandas_frame(obj, options)
            return
        if module == "pyspark":
            self._write_spark_frame(obj, options)
            return

        if isinstance(obj, TabularIO):
            self._write_io(obj, options)
            return

        if isinstance(obj, (list, tuple)):
            if not obj:
                return
            if all(isinstance(row, dict) for row in obj):
                self._write_pylist(obj, options)
                return

        if isinstance(obj, dict):
            self._write_pydict(obj, options)
            return

        if isinstance(obj, AbcIterator) or hasattr(obj, "__iter__"):
            self._write_arrow_batches(obj, options)
            return

        raise TypeError(
            f"Unsupported tabular object for write_table: {type(obj)!r}. "
            "Expected one of: pyarrow.Table, pyarrow.RecordBatch, "
            "pandas.DataFrame, polars.DataFrame, polars.LazyFrame, "
            "pyspark.sql.DataFrame, list[dict], dict[str, list], "
            "TabularIO, or Iterable[pyarrow.RecordBatch]."
        )

    # ==================================================================
    # Pure-function merge helpers — no IO state, easy to unit-test
    # ==================================================================

    def merge_upsert_tables(
        self,
        existing: pa.Table,
        incoming: pa.Table,
        *,
        match_by: Sequence[str],
        update_column_names: Sequence[str] | None = None,
    ) -> pa.Table:
        """Outer-merge with incoming-wins-on-overlap.

        With ``update_column_names``, only those columns are taken from
        the incoming side on a key match; non-key columns outside the
        update list keep their existing values. Mirrors the warehouse
        ``MERGE … WHEN MATCHED THEN UPDATE SET <cols>`` semantics so a
        user can switch between Delta and a buffer-backed target without
        rewriting the upsert payload.

        ``update_column_names = None`` keeps the historical behaviour:
        every non-key column on the incoming side overwrites the
        existing one.
        """
        match_by = tuple(match_by)
        if not match_by:
            raise ValueError("match_by must contain at least one column name")

        missing_in_incoming = [
            c for c in match_by if c not in incoming.column_names
        ]
        if missing_in_incoming:
            raise ValueError(
                f"Match-by columns missing from incoming table: "
                f"{missing_in_incoming}. Available columns: "
                f"{list(incoming.column_names)}."
            )

        if existing.num_rows == 0 and incoming.num_rows == 0:
            return self.concat_with_schema_union([existing, incoming])

        if incoming.num_rows == 0:
            return existing
        if existing.num_rows == 0:
            return incoming

        if update_column_names is not None:
            incoming = self._restrict_update_columns(
                existing, incoming,
                match_by=match_by,
                update_column_names=tuple(update_column_names),
            )

        survivors = self._filter_out_matches(existing, incoming, match_by)
        return self.concat_with_schema_union([survivors, incoming])

    def _restrict_update_columns(
        self,
        existing: pa.Table,
        incoming: pa.Table,
        *,
        match_by: Sequence[str],
        update_column_names: Sequence[str],
    ) -> pa.Table:
        """Replace incoming non-update columns with values pulled from
        existing rows that share the same match key.

        For incoming rows whose key has no existing match, columns
        outside ``update_column_names`` are left as nulls — the row is a
        plain INSERT and there's nothing to preserve from the target
        side.
        """
        update_set = set(update_column_names)
        match_set = set(match_by)
        preserved_cols = [
            name for name in incoming.column_names
            if name not in match_set and name not in update_set
            and name in existing.column_names
        ]
        if not preserved_cols:
            return incoming

        if len(match_by) == 1:
            col = match_by[0]
            existing_keys = existing[col].to_pylist()
            key_to_row = {k: i for i, k in enumerate(existing_keys) if k is not None}
            incoming_keys = incoming[col].to_pylist()
        else:
            existing_keys = list(zip(
                *[existing[c].to_pylist() for c in match_by]
            ))
            key_to_row = {k: i for i, k in enumerate(existing_keys)}
            incoming_keys = list(zip(
                *[incoming[c].to_pylist() for c in match_by]
            ))

        preserved_arrays: dict[str, pa.Array] = {}
        for name in preserved_cols:
            existing_col = existing[name].to_pylist()
            preserved_arrays[name] = pa.array(
                [
                    existing_col[key_to_row[k]] if k in key_to_row else None
                    for k in incoming_keys
                ],
                type=existing.schema.field(name).type,
            )

        new_columns = []
        for i, name in enumerate(incoming.column_names):
            if name in preserved_arrays:
                new_columns.append(preserved_arrays[name])
            else:
                new_columns.append(incoming.column(i))
        return pa.table(new_columns, names=list(incoming.column_names))

    def concat_with_schema_union(
        self,
        tables: Sequence[pa.Table],
    ) -> pa.Table:
        """Concat tables with a column-union schema, filling missing
        columns with nulls."""
        real_tables = [
            t for t in tables if t.num_columns > 0 or t.num_rows > 0
        ]
        if not real_tables:
            return pa.table({})

        if len(real_tables) == 1:
            return real_tables[0]

        union_names: list[str] = []
        union_types: dict[str, pa.DataType] = {}
        for t in real_tables:
            for i, name in enumerate(t.column_names):
                if name not in union_types:
                    union_names.append(name)
                    union_types[name] = t.schema.field(i).type

        aligned: list[pa.Table] = []
        for t in real_tables:
            if t.column_names == union_names:
                aligned.append(t)
                continue
            cols = []
            for name in union_names:
                if name in t.column_names:
                    cols.append(t[name])
                else:
                    cols.append(pa.nulls(t.num_rows, type=union_types[name]))
            aligned.append(pa.table(dict(zip(union_names, cols))))

        return pa.concat_tables(aligned, promote_options="default")

    def _filter_out_matches(
        self,
        existing: pa.Table,
        incoming: pa.Table,
        match_by: Sequence[str],
    ) -> pa.Table:
        """Return *existing* with rows whose match-by key is in *incoming* dropped."""
        missing_in_existing = [
            c for c in match_by if c not in existing.column_names
        ]
        if missing_in_existing:
            return existing

        if len(match_by) == 1:
            col = match_by[0]
            mask = pc.invert(pc.is_in(existing[col], value_set=incoming[col]))
            return existing.filter(mask)

        incoming_keys = set(
            zip(*[incoming[c].to_pylist() for c in match_by])
        )
        if not incoming_keys:
            return existing

        existing_cols = [existing[c].to_pylist() for c in match_by]
        keep_mask = [
            tuple(c[i] for c in existing_cols) not in incoming_keys
            for i in range(existing.num_rows)
        ]
        return existing.filter(pa.array(keep_mask))

    # ==================================================================
    # Append / upsert via rewrite
    # ==================================================================

    def _arrow_append_via_rewrite(
        self,
        batches: Any,
        options: O,
    ) -> None:
        """Read all existing batches, concat with *batches*, OVERWRITE."""
        persisted = self._read_arrow_table(options.copy(read_seek=0))

        options = options.copy(
            mode=Mode.OVERWRITE,
        )

        batches = any_to_arrow_batch_iterator(batches, options)

        if persisted.num_rows == 0:
            self._write_arrow_batches(batches, options)
            return

        def new_iterator(f=persisted, n=batches) -> Iterator[pa.RecordBatch]:
            yield from f.to_batches(max_chunksize=options.row_size)
            yield from n

        self._write_arrow_batches(
            new_iterator(),
            options=options,
        )

    def _arrow_upsert_via_rewrite(
        self,
        batches: Any,
        options: O,
    ) -> None:
        """Read existing, merge with incoming on
        ``options.match_by_names``, OVERWRITE."""
        match_by = options.match_by_names
        if not match_by:
            raise ValueError(
                f"{type(self).__name__} UPSERT requires "
                "options.match_by_names to be a non-empty sequence of "
                "column names. For 'replace everything,' use "
                "Mode.OVERWRITE instead."
            )

        existing_table = self._read_arrow_table(options.copy(read_seek=0))
        incoming_table = any_to_arrow_table(batches, options)

        merged = self.merge_upsert_tables(
            existing_table, incoming_table,
            match_by=match_by,
            update_column_names=options.update_column_names,
        )

        self._write_arrow_batches(
            merged.to_batches(max_chunksize=options.row_size),
            options.copy(mode=Mode.OVERWRITE),
        )