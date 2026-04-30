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

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator as AbcIterator
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
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.disposable import Disposable
from yggdrasil.environ import PyEnv
from yggdrasil.io.enums import MediaType, MimeTypes, MimeType, MediaTypes, Mode
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module, path_class,
)

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame


__all__ = ["TabularIO"]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

O = TypeVar("O", bound=CastOptions)

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
        """Auto-register concrete subclasses against their mime type.

        Intermediate abstract subclasses return ``None`` from
        :meth:`default_mime_type` and don't register. A duplicate
        registration emits a warning rather than raising — picking
        a winner is the right move when, say, a test module defines
        a sub-handler that should override.
        """
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
        if media_type is not None:
            media_type = MediaType.from_(media_type)
        elif path is not None:
            path = path_class().from_(path)
            media_type = path.url.infer_media_type(default=None)

        # Bug: when media_type is still None here, target = cls.media_type_class(None)
        # falls through to MediaType.from_(None, default=OCTET_STREAM) and may
        # return BytesIO (not a TabularIO subclass). object.__new__(cls) on the
        # outer cls then succeeds but skips dispatch entirely — fine — except the
        # logic below ALSO calls media_type_class(None) a second time. Collapse it.

        if cls._FINAL_TABULAR_IO:
            return object.__new__(cls)

        if media_type is None:
            # No media type, no path hint — caller wants this exact class.
            # Don't probe the registry; that would either route to BytesIO
            # (not a TabularIO) or raise.
            return object.__new__(cls)

        target = cls.media_type_class(media_type, default=cls)
        if not issubclass(target, cls) and target is not cls:
            # Registry returned an unrelated class (e.g. BytesIO fallback).
            # Don't switch — keep cls.
            target = cls
        return object.__new__(target)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any
    ) -> None:
        # Cooperative super; concrete subclasses (DataIO) layer
        # storage-aware __init__ on top of this.
        Disposable.__init__(self)
        self._arrow_table = None
        self._spark_frame = None

    @classmethod
    def media_type_class(
        cls,
        media_type: "MediaType | None",
        *,
        default: "type[TabularIO[CastOptions]] | EllipsisType" = ...,
    ) -> "type[TabularIO[CastOptions]]":
        """Resolve a media-type-like value to the IO class that handles it.

        Returns the registered TabularIO subclass for the given mime type.
        On miss, returns `default` if provided, else raises RuntimeError.
        No BytesIO fallback — BytesIO is not a TabularIO.
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

        try:
            import yggdrasil.io.buffer.primitive  # noqa: F401
        except ImportError:
            pass
        else:
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

    def _release(self, committed: bool) -> None:
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
    def options_class(cls) -> type[O]:
        """The :class:`CastOptions` subclass this IO consumes.

        Default is :class:`CastOptions` itself; concrete subclasses
        with format-specific options (``ParquetOptions``,
        ``ArrowIPCOptions``) override.
        """
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def check_options(
        cls,
        options: "O | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> O:
        """Validate and merge caller kwargs into a resolved options instance.

        The standard pattern in the public surface is::

            def read_arrow_table(self, options=None, **kwargs):
                return self._read_arrow_table(
                    self.check_options(options, overrides=locals())
                )

        ``overrides`` is the ``locals()`` of the caller — a mix of
        ``options``, declared kwargs, and ``self``. We strip
        ``self``, pop ``options`` (already passed), drop ``...``
        sentinels, fold any ``**kwargs`` collected in the caller,
        and hand the rest to :meth:`CastOptions.check`.
        """
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))

        return cls.options_class().check(options, **kwargs)

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the underlying source.

        The single hook every other read surface routes through.
        Implementations should respect ``options.target_field`` for
        per-batch casting, ``options.row_size`` / ``options.byte_size``
        for batch sizing where the format permits, and
        ``options.column_names`` for column projection where natively
        supported.
        """

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
        """Consume Arrow record batches and persist them.

        The single hook every other write surface routes through.
        Implementations should resolve save semantics and respect
        ``options.target_field`` for output schema control.
        """

    # ==================================================================
    # Static helpers
    # ==================================================================

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> list[dict]:
        """Backfill every row to the union of keys seen across all rows.

        :func:`pa.Table.from_pylist` infers the schema from the first
        row only — sparse list-of-dicts inputs lose columns that
        appear later. We pre-walk and rewrite so every row carries
        every key (with ``None`` where missing) in first-seen order.
        """
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
        """Return the yggdrasil :class:`Schema` of the source.

        Reads the first batch's schema and (under ``options.safe``)
        merges schemas across all batches to handle drift.
        """
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

        # Default RangeIndex with no name → don't preserve. Otherwise
        # the user named the index for a reason — keep it.
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
        # Spark 3.5+ has ``toArrow``; older versions need the
        # toPandas() detour.
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
        """Write any supported tabular object.

        Strict ordering (so polars/pandas/spark frames don't fall
        into the iterable fallback): pyarrow → polars → pandas →
        spark → list[dict] / dict[list] → iterable of batches.
        """
        self._write_table(obj, self.check_options(options, overrides=locals()))

    def _write_table(self, obj: Any, options: O) -> None:
        if isinstance(obj, pa.Table):
            self._write_arrow_table(obj, options)
            return
        if isinstance(obj, pa.RecordBatch):
            self._write_arrow_table(pa.Table.from_batches([obj]), options)
            return

        # Module-prefix routing for engines we don't want to import
        # eagerly. ``polars.DataFrame.__class__.__module__`` is
        # ``polars.dataframe.frame``; the head module check is
        # robust against polars internal renames.
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
            # Last-resort: assume an iterable of pa.RecordBatch.
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
    ) -> pa.Table:
        """Outer-merge with incoming-wins-on-overlap.

        See :meth:`_arrow_upsert_via_rewrite` for the user-facing
        semantics. Pulled out so unit tests can exercise the merge
        directly without setting up an IO instance.

        :param existing: table currently on disk.
        :param incoming: table being written.
        :param match_by: columns whose values define row identity.
            Must be present in *incoming* (raises if not). May be
            absent in *existing* (treated as "no overlap").
        :returns: merged table. Column order: existing's columns
            first, then any incoming-only columns appended in
            incoming's order. Row order: surviving existing rows
            first, then incoming rows.
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

        # Both empty → empty result with the union of schemas.
        if existing.num_rows == 0 and incoming.num_rows == 0:
            return self.concat_with_schema_union([existing, incoming])

        # Only one side populated → no merge needed.
        if incoming.num_rows == 0:
            return existing
        if existing.num_rows == 0:
            return incoming

        # Both populated — drop existing rows whose match-by tuple
        # is in incoming, then schema-union concat.
        survivors = self._filter_out_matches(existing, incoming, match_by)
        return self.concat_with_schema_union([survivors, incoming])

    def concat_with_schema_union(
        self,
        tables: Sequence[pa.Table],
    ) -> pa.Table:
        """Concat tables with a column-union schema, filling missing
        columns with nulls.

        pyarrow's :func:`pa.concat_tables` requires identical schemas
        by default and supports ``promote_options="default"`` for
        some compatibility, but its handling of "left has columns
        right doesn't and vice-versa" is fragile across versions.
        This helper does the alignment explicitly:

        1. Build the union of column names, preserving first-seen
           order.
        2. For each table, add missing columns as null arrays of
           the union's inferred type.
        3. Concat the aligned tables.

        Skips empty tables (empty-schema and zero-row).
        """
        real_tables = [
            t for t in tables if t.num_columns > 0 or t.num_rows > 0
        ]
        if not real_tables:
            return pa.table({})

        if len(real_tables) == 1:
            return real_tables[0]

        # Column union (first-seen order).
        union_names: list[str] = []
        union_types: dict[str, pa.DataType] = {}
        for t in real_tables:
            for i, name in enumerate(t.column_names):
                if name not in union_types:
                    union_names.append(name)
                    union_types[name] = t.schema.field(i).type

        # Schema-align each table.
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
        """Return *existing* with rows whose match-by key is in
        *incoming* dropped.

        Two paths:

        - **Single-column match_by** → :func:`pyarrow.compute.is_in`
          directly. The fast path; runs in pyarrow C++ over a flat
          array.
        - **Multi-column match_by** → build tuple sets in Python.
          O(rows) in both tables but the constant is small; in
          practice this beats the alternative pyarrow shapes
          (struct-array is_in, hash join) for typical match-by
          sizes.
        """
        # Existing missing one of the match-by columns → no rows
        # can match (NULL vs anything in incoming is a non-match
        # for equality). Return existing unchanged.
        missing_in_existing = [
            c for c in match_by if c not in existing.column_names
        ]
        if missing_in_existing:
            return existing

        if len(match_by) == 1:
            col = match_by[0]
            mask = pc.invert(pc.is_in(existing[col], value_set=incoming[col]))
            return existing.filter(mask)

        # Multi-column path.
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
        """Read all existing batches, concat with *batches*, OVERWRITE.

        Used by leaves whose format doesn't support partial appends.
        ``self._read_arrow_batches`` is called BEFORE consuming the
        incoming iterator, so the read sees the pre-write buffer
        state.

        Schema handling: incoming and existing schemas may differ.
        They're unified column-wise via
        :meth:`concat_with_schema_union` (left columns first,
        right-only columns appended; missing columns filled with
        nulls of the appropriate type).

        The recursion-into-OVERWRITE is done via ``options.copy``
        with ``mode=Mode.OVERWRITE``.
        """
        # Materialize existing first — the read MUST happen before
        # we start writing or we'll feed the writer its own output.
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
        ``options.match_by_names``, OVERWRITE.

        Merge semantics:

        - Outer-join existing ⨝ incoming on match_by_names.
        - For rows in both, incoming columns replace existing
          columns (other than match_by, which is the join key).
        - For rows only in existing, the row is kept verbatim.
        - For rows only in incoming, the row is kept verbatim.
        - Schema is the column union; columns missing in either
          side are filled with nulls.

        Implementation note: equivalent to "delete existing rows
        whose match-by key is in incoming, then concat existing
        survivors with incoming, with schema union." Cheaper than a
        real full join because no key duplication and no per-column
        coalesce.

        Requires ``options.match_by_names`` to be a non-empty
        sequence of column names. Raises :class:`ValueError`
        otherwise — UPSERT without a key is ambiguous.
        """
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
            existing_table, incoming_table, match_by=match_by,
        )

        self._write_arrow_batches(
            merged.to_batches(max_chunksize=options.row_size),
            options.copy(mode=Mode.OVERWRITE),
        )
