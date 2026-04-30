"""Abstract tabular I/O over Apache Arrow record batches.

:class:`DataIO` is the abstract data contract: subclasses implement
how to read/write Arrow record batches; everything else (Polars,
Pandas, Spark, Python dict/list, dataset views, generic
``write_table`` dispatch, SaveMode resolution, codec round-tripping,
append/upsert via rewrite) derives from those two hooks.

Inheritance shape
-----------------

:class:`DataIO` inherits :class:`BytesIO` directly. Practically this
means:

- The bytes are right there: ``self.read()``, ``self.write()``,
  ``self.size``, ``self.seek()``, the spill machinery — every
  :class:`BytesIO` op works on a :class:`DataIO`.
- Path binding (``__init__(path=...)``) materializes the path's
  bytes into the buffer on :meth:`_acquire`, writes them back on
  committed :meth:`_release`.
- :class:`NestedIO` (folder-of-files formats, not single-buffer)
  inherits a degenerate :class:`BytesIO` it doesn't really use.
  This is a known structural compromise — the alternative was a
  multiple-inheritance MRO that wouldn't linearize. For
  :class:`NestedIO` we override the byte-level methods to raise.

The contract
------------

A subclass implements four methods total:

- :meth:`options_class`        — what :class:`CastOptions` subtype it
                                  consumes. Default: :class:`CastOptions`
                                  itself.
- :meth:`_read_arrow_batches`  — yield :class:`pa.RecordBatch`.
- :meth:`_write_arrow_batches` — consume an iterable of batches.
- :meth:`default_mime_type`    — what :class:`MimeType` slots it into
                                  the registry. Default ``None``
                                  means "abstract / not registered."

Two more hooks are optional, with sensible defaults:

- :meth:`to_fragment`     — project this IO into one fragment. Default
                            uses the IO's path + schema.
- :meth:`_read_fragments` — yield child fragments. Default: empty.

Lifecycle helpers
-----------------

Read and write methods funnel through :meth:`_reading_context` /
:meth:`_writing_context`. These context managers handle:

- Opening the buffer if the caller hadn't, and closing it on exit
  iff this context opened it.
- Cursor pre-positioning and post-restore (``options.read_seek``,
  ``options.write_seek``, ``options.reset_seek``).
- Truncate-before-write (``options.truncate_before_write``).
- Dirty-marking (``options.mark_dirty_on_write``).
- **Codec round-trip** — when ``self.codec`` is set, the context
  yields a transient sibling carrying the uncompressed bytes,
  writes the body's output back through the codec, and replaces
  ``self``'s payload with the compressed result. Leaves never
  see a codec-aware code path; they always operate on the yielded
  IO as if it were a plain uncompressed buffer of their format.

Public surface
--------------

Every non-abstract method (``read_arrow_table``, ``read_polars_frame``,
``write_pandas_frame``, ``write_table``, ``read_arrow_dataset``, …)
funnels through :meth:`check_options` to land on a single resolved
options instance, then dispatches to the format's ``_read_*`` /
``_write_*`` building block.

``check_options`` accepts ``options=`` plus any number of overrides;
``CastOptions.check`` handles the merge.

SaveMode + append/upsert
------------------------

:meth:`_resolve_save_mode` collapses the eight-value :class:`SaveMode`
to a four-value subset a writer can branch on
(OVERWRITE / APPEND / IGNORE / UPSERT). Class-level switches
``_SUPPORTED_APPEND`` / ``_SUPPORTED_UPSERT`` gate the modes that
aren't universally legal. :meth:`_arrow_append_via_rewrite` /
:meth:`_arrow_upsert_via_rewrite` are read-modify-write helpers for
formats whose footer-bound layout makes partial writes infeasible
(parquet, IPC) — they bottom out by recursing into
``_write_arrow_batches`` with ``mode=OVERWRITE``.
"""

from __future__ import annotations

import contextlib
from abc import abstractmethod
from collections.abc import Iterable, Iterator as AbcIterator
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
)

import pyarrow as pa
import pyarrow.compute as pc

from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.buffer import BytesIO
from yggdrasil.io.enums import MimeType, SaveMode
from yggdrasil.io.fragment import Fragment
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession


__all__ = ["DataIO"]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

O = TypeVar("O", bound=CastOptions)


# ---------------------------------------------------------------------------
# DataIO
# ---------------------------------------------------------------------------


class DataIO(BytesIO, Generic[O]):
    """Abstract tabular I/O with Arrow record batches as the primitive.

    Inherits :class:`BytesIO` so the bytes are directly addressable;
    inherits :class:`Generic[O]` so subclasses can declare their
    options-class shape. Uses :class:`ABCMeta` (via
    :func:`abstractmethod`) so concrete subclasses must implement
    ``_read_arrow_batches`` and ``_write_arrow_batches``.

    See the module docstring for the contract.
    """

    _FINAL_IO: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # SaveMode + codec class-level config
    # ------------------------------------------------------------------

    # Whether the subclass supports streaming append. Single-buffer
    # formats with a footer (parquet, IPC) generally don't — the
    # whole stream needs a rewrite. CSV/JSON-as-NDJSON can; those
    # subclasses override.
    _SUPPORTED_APPEND: ClassVar[bool] = False

    # Whether the subclass supports row-level upsert. Single-file
    # formats can't — there's no row identity. Set True only on
    # database-backed or transactional-table backends, or on the
    # rewrite-via-merge path.
    _SUPPORTED_UPSERT: ClassVar[bool] = False

    # Hint dropped into the rejection error for APPEND on formats
    # that don't support it. Subclasses override to point at the
    # right alternative.
    _APPEND_REJECTED_HINT: ClassVar[str] = (
        "use a folder-oriented writer (FolderIO / NestedIO) to "
        "accumulate files alongside instead."
    )

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

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Consume Arrow record batches and persist them.

        The single hook every other write surface routes through.
        Implementations should resolve ``options.mode`` for save
        semantics (overwrite/append/ignore/error-if-exists/upsert)
        via :meth:`_resolve_save_mode` and respect
        ``options.target_field`` for output schema control.
        """

    # ==================================================================
    # Optional protocol — fragments
    # ==================================================================

    def has_fragments(self) -> bool:
        """Whether this IO subdivides into fragments.

        ``True`` for nested IOs (folders, multi-piece formats),
        ``False`` for true leaves.
        """
        return False

    def to_fragment(self) -> Fragment:
        """Project this IO as a single :class:`Fragment`.

        Default uses the IO's path + collected schema + own mtime.
        Subclasses with cheap schema access (a parsed footer) can
        override to skip the schema collection round trip.
        """
        return Fragment(
            url=self.url,
            schema=self.collect_schema(),
            mtime=self.stat().mtime,
            io=self,
        )

    def _read_fragments(self, options: O) -> Iterator[Fragment]:
        """Yield child fragments. Default: empty (leaf IO)."""
        return iter(())

    def _write_fragments(
        self,
        fragments: Iterator[Fragment],
        options: O,
    ) -> None:
        """Consume fragments by draining each into self."""
        read_options = options.copy()

        def fetcher() -> Iterator[pa.RecordBatch]:
            for frag in fragments:
                yield from frag.io.read_arrow_batches(options=read_options)

        # Strip target so we don't double-cast (the source already
        # produced typed batches via its own options).
        self.write_arrow_batches(fetcher(), options.copy(target=None))

    # ==================================================================
    # Identity — path/buffer/url accessors
    # ==================================================================

    @property
    def url(self) -> Any:
        """Source URL. Falls back to the path's URL when bound."""
        p = self.path
        return p.url if p is not None else None

    @property
    def buffer(self) -> "BytesIO":
        """The underlying byte buffer — which is ``self``.

        Some legacy callers reach for ``io.buffer`` to get the bytes;
        this keeps them working now that :class:`DataIO` IS-A
        :class:`BytesIO`.
        """
        return self

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

    def read_arrow_batches(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches. Primary streaming read path."""
        yield from self._read_arrow_batches(
            self.check_options(options, overrides=locals())
        )

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
            options.copy(row_size=None) if row_size else options,
        )

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
        schema = options.check_target(obj=self.collect_schema()).target_schema
        arrow_schema = schema.to_arrow_schema()
        reader = pa.RecordBatchReader.from_batches(
            arrow_schema,
            self._read_arrow_batches(options),
        )
        return pds.dataset(reader, schema=arrow_schema)

    # ==================================================================
    # Polars
    # ==================================================================

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

    def read_spark_frame(
        self,
        spark: "SparkSession",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> "SparkDataFrame":
        """Materialize on the driver and build a Spark DataFrame."""
        return self._read_spark_frame(
            spark, self.check_options(options, overrides=locals())
        )

    def _read_spark_frame(
        self,
        spark: "SparkSession",
        options: O,
    ) -> "SparkDataFrame":
        return spark.createDataFrame(self._read_pandas_frame(options))

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

    # ``write_any`` is a legacy alias of :meth:`write_table`. Kept so
    # existing callers don't break; new code should use ``write_table``.
    write_any = write_table

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
            "or Iterable[pyarrow.RecordBatch]."
        )

    # ==================================================================
    # Fragments — public entry points
    # ==================================================================

    def read_fragments(
        self,
        options: "O | None" = None,
        **kwargs: Any,
    ) -> Iterator[Fragment]:
        """Enumerate child fragments. Default: empty (leaf IO)."""
        yield from self._read_fragments(
            self.check_options(options, overrides=locals())
        )

    def write_fragments(
        self,
        fragments: Iterator[Fragment],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        """Persist fragments. Default: drain each into self."""
        return self._write_fragments(
            fragments, self.check_options(options, overrides=locals())
        )

    # ==================================================================
    # SaveMode resolution
    # ==================================================================

    def _resolve_save_mode(self, mode: Any) -> SaveMode:
        """Resolve any :class:`SaveMode` to one a writer can branch on.

        Returns one of:

        - :attr:`SaveMode.OVERWRITE` — truncate and write fresh.
          Includes AUTO/TRUNCATE, IGNORE-with-empty-buffer,
          ERROR_IF_EXISTS-with-empty-buffer.
        - :attr:`SaveMode.APPEND` — only when ``_SUPPORTED_APPEND``.
        - :attr:`SaveMode.IGNORE` — buffer non-empty, caller wants
          to skip.
        - :attr:`SaveMode.UPSERT` — only when ``_SUPPORTED_UPSERT``.

        Raises :class:`ValueError` for unsupported APPEND/UPSERT
        with a subclass-specific hint, :class:`FileExistsError` for
        ERROR_IF_EXISTS on a non-empty buffer.
        """
        m = mode if isinstance(mode, SaveMode) else SaveMode.parse(mode)

        if m in (SaveMode.AUTO, SaveMode.OVERWRITE, SaveMode.TRUNCATE):
            return SaveMode.OVERWRITE

        if m is SaveMode.APPEND:
            if self._SUPPORTED_APPEND:
                return SaveMode.APPEND
            raise ValueError(
                f"{type(self).__name__} does not support SaveMode.APPEND. "
                + self._APPEND_REJECTED_HINT
            )

        if m is SaveMode.IGNORE:
            return SaveMode.IGNORE if not self.is_empty() else SaveMode.OVERWRITE

        if m is SaveMode.UPSERT:
            if self._SUPPORTED_UPSERT:
                return SaveMode.UPSERT
            raise ValueError(
                f"{type(self).__name__} does not support SaveMode.UPSERT. "
                "Single-file formats have no row identity; use a "
                "database, Iceberg, or Delta backend for upsert "
                "semantics."
            )

        if m is SaveMode.ERROR_IF_EXISTS:
            if not self.is_empty():
                raise FileExistsError(
                    f"{type(self).__name__} write with "
                    f"SaveMode.ERROR_IF_EXISTS but buffer is non-empty "
                    f"({self.size} bytes). Path: {self.path!r}"
                )
            return SaveMode.OVERWRITE

        raise ValueError(f"Unhandled SaveMode: {m!r}")

    # ==================================================================
    # Codec siblings
    # ==================================================================

    def _make_uncompressed_sibling(self) -> "DataIO":
        """Build an uncompressed sibling carrying self's bytes decompressed.

        The sibling is the same concrete class as ``self``; it gets
        ``default_mime_type()`` (no codec) as its media type so a
        downstream lookup of ``codec`` on the sibling returns
        ``None`` and any recursion through the codec branch
        terminates.
        """
        codec = self.codec
        if codec is None:
            raise RuntimeError(
                f"_make_uncompressed_sibling called on {type(self).__name__} "
                "with no codec; this is a bug in the caller."
            )

        decompressed_buf = codec.decompress(self, copy=True)
        return type(self)(
            decompressed_buf,
            media_type=type(self).default_mime_type(),
        )

    def _make_empty_sibling(self) -> "DataIO":
        """Empty sibling, no source bytes — same format minus the codec.

        Used by the write codec branch: the body fills the sibling
        with raw format bytes, then we compress on the way out.
        Deliberately not via :meth:`_make_uncompressed_sibling` —
        that decompresses self's current bytes, which for a write
        target are either empty or the previous compressed version
        we're about to overwrite.
        """
        return type(self)(
            media_type=type(self).default_mime_type(),
        )

    # ==================================================================
    # Lifecycle context managers — open/seek/codec
    # ==================================================================

    @contextlib.contextmanager
    def _reading_context(self, options: O) -> Iterator["DataIO"]:
        """Open an IO for reading; yield the IO the body should read from.

        With no codec, yields ``self``. With a codec, yields a
        transient decompressed sibling whose lifetime is bounded by
        this context — the sibling is opened on entry and closed
        (scratch buffer unlinked) on exit, including the unhappy-path
        exit where the consumer breaks out early or an exception
        propagates.

        Driven by *options*:

        - ``options.read_seek`` — cursor to seek to before the body
          runs on the yielded IO. ``None`` leaves it untouched.
          Defaults to ``0`` on CastOptions (read from byte zero).
        - ``options.reset_seek`` — restore the pre-entry cursor on
          exit (only when the IO was already open; one we opened
          ourselves is closed instead).
        """
        with contextlib.ExitStack() as stack:
            if self.codec is not None:
                target = stack.enter_context(self._make_uncompressed_sibling())
            else:
                target = self
                if not target.opened:
                    target.open()
                    stack.callback(target.close)
                elif options.reset_seek and target.seekable():
                    stack.callback(target.seek, target.tell())

            if options.read_seek is not None and target.seekable():
                target.seek(options.read_seek)

            yield target

    @contextlib.contextmanager
    def _writing_context(self, options: O) -> Iterator["DataIO"]:
        """Open an IO for writing; yield the IO the body should write to.

        With no codec, yields ``self``. With a codec, yields a
        transient uncompressed sibling — the body writes the raw
        format bytes into the sibling, and on successful exit the
        sibling's bytes are compressed back into ``self`` and
        ``self`` is marked dirty so the bound path's write-back
        fires on close.

        On exception inside the body during the codec branch,
        ``self`` is left untouched (the sibling is discarded).

        Driven by *options*:

        - ``options.truncate_before_write`` — truncate the yielded
          IO to zero before the body. Set by OVERWRITE; cleared by
          APPEND.
        - ``options.write_seek`` — cursor on the yielded IO before
          the body. ``None`` leaves it untouched, ``0`` rewinds,
          ``-1`` seeks to end (SEEK_END). APPEND sets ``-1``.
        - ``options.mark_dirty_on_write`` — if True (default), mark
          the yielded IO dirty after the body. In the codec branch
          ``self`` is additionally marked dirty after compression
          replaces its payload, regardless of this flag.
        - ``options.reset_seek`` — restore the pre-entry cursor on
          exit (only when the IO stays open).
        """
        if self.codec is not None:
            yield from self._writing_context_compressed(options)
            return

        with contextlib.ExitStack() as stack:
            if not self.opened:
                self.open()
                stack.callback(self.close)
            elif options.reset_seek and self.seekable():
                stack.callback(self.seek, self.tell())

            if options.truncate_before_write:
                self.truncate(0)

            if options.write_seek is not None and self.seekable():
                if options.write_seek < 0:
                    self.seek(0, 2)  # SEEK_END
                else:
                    self.seek(options.write_seek)

            if options.mark_dirty_on_write:
                self.mark_dirty()

            yield self

    def _writing_context_compressed(self, options: O) -> Iterator["DataIO"]:
        """Codec branch of :meth:`_writing_context`.

        Pulled out so the no-codec fast path stays flat. Yields the
        sibling the body should write to; on successful exit
        compresses sibling bytes into ``self``.
        """
        codec = self.codec
        assert codec is not None

        sibling = self._make_empty_sibling()
        with sibling:
            if options.write_seek is not None and sibling.seekable():
                if options.write_seek < 0:
                    sibling.seek(0, 2)
                else:
                    sibling.seek(options.write_seek)

            yield sibling

            sibling.seek(0)
            compressed = codec.compress(sibling)

        # Sibling closed and scratch unlinked. Replace self's payload.
        self.truncate(0)
        self.seek(0)
        self.replace_with_payload(compressed)
        self.mark_dirty()


