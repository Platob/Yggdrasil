"""Pure interface for tabular I/O over Apache Arrow record batches.

:class:`Tabular[O]` declares "I yield and consume Arrow record
batches" — nothing else. No state, no lifecycle, no holder, no
cursor. Two abstract hooks plus an options-class declaration; the
engine fan-out (Arrow / Polars / Pandas / Spark / Python-native) is
derived from those hooks.

Implementers compose :class:`Tabular` with whatever concrete
substrate they need:

- :class:`yggdrasil.io.buffer.bytes_io.BytesIO` mixes Tabular with
  :class:`Disposable` and a :class:`Holder` cursor — the default
  byte-backed implementation.
- A hypothetical :class:`SparkCatalogTabular` could mix Tabular with
  a Spark-session reference and skip byte-level concerns entirely.

The contract
------------

A leaf implements three things:

- :meth:`options_class`        — :class:`CastOptions` subtype.
- :meth:`_read_arrow_batches`  — yield :class:`pa.RecordBatch`.
- :meth:`_write_arrow_batches` — consume an iterable of batches.

Public methods route caller kwargs through :meth:`check_options` to
a single resolved options instance, then dispatch to one of the
hooks.

Why not inherit from Holder
---------------------------

Tabular is an interface, not a substrate. Most things that satisfy
"yield Arrow batches" don't have a meaningful byte representation
(a Spark catalog table, a JDBC cursor, a remote dataset service).
Forcing Holder's byte primitives into the contract would either
require those backends to fake a buffer or split them off into a
parallel hierarchy. Keeping Tabular pure lets every backend
satisfy it without lying.

Format registry
---------------

Concrete byte-backed leaves (ParquetIO, CsvIO, ArrowIPCIO, …)
declare :attr:`Tabular.mime_type` at the class level and the
:meth:`__init_subclass__` hook auto-registers them in
:data:`_TABULAR_REGISTRY`. :meth:`Tabular.for_holder` resolves a
holder's :class:`MediaType` to the right concrete leaf and
constructs ``Cls(holder=holder)``. Mirror of the ``scheme``-based
dispatch on :class:`Holder`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Iterator, TypeVar

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.data.enums import MediaType, MimeType
from yggdrasil.lazy_imports import polars_module, pyarrow_dataset_module

if TYPE_CHECKING:
    import pandas
    import polars as pl
    import pyarrow.dataset as pds
    from pyspark.sql import DataFrame as SparkDataFrame
    from yggdrasil.io.holder import Holder


__all__ = ["Tabular"]


O = TypeVar("O", bound=CastOptions)
_ChildT = TypeVar("_ChildT", bound="Tabular")


# Process-wide registry mapping :class:`MimeType` name → concrete
# :class:`Tabular` subclass. Populated by :meth:`Tabular.__init_subclass__`
# when a subclass declares :attr:`Tabular.mime_type`. Mirror of
# :data:`yggdrasil.io.holder._HOLDER_SCHEMES` — same shape, same
# rules: declarative class-level discriminator, populated at import
# time, looked up by the factory classmethods.
_TABULAR_REGISTRY: "dict[str, type[Tabular]]" = {}


class Tabular(ABC, Generic[O]):
    """Pure interface — Arrow record-batch source/sink + engine fan-out.

    No state, no lifecycle, with the single exception of a
    :attr:`parent` back-pointer for tree-shaped sources (folders,
    archives, partitioned tables) where a child Tabular wants to
    walk back up to whatever yielded it. :meth:`adopt_child` is the
    parent-stamp helper aggregators call when handing back a child.

    Concrete implementers add whatever substrate they need (a
    holder + cursor for byte-backed shapes, a session reference
    for catalog-backed shapes, etc.) and override the two batch
    hooks. Format-specific leaves (ParquetIO, CsvIO, ArrowIPCIO, …)
    additionally declare :attr:`mime_type` to register against the
    process-wide :data:`_TABULAR_REGISTRY` — :meth:`for_holder` uses
    that registry to dispatch from a holder's :class:`MediaType` to
    the right concrete leaf at runtime.
    """

    #: Format identity for the registry. Subclasses set to a concrete
    #: :class:`MimeType` (``MimeTypes.PARQUET``, ``MimeTypes.CSV``, …)
    #: to claim that mime in :data:`_TABULAR_REGISTRY`. ``None`` (the
    #: abstract default) opts out of registration — :class:`Tabular`
    #: itself and intermediate abstracts (:class:`BytesIO`,
    #: :class:`NestedIO`) leave it unset so they don't shadow the
    #: real format leaves. Mirrors :attr:`Holder.scheme`.
    mime_type: "ClassVar[MimeType | None]" = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-register concrete subclasses keyed on :attr:`mime_type`."""
        super().__init_subclass__(**kwargs)
        mt = cls.mime_type
        if mt is None:
            return
        key = mt.name
        existing = _TABULAR_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise RuntimeError(
                f"Duplicate Tabular mime_type {mt.value!r}: "
                f"{cls.__name__} clashes with {existing.__name__}. "
                "If the override is intentional, clear the slot first "
                "via _TABULAR_REGISTRY.pop(...) at module-load time."
            )
        _TABULAR_REGISTRY[key] = cls

    def __init__(self, *, parent: "Tabular | None" = None, **kwargs: Any) -> None:
        """Initialize the parent slot.

        Subclasses chain ``super().__init__(**kwargs)`` and don't
        otherwise touch :attr:`parent` — aggregators set it via
        :meth:`adopt_child` when they yield / mint the child.
        """
        super().__init__(**kwargs)
        self.parent: "Tabular | None" = parent

    # ==================================================================
    # Parent / child linkage
    # ==================================================================

    def adopt_child(self, child: "_ChildT") -> "_ChildT":
        """Stamp ``child.parent = self`` and return *child*.

        Used by aggregator Tabulars (folders, archives, partitioned
        tables) inside their child-yielding hooks (``iter_children``,
        ``make_child``) so consumers can walk back up the tree
        without the aggregator scattering the same line of attribute
        assignment across every implementation::

            for entry in self.path.iterdir():
                yield self.adopt_child(LeafIO(holder=entry))

        Re-adopting an already-attached child is fine — the slot
        is just overwritten. No checks, no ceremony.
        """
        child.parent = self
        return child

    # ==================================================================
    # Format registry — MediaType → Tabular subclass dispatch
    # ==================================================================

    @classmethod
    def class_for_media_type(
        cls,
        media_type: "MediaType | MimeType | str | Any",
        *,
        default: Any = ...,
    ) -> "type[Tabular]":
        """Resolve a :class:`MediaType` (or coercible) to its Tabular class.

        Looks up :attr:`MediaType.mime_type`'s name in
        :data:`_TABULAR_REGISTRY`. Codec is orthogonal — Parquet
        compressed with zstd or snappy still resolves to
        :class:`ParquetIO`; the codec layer is the holder's concern.

        Returns *default* on miss when supplied; otherwise raises
        :class:`KeyError` with a list of registered names.
        """
        mt = MediaType.from_(media_type, default=None)
        if mt is None:
            if default is ...:
                raise KeyError(
                    f"Cannot coerce {media_type!r} to a MediaType "
                    "for Tabular registry lookup."
                )
            return default

        hit = _TABULAR_REGISTRY.get(mt.mime_type.name)
        if hit is not None:
            return hit

        if default is ...:
            raise KeyError(
                f"No Tabular registered for {mt.mime_type.value!r}. "
                f"Registered: {sorted(_TABULAR_REGISTRY)}."
            )
        return default

    @classmethod
    def for_holder(
        cls,
        holder: "Holder",
        *,
        media_type: "MediaType | MimeType | str | None" = None,
        default: Any = ...,
        **kwargs: Any,
    ) -> "Tabular":
        """Build the right :class:`Tabular` subclass for *holder*.

        Resolution order for the format discriminator:

        1. The explicit *media_type* kwarg, when supplied.
        2. ``holder.stat().media_type`` — set by the holder from its
           URL extension, magic-byte sniff, or content-type header.

        The resolved class is instantiated as ``Cls(holder=holder,
        **kwargs)`` — every registered Tabular leaf is expected to
        accept ``holder=`` (true for :class:`BytesIO` subclasses,
        which is what the registry actually contains).

        On lookup miss, falls back to *default* when supplied. With
        no default, raises :class:`KeyError`.
        """
        mt = media_type
        if mt is None:
            stats = getattr(holder, "stat", None)
            if callable(stats):
                mt = getattr(stats(), "media_type", None)

        if mt is None:
            if default is ...:
                raise KeyError(
                    f"No media_type on {holder!r}; pass media_type= "
                    "explicitly or seed the holder's IOStats."
                )
            return default

        target = cls.class_for_media_type(mt, default=default)
        if target is default and default is not ...:
            return default
        return target(holder=holder, **kwargs)

    @classmethod
    def registered_classes(cls) -> "dict[str, type[Tabular]]":
        """Snapshot of the registry — debugging / introspection only."""
        return dict(_TABULAR_REGISTRY)

    # ==================================================================
    # Options
    # ==================================================================

    @classmethod
    def options_class(cls) -> "type[O]":
        """The :class:`CastOptions` subclass this implementer consumes.

        Default :class:`CastOptions`. Format-specific leaves with
        their own knobs (Parquet compression, CSV delimiter, …)
        override.
        """
        return CastOptions  # type: ignore[return-value]

    @classmethod
    def check_options(
        cls,
        options: "O | None" = None,
        overrides: "dict | None" = None,
        **kwargs: Any,
    ) -> O:
        """Validate and merge caller kwargs into a resolved options.

        Canonical pattern: a public method passes ``overrides=locals()``
        and the ``...``-defaulted entries are stripped, the rest merged.
        """
        if overrides:
            overrides = {k: v for k, v in overrides.items() if v is not ...}
            overrides.pop("self", None)
            options = overrides.pop("options", options)
            kwargs.update(overrides)
            kwargs.update(kwargs.pop("kwargs", {}))
        return cls.options_class().check(options, **kwargs)

    # ==================================================================
    # Abstract batch hooks — the two things every implementer overrides
    # ==================================================================

    @abstractmethod
    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Yield Arrow record batches from the underlying source."""

    @abstractmethod
    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Consume Arrow record batches and persist them."""

    # ==================================================================
    # Schema inspection
    # ==================================================================

    def collect_schema(self, options: "O | None" = None, **kwargs: Any) -> Schema:
        return self._collect_schema(self.check_options(options, overrides=locals()))

    def _collect_schema(self, options: O) -> Schema:
        """First batch's schema by default; merged across batches when
        ``options.safe`` is set, so heterogeneous sources surface a
        union schema instead of "whichever batch came first"."""
        batches = self._read_arrow_batches(options)
        first = next(iter(batches), None)
        if first is None:
            return Schema.empty()
        schema = Schema.from_arrow(first.schema)
        if not getattr(options, "safe", False):
            return schema
        for batch in batches:
            schema = schema.merge_with(
                Schema.from_arrow(batch.schema), inplace=True,
            )
        return schema

    # ==================================================================
    # Static helpers
    # ==================================================================

    @staticmethod
    def _normalize_records(data: Iterable[dict]) -> "list[dict]":
        """Backfill every row to the union of keys seen across all rows.

        :meth:`pa.Table.from_pylist` infers schema from the first row
        and silently drops keys that show up only later; callers that
        build rows incrementally (e.g. JSON parsers) route through
        here first.
        """
        rows = list(data) if not isinstance(data, list) else data
        if not rows:
            return []

        all_keys: "dict[str, None]" = {}
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
    # Arrow surface
    # ==================================================================

    def read_arrow_batches(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> Iterator[pa.RecordBatch]:
        yield from self._read_arrow_batches(
            self.check_options(options, overrides=locals())
        )

    def read_arrow_table(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> pa.Table:
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

    def read_arrow_batch_reader(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pa.RecordBatchReader":
        return self._read_arrow_batch_reader(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_batch_reader(self, options: O) -> "pa.RecordBatchReader":
        schema = options.check_target(obj=self.collect_schema).merged_schema
        return pa.RecordBatchReader.from_batches(
            schema.to_arrow_schema(), self._read_arrow_batches(options),
        )

    def read_arrow_dataset(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pds.Dataset":
        return self._read_arrow_dataset(
            self.check_options(options, overrides=locals())
        )

    def _read_arrow_dataset(self, options: O) -> "pds.Dataset":
        pds = pyarrow_dataset_module()
        reader = self._read_arrow_batch_reader(options)
        return pds.dataset(reader, schema=reader.schema)

    def write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_arrow_batches(
            batches, self.check_options(options, overrides=locals()),
        )

    def write_arrow_table(
        self, table: pa.Table, options: "O | None" = None, **kwargs: Any,
    ) -> None:
        self._write_arrow_table(
            table, self.check_options(options, overrides=locals()),
        )

    def _write_arrow_table(self, table: pa.Table, options: O) -> None:
        row_size = getattr(options, "row_size", None) or None
        self._write_arrow_batches(
            table.to_batches(max_chunksize=row_size),
            options.check_source(table).copy(row_size=None) if row_size else options,
        )

    # ==================================================================
    # Polars
    # ==================================================================

    def read_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.DataFrame":
        return self._read_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frame(self, options: O) -> "pl.DataFrame":
        return polars_module().from_arrow(self._read_arrow_table(options))  # type: ignore[return-value]

    def read_polars_frames(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[pl.DataFrame]":
        """One Polars frame per Arrow batch — streaming."""
        yield from self._read_polars_frames(
            self.check_options(options, overrides=locals())
        )

    def _read_polars_frames(self, options: O) -> "Iterator[pl.DataFrame]":
        pl = polars_module()
        for batch in self._read_arrow_batches(options):
            yield pl.from_arrow(batch, rechunk=False)  # type: ignore[misc]

    def scan_polars_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pl.LazyFrame":
        return self._scan_polars_frame(
            self.check_options(options, overrides=locals())
        )

    def _scan_polars_frame(self, options: O) -> "pl.LazyFrame":
        return polars_module().scan_pyarrow_dataset(self._read_arrow_dataset(options))

    def write_polars_frame(
        self,
        frame: "pl.DataFrame | pl.LazyFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_polars_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_polars_frame(
        self, frame: "pl.DataFrame | pl.LazyFrame", options: O,
    ) -> None:
        pl = polars_module()
        if isinstance(frame, pl.LazyFrame):
            frame = frame.collect()
        if frame.height == 0:
            return

        row_size = getattr(options, "row_size", None) or 0
        byte_size = getattr(options, "byte_size", None) or 0
        if row_size > 0:
            chunks: "Iterator[pl.DataFrame]" = frame.iter_slices(n_rows=row_size)
        elif byte_size > 0:
            total = frame.estimated_size(unit="b")
            if total == 0:
                chunks = iter((frame,))
            else:
                rows_per_chunk = max(1, int(frame.height * byte_size / total))
                chunks = frame.iter_slices(n_rows=rows_per_chunk)
        else:
            chunks = iter((frame,))

        def gen() -> Iterator[pa.RecordBatch]:
            for f in chunks:
                yield from f.to_arrow().to_batches()

        self._write_arrow_batches(gen(), options)

    # ==================================================================
    # Pandas
    # ==================================================================

    def read_pandas_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "pandas.DataFrame":
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
        self._write_pandas_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_pandas_frame(self, frame: "pandas.DataFrame", options: O) -> None:
        import pandas as pd
        is_default_range = (
            isinstance(frame.index, pd.RangeIndex) and frame.index.name is None
        )
        table = pa.Table.from_pandas(frame, preserve_index=not is_default_range)
        self._write_arrow_table(table, options)

    # ==================================================================
    # Spark — driver-side materialize, no streaming spill
    # ==================================================================

    def read_spark_frame(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "SparkDataFrame":
        return self._read_spark_frame(
            self.check_options(options, overrides=locals())
        )

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        from yggdrasil.environ import PyEnv
        spark = PyEnv.spark_session(create=True)
        return options.cast_spark(
            spark.createDataFrame(self._read_arrow_table(options))
        )

    def write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_spark_frame(
            frame, self.check_options(options, overrides=locals()),
        )

    def _write_spark_frame(self, frame: "SparkDataFrame", options: O) -> None:
        to_arrow = getattr(frame, "toArrow", None)
        if to_arrow is not None:
            self._write_arrow_table(to_arrow(), options)
            return
        self._write_pandas_frame(frame.toPandas(), options)

    # ==================================================================
    # Python-native
    # ==================================================================

    def read_pylist(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "list[dict]":
        return self._read_pylist(self.check_options(options, overrides=locals()))

    def _read_pylist(self, options: O) -> "list[dict]":
        return self._read_arrow_table(options).to_pylist()

    def write_pylist(
        self,
        data: Iterable[dict],
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pylist(data, self.check_options(options, overrides=locals()))

    def _write_pylist(self, data: Iterable[dict], options: O) -> None:
        rows = self._normalize_records(data)
        if not rows:
            return
        self._write_arrow_table(pa.Table.from_pylist(rows), options)

    def read_pydict(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "dict[str, list]":
        return self._read_pydict(self.check_options(options, overrides=locals()))

    def _read_pydict(self, options: O) -> "dict[str, list]":
        return self._read_arrow_table(options).to_pydict()

    def write_pydict(
        self,
        data: "dict[str, list]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_pydict(data, self.check_options(options, overrides=locals()))

    def _write_pydict(self, data: "dict[str, list]", options: O) -> None:
        self._write_arrow_table(pa.Table.from_pydict(data), options)

    def read_record_iterator(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Mapping[str, Any]]":
        """Stream rows as plain ``dict``. True streaming — the full
        table never materializes; ``batch.to_pylist()`` does the
        column→row rotation in pyarrow C++ once per batch."""
        return self._read_record_iterator(
            self.check_options(options, overrides=locals())
        )

    def _read_record_iterator(self, options: O) -> "Iterator[Mapping[str, Any]]":
        for batch in self._read_arrow_batches(options):
            yield from batch.to_pylist()

    def read_records(
        self, options: "O | None" = None, **kwargs: Any,
    ) -> "Iterator[Any]":
        """Stream rows as :class:`yggdrasil.data.record.Record`. Lower
        per-row allocation than :meth:`read_pylist` for stable-schema
        sources — the underlying :class:`Schema` is materialized once
        and shared by reference across every record."""
        return self._read_records(
            self.check_options(options, overrides=locals())
        )

    def _read_records(self, options: O) -> "Iterator[Any]":
        from yggdrasil.data.record import Record
        yield from Record.from_arrow_batches(self._read_arrow_batches(options))

    def write_records(
        self,
        records: "Iterable[Any]",
        options: "O | None" = None,
        **kwargs: Any,
    ) -> None:
        self._write_records(
            records, self.check_options(options, overrides=locals()),
        )

    def _write_records(self, records: "Iterable[Any]", options: O) -> None:
        """Bucket records by ``options.row_size`` and delegate to
        :meth:`_write_arrow_batches`. The first record's schema becomes
        the writer's target — heterogeneous-schema streams are silently
        re-aligned via dict-order iteration, so callers that need
        strictness should pre-validate. Subclasses with a row-native
        sink (SQL bulk-insert, Spark createDataFrame) should override
        to skip the row → Arrow round-trip.
        """
        from yggdrasil.data.record import Record

        chunk_size = max(1, getattr(options, "row_size", None) or 1024)
        chunk_rows: "list[dict]" = []
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