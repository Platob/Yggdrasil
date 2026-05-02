"""Arrow conversion entry points with native fast-path preference.

Design principles
-----------------

1. **Cast in the source engine, then serialize.** Polars/Spark/pandas
   each have engine-native cast machinery exposed on
   :class:`CastOptions` as :meth:`cast_polars` / :meth:`cast_spark` /
   :meth:`cast_pandas`. We use those *before* the Arrow conversion —
   it's faster (no round-trip rebuild), preserves engine-specific
   dtypes (polars Categoricals, pandas ExtensionArrays), and lets
   lazy engines push the cast into their query plan. Arrow-side
   casting is reserved for sources that are already Arrow or have no
   native cast path.

2. **Bulk over iterate.** Vectorized native methods over per-row
   iteration. The streaming entry point is reserved for sources that
   are themselves streams, or for materialized tables that need
   chunking via ``options.row_size`` / ``options.byte_size``.

3. **Bind source schemas, don't peek.** When we infer a source
   schema, we bind it onto ``options.source_field`` so it propagates
   downstream and drives :meth:`CastOptions.need_cast` without
   re-inference.

4. **Emit the merged schema.** When both source and target are bound,
   the output schema is :attr:`CastOptions.merged_schema` —
   reconciled per ``schema_mode``. ``RecordBatchReader`` /
   iterator declarations use this.

5. **Honor every options knob.** ``column_names`` projects on the way
   in; ``arrow_memory_pool`` threads through pyarrow allocators;
   ``safe`` flows into engine cast methods; ``row_size``/``byte_size``
   drive output chunking. ``byte_size`` is resolved against actual
   ``nbytes`` measurements — exact for bounded objects (Table,
   RecordBatch), cumulative running average for streams.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Iterator
from dataclasses import is_dataclass
from typing import Optional, Union, Any, Generator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.schema import Schema, Field
from yggdrasil.lazy_imports import (
    pandas_module,
    spark_sql_module,
    polars_module,
)
from yggdrasil.pickle.serde import ObjectSerde
from .python_defaults import default_arrow_scalar

__all__ = [
    "ArrowDataType",
    "any_to_arrow_scalar",
    "any_to_arrow_field",
    "any_to_arrow_schema",
    "any_to_arrow_table",
    "any_to_arrow_record_batch",
    "any_to_arrow_batch_iterator",
    "any_to_arrow_record_batch_reader",
    "cast_arrow_array",
    "cast_arrow_scalar",
    "cast_arrow_tabular",
    "cast_arrow_record_batch_reader",
    "default_arrow_scalar",
]

logger = logging.getLogger(__name__)

#: Union alias covering all Arrow DataType variants used throughout this module.
ArrowDataType = Union[
    pa.DataType,
    pa.Decimal128Type,
    pa.TimestampType,
    pa.DictionaryType,
    pa.MapType,
    pa.StructType,
    pa.FixedSizeListType,
]


# ---------------------------------------------------------------------------
# Source-binding & projection helpers
# ---------------------------------------------------------------------------


def _bind_source(options: CastOptions, obj: Any) -> CastOptions:
    """Bind an inferred source schema onto ``options.source_field``.

    Idempotent. Coercion failures are swallowed — "couldn't infer"
    is not a hard error here.
    """
    if options.source_field is not None:
        return options
    try:
        return options.check_source(obj=obj, copy=False)
    except Exception:
        return options


def _resolve_projection(options: CastOptions) -> Optional[list[str]]:
    """Resolve the effective column projection.

    ``column_names`` is auto-populated from ``target_field.names`` by
    :meth:`CastOptions.__post_init__`, so explicit and implicit
    projection collapse into one source.
    """
    return list(options.column_names) if options.column_names else None


def _project(table: pa.Table, options: CastOptions) -> pa.Table:
    """Apply the resolved projection to a ``pa.Table``. Zero-copy."""
    cols = _resolve_projection(options)
    if not cols:
        return table
    if list(table.column_names) == cols:
        return table
    present = [c for c in cols if c in table.column_names]
    return table.select(present) if present else table


def _resolve_row_size(options: CastOptions, nbytes: int, num_rows: int) -> CastOptions:
    """Derive effective row_size from byte_size using measured nbytes.

    Exact for bounded objects (whole-table or whole-batch ``nbytes``).
    Returns options with byte_size stripped — downstream sees rows only.
    """
    if not options.byte_size or num_rows <= 0 or nbytes <= 0:
        return options.copy(byte_size=None) if options.byte_size else options

    bytes_per_row = max(1, nbytes // num_rows)
    derived = max(1, options.byte_size // bytes_per_row)
    effective = min(options.row_size, derived) if options.row_size else derived
    return options.copy(row_size=effective, byte_size=None)


# ---------------------------------------------------------------------------
# Engine-specific bulk converters — cast in-engine, then serialize
# ---------------------------------------------------------------------------


def _pandas_to_arrow(obj: Any, options: CastOptions) -> tuple[pa.Table, CastOptions]:
    """Convert pandas DataFrame/Series to Arrow.

    1. Bind source.
    2. Project.
    3. **Cast in-engine** via :meth:`CastOptions.cast_pandas` —
       handles dtype conversions and ExtensionArrays before Arrow
       sees them. Avoids the "convert via numpy → lose precision →
       fail to cast" trap that hits when pandas extension dtypes
       (Int64, Float64, string[pyarrow]) hit ``Table.from_pandas``.
    4. Serialize via ``Table.from_pandas(memory_pool=, safe=)``.
    """
    pd = pandas_module()

    if isinstance(obj, pd.Series):
        obj = obj.to_frame()

    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Unsupported pandas object: %s" % (obj,))

    options = _bind_source(options, obj)

    projection = _resolve_projection(options)
    if projection:
        keep = [c for c in projection if c in obj.columns]
        if keep and list(obj.columns) != keep:
            obj = obj.loc[:, keep]

    if options.target_field is not None and options.need_cast():
        obj = options.cast_pandas(obj)

    table = pa.Table.from_pandas(
        obj,
        preserve_index=bool(obj.index.name),
        memory_pool=options.arrow_memory_pool,
        safe=options.safe,
    )
    return table, options.copy(target_field=None)


def _spark_to_arrow(obj: Any, options: CastOptions) -> tuple[pa.Table, CastOptions]:
    """Convert Spark DataFrame to Arrow.

    1. Bind source from Spark's ``StructType`` (cheap, no materialization).
    2. Project via ``DataFrame.select`` — pushed into the physical plan.
    3. **Cast in-engine** via :meth:`CastOptions.cast_spark` — the
       cast becomes part of the Spark plan, fused with projection
       and any pushdown filters. Vastly preferable to Arrow-side
       cast, which would materialize the whole frame before casting.
    4. Trigger ``toArrow()``.
    """
    pyspark_sql = spark_sql_module()

    if not isinstance(obj, pyspark_sql.DataFrame):
        raise TypeError("Unsupported Spark object: %s" % (obj,))

    options = _bind_source(options, obj)

    projection = _resolve_projection(options)
    if projection:
        keep = [c for c in projection if c in obj.columns]
        if keep and list(obj.columns) != keep:
            obj = obj.select(*keep)

    if options.target_field is not None and options.need_cast():
        obj = options.cast_spark(obj)

    return obj.toArrow(), options.copy(target_field=None)


# ---------------------------------------------------------------------------
# Any-to-Arrow table / batch — bulk path
# ---------------------------------------------------------------------------


@register_converter(Any, pa.Table)
def any_to_arrow_table(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Table:
    """Convert *any* supported object to a ``pa.Table``, with engine-native
    casting applied upstream when possible.

    Casting strategy:

    * **Arrow inputs** (Table/RecordBatch/Array/Reader) — cast on the
      Arrow side via :meth:`CastOptions.cast_arrow_tabular`, with the
      ``need_cast`` skip optimization.
    * **Pandas / Spark / Polars inputs** — cast in-engine first via
      ``cast_pandas`` / ``cast_spark`` / ``cast_polars``, then
      serialize to Arrow. The serialized table needs no further cast.
    * **Generic Python inputs** — wrapped via ``pl.DataFrame(obj)``
      to get a polars cast path.
    """
    options = CastOptions.check(options)

    if isinstance(obj, pa.Table):
        options = _bind_source(options, obj)
        table = _project(obj, options)

    elif isinstance(obj, pa.RecordBatch):
        options = _bind_source(options, obj)
        table = _project(pa.Table.from_batches([obj]), options)

    elif isinstance(obj, (pa.Array, pa.ChunkedArray)):
        name = (
            options.target_field.name
            if options.target_field and options.target_field.name
            else "value"
        )
        table = pa.table({name: obj})
        options = _bind_source(options, table)

    elif isinstance(obj, pa.RecordBatchReader):
        options = _bind_source(options, obj.schema)
        table = _project(obj.read_all(), options)

    elif isinstance(obj, (Generator, Iterator)):
        batches = list(_stream_from_iterator(iter(obj), options))
        table = pa.Table.from_batches(batches)

    elif isinstance(obj, (list, tuple)):
        if not obj:
            merged_schema = options.merged_schema or Schema.empty()
            return pa.Table.from_batches([], schema=merged_schema.to_arrow_schema())

        try:
            table = pa.Table.from_pylist(obj)
        except Exception:
            batches = list(_stream_from_iterator(iter(obj), options))
            table = pa.Table.from_batches(batches)

    else:
        namespace = ObjectSerde.full_namespace(obj)

        if namespace.startswith("pandas."):
            table, options = _pandas_to_arrow(obj, options)
        elif namespace.startswith("pyspark."):
            table, options = _spark_to_arrow(obj, options)
        elif namespace.startswith("polars."):
            table, options = _polars_to_arrow(obj, options)
        else:
            pl = polars_module()
            table, options = _polars_to_arrow(pl.DataFrame(obj), options)

    return options.cast_arrow_tabular(table)


@register_converter(Any, pa.RecordBatch)
def any_to_arrow_record_batch(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatch:
    """Convert to a single ``pa.RecordBatch``."""
    options = CastOptions.check(options)

    if isinstance(obj, pa.RecordBatch):
        options = _bind_source(options, obj)
        return options.cast_arrow_tabular(obj)

    table = any_to_arrow_table(obj, options)
    batches = table.to_batches()
    if not batches:
        return pa.RecordBatch.from_pylist([], schema=table.schema)
    if len(batches) == 1:
        return batches[0]
    return table.combine_chunks().to_batches()[0]


# ---------------------------------------------------------------------------
# Streaming entry points
# ---------------------------------------------------------------------------


@register_converter(Any, Iterator[pa.RecordBatch])
def any_to_arrow_batch_iterator(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> Iterator[pa.RecordBatch]:
    """Convert *any* supported object to a lazy iterator of ``pa.RecordBatch``.

    For Polars LazyFrame and Spark DataFrame, the engine-native cast
    is applied before the streaming serialization — the iterator
    yields already-cast batches with no Arrow-side rework.
    """
    options = CastOptions.check(options)

    if isinstance(obj, pa.Table):
        options = _bind_source(options, obj)
        return _stream_from_table(obj, options, already_cast=False)

    if isinstance(obj, pa.RecordBatch):
        options = _bind_source(options, obj)
        casted = options.cast_arrow_tabular(obj)
        options = _resolve_row_size(options, casted.nbytes, casted.num_rows)
        return _slice_batch(casted, options.row_size)

    if isinstance(obj, pa.RecordBatchReader):
        options = _bind_source(options, obj.schema)
        return _stream_from_reader(obj, options)

    namespace = ObjectSerde.full_namespace(obj)

    if namespace.startswith("pyspark."):
        pyspark_sql = spark_sql_module()
        if isinstance(obj, pyspark_sql.DataFrame):
            options = _bind_source(options, obj)
            projection = _resolve_projection(options)
            if projection:
                keep = [c for c in projection if c in obj.columns]
                if keep and list(obj.columns) != keep:
                    obj = obj.select(*keep)

            # In-engine cast before streaming — fuses into Spark plan.
            if options.target_field is not None and options.need_cast():
                obj = options.cast_spark(obj)
                options = options.copy(target_field=None)

            to_iter = getattr(obj, "toArrowBatchIterator", None)
            if callable(to_iter):
                return _stream_from_iterator(iter(to_iter()), options)
            # No batch-iter API — fall through to generic bulk fallback.

    if namespace.startswith("polars."):
        pl = polars_module()
        if isinstance(obj, pl.LazyFrame):
            return _polars_lazy_to_batch_iterator(obj, options)
        # Eager polars (DataFrame/Series) falls through to the bulk
        # path — no streaming benefit since data is already materialized.

    if isinstance(obj, (Generator, Iterator)):
        return _stream_from_iterator(obj, options)

    if isinstance(obj, (list, tuple)):
        if obj:
            return _stream_from_iterator(iter(obj), options)
        return iter(())

    # Generic fallback — bulk-convert (cast applied), then slice.
    table = any_to_arrow_table(obj, options)
    return _stream_from_table(table, options, already_cast=True)


def _stream_from_reader(
    reader: pa.RecordBatchReader,
    options: CastOptions,
) -> Iterator[pa.RecordBatch]:
    """Cast each upstream batch on the Arrow side, then re-chunk.

    Arrow-side cast here because a ``RecordBatchReader`` is already
    Arrow — there's no upstream engine to push the cast into. The
    cast stream feeds :func:`_stream_from_arrow_batches` for
    cumulative byte-aware re-chunking.
    """
    needs_cast = options.target_field is not None and options.need_cast()

    def _cast_stream():
        for batch in reader:
            yield options.cast_arrow_tabular(batch) if needs_cast else batch

    yield from _stream_from_arrow_batches(_cast_stream(), options)


def _stream_from_iterator(
    items: Iterator[Any],
    options: CastOptions,
) -> Iterator[pa.RecordBatch]:
    """Lazily flatten + cast a stream of mixed tabular objects.

    Per-item shape may be Arrow (cast Arrow-side) or engine-native
    (route through ``any_to_arrow_table`` which dispatches in-engine).
    Flattened cast batches feed :func:`_stream_from_arrow_batches` for
    cumulative byte-aware re-chunking — only one coalescer in the
    chain, regardless of nesting depth.
    """

    def _flatten(it: Iterator[Any], opts: CastOptions, bound: bool):
        for item in it:
            if not bound and isinstance(item, (pa.RecordBatch, pa.Table)):
                opts = _bind_source(opts, item)
                bound = True

            if isinstance(item, pa.RecordBatch):
                yield opts.cast_arrow_tabular(item)
            elif isinstance(item, pa.Table):
                casted = opts.cast_arrow_tabular(item)
                yield from casted.to_batches()
            elif isinstance(item, (Generator, Iterator)):
                yield from _flatten(item, opts, bound)
            elif isinstance(item, (list, tuple)):
                yield from _flatten(iter(item), opts, bound)
            else:
                # Engine-native item — any_to_arrow_table casts in-engine.
                sub = any_to_arrow_table(item, opts)
                yield from sub.to_batches()

    yield from _stream_from_arrow_batches(
        _flatten(items, options, options.source_field is not None),
        options,
    )


def _stream_from_table(
    table: pa.Table,
    options: CastOptions,
    *,
    already_cast: bool,
) -> Iterator[pa.RecordBatch]:
    """Slice a table into target-sized batches, casting per-batch when needed.

    ``byte_size`` is resolved against the table's actual ``nbytes``
    (exact, single-pass) and stripped — downstream sees only row_size.
    """
    options = _resolve_row_size(options, table.nbytes, table.num_rows)

    if already_cast or options.target_field is None:
        yield from table.to_batches(max_chunksize=options.row_size)
        return

    options = _bind_source(options, table)
    if not options.need_cast():
        yield from table.to_batches(max_chunksize=options.row_size)
        return

    for batch in table.to_batches(max_chunksize=options.row_size):
        yield options.cast_arrow_tabular(batch)


def _stream_from_arrow_batches(
    batches: Iterator[pa.RecordBatch],
    options: CastOptions,
) -> Iterator[pa.RecordBatch]:
    """Re-chunk a stream of already-cast Arrow batches to target size.

    Passthrough when no sizing constraint. Otherwise the row target is
    derived from a cumulative bytes/rows running average — refined as
    more batches flow through — and batches are coalesced/sliced toward
    that target. ``byte_size``, once resolved into a row target each
    pass, is honored continuously; ``row_size`` (if set) is a hard cap.
    """
    if not options.row_size and not options.byte_size:
        yield from batches
        return

    user_row_cap = options.row_size
    byte_size = options.byte_size

    cum_bytes = 0
    cum_rows = 0

    def _current_target() -> Optional[int]:
        if not byte_size:
            return user_row_cap
        if cum_rows <= 0 or cum_bytes <= 0:
            return user_row_cap
        bytes_per_row = max(1, cum_bytes // cum_rows)
        derived = max(1, byte_size // bytes_per_row)
        return min(user_row_cap, derived) if user_row_cap else derived

    buffer: list[pa.RecordBatch] = []
    buffered_rows = 0

    def _ingest(batch):
        nonlocal buffer, buffered_rows, cum_bytes, cum_rows
        cum_bytes += batch.nbytes
        cum_rows += batch.num_rows
        target = _current_target()

        if target is None:
            # Estimation degenerated; pass through.
            yield batch
            return

        # Oversized incoming, nothing buffered → zero-copy slice direct.
        if batch.num_rows >= target and not buffer:
            yield from _slice_batch(batch, target)
            return

        buffer.append(batch)
        buffered_rows += batch.num_rows
        if buffered_rows < target:
            return

        combined = pa.Table.from_batches(buffer).to_batches(max_chunksize=target)
        for b in combined[:-1]:
            yield b
        tail = combined[-1]
        if tail.num_rows == target:
            yield tail
            buffer = []
            buffered_rows = 0
        else:
            buffer = [tail]
            buffered_rows = tail.num_rows

    for batch in batches:
        yield from _ingest(batch)

    if buffer:
        target = _current_target() or buffered_rows
        yield from pa.Table.from_batches(buffer).to_batches(max_chunksize=target)


def _slice_batch(
    batch: pa.RecordBatch,
    chunksize: Optional[int],
) -> Iterator[pa.RecordBatch]:
    """Slice a batch to ``chunksize`` rows. Zero-copy."""
    if chunksize is None or batch.num_rows <= chunksize:
        yield batch
        return
    for offset in range(0, batch.num_rows, chunksize):
        yield batch.slice(offset, chunksize)


@register_converter(Any, pa.RecordBatchReader)
def any_to_arrow_record_batch_reader(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Wrap ``any_to_arrow_batch_iterator`` behind a ``RecordBatchReader``.

    Output schema: ``merged_schema`` → ``target_schema`` →
    first-batch peek.
    """
    options = CastOptions.check(options)

    if isinstance(obj, pa.RecordBatchReader):
        return cast_arrow_record_batch_reader(obj, options)

    iterator = any_to_arrow_batch_iterator(obj, options)

    merged_schema = options.merged_schema
    if merged_schema is not None:
        return pa.RecordBatchReader.from_batches(merged_schema.to_arrow_schema(), iterator)

    try:
        first = next(iterator)
    except StopIteration:
        return pa.RecordBatchReader.from_batches(pa.schema([]), iter(()))

    def _chained(head=first, tail=iterator):
        yield head
        yield from tail

    return pa.RecordBatchReader.from_batches(first.schema, _chained())


# ---------------------------------------------------------------------------
# Scalar casting
# ---------------------------------------------------------------------------


@register_converter(Any, pa.Scalar)
def any_to_arrow_scalar(
    scalar: Any,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Convert a Python value to an Arrow scalar, then cast to target type."""
    options = CastOptions.check(options)

    if isinstance(scalar, pa.Scalar):
        return cast_arrow_scalar(scalar, options)

    target_field = options.target_field

    if scalar is None:
        return default_arrow_scalar(
            target_field,
            nullable=True if target_field is None else target_field.nullable,
        )

    if isinstance(scalar, enum.Enum):
        scalar = scalar.value

    if is_dataclass(scalar):
        scalar = dataclasses.asdict(scalar)

    if target_field is None:
        return pa.scalar(scalar)

    try:
        scalar = pa.scalar(scalar, type=target_field.type)
    except pa.ArrowInvalid:
        scalar = pa.scalar(scalar)

    return cast_arrow_scalar(scalar, options)


@register_converter(pa.Scalar, pa.Scalar)
def cast_arrow_scalar(
    scalar: pa.Scalar,
    options: Optional[CastOptions] = None,
) -> pa.Scalar:
    """Cast an Arrow scalar via the array path."""
    options = CastOptions.check(options)
    if options.target_field is None:
        return scalar
    arr = pa.array([scalar])
    return cast_arrow_array(arr, options)[0]


# ---------------------------------------------------------------------------
# Array / Tabular casting — thin wrappers
# ---------------------------------------------------------------------------


@register_converter(pa.Array, pa.Array)
@register_converter(pa.ChunkedArray, pa.ChunkedArray)
def cast_arrow_array(
    array: Union[pa.ChunkedArray, pa.Array],
    options: Optional[CastOptions] = None,
) -> Union[pa.ChunkedArray, pa.Array]:
    """Cast a pyarrow Array/ChunkedArray."""
    return CastOptions.check(options).cast_arrow_array(array)


@register_converter(pa.Table, pa.Table)
@register_converter(pa.RecordBatch, pa.RecordBatch)
def cast_arrow_tabular(
    data: Union[pa.Table, pa.RecordBatch],
    options: Optional[CastOptions] = None,
) -> Union[pa.Table, pa.RecordBatch]:
    """Cast pyarrow Table/RecordBatch with skip-cast on schema match."""
    return CastOptions.check(options).cast_arrow_tabular(data)


@register_converter(pa.RecordBatchReader, pa.RecordBatchReader)
def cast_arrow_record_batch_reader(
    data: pa.RecordBatchReader,
    options: Optional[CastOptions] = None,
) -> pa.RecordBatchReader:
    """Lazily wrap a ``RecordBatchReader`` with on-the-fly cast.

    Pure passthrough when the source schema matches target and no
    chunking is requested.
    """
    options = CastOptions.check(options)
    options = _bind_source(options, data.schema)

    needs_cast = options.target_field is not None and options.need_cast()
    needs_chunk = bool(options.row_size or options.byte_size)

    if not needs_cast and not needs_chunk:
        return data

    merged_schema = options.check_target(data.schema).merged_schema

    def casted_batches(opt=options):
        yield from _stream_from_reader(data, opt)

    return pa.RecordBatchReader.from_batches(merged_schema.to_arrow_schema(), casted_batches())


# ---------------------------------------------------------------------------
# Field / Schema converters
# ---------------------------------------------------------------------------


@register_converter(Any, pa.Field)
def any_to_arrow_field(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Field:
    if isinstance(obj, pa.Field):
        return obj
    if isinstance(obj, Field):
        return obj.to_arrow_field()
    return Field.from_any(obj).to_arrow_field()


@register_converter(Any, pa.Schema)
def any_to_arrow_schema(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pa.Schema:
    if isinstance(obj, pa.Schema):
        return obj
    if isinstance(obj, Schema):
        return obj.to_arrow_schema()
    return Schema.from_any(obj).to_arrow_schema()


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def _polars_lazyframe_prep(
    lf: Any,
    options: CastOptions,
) -> tuple[Any, CastOptions]:
    """Prepare a LazyFrame: bind source, project, and apply in-engine cast.

    Returns the LazyFrame *unmaterialized* — caller decides when to
    collect. The cast is folded into the plan via :meth:`cast_polars`,
    so it executes as part of streaming collection, fused with
    projection pushdown and any upstream scan filters.
    """
    pl = polars_module()
    projection = _resolve_projection(options)

    # Bind source from the lazy plan schema — cheap, no materialization.
    if options.source_field is None:
        try:
            options = options.check_source(obj=lf.collect_schema(), copy=False)
        except Exception:
            pass

    if projection:
        available = set(lf.collect_schema().names())
        keep = [c for c in projection if c in available]
        if keep:
            lf = lf.select(keep)

    # Plan-level cast — fuses into streaming execution.
    if options.target_field is not None and options.need_cast():
        lf = options.cast_polars(lf)
        options = options.copy(target_field=None)

    return lf, options


def _polars_eager_to_arrow(
    df: Any,
    options: CastOptions,
) -> tuple[pa.Table, CastOptions]:
    """Convert an *already eager* polars DataFrame/Series to Arrow.

    Applies projection + in-engine cast, then serializes. Used by both
    the bulk path (after collecting a LazyFrame) and the direct
    DataFrame/Series entry point.
    """
    pl = polars_module()
    projection = _resolve_projection(options)

    if isinstance(df, pl.Series):
        options = _bind_source(options, df)
        if options.target_field is not None and options.need_cast():
            df = options.cast_polars(df)
            options = options.copy(target_field=None)
        df = df.to_frame()

    elif isinstance(df, pl.DataFrame):
        options = _bind_source(options, df)
        if projection:
            keep = [c for c in projection if c in df.columns]
            if keep and list(df.columns) != keep:
                df = df.select(keep)
        if options.target_field is not None and options.need_cast():
            df = options.cast_polars(df)
            options = options.copy(target_field=None)

    else:
        raise TypeError("Unsupported eager Polars object: %s" % (df,))

    try:
        table = df.to_arrow(compat_level=pl.CompatLevel.newest())
    except Exception:
        table = df.rechunk().to_arrow(compat_level=pl.CompatLevel.newest())

    return table, options


def _polars_to_arrow(obj: Any, options: CastOptions) -> tuple[pa.Table, CastOptions]:
    """Bulk-convert any polars object to a ``pa.Table``.

    Used only by the bulk entry points. LazyFrames *must* materialize
    here — caller asked for a Table — so we use the streaming engine
    to keep memory bounded. For lazy-preserving iteration, callers
    should reach for :func:`_polars_lazy_to_batch_iterator` instead.
    """
    pl = polars_module()

    if isinstance(obj, pl.LazyFrame):
        lf, options = _polars_lazyframe_prep(obj, options)
        try:
            df = lf.collect(engine="streaming")
        except TypeError:
            df = lf.collect(streaming=True)
        return _polars_eager_to_arrow(df, options)

    return _polars_eager_to_arrow(obj, options)


def _polars_lazy_to_batch_iterator(
    lf: Any,
    options: CastOptions,
) -> Iterator[pa.RecordBatch]:
    """Stream a LazyFrame as Arrow batches without eager materialization.

    Uses ``LazyFrame.collect_batches(lazy=True)`` — a true pull-based
    streaming iterator over the plan's output. Each yielded polars
    ``DataFrame`` is one streaming-engine chunk; we convert per-chunk
    to Arrow batches and feed the result into
    :func:`_stream_from_arrow_batches` for ``row_size`` / ``byte_size``
    re-chunking.

    Falls back to the older ``collect(streaming=True)`` path on polars
    versions that lack ``collect_batches``. ``collect_batches`` is
    flagged unstable upstream but is the only API that streams without
    materializing the full result.
    """
    pl = polars_module()
    lf, options = _polars_lazyframe_prep(lf, options)

    def _native_batches():
        # Pull-based streaming when available — never materializes the
        # full result. lazy=True defers query start to first next().
        collect_batches = getattr(lf, "collect_batches", None)
        if callable(collect_batches):
            chunks = collect_batches(lazy=True, engine="streaming")
        else:
            # Old polars: fall back to materialize-then-slice. Bounded
            # by the streaming engine's internal chunking, but the
            # result DataFrame is held until iteration completes.
            try:
                df = lf.collect(engine="streaming")
            except TypeError:
                df = lf.collect(streaming=True)
            chunks = iter((df,))

        for chunk_df in chunks:
            try:
                chunk_table = chunk_df.to_arrow(
                    compat_level=pl.CompatLevel.newest()
                )
            except Exception:
                chunk_table = chunk_df.rechunk().to_arrow(
                    compat_level=pl.CompatLevel.newest()
                )
            yield from chunk_table.to_batches()

    yield from _stream_from_arrow_batches(_native_batches(), options)