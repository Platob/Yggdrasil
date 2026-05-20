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

3. **One streaming pipeline.** Per-batch Arrow cast and
   ``byte_size`` / ``row_size`` rechunking are owned by the nested
   struct helpers in
   :mod:`yggdrasil.data.types.nested.struct_arrow` (reachable via
   :meth:`CastOptions.cast_arrow_batch_iterator`). Every streaming
   entry point here flattens its input to ``pa.RecordBatch`` and
   hands it off to that pipeline — no parallel chunkers in this
   module.

4. **Bind source schemas, don't peek.** When we infer a source
   schema, we bind it onto ``options.source`` so it propagates
   downstream and drives :meth:`CastOptions.need_cast` without
   re-inference.

5. **Emit the merged schema.** When both source and target are bound,
   the output schema is :attr:`CastOptions.merged_schema` —
   reconciled per ``schema_mode``. ``RecordBatchReader`` /
   iterator declarations use this.

6. **Honor every options knob.** ``column_names`` projects on the way
   in; ``arrow_memory_pool`` threads through pyarrow allocators;
   ``safe`` flows into engine cast methods; ``row_size`` /
   ``byte_size`` drive output chunking via the nested rechunker.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections.abc import Iterable, Iterator
from dataclasses import is_dataclass
from typing import Optional, Union, Any, Generator

import pyarrow as pa

from yggdrasil.data.options import CastOptions
from yggdrasil.data.cast.registry import register_converter
from yggdrasil.data.schema import Schema, Field
from yggdrasil.lazy_imports import (
    pandas_module,
    pyarrow_dataset_module,
    spark_dataframe_classes,
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
    "rechunk_arrow_batches",
    "rechunk_arrow_table",
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
    """Bind an inferred source schema onto ``options.source``.

    Idempotent. Coercion failures are swallowed — "couldn't infer"
    is not a hard error here.
    """
    if options.source is not None:
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


def _is_arrow_dataset(obj: Any) -> bool:
    """Return ``True`` if *obj* is a :class:`pyarrow.dataset.Dataset`.

    Lazy-imports ``pyarrow.dataset`` so the base install (pyarrow only)
    isn't paying for the dataset submodule unless something actually
    feeds a Dataset into the cast pipeline.
    """
    try:
        ds = pyarrow_dataset_module()
    except Exception:
        return False
    return isinstance(obj, ds.Dataset)


def _is_yggdrasil_tabular(obj: Any) -> bool:
    """Return ``True`` if *obj* is a :class:`yggdrasil.io.tabular.Tabular`.

    Imported lazily to avoid pulling the io subpackage into the import
    graph of every Arrow consumer.
    """
    from yggdrasil.io.tabular import Tabular
    return isinstance(obj, Tabular)


def _is_tabular_source(obj: Any) -> bool:
    """Return ``True`` if *obj* can be coerced via :meth:`Tabular.from_`.

    Lifted to a helper so the cast entry points share the same
    path-string heuristic as :func:`yggdrasil.io.tabular.is_tabular_source`
    — keeps ``any_to_arrow_table('data.parquet')`` and the engine
    siblings (``any_to_polars_dataframe`` / ``any_to_spark_dataframe``
    / ``any_to_pandas_dataframe``) agreeing on what counts as a path.
    """
    from yggdrasil.io.tabular import is_tabular_source
    return is_tabular_source(obj)


# ---------------------------------------------------------------------------
# Streaming rechunker — pure pyarrow, used by every engine entry point that
# needs ``byte_size`` / ``row_size`` sizing on the way out.
# ---------------------------------------------------------------------------


#: Saturating cap for size estimates. ``2**62`` is well within Python's
#: arbitrary-precision int range yet larger than any realistic Arrow
#: payload (a 4 EiB ceiling), so callers can sum many estimates without
#: ever overflowing into pathological territory.
_MAX_NBYTES = 1 << 62

#: Flat byte estimate returned for view-typed arrays
#: (``string_view`` / ``binary_view`` / list-view variants). ``.nbytes``
#: on these arrays is unreliable (see :func:`get_arrow_nbytes`); rather
#: than walk per-element to compute the real referenced size, we return
#: a coarse 1 MiB constant. That's the same threshold the IPC writer
#: uses to flip ``compression="auto"`` to LZ4, so a view-typed array
#: lands on the compressed path by default — which is exactly what the
#: data-density of these types warrants.
_VIEW_DEFAULT_NBYTES = 1 << 20  # 1 MiB


def _saturating_add(a: int, b: int) -> int:
    """``a + b`` clamped to ``[0, _MAX_NBYTES]``.

    Sum-of-children paths use this so a pathological input (millions
    of chunks, capped children) cannot drift past
    :data:`_MAX_NBYTES` and into territory that overflows downstream
    integer fields (parquet row-group sizing, IPC body length, etc.).
    """
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    s = a + b
    return _MAX_NBYTES if s > _MAX_NBYTES else s


def _resolve_view_probes() -> tuple:
    """Bind ``pa.types.is_*_view`` predicates once at import time.

    Old impl re-did the four ``getattr(pa.types, ...)`` lookups on
    every call. Hot rechunk passes call :func:`_is_view_type` once
    per Array leaf, so the bound-once form removes 4 attribute hops
    per leaf invocation.
    """
    types_mod = getattr(pa, "types", None)
    if types_mod is None:
        return ()
    out = []
    for name in (
        "is_string_view", "is_binary_view",
        "is_list_view", "is_large_list_view",
    ):
        fn = getattr(types_mod, name, None)
        if fn is not None:
            out.append(fn)
    return tuple(out)


_VIEW_PROBES: tuple = _resolve_view_probes()

#: Memo of view-ness keyed by Arrow DataType. ``pa`` interns primitive
#: types (``pa.int64() is pa.int64()`` is True) so this dict caps at
#: O(#distinct schemas seen) — bounded in practice. Avoids the 4-probe
#: scan on every leaf of every rechunk pass.
_VIEW_TYPE_CACHE: "dict[Any, bool]" = {}

#: Memo of "does this schema have any top-level view-typed column?"
#: Keyed by the lifted :class:`yggdrasil.data.Schema` (a.k.a.
#: :class:`StructField`), which is hashable by design — unlike
#: :class:`pa.Schema`, whose ``__hash__`` raises ``TypeError:
#: unhashable type: 'dict'`` whenever schema metadata carries dict
#: values (routine after a Spark Connect round trip). The lifted
#: ``Schema`` strips the unhashable-shaped metadata via its own
#: ``__hash__`` so the cache key stays stable across calls that
#: share the same logical schema even when ``pa.Schema`` itself
#: refuses to hash.
_SCHEMA_TOP_VIEW_CACHE: "dict[Any, bool]" = {}


def _schema_cache_key(schema: Any) -> Any:
    """Hashable key for *schema* — lifts to ygg :class:`Schema` on demand.

    Tries the cheap path first (``pa.Schema`` is usually hashable),
    falls back to ``Schema.from_arrow_schema(schema)`` for the unhashable-
    metadata case. Returns ``None`` when neither path produces a
    hashable key — the caller then skips the cache entirely.
    """
    try:
        hash(schema)
        return schema
    except TypeError:
        pass
    except Exception:
        return None
    try:
        from yggdrasil.data.schema import Schema
        return Schema.from_arrow_schema(schema)
    except Exception:
        return None


def _schema_has_top_view(schema: Any) -> bool:
    """``True`` iff *schema* has any top-level view-typed field.

    Mirrors the existing :func:`get_arrow_nbytes` semantics: only the
    top-level type of each column matters (a nested type wrapping a
    view-typed leaf is *not* caught here — same as the recursive
    walk, which short-circuits on view-ness only at the top of each
    leaf array). Result is memoized via :func:`_schema_cache_key`.
    """
    key = _schema_cache_key(schema)
    if key is not None:
        cached = _SCHEMA_TOP_VIEW_CACHE.get(key)
        if cached is not None:
            return cached
    result = False
    try:
        for field in schema:
            if _is_view_type(field.type):
                result = True
                break
    except Exception:
        # Defensive — non-Schema objects shouldn't reach here, but a
        # walk-failure shouldn't crash the rechunker.
        result = True  # bail to the safe recursive path
    if key is not None:
        try:
            _SCHEMA_TOP_VIEW_CACHE[key] = result
        except TypeError:
            pass
    return result


def _is_view_type(arr_type: Any) -> bool:
    """True if ``arr_type`` is one of the Arrow view types whose
    ``.nbytes`` either raises or over-counts on slices.

    Routes through cached ``pa.types.is_*_view`` predicates (pyarrow
    ≥ 17 exposes ``is_string_view`` / ``is_binary_view`` / list-view
    helpers); falls back to a string match on ``str(arr_type)`` for
    older builds and for list-view variants the helper module may
    not enumerate.

    Result is memoized per ``arr_type`` so a hot rechunk path pays
    the probe cost once per distinct schema, not once per batch.
    """
    cached = _VIEW_TYPE_CACHE.get(arr_type)
    if cached is not None:
        return cached
    result = False
    for probe in _VIEW_PROBES:
        try:
            if probe(arr_type):
                result = True
                break
        except Exception:
            continue
    if not result:
        # Last-resort textual match — covers exotic / future view types
        # whose ``is_*`` helper isn't exposed yet.
        try:
            result = "view" in str(arr_type).lower()
        except Exception:
            result = False
    try:
        _VIEW_TYPE_CACHE[arr_type] = result
    except TypeError:
        # Unhashable type — return without caching.
        pass
    return result


def get_arrow_nbytes(obj: Any, default: int = 0) -> int:
    """Best-effort byte size of an Arrow object.

    ``obj.nbytes`` is the fast path but is unreliable for the Arrow
    *view* types (``string_view``, ``binary_view``, list-view variants):

    * In older pyarrow it raises ``NotImplementedError`` outright.
    * In newer pyarrow it returns the **physical** sum of buffer sizes,
      which over-counts dramatically for sliced views — variadic data
      buffers are shared with the parent and the slice's logical
      payload may be a tiny fraction of what ``nbytes`` reports
      (a 1-row slice of a 1k-row view array reports the full
      variadic buffer, not the one referenced string).

    Resolution order:

    1. **Container recursion** — ``ChunkedArray`` over ``chunks``,
       ``Table`` / ``RecordBatch`` over ``columns``. Recurse first so
       view-typed children get the 1 MiB treatment per-chunk; the
       container's own ``.nbytes`` would over-count slices.
    2. **View-typed leaf** (string_view / binary_view, list-view
       variants) — return a flat :data:`_VIEW_DEFAULT_NBYTES` (1 MiB)
       per array. We deliberately do **not** scan the data
       (no ``binary_length`` aggregation, no per-element walk) — the
       caller uses this for chunking / threshold decisions, not
       accounting, and a coarse "treat view arrays as ~1 MiB" is
       enough to keep them on the compressed path.
    3. ``obj.nbytes`` — used as-is for non-view leaves. Negative or
       overflow-prone values are clamped to ``[0, _MAX_NBYTES]``.
    4. **Buffer walk** — sum ``buf.size`` for every non-null buffer
       returned by ``Array.buffers()``. Last resort for non-view
       leaves whose ``nbytes`` raised.
    5. ``default`` — never raises.

    Always returns an ``int`` in ``[0, _MAX_NBYTES]``; never
    propagates exceptions from Arrow internals (the caller is sizing
    batches for chunking, not doing accounting — a slightly off
    estimate is fine, a crash is not).
    """
    if obj is None:
        return default

    # Container dispatch via ``isinstance`` rather than ``getattr``
    # probing — Arrow's container hierarchy is closed (ChunkedArray /
    # Table / RecordBatch) and the explicit check skips two
    # missing-attribute lookups on every leaf call.
    #
    # Fast path: when the container has no top-level view-typed
    # column, the native ``.nbytes`` (sum of all buffers, computed
    # C-side) is both correct and an order of magnitude cheaper than
    # the per-column Python recursion. The "any view column?" check
    # is memoized per ``pa.Schema`` / ``pa.DataType`` so a streaming
    # pipeline pays it once.
    if isinstance(obj, pa.ChunkedArray):
        if not _is_view_type(obj.type):
            try:
                nb = int(obj.nbytes)
            except (NotImplementedError, AttributeError, TypeError):
                nb = None
            except Exception:
                nb = None
            if nb is not None:
                if nb < 0:
                    return default
                return _MAX_NBYTES if nb > _MAX_NBYTES else nb
        total = 0
        for c in obj.chunks:
            total = _saturating_add(total, get_arrow_nbytes(c, default=default))
            if total >= _MAX_NBYTES:
                break
        return total

    if isinstance(obj, (pa.Table, pa.RecordBatch)):
        if not _schema_has_top_view(obj.schema):
            try:
                nb = int(obj.nbytes)
            except (NotImplementedError, AttributeError, TypeError):
                nb = None
            except Exception:
                nb = None
            if nb is not None:
                if nb < 0:
                    return default
                return _MAX_NBYTES if nb > _MAX_NBYTES else nb
        total = 0
        for c in obj.columns:
            total = _saturating_add(total, get_arrow_nbytes(c, default=default))
            if total >= _MAX_NBYTES:
                break
        return total

    # Leaf path — Array, Scalar, or any object with ``.nbytes``. Check
    # the view-type short-circuit first so a sliced view doesn't fall
    # through to the over-counting ``.nbytes`` path.
    arr_type = getattr(obj, "type", None)
    if arr_type is not None and _is_view_type(arr_type):
        return _VIEW_DEFAULT_NBYTES

    try:
        nb = int(obj.nbytes)
    except (NotImplementedError, AttributeError, TypeError):
        nb = None
    except Exception:
        nb = None
    if nb is not None:
        if nb < 0:
            nb = default
        elif nb > _MAX_NBYTES:
            nb = _MAX_NBYTES
        return nb

    # Leaf array fallback: walk every buffer. For view types this is a
    # known over-counter (see the docstring) — the dedicated branch
    # above runs first; this path is the last resort for non-view
    # leaves whose ``.nbytes`` failed.
    buffers = getattr(obj, "buffers", None)
    if callable(buffers):
        try:
            total = 0
            for buf in buffers():
                if buf is None:
                    continue
                sz = int(buf.size)
                if sz <= 0:
                    continue
                total = _saturating_add(total, sz)
                if total >= _MAX_NBYTES:
                    break
            return total
        except Exception:
            pass

    return default


def rechunk_arrow_batches(
    batches: Iterable[pa.RecordBatch],
    *,
    byte_size: int | None = None,
    row_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> Iterator[pa.RecordBatch]:
    """Stream-coalesce/slice batches to ~``byte_size`` bytes / ``row_size`` rows.

    Both knobs are optional:

    * Neither set → passthrough.
    * ``row_size`` only → emit fixed-size chunks of at most
      ``row_size`` rows; no buffering, zero-copy slices.
    * ``byte_size`` only → emit ~``byte_size``-byte chunks using the
      per-segment bytes/row ratio to derive a row target.
    * Both set → ``byte_size`` drives the row target; ``row_size``
      caps it (final ``target_rows = min(row_size, derived)``).

    Byte sizing routes through :func:`get_arrow_nbytes` so view-typed
    arrays (``string_view`` / ``binary_view``) — which raise from
    ``RecordBatch.nbytes`` in current pyarrow — fall back to a buffer
    walk instead of crashing the rechunker.

    Algorithm (byte_size path):

    * Empty incoming batch → drop (no schema gymnastics on zero-row
      flushes — the consumer already saw a schema in an earlier batch
      or will get one from the upstream reader).
    * Buffer empty + incoming batch already at/over target → slice it
      directly into target-sized chunks (zero-copy).
    * Otherwise accumulate; once buffered ``nbytes`` crosses the
      target, concat + slice the buffer to target-sized chunks. Yield
      everything except a possibly-undersized tail; carry the tail
      forward.
    * On exhaustion → flush whatever is left as a single concat'd
      batch (may be smaller than ``byte_size``).
    """
    has_byte = bool(byte_size and byte_size > 0)
    has_row = bool(row_size and row_size > 0)

    if not has_byte and not has_row:
        yield from batches
        return

    if not has_byte:
        for batch in batches:
            if batch.num_rows == 0:
                continue
            if batch.num_rows <= row_size:
                yield batch
                continue
            for offset in range(0, batch.num_rows, row_size):
                yield batch.slice(offset, row_size)
        return

    def _target_rows(batch: pa.RecordBatch, nbytes: int | None = None) -> int:
        if nbytes is None:
            nbytes = get_arrow_nbytes(batch)
        bytes_per_row = max(1, nbytes // max(1, batch.num_rows))
        derived = max(1, byte_size // bytes_per_row)
        return min(row_size, derived) if has_row else derived

    def _slice_to_target(
        batch: pa.RecordBatch, nbytes: int | None = None
    ) -> Iterator[pa.RecordBatch]:
        target = _target_rows(batch, nbytes=nbytes)
        if batch.num_rows <= target:
            yield batch
            return
        for offset in range(0, batch.num_rows, target):
            yield batch.slice(offset, target)

    buffer: list[pa.RecordBatch] = []
    buffered_bytes = 0

    for batch in batches:
        if batch.num_rows == 0:
            continue

        nbytes = get_arrow_nbytes(batch)

        if not buffer and nbytes >= byte_size:
            yield from _slice_to_target(batch, nbytes=nbytes)
            continue

        buffer.append(batch)
        buffered_bytes += nbytes

        if buffered_bytes < byte_size:
            continue

        combined = pa.concat_batches(buffer, memory_pool=memory_pool)
        combined_nbytes = get_arrow_nbytes(combined)
        target = _target_rows(combined, nbytes=combined_nbytes)

        if combined.num_rows <= target:
            # Estimator pushed the row-cap above the combined batch
            # (under-estimate of bytes/row from skewed inputs). Emit as
            # one batch and reset.
            yield combined
            buffer = []
            buffered_bytes = 0
            continue

        sliced = list(_slice_to_target(combined, nbytes=combined_nbytes))
        for chunk in sliced[:-1]:
            yield chunk

        tail = sliced[-1]
        tail_nbytes = get_arrow_nbytes(tail)
        if tail_nbytes >= byte_size:
            yield tail
            buffer = []
            buffered_bytes = 0
        else:
            buffer = [tail]
            buffered_bytes = tail_nbytes

    if buffer:
        combined = pa.concat_batches(buffer, memory_pool=memory_pool)
        if has_row and combined.num_rows > row_size:
            yield from _slice_to_target(combined)
        else:
            yield combined


def rechunk_arrow_table(
    table: pa.Table,
    *,
    byte_size: int | None = None,
    row_size: int | None = None,
    memory_pool: pa.MemoryPool | None = None,
) -> pa.Table:
    """Re-chunk *table* to ~``byte_size`` bytes / ``row_size`` rows per chunk.

    Thin :class:`pa.Table`-shaped wrapper over
    :func:`rechunk_arrow_batches` — runs ``table.to_batches()`` through
    the same streaming chunker and rebuilds a :class:`pa.Table` from
    the result. Schema (including metadata) is preserved end-to-end so
    callers can drop this in front of any sink that prefers a
    particular chunk shape without losing field annotations.

    Both knobs are optional:

    * Neither set → returned table is the input (no copy).
    * ``row_size`` only → chunks contain at most ``row_size`` rows;
      zero-copy slices.
    * ``byte_size`` only → chunks target ~``byte_size`` bytes via the
      per-segment bytes/row ratio.
    * Both set → ``byte_size`` drives the row target; ``row_size``
      caps it.

    See :func:`rechunk_arrow_batches` for the underlying algorithm.
    """
    has_byte = bool(byte_size and byte_size > 0)
    has_row = bool(row_size and row_size > 0)
    if not has_byte and not has_row:
        return table

    batches = list(
        rechunk_arrow_batches(
            table.to_batches(),
            byte_size=byte_size,
            row_size=row_size,
            memory_pool=memory_pool,
        )
    )
    return pa.Table.from_batches(batches, schema=table.schema)


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

    if options.target is not None and options.need_cast():
        obj = options.cast_pandas(obj)

    table = pa.Table.from_pandas(
        obj,
        preserve_index=bool(obj.index.name),
    )
    return table, options.copy(target=None)


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
    if not isinstance(obj, spark_dataframe_classes()):
        raise TypeError("Unsupported Spark object: %s" % (obj,))

    options = _bind_source(options, obj)

    projection = _resolve_projection(options)
    if projection:
        keep = [c for c in projection if c in obj.columns]
        if keep and list(obj.columns) != keep:
            obj = obj.select(*keep)

    if options.target is not None and options.need_cast():
        obj = options.cast_spark(obj)

    from yggdrasil.spark.cast import spark_dataframe_to_arrow
    return spark_dataframe_to_arrow(obj), options.copy(target=None)


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
            options.target.name
            if options.target and options.target.name
            else "value"
        )
        table = pa.table({name: obj})
        options = _bind_source(options, table)

    elif isinstance(obj, pa.RecordBatchReader):
        options = _bind_source(options, obj.schema)
        table = _project(obj.read_all(), options)

    elif _is_tabular_source(obj):
        # ``Tabular`` (Response, StatementResult, ParquetFile, …) owns
        # its own engine fan-out — let it produce the table so format
        # leaves use their native scanners (Parquet predicate pushdown,
        # StatementResult's persisted-frame short-circuit) instead of
        # the polars-wrap fallback further down. Path-shaped strings
        # and ``os.PathLike`` inputs are coerced through
        # :meth:`Tabular.from_`, which wraps them in a :class:`Path`
        # holder and lets :meth:`Path.read_arrow_table` dispatch to
        # the right format leaf (ParquetFile / CSVFile / …) via the
        # MediaType registry.
        from yggdrasil.io.tabular import Tabular
        tabular = obj if _is_yggdrasil_tabular(obj) else Tabular.from_(obj)
        table = tabular.read_arrow_table(options)
        options = _bind_source(options, table)

    elif _is_arrow_dataset(obj):
        # ``Dataset.to_table()`` runs the scanner with column projection
        # pushed into the file format readers — strictly better than
        # materializing then projecting on the Arrow side.
        options = _bind_source(options, obj.schema)
        projection = _resolve_projection(options)
        if projection:
            present = [c for c in projection if c in obj.schema.names]
            table = obj.to_table(columns=present) if present else obj.to_table()
        else:
            table = obj.to_table()

    elif isinstance(obj, (Generator, Iterator)):
        batches = list(_flatten_to_arrow_batches(iter(obj), options))
        if not batches:
            merged_schema = options.merged or Schema.empty()
            return pa.Table.from_batches([], schema=merged_schema.to_arrow_schema())
        table = pa.Table.from_batches(batches)

    elif isinstance(obj, (list, tuple)):
        if not obj:
            merged_schema = options.merged or Schema.empty()
            return pa.Table.from_batches([], schema=merged_schema.to_arrow_schema())

        # Dataclass rows aren't mappings: ``pa.Table.from_pylist`` rejects
        # them, and the per-item fallback below would route each scalar
        # through ``pl.DataFrame(single_dataclass)`` — also unsupported.
        # Normalize to dicts up front when the first item is a dataclass.
        if is_dataclass(obj[0]) and not isinstance(obj[0], type):
            obj = [
                dataclasses.asdict(x)
                if is_dataclass(x) and not isinstance(x, type)
                else x
                for x in obj
            ]

        try:
            table = pa.Table.from_pylist(obj)
        except Exception:
            batches = list(_flatten_to_arrow_batches(iter(obj), options))
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

    Per-batch Arrow cast and ``byte_size`` / ``row_size`` rechunking
    are owned by :meth:`CastOptions.cast_arrow_batch_iterator`, which
    delegates to the nested struct rechunker. The job here is to
    *produce* the source batch stream — engine-native casts happen
    upstream when an engine (polars / spark / pandas) owns the data.

    For Polars LazyFrame and Spark DataFrame, the engine-native cast
    is applied before serialization, so the rechunker sees already-
    cast batches and skips per-batch Arrow rework.
    """
    options = CastOptions.check(options)

    if isinstance(obj, pa.Table):
        options = _bind_source(options, obj)
        return options.cast_arrow_batch_iterator(_project(obj, options).to_batches())

    if isinstance(obj, pa.RecordBatch):
        options = _bind_source(options, obj)
        return options.cast_arrow_batch_iterator(iter([obj]))

    if isinstance(obj, pa.RecordBatchReader):
        options = _bind_source(options, obj.schema)
        return options.cast_arrow_batch_iterator(iter(obj))

    if _is_arrow_dataset(obj):
        options = _bind_source(options, obj.schema)
        projection = _resolve_projection(options)
        if projection:
            present = [c for c in projection if c in obj.schema.names]
            scanner = obj.scanner(columns=present) if present else obj.scanner()
        else:
            scanner = obj.scanner()
        return options.cast_arrow_batch_iterator(scanner.to_batches())

    namespace = ObjectSerde.full_namespace(obj)

    if namespace.startswith("pyspark."):
        if isinstance(obj, spark_dataframe_classes()):
            options = _bind_source(options, obj)
            projection = _resolve_projection(options)
            if projection:
                keep = [c for c in projection if c in obj.columns]
                if keep and list(obj.columns) != keep:
                    obj = obj.select(*keep)

            # In-engine cast before streaming — fuses into Spark plan.
            if options.target is not None and options.need_cast():
                obj = options.cast_spark(obj)
                options = options.copy(target=None)

            to_iter = getattr(obj, "toArrowBatchIterator", None)
            if callable(to_iter):
                return options.cast_arrow_batch_iterator(iter(to_iter()))
            # No batch-iter API — fall through to generic bulk fallback.

    if namespace.startswith("polars."):
        pl = polars_module()
        if isinstance(obj, pl.LazyFrame):
            return _polars_lazy_to_batch_iterator(obj, options)
        # Eager polars (DataFrame/Series) falls through to the bulk
        # path — no streaming benefit since data is already materialized.

    if isinstance(obj, (Generator, Iterator)):
        return options.cast_arrow_batch_iterator(_flatten_to_arrow_batches(obj, options))

    if isinstance(obj, (list, tuple)):
        if not obj:
            return iter(())
        return options.cast_arrow_batch_iterator(_flatten_to_arrow_batches(iter(obj), options))

    # Generic fallback — bulk-convert in-engine (cast applied), then
    # re-stream the already-cast table through the rechunker.
    table = any_to_arrow_table(obj, options)
    return options.copy(target=None).cast_arrow_batch_iterator(table.to_batches())


def _flatten_to_arrow_batches(
    items: Iterator[Any],
    options: CastOptions,
) -> Iterator[pa.RecordBatch]:
    """Flatten a (possibly nested) iterator of tabular things into ``pa.RecordBatch``.

    Arrow inputs flow through unchanged — the rechunker downstream
    runs the per-batch cast. Engine-native items (pandas / polars /
    spark frames, dicts, dataclasses, pylist rows) are routed through
    :func:`any_to_arrow_table` so the in-engine cast still applies
    and the result hits the rechunker as already-cast batches.
    """
    for item in items:
        if isinstance(item, pa.RecordBatch):
            yield item
        elif isinstance(item, pa.Table):
            yield from item.to_batches()
        elif isinstance(item, pa.RecordBatchReader):
            yield from item
        elif _is_arrow_dataset(item):
            yield from item.to_batches()
        elif isinstance(item, (Generator, Iterator)):
            yield from _flatten_to_arrow_batches(item, options)
        elif isinstance(item, (list, tuple)):
            yield from _flatten_to_arrow_batches(iter(item), options)
        else:
            sub = any_to_arrow_table(item, options)
            yield from sub.to_batches()


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

    merged_schema = options.merged
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

    target_field = options.target

    if scalar is None:
        return default_arrow_scalar(
            target_field.arrow_type if target_field is not None else None,
            nullable=True if target_field is None else target_field.nullable,
        )

    if isinstance(scalar, enum.Enum):
        scalar = scalar.value

    if is_dataclass(scalar):
        scalar = dataclasses.asdict(scalar)

    if target_field is None:
        return pa.scalar(scalar)

    try:
        scalar = pa.scalar(scalar, type=target_field.arrow_type)
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
    if options.target is None:
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

    needs_cast = options.target is not None and options.need_cast()
    needs_chunk = bool(options.row_size or options.byte_size)

    if not needs_cast and not needs_chunk:
        return data

    merged_schema = options.check_target(data.schema).merged

    return pa.RecordBatchReader.from_batches(
        merged_schema.to_arrow_schema(),
        options.cast_arrow_batch_iterator(iter(data)),
    )


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
    if options.source is None:
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
    if options.target is not None and options.need_cast():
        lf = options.cast_polars(lf)
        options = options.copy(target=None)

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
        # Wrap into a single-column frame *before* binding so the
        # source schema is struct-shaped — matches the pandas Series
        # path and lets struct-shaped targets (a Schema or struct
        # Field) cast in-engine without tripping the leaf↔struct
        # mismatch in the polars cast dispatcher.
        df = df.to_frame()

    if isinstance(df, pl.DataFrame):
        options = _bind_source(options, df)
        if projection:
            keep = [c for c in projection if c in df.columns]
            if keep and list(df.columns) != keep:
                df = df.select(keep)
        if options.target is not None and options.need_cast():
            df = options.cast_polars(df)
            options = options.copy(target=None)

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
    to Arrow batches and hand them to
    :meth:`CastOptions.cast_arrow_batch_iterator`, which routes
    through the nested struct rechunker for ``row_size`` /
    ``byte_size`` re-chunking. Cast is already done in-engine inside
    :func:`_polars_lazyframe_prep`, so the rechunker only resizes.

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

    yield from options.cast_arrow_batch_iterator(_native_batches())