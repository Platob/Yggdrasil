"""Spark <-> Arrow casting helpers and converters.

This module provides bidirectional type-mapping and data-casting between
Apache Spark (PySpark) and Apache Arrow (PyArrow).  Every public function
accepts an optional ``CastOptions`` argument that carries the target schema
**and** a ``safe`` flag:

- ``safe=True``  (default)  – best-effort: invalid casts become ``null`` /
  empty values; missing columns are filled with type-appropriate defaults.
- ``safe=False`` – strict: any type mismatch or missing column raises
  immediately.

The module is structured in three layers:

1. **Type converters** – stateless, pure functions that map Arrow ↔ Spark
   type objects without touching any data.
2. **Field / schema converters** – wrap type converters to also carry name,
   nullability, and metadata.
3. **Data converters** – operate on live Spark DataFrames / Columns using the
   type information from layers 1-2.

JSON decoding for compound targets
-----------------------------------
When the *source* column is ``StringType`` or ``BinaryType`` and the *target*
type is a compound type (struct, array, or map), the casters automatically
attempt to parse the column as JSON using ``from_json`` **before** applying
field-level / element-level casting.  This handles the common commodity-data
pattern where nested structures are stored as JSON strings in a flat column
(e.g. a ``STRING`` column containing ``'{"bid":99.5,"ask":100.0}'``).

- ``BinaryType`` sources are decoded to UTF-8 string first via ``CAST(... AS STRING)``.
- Spark's ``from_json`` silently returns ``null`` for rows whose JSON is
  malformed or structurally incompatible — consistent with ``safe=True``
  semantics.  In ``safe=False`` mode callers can enforce non-nullability on
  the result via the standard null-fill machinery.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pyarrow as pa
import pyspark.sql as pyspark_sql
import pyspark.sql.types as T

from yggdrasil.arrow.cast import (
    any_to_arrow_table,
    rechunk_arrow_batches,
)
from yggdrasil.data.cast import register_converter
from yggdrasil.data.options import CastOptions
from yggdrasil.environ import PyEnv

__all__ = [
    "any_to_spark_field",
    "any_to_spark_schema",
    "any_to_spark_dataframe",
    "cast_spark_column",
    "cast_spark_dataframe",
    "spark_dataframe_to_arrow",
    "spark_dataframe_to_pandas",
]


logger = logging.getLogger(__name__)

#: Cap on per-batch Arrow size handed to ``createDataFrame``. 128 MiB matches
#: Spark's preferred Arrow batch size and keeps a single oversized chunk from
#: pinning a partition's working memory.
_SPARK_ARROW_BATCH_BYTE_LIMIT: int = 128 * 1024 * 1024


def spark_dataframe_to_arrow(df: "pyspark_sql.DataFrame") -> pa.Table:
    """Materialize *df* as a ``pa.Table``, working on PySpark 3 and 4.

    PySpark 4.x added the native :meth:`pyspark.sql.DataFrame.toArrow`
    helper that we prefer when available — single driver-side collect
    that round-trips through Arrow IPC.  Older releases (3.x) don't
    expose it; we fall back to the streaming
    :meth:`toArrowBatchIterator` API when present, and finally to a
    pandas round-trip.

    Both Arrow paths can fail at the JVM level on Java 17+ when the
    cluster JVM was started without the ``sun.misc.Unsafe`` /
    ``java.nio.DirectByteBuffer`` ``--add-opens`` flags
    (``IllegalAccessException`` deep inside the Spark allocator).  We
    catch that and fall back to ``toPandas`` so callers see a working
    table instead of a JVM stack trace.  The pandas path itself
    benefits from Spark's Arrow optimization when the session has
    ``spark.sql.execution.arrow.pyspark.enabled=true``.

    Centralizing the polyfill here keeps callers from sprinkling
    ``hasattr(df, "toArrow")`` and exception-handling boilerplate
    throughout the codebase.
    """
    to_arrow = getattr(df, "toArrow", None)
    if callable(to_arrow):
        try:
            return to_arrow()
        except Exception as exc:
            # Java 17+ without the Unsafe ``--add-opens`` flags surfaces
            # here; fall back rather than fail the whole call.
            _log_arrow_path_fallback("toArrow", exc)

    to_iter = getattr(df, "toArrowBatchIterator", None)
    if callable(to_iter):
        try:
            batches = list(to_iter())
        except Exception as exc:
            _log_arrow_path_fallback("toArrowBatchIterator", exc)
        else:
            if not batches:
                # Empty result — synthesize an empty table with the
                # right schema rather than returning ``None``.
                from yggdrasil.data.schema import Schema
                return pa.Table.from_batches(
                    [], schema=Schema.from_(df.schema).to_arrow_schema(),
                )
            return pa.Table.from_batches(batches)

    # Last resort: pandas round-trip.  The session-level Arrow
    # optimization (``spark.sql.execution.arrow.pyspark.enabled``)
    # accelerates this transparently when it works, but trips on
    # Java 17+ if the JVM was started without the
    # ``--add-opens java.base/sun.misc=ALL-UNNAMED`` flag (the JVM-side
    # Arrow allocator can't reach ``sun.misc.Unsafe``).  Force-disable
    # arrow.pyspark for the duration of the collect so we use the
    # plain row-based path, then restore the previous setting.
    return pa.Table.from_pandas(_to_pandas_no_arrow(df), preserve_index=False)


def spark_dataframe_to_pandas(df: "pyspark_sql.DataFrame"):
    """Public alias for :func:`_to_pandas_no_arrow`.

    Same JVM-Arrow-bypass behavior; exposed so tests and callers that
    just want a pandas DataFrame don't have to round-trip through Arrow
    or hand-roll the same conf-toggle dance.
    """
    return _to_pandas_no_arrow(df)


def _to_pandas_no_arrow(df: "pyspark_sql.DataFrame"):
    """Run ``df.toPandas()`` with Spark's Arrow optimization disabled.

    PySpark's ``spark.sql.execution.arrow.pyspark.fallback.enabled``
    catches Python-level conversion errors but propagates JVM-side
    failures (e.g. the Java 17+ ``IllegalAccessException`` against
    ``sun.misc.Unsafe``) directly, which means a single broken JVM can
    take down every ``toPandas`` call in the process.  Toggling the
    arrow conf for the duration of this one call sidesteps that.
    """
    spark = df.sparkSession
    arrow_key = "spark.sql.execution.arrow.pyspark.enabled"
    try:
        prev = spark.conf.get(arrow_key)
    except Exception:
        prev = None
    try:
        spark.conf.set(arrow_key, "false")
        return df.toPandas()
    finally:
        if prev is None:
            try:
                spark.conf.unset(arrow_key)
            except Exception:
                pass
        else:
            spark.conf.set(arrow_key, prev)


def _log_arrow_path_fallback(api: str, exc: BaseException) -> None:
    """One-line debug note when an Arrow fast path falls back to pandas.

    Kept private and at DEBUG so the fallback is visible to anyone who
    enables the ``yggdrasil.spark`` logger but doesn't spam the default
    output.  The call site continues with the pandas path; see
    :func:`spark_dataframe_to_arrow`.
    """
    logger.debug(
        "spark_dataframe_to_arrow: %s failed (%s); falling back",
        api, exc.__class__.__name__,
    )


@register_converter(Any, T.StructField)
def any_to_spark_field(
    field: pa.Field,
    options: Optional[CastOptions] = None,
) -> T.StructField:
    return options.check_source(field).merged.to_pyspark_field()


@register_converter(Any, T.StructType)
def any_to_spark_schema(
    schema: pa.Schema,
    options: Optional[CastOptions] = None,
) -> T.StructType:
    return options.check_source(schema).merged.to_spark_schema()


@register_converter(pyspark_sql.Column, pyspark_sql.Column)
def cast_spark_column(
    column: pyspark_sql.Column,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.Column:
    return CastOptions.check(options).cast_spark_column(column)


@register_converter(pyspark_sql.DataFrame, pyspark_sql.DataFrame)
def cast_spark_dataframe(
    dataframe: pyspark_sql.DataFrame,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    return CastOptions.check(options).cast_spark_tabular(dataframe)


@register_converter(Any, pyspark_sql.DataFrame)
def any_to_spark_dataframe(
    obj: Any,
    options: Optional[CastOptions] = None,
) -> pyspark_sql.DataFrame:
    opts = CastOptions.check(options)

    if isinstance(obj, pyspark_sql.DataFrame):
        return opts.cast_spark_tabular(obj)

    # ``Tabular`` (Response, StatementResult, ParquetFile, …) owns its
    # own Spark fan-out — :meth:`Tabular.read_spark_frame` short-circuits
    # to a persisted Spark frame when one is on hand (see
    # ``StatementResult._read_spark_frame``) and otherwise runs the
    # Arrow round-trip with the same cast options. The leaf already
    # applies ``cast_spark`` / ``cast_spark_tabular`` against ``opts``,
    # so we don't re-cast at the call site. Path-shaped strings and
    # ``os.PathLike`` inputs are wrapped into a :class:`Path` via
    # :meth:`Tabular.from_` so callers can ``convert("s3://b/k.parquet",
    # pyspark_sql.DataFrame)`` without hand-rolling the holder.
    from yggdrasil.io.tabular import Tabular, is_tabular_source
    if is_tabular_source(obj):
        tabular = obj if isinstance(obj, Tabular) else Tabular.from_(obj)
        return tabular.read_spark_frame(opts)

    spark = PyEnv.spark_session(
        create=True,
        import_error=True,
        install_spark=False,
    )

    if obj is None:
        return spark.createDataFrame([], schema=opts.merged.to_spark_schema())

    arrow_table = any_to_arrow_table(obj, options=opts)
    rechunked = list(rechunk_arrow_batches(
        arrow_table.to_batches(),
        byte_size=_SPARK_ARROW_BATCH_BYTE_LIMIT,
        memory_pool=opts.arrow_memory_pool,
    ))
    arrow_table = pa.Table.from_batches(rechunked, schema=arrow_table.schema)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "spark.createDataFrame from %d batches / %d rows",
            len(rechunked), arrow_table.num_rows,
        )
    df = spark.createDataFrame(arrow_table, schema=opts.merged.to_spark_schema())
    return opts.cast_spark(df)
