import logging
from typing import Any, Iterator

import cloudpickle
import pyarrow as pa
from yggdrasil.arrow.cast import any_to_arrow_batch_iterator, any_to_arrow_table
from yggdrasil.data import schema as schema_builder, field as field_builder, Schema
from yggdrasil.data.options import CastOptions

# Dynamic-frame row payloads and the user function are serialized with
# cloudpickle — the serializer pyspark itself ships — so executors need
# nothing ygg-specific (no install step, no custom envelope) to read a
# dynamic batch back.

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Dataset",  # noqa: F822  -- lazy module attribute via ``__getattr__``
    "is_dynamic_schema",
]

PICKLE_COLUMN_NAME = "_pickle"
DYNAMIC_SCHEMA = schema_builder(
    [
        field_builder(
            name=PICKLE_COLUMN_NAME,
            arrow_type=pa.binary(),
            nullable=False,
            metadata={"format": "binary"},
            tags={"namespace": "yggdrasil.spark.frame"},
        )
    ]
)
_ARROW_DYNAMIC_SCHEMA = DYNAMIC_SCHEMA.to_arrow_schema()


def is_dynamic_schema(obj: Any) -> bool:
    schema = Schema.from_any(obj)
    if len(schema) != 1:
        return False
    first = schema.field(index=0)
    return first.name == PICKLE_COLUMN_NAME and pa.types.is_binary(first.arrow_type)


# ---------------------------------------------------------------------------
# Per-partition helpers
# ---------------------------------------------------------------------------

def _emit_pickled(
    objects: Iterator[Any],
    *,
    byte_size: int,
) -> Iterator[pa.RecordBatch]:
    """Pickle a stream of Python objects into dynamic-schema record batches."""
    out: list[dict[str, bytes]] = []
    out_bytes = 0
    for obj in objects:
        ser = cloudpickle.dumps(obj)
        if out and out_bytes + len(ser) > byte_size:
            yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)
            out = []
            out_bytes = 0
        out.append({PICKLE_COLUMN_NAME: ser})
        out_bytes += len(ser)
    if out:
        yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)


def spark_typed_cast(
    objects_per_batch: Iterator[list[Any]],
    schema: Schema,
    *,
    byte_size: int,
) -> Iterator[pa.RecordBatch]:
    """Cast batched Python objects into Arrow batches matching ``schema``."""
    options = CastOptions(target=schema, safe=False, byte_size=byte_size)

    def _tables() -> Iterator[pa.Table]:
        for group in objects_per_batch:
            yield any_to_arrow_table(group)

    return any_to_arrow_batch_iterator(_tables(), options=options)


def _dynamic_rows(batches: Iterator[pa.RecordBatch]) -> Iterator[Any]:
    """Yield unpickled inner objects from a dynamic-schema batch stream."""
    for batch in batches:
        col = batch.column(0)
        for i in range(batch.num_rows):
            yield cloudpickle.loads(col[i].as_py())


def _typed_rows(batches: Iterator[pa.RecordBatch]) -> Iterator[dict[str, Any]]:
    """Yield row-dicts from a typed batch stream."""
    for batch in batches:
        for row in batch.to_pylist():
            yield row


# ---------------------------------------------------------------------------
# Dataset re-export
# ---------------------------------------------------------------------------
#
# Historically :class:`Dataset` lived here as a standalone Spark-DataFrame
# wrapper while :class:`yggdrasil.io.tabular.SparkTabular` carried the
# :class:`Tabular` contract. They've been merged into one class,
# :class:`yggdrasil.spark.tabular.Dataset`, so:
#
# * the Tabular surface (read_arrow_batches / write_arrow_batches /
#   read_spark_frame / write_spark_frame) and
# * the rich Dataset surface (map / apply / filter / explode / cast,
#   executor module shipping, schema inference, the
#   :meth:`__getattr__` DataFrame proxy)
#
# live on one type. Existing call sites that import :class:`Dataset` from
# ``yggdrasil.spark.frame`` or :class:`SparkTabular` from
# ``yggdrasil.io.tabular`` keep working unchanged — both names resolve
# to the same class. The module-level helpers above
# (``is_dynamic_schema``, ``_emit_pickled``, ``_typed_cast``,
# ``_dynamic_rows``, ``_typed_rows``,
# ``DYNAMIC_SCHEMA``, ``PICKLE_COLUMN_NAME``) stay here —
# :class:`Dataset` imports them where needed.

def __getattr__(name: str) -> Any:
    """Lazy ``Dataset`` accessor — defers the import to break the cycle.

    Importing :class:`Dataset` at module top-level would form a cycle:
    ``yggdrasil.spark.tabular`` (where :class:`Dataset` lives) imports
    :class:`yggdrasil.io.tabular.Tabular`, whose package ``__init__``
    re-imports :class:`Dataset` — and our module is only half-loaded at
    that point. Resolving the attribute on first access (PEP 562) breaks
    the cycle without forcing every caller to spell out the new import
    path. ``SparkTabular`` lands on the same class for back-compat.
    """
    if name in ("Dataset", "SparkTabular"):
        from yggdrasil.spark.tabular import SparkDataset as _Dataset

        # Cache on the module so subsequent lookups skip the resolver.
        globals()[name] = _Dataset
        return _Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
