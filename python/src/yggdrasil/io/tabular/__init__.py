"""In-memory :class:`Tabular` holders.

Two leaf classes that wrap data already on the driver / executor
without serializing to bytes:

- :class:`ArrowTabular` — holds Arrow record batches plus the schema.
  Lives at :mod:`yggdrasil.arrow.tabular`; re-exported here.
- :class:`SparkDataset` — holds a (mutable) Spark DataFrame. Lives at
  :mod:`yggdrasil.spark.tabular`; re-exported here.

Both implement the full :class:`Tabular` contract; reads return the
held data with no copy, writes mutate it in place subject to the save
mode.

The engine-tabular re-exports are wired through :pep:`562` module
``__getattr__`` so the engine leaf modules can load standalone without
a circular-import dance.
"""

from .base import O, Tabular, is_tabular_source

__all__ = [
    "O",
    "Tabular",
    "ArrowTabular",
    "LazyTabular",
    "SparkDataset",
    "SparkTabular",
    "is_tabular_source",
]


def __getattr__(name: str):
    if name == "ArrowTabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        return ArrowTabular
    if name == "LazyTabular":
        from yggdrasil.saga.plan.lazy import LazyTabular
        return LazyTabular
    if name in ("SparkDataset", "SparkTabular"):
        from yggdrasil.spark.tabular import SparkDataset
        return SparkDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
