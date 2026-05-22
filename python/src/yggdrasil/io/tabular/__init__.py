"""In-memory :class:`Tabular` holders.

Two leaf classes that wrap data already on the driver / executor
without serializing to bytes:

- :class:`ArrowTabular` — holds Arrow record batches plus the schema.
  Lives at :mod:`yggdrasil.arrow.tabular`; re-exported here.
- :class:`Dataset` — holds a (mutable) Spark DataFrame. Lives at
  :mod:`yggdrasil.spark.tabular`; re-exported here under both the
  ``Dataset`` and legacy ``SparkTabular`` spellings (same class).

Both implement the full :class:`Tabular` contract; reads return the
held data with no copy, writes mutate it in place subject to the save
mode. Use these when you want a :class:`Tabular` over data you
already have in memory and don't want to round-trip through IPC bytes.

The engine-tabular re-exports are wired through :pep:`562` module
``__getattr__`` so the engine leaf modules (which themselves need
``Tabular`` from this package) can load standalone without a
circular-import dance — and so a base install that never touches
Spark doesn't pay the import cost of :mod:`yggdrasil.spark.tabular`.
"""

from .base import O, Tabular, is_tabular_source
from yggdrasil.io.tabular.lazy import LazyTabular
from yggdrasil.io.tabular.union import UnionTabular
from yggdrasil.io.tabular.engine import (
    SYSTEM_ENGINE,
    TabularEngine,
    TabularEntry,
)

__all__ = [
    "O",
    "Tabular",
    "ArrowTabular",
    "Dataset",
    "SparkTabular",
    "LazyTabular",
    "UnionTabular",
    "TabularEngine",
    "TabularEntry",
    "SYSTEM_ENGINE",
    "is_tabular_source",
]


def __getattr__(name: str):
    # Lazy engine-leaf re-exports. Imports trigger the engine module
    # only when the attribute is actually accessed, so:
    #   - ``from yggdrasil.io.tabular import ArrowTabular`` works exactly
    #     like before (one extra attribute lookup on first access).
    #   - Direct ``from yggdrasil.arrow.tabular import ArrowTabular`` /
    #     ``from yggdrasil.spark.tabular import Dataset`` loads the leaf
    #     without triggering its sibling — which would otherwise drag
    #     pyspark in on a base install that never asked for it.
    if name == "ArrowTabular":
        from yggdrasil.arrow.tabular import ArrowTabular
        return ArrowTabular
    if name in ("Dataset", "SparkTabular"):
        from yggdrasil.spark.tabular import Dataset
        return Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
