"""In-memory :class:`Tabular` holders.

Two leaf classes that wrap data already on the driver / executor
without serializing to bytes:

- :class:`ArrowTabular` — holds Arrow record batches plus the schema.
- :class:`SparkTabular` — holds a (mutable) Spark DataFrame.

Both implement the full :class:`Tabular` contract; reads return the
held data with no copy, writes mutate it in place subject to the save
mode. Use these when you want a :class:`Tabular` over data you
already have in memory and don't want to round-trip through IPC bytes.
"""

from .base import O, Tabular
from yggdrasil.io.tabular.arrow import ArrowTabular
from yggdrasil.io.tabular.spark import SparkTabular
from yggdrasil.io.tabular.lazy import LazyTabular
from yggdrasil.io.tabular.union import UnionTabular

__all__ = [
    "O",
    "Tabular",
    "ArrowTabular",
    "SparkTabular",
    "LazyTabular",
    "UnionTabular",
]
