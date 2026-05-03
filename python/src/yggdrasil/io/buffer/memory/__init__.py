"""In-memory :class:`TabularIO` holders.

Two leaf classes that wrap data already on the driver / executor
without serializing to bytes:

- :class:`MemoryArrowIO` — holds Arrow record batches plus the schema.
- :class:`MemorySparkIO` — holds a (mutable) Spark DataFrame.

Both implement the full :class:`TabularIO` contract; reads return the
held data with no copy, writes mutate it in place subject to the save
mode. Use these when you want a :class:`TabularIO` over data you
already have in memory and don't want to round-trip through IPC bytes.
"""

from .arrow import MemoryArrowIO
from .spark import MemorySparkIO

__all__ = ["MemoryArrowIO", "MemorySparkIO"]
