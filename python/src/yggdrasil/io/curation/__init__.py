"""Curation — auto-typing rules for raw Arrow columns.

A *curator* turns a "we don't know what this column is" Arrow array
into "this is actually an int64 / date / tz-aware timestamp / …"
and hands back both the inferred :class:`DataType` and the array
re-cast to it.

Public surface:

* :class:`Curator` — abstract base, fixes the
  ``handles`` / ``infer`` / ``curate`` contract.
* :class:`StringCurator` — concrete subclass for UTF-8 string columns
  (CSV / JSON-as-text / dict payloads). Handles trim + null-token
  normalization, then probes booleans → integers → floats → date →
  time → timestamp, picking the first family that absorbs every
  non-null cell. Timestamps are uniformized — mixed offsets collapse
  to a single ``timestamp(us, tz="UTC")``.

Pick a curator with :meth:`Curator.pick` when you have an array and
want the right subclass instantiated automatically:

>>> import pyarrow as pa
>>> from yggdrasil.io.curation import Curator
>>> arr = pa.array(["1", "2", "3"])
>>> result = Curator.pick(arr).curate(arr)
>>> result.dtype
Int64Type()
"""

from __future__ import annotations

from .base import ArrayLike, CurationResult, Curator, TabularLike
from .nested import NestedCurator
from .string import StringCurator

__all__ = [
    "ArrayLike",
    "CurationResult",
    "Curator",
    "NestedCurator",
    "StringCurator",
    "TabularLike",
]
