"""Abstract :class:`Curator` — the auto-typing rule engine surface.

A *curator* takes a raw Arrow array — the kind you get out of a CSV
reader, an HTTP JSON body, a Power Query payload, any boundary where
the schema is unknown — cleans the cells, picks the most specific
:class:`~yggdrasil.data.types.DataType` that still holds every non-null
value, and returns the array re-cast to that type.

The split is by *source* dtype on purpose: strings, bytes, and Python
objects all want different rule sets, but once a Curator has decided
what the column actually is, the rest of the pipeline (Schema, cast
registry, IO writers) takes over. Subclass :class:`Curator` to add a
new source family; do not reach in and special-case a dtype on the
base class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

import pyarrow as pa

if TYPE_CHECKING:
    from yggdrasil.data.types import DataType


__all__ = ["Curator", "CurationResult", "ArrayLike"]


ArrayLike = Union[pa.Array, pa.ChunkedArray]


@dataclass(frozen=True)
class CurationResult:
    """The outcome of :meth:`Curator.curate`.

    Carries both the re-cast Arrow array and the inferred
    :class:`DataType`. Callers usually want the array, but the type
    is the load-bearing piece for schema inference flows that record
    "this column is actually a timestamp(us, UTC)" and re-apply the
    cast on later batches.
    """

    array: ArrayLike
    dtype: "DataType"

    def __iter__(self):
        # Lets callers unpack: ``arr, dtype = curator.curate(arr)``.
        yield self.array
        yield self.dtype


class Curator(ABC):
    """Auto-typing rule engine for a single source Arrow family.

    Subclasses fix three things:

    * :meth:`handles` — which Arrow source dtypes this curator accepts.
    * :meth:`infer`   — the rule order that decides the inferred type.
    * :meth:`curate`  — the actual clean-and-cast pipeline.

    The base class only owns the dispatch + the friendly
    :class:`TypeError` you get when the wrong curator meets the wrong
    column. Concrete rules live in subclasses (``StringCurator`` for
    text columns; future ``BytesCurator`` / ``ObjectCurator`` for the
    binary and ``object`` families).
    """

    # -------------------------------------------------------------- API

    @classmethod
    @abstractmethod
    def handles(cls, dtype: pa.DataType) -> bool:
        """True iff this curator can curate arrays of *dtype*."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, array: ArrayLike) -> "DataType":
        """Pick the most specific :class:`DataType` for *array*.

        Cheaper than :meth:`curate` when the caller only needs the
        type — schema discovery, dtype probing, registry routing.
        """
        raise NotImplementedError

    @abstractmethod
    def curate(self, array: ArrayLike) -> CurationResult:
        """Clean *array* and re-cast it into the inferred type."""
        raise NotImplementedError

    def __call__(self, array: ArrayLike) -> CurationResult:
        if not self.handles(array.type):
            raise TypeError(
                f"{type(self).__name__} cannot curate Arrow type {array.type!r}. "
                f"Pick a Curator subclass whose .handles() matches your source "
                f"dtype, or call the right subclass directly."
            )
        return self.curate(array)

    # ------------------------------------------------------------- utils

    @classmethod
    def pick(cls, array: ArrayLike, **kwargs: Any) -> "Curator":
        """Return a Curator instance that handles *array*'s dtype.

        Walks the subclass tree top-down and instantiates the first
        match with *kwargs*. New subclasses register automatically by
        existing — no manual registry to keep in sync.
        """
        for subclass in cls._iter_subclasses():
            if subclass.handles(array.type):
                return subclass(**kwargs)
        raise TypeError(
            f"No Curator subclass handles Arrow type {array.type!r}. "
            f"Register one by subclassing Curator and implementing .handles()."
        )

    @classmethod
    def _iter_subclasses(cls):
        seen: set[type] = set()
        stack = list(cls.__subclasses__())
        while stack:
            sub = stack.pop()
            if sub in seen:
                continue
            seen.add(sub)
            stack.extend(sub.__subclasses__())
            yield sub
