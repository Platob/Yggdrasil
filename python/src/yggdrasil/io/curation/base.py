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

from yggdrasil.data.constants import DEFAULT_FIELD_NAME

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.schema import Schema
    from yggdrasil.data.types import DataType


__all__ = ["Curator", "CurationResult", "ArrayLike", "TabularLike"]


ArrayLike = Union[pa.Array, pa.ChunkedArray]
TabularLike = Union[pa.Table, pa.RecordBatch]


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

    # ------------------------------------------------------ Field / Schema

    def curate_arrow_array(
        self,
        array: ArrayLike,
        name: str = DEFAULT_FIELD_NAME,
        nullable: bool = True,
    ) -> "tuple[Field, ArrayLike]":
        """Curate *array* and wrap the inferred dtype into a :class:`Field`.

        Returns ``(Field(name, inferred_dtype, nullable), curated_array)``
        so the caller gets the schema piece + the data piece in one call.
        The Field name defaults to :data:`DEFAULT_FIELD_NAME` (``""``) —
        pass it explicitly when you're naming a column.
        """
        from yggdrasil.data.data_field import Field

        result = self(array)
        return Field(name=name, dtype=result.dtype, nullable=nullable), result.array

    @classmethod
    def curate_arrow_tabular(
        cls,
        tabular: TabularLike,
        nullable: bool = True,
        **curator_kwargs: Any,
    ) -> "tuple[Schema, TabularLike]":
        """Curate every column of *tabular* and return ``(Schema, table)``.

        Picks the right :class:`Curator` subclass per column via
        :meth:`pick`, so a mixed table (some strings, some bytes, some
        already-typed numerics) all goes through one call. Columns whose
        dtype no subclass handles pass through unchanged with the
        original Arrow type wrapped in a :class:`Field`.

        Input shape is preserved: a :class:`pa.Table` comes back as a
        Table, a :class:`pa.RecordBatch` comes back as a RecordBatch.
        ``curator_kwargs`` are forwarded to each curator constructor.
        """
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import StructField

        if isinstance(tabular, pa.RecordBatch):
            columns = list(tabular.columns)
            names = list(tabular.schema.names)
            is_batch = True
        elif isinstance(tabular, pa.Table):
            columns = [tabular.column(i) for i in range(tabular.num_columns)]
            names = list(tabular.schema.names)
            is_batch = False
        else:
            raise TypeError(
                f"curate_arrow_tabular expects a pyarrow Table or RecordBatch; "
                f"got {type(tabular).__name__}. Wrap the data in "
                f"``pa.Table.from_pydict`` (or .from_arrays) first."
            )

        fields: list[Field] = []
        curated_columns: list[ArrayLike] = []
        for name, column in zip(names, columns):
            try:
                curator = cls.pick(column, **curator_kwargs)
            except TypeError:
                # No subclass claims this dtype — pass it through with
                # the existing Arrow type. Better than silently dropping
                # the column or forcing the caller into a typecheck.
                fields.append(Field(name=name, dtype=column.type, nullable=nullable))
                curated_columns.append(column)
                continue
            field, curated = curator.curate_arrow_array(
                column, name=name, nullable=nullable
            )
            fields.append(field)
            curated_columns.append(curated)

        schema: "Schema" = StructField(fields)
        arrow_schema = pa.schema(
            [pa.field(f.name, f.dtype.to_arrow(), nullable=f.nullable) for f in fields]
        )
        if is_batch:
            # RecordBatch needs flat Arrays, not ChunkedArrays. Combine
            # if any curator handed back a chunked result.
            flat = [
                c.combine_chunks() if isinstance(c, pa.ChunkedArray) else c
                for c in curated_columns
            ]
            return schema, pa.RecordBatch.from_arrays(flat, schema=arrow_schema)
        return schema, pa.Table.from_arrays(curated_columns, schema=arrow_schema)

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
