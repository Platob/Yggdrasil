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

    # ----------------------------------------------------- Engine wrappers
    #
    # Each engine method round-trips through Arrow — that's the whole
    # point of having one curation pipeline. Series-shaped wrappers go
    # through ``curate_arrow_array`` (instance method, single curator);
    # frame-shaped wrappers go through ``curate_arrow_tabular``
    # (classmethod, picks per column, falls back to as-is for any
    # column no subclass claims).

    def curate_polars_series(
        self,
        series: Any,
        name: str | None = None,
        nullable: bool = True,
    ) -> "tuple[Field, Any]":
        """Curate a polars Series via the Arrow bridge.

        Returns ``(Field, polars.Series)``. The Series name on the way
        in becomes the Field name by default; pass *name* to override.
        """
        from yggdrasil.lazy_imports import polars_module

        pl = polars_module()
        arrow_arr = series.to_arrow()
        resolved_name = (
            name if name is not None else (series.name or DEFAULT_FIELD_NAME)
        )
        field, curated = self.curate_arrow_array(
            arrow_arr, name=resolved_name, nullable=nullable
        )
        return field, pl.Series(name=resolved_name, values=curated)

    @classmethod
    def curate_polars_dataframe(
        cls,
        df: Any,
        nullable: bool = True,
        **curator_kwargs: Any,
    ) -> "tuple[Schema, Any]":
        """Curate every column of a polars DataFrame.

        Returns ``(Schema, polars.DataFrame)``. Columns whose dtype no
        Curator subclass handles pass through unchanged, matching the
        contract of :meth:`curate_arrow_tabular`.
        """
        from yggdrasil.lazy_imports import polars_module

        pl = polars_module()
        arrow_table = df.to_arrow()
        schema, curated = cls.curate_arrow_tabular(
            arrow_table, nullable=nullable, **curator_kwargs
        )
        return schema, pl.from_arrow(curated)

    def curate_pandas_series(
        self,
        series: Any,
        name: str | None = None,
        nullable: bool = True,
    ) -> "tuple[Field, Any]":
        """Curate a pandas Series via the Arrow bridge.

        Returns ``(Field, pandas.Series)``. The Series ``.name`` is the
        default Field name and survives the round-trip.
        """
        arrow_arr = pa.Array.from_pandas(series)
        resolved_name = (
            name
            if name is not None
            else (series.name if series.name is not None else DEFAULT_FIELD_NAME)
        )
        field, curated = self.curate_arrow_array(
            arrow_arr, name=resolved_name, nullable=nullable
        )
        pandas_series = curated.to_pandas()
        pandas_series.name = resolved_name or None
        return field, pandas_series

    @classmethod
    def curate_pandas_dataframe(
        cls,
        df: Any,
        nullable: bool = True,
        **curator_kwargs: Any,
    ) -> "tuple[Schema, Any]":
        """Curate every column of a pandas DataFrame.

        Returns ``(Schema, pandas.DataFrame)``. The pandas index is
        dropped — re-attach it on the caller side if you need it
        (``preserve_index=False`` mirrors the canonical Arrow ↔ pandas
        bridge behaviour).
        """
        arrow_table = pa.Table.from_pandas(df, preserve_index=False)
        schema, curated = cls.curate_arrow_tabular(
            arrow_table, nullable=nullable, **curator_kwargs
        )
        return schema, curated.to_pandas()

    @classmethod
    def curate_spark_dataframe(
        cls,
        df: Any,
        nullable: bool = True,
        **curator_kwargs: Any,
    ) -> "tuple[Schema, Any]":
        """Curate every column of a Spark DataFrame via the pandas bridge.

        Returns ``(Schema, pyspark.sql.DataFrame)``. PySpark has no
        standalone column object you can curate in isolation
        (``Column`` is a reference, not data), so only the
        DataFrame-shaped entry point exists for Spark — pull a single
        column out as ``df.select("x")`` if you need it.

        Round-trips through ``df.toPandas()`` + ``spark.createDataFrame``,
        which uses Arrow under the hood when
        ``spark.sql.execution.arrow.pyspark.enabled`` is set (the
        default on modern PySpark).
        """
        spark_session = df.sparkSession
        pandas_df = df.toPandas()
        schema, curated = cls.curate_pandas_dataframe(
            pandas_df, nullable=nullable, **curator_kwargs
        )
        new_df = spark_session.createDataFrame(curated)
        return schema, new_df

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
