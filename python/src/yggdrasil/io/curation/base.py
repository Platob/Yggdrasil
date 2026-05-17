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
from typing import TYPE_CHECKING, Any, Iterable, Union

import pyarrow as pa

from yggdrasil.data.constants import DEFAULT_FIELD_NAME

if TYPE_CHECKING:
    from yggdrasil.data.data_field import Field
    from yggdrasil.data.schema import Schema
    from yggdrasil.data.types import DataType


__all__ = ["Curator", "CurationResult", "ArrayLike", "TabularLike"]


ArrayLike = Union[pa.Array, pa.ChunkedArray]
TabularLike = Union[pa.Table, pa.RecordBatch]


def _drop_all_null_rows(columns: list[ArrayLike]) -> list[ArrayLike]:
    """Filter *columns* to rows where at least one column is non-null.

    Vectorised: AND together the per-column null masks, invert, and
    pass the result to ``pa.compute.filter`` for each column. No
    Python row loop.
    """
    import pyarrow.compute as pc

    iterator = iter(columns)
    first = next(iterator)
    all_null_mask = pc.is_null(first)
    for col in iterator:
        all_null_mask = pc.and_(all_null_mask, pc.is_null(col))
    keep = pc.invert(all_null_mask)
    # Short-circuit: nothing to drop.
    if pc.all(keep).as_py():
        return columns
    return [pc.filter(col, keep) for col in columns]


def _align_batch_to_schema(batch: pa.RecordBatch, target: pa.Schema) -> pa.RecordBatch:
    """Project *batch* onto *target* — add missing columns as nulls,
    drop extras, cast overlapping dtypes to the target width.

    Used by the Spark Pass 2 path to land every cached partition's
    curated batch on the merged schema, regardless of which subset of
    columns / which (narrower) dtypes that partition originally
    inferred.
    """
    columns: list[pa.Array] = []
    for field in target:
        if field.name in batch.schema.names:
            col = batch.column(field.name)
            if col.type != field.type:
                col = pa.compute.cast(col, field.type, safe=False)
            columns.append(col)
        else:
            columns.append(pa.nulls(batch.num_rows, type=field.type))
    return pa.RecordBatch.from_arrays(columns, schema=target)


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
        drop_all_null_columns: bool = True,
        drop_all_null_rows: bool = True,
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
        ``curator_kwargs`` are forwarded to each curator constructor;
        unknown kwargs are filtered per curator by :meth:`pick`.

        ``purge_nulls`` is forced to ``False`` for the duration of this
        call — dropping null cells per column would shift the row
        alignment between columns. Override by passing
        ``purge_nulls=True`` if you really want the per-column purge.

        ``drop_all_null_columns`` (default ``True``) removes any column
        that inferred as :class:`NullType` — i.e., every cell was
        null. These columns carry no information and shipping them
        downstream tends to cause more grief (Spark / Parquet writers
        refuse ``null`` columns, schema diff tools flag them, etc.)
        than keeping them ever helps. Pass ``False`` to preserve them.

        ``drop_all_null_rows`` (default ``True``) drops rows where every
        remaining column is null. Same rationale — an all-null row is
        the tabular shape of "no data", and downstream code usually
        wants it gone. Pass ``False`` to keep them.
        """
        from yggdrasil.data.data_field import Field
        from yggdrasil.data.schema import StructField
        from yggdrasil.data.types import NullType

        # Tabular usage needs per-column row alignment; auto-purge
        # would tear that apart. Pin the safer default unless the
        # caller said otherwise.
        curator_kwargs.setdefault("purge_nulls", False)

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
                if drop_all_null_columns and pa.types.is_null(column.type):
                    continue
                fields.append(Field(name=name, dtype=column.type, nullable=nullable))
                curated_columns.append(column)
                continue
            field, curated = curator.curate_arrow_array(
                column, name=name, nullable=nullable
            )
            if drop_all_null_columns and isinstance(field.dtype, NullType):
                # All-null column → nothing the downstream can do with
                # it. Drop the whole pair before the schema gets built.
                continue
            fields.append(field)
            curated_columns.append(curated)

        schema: "Schema" = StructField(fields)
        arrow_schema = pa.schema(
            [pa.field(f.name, f.dtype.to_arrow(), nullable=f.nullable) for f in fields]
        )

        if drop_all_null_rows and curated_columns:
            curated_columns = _drop_all_null_rows(curated_columns)

        if is_batch:
            # RecordBatch needs flat Arrays, not ChunkedArrays. Combine
            # if any curator handed back a chunked result.
            flat = [
                c.combine_chunks() if isinstance(c, pa.ChunkedArray) else c
                for c in curated_columns
            ]
            # Empty schema (all columns dropped) needs an explicit
            # row count — pyarrow infers 0 when no arrays are given.
            if not flat:
                return schema, pa.RecordBatch.from_arrays([], schema=arrow_schema)
            return schema, pa.RecordBatch.from_arrays(flat, schema=arrow_schema)
        if not curated_columns:
            return schema, pa.Table.from_arrays([], schema=arrow_schema)
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
        drop_all_null_columns: bool = True,
        **curator_kwargs: Any,
    ) -> "tuple[Schema, Any]":
        """Curate a Spark DataFrame in two distributed ``mapInArrow`` passes.

        Returns ``(Schema, pyspark.sql.DataFrame)``. PySpark has no
        standalone column object you can curate in isolation
        (``Column`` is a reference, not data), so only the
        DataFrame-shaped entry point exists for Spark — pull a single
        column out as ``df.select("x")`` if you need it.

        Layout:

        1. **Pass 1** — ``df.mapInArrow`` runs :meth:`curate_arrow_tabular`
           per partition's RecordBatch. Each curated batch is serialized
           to an Arrow IPC stream and emitted alongside the partition's
           inferred yggdrasil :class:`Schema` (pickled). The intermediate
           DataFrame is cached so Pass 2 can replay it without re-running
           the curation.
        2. **Driver schema merge** — collect the pickled schemas, union
           the fields, and upcast overlapping dtypes so every column
           lands at the widest width the data needed across all
           partitions. (``int8`` in one partition + ``int32`` in another
           ⇒ ``int32`` in the output.)
        3. **Pass 2** — ``cached.mapInArrow`` reads each cached IPC blob,
           rebuilds the RecordBatch, fills missing columns with nulls,
           casts to the merged Arrow schema, and yields the result. The
           outer DataFrame carries the curated Spark schema so downstream
           DataFrame ops see the inferred types directly.

        Keeping the curation distributed (no ``toPandas()`` collect) is
        the whole point — on production-size frames the driver round-trip
        is the difference between "a few seconds" and "your YARN slot
        gets killed for OOM."
        """
        from yggdrasil.data.schema import StructField
        from yggdrasil.data.enums import Mode
        from yggdrasil.lazy_imports import spark_sql_module

        # ``pyspark.cloudpickle`` is the canonical serializer for
        # everything that crosses the Spark driver / worker boundary
        # in this codebase (see ``spark/frame.py``). It handles
        # closures + arbitrary Python objects (yggdrasil ``Schema``)
        # the stock pickle module can't.
        from pyspark.cloudpickle import dumps as _ser_dumps
        from pyspark.cloudpickle import loads as _ser_loads

        pst = spark_sql_module().types

        # Bind curator_kwargs into a closure-friendly form.
        kwargs = dict(curator_kwargs)
        nullable_arg = nullable
        drop_nulls = drop_all_null_columns

        # ---- Pass 1: per-partition curate + serialize ----------------

        cache_schema = pst.StructType(
            [
                pst.StructField("schema_pickle", pst.BinaryType(), False),
                pst.StructField("batch_ipc", pst.BinaryType(), False),
            ]
        )

        def _pass_one(
            batches: "Iterable[pa.RecordBatch]",
        ) -> "Iterable[pa.RecordBatch]":
            for batch in batches:
                if batch.num_rows == 0:
                    continue
                schema, curated = cls.curate_arrow_tabular(
                    batch,
                    nullable=nullable_arg,
                    drop_all_null_columns=drop_nulls,
                    **kwargs,
                )
                schema_pickle = _ser_dumps(schema)
                sink = pa.BufferOutputStream()
                with pa.ipc.new_stream(sink, curated.schema) as writer:
                    writer.write_batch(curated)
                ipc_bytes = sink.getvalue().to_pybytes()
                yield pa.RecordBatch.from_arrays(
                    [
                        pa.array([schema_pickle], type=pa.binary()),
                        pa.array([ipc_bytes], type=pa.binary()),
                    ],
                    names=["schema_pickle", "batch_ipc"],
                )

        # Two passes over the data: first to infer the schema from each
        # partition's batches, then to emit the typed result. ``.cache()``
        # avoids re-running pass one. Databricks Connect serverless rejects
        # ``PERSIST TABLE`` (``[NOT_SUPPORTED_WITH_SERVERLESS]``), so fall
        # back to the un-cached frame — pass one then runs twice, but the
        # alternative is a hard crash on serverless.
        intermediate = df.mapInArrow(_pass_one, schema=cache_schema)
        try:
            cached = intermediate.cache()
        except Exception:
            cached = intermediate

        # ---- Driver-side schema merge --------------------------------

        merged: "Schema | None" = None
        for row in cached.select("schema_pickle").toLocalIterator():
            partition_schema = _ser_loads(row["schema_pickle"])
            if merged is None:
                merged = partition_schema
            else:
                # ``Mode.APPEND`` keeps every field name from either side
                # and ``upcast=True`` widens overlapping dtypes (int8 +
                # int32 → int32, etc) — the right semantic for "all the
                # rows must fit".
                merged = merged.merge_with(
                    partition_schema, mode=Mode.APPEND, upcast=True
                )

        if merged is None:
            # No batches → empty frame. Return an empty Spark frame with
            # an empty schema so callers get a stable shape.
            empty_schema = StructField([])
            empty_spark_schema = pst.StructType([])
            return empty_schema, df.sparkSession.createDataFrame(
                [], schema=empty_spark_schema
            )

        merged_fields: list[Field] = list(merged.fields)
        merged_arrow_schema = pa.schema(
            [
                pa.field(f.name, f.dtype.to_arrow(), nullable=f.nullable)
                for f in merged_fields
            ]
        )
        merged_spark_schema = merged.to_spark_schema()

        # ---- Pass 2: rehydrate + cast to merged schema ---------------

        def _pass_two(
            batches: "Iterable[pa.RecordBatch]",
        ) -> "Iterable[pa.RecordBatch]":
            for meta_batch in batches:
                ipc_col = meta_batch.column("batch_ipc")
                for i in range(meta_batch.num_rows):
                    blob = ipc_col[i].as_py()
                    reader = pa.ipc.open_stream(pa.BufferReader(blob))
                    partition_batch = reader.read_next_batch()
                    yield _align_batch_to_schema(partition_batch, merged_arrow_schema)

        return merged, cached.mapInArrow(_pass_two, schema=merged_spark_schema)

    # ------------------------------------------------------------- utils

    @classmethod
    def pick(cls, array: ArrayLike, **kwargs: Any) -> "Curator":
        """Return a Curator instance that handles *array*'s dtype.

        Walks the subclass tree top-down and instantiates the first
        match with *kwargs*. Kwargs are filtered to the subclass's
        ``__init__`` signature so dataclass curators (with fixed
        fields like ``IntegerCurator(allow_unsigned=...)``) don't trip
        on options that only make sense for another family (e.g.
        ``purge_nulls`` on :class:`StringCurator`). Curators that
        accept ``**kwargs`` (``NestedCurator``) get the full forward.
        """
        for subclass in cls._iter_subclasses():
            if subclass.handles(array.type):
                return subclass(**cls._filter_kwargs(subclass, kwargs))
        raise TypeError(
            f"No Curator subclass handles Arrow type {array.type!r}. "
            f"Register one by subclassing Curator and implementing .handles()."
        )

    @staticmethod
    def _filter_kwargs(
        subclass: "type[Curator]", kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        import inspect

        params = inspect.signature(subclass.__init__).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

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
