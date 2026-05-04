from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator

import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from pyspark.cloudpickle import dumps, loads

from yggdrasil.arrow.cast import any_to_arrow_batch_iterator, any_to_arrow_table
from yggdrasil.data import schema as schema_builder, field as field_builder, Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.environ import PyEnv

__all__ = [
    "DynamicFrame",
    "is_dynamic_schema"
]


PICKLE_COLUMN_NAME = "_pickle"
DYNAMIC_SCHEMA = schema_builder(
    [
        field_builder(
            name=PICKLE_COLUMN_NAME,
            arrow_type=pa.binary(),
            nullable=False,
            metadata={"format": "binary"},
            tags={
                "namespace": "yggdrasil.spark.frame"
            }
        )
    ]
)


_ARROW_DYNAMIC_SCHEMA = pa.schema([
    pa.field(PICKLE_COLUMN_NAME, pa.binary(), nullable=False),
])


def _spark_dynamic_schema() -> StructType:
    return StructType([
        StructField(PICKLE_COLUMN_NAME, BinaryType(), nullable=False),
    ])


def is_dynamic_schema(obj: Any) -> bool:
    schema = Schema.from_any(obj)

    if len(schema) != 1:
        return False

    first = schema.get(0)

    return first.name == PICKLE_COLUMN_NAME and pa.types.is_binary(first.arrow_type)


def inputs_map_partition(
    function_pickle: bytes,
    batches: Iterator[pa.RecordBatch],
    *,
    byte_size: int = 128 * 1024 * 1024,
) -> Iterator[pa.RecordBatch]:
    func = loads(function_pickle)

    out: list[dict[str, bytes]] = []
    out_bytes = 0

    for batch in batches:
        col = batch.column(0)

        for i in range(batch.num_rows):
            inp = loads(col[i].as_py())
            result = func(inp)
            ser = dumps(result)

            if out and (out_bytes + len(ser) > byte_size):
                yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA) # noqa
                out = []
                out_bytes = 0

            out.append({PICKLE_COLUMN_NAME: ser})
            out_bytes += len(ser)

    if out:
        yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA) # noqa


def _iter_unpickled(batches: Iterator[pa.RecordBatch]) -> Iterator[Any]:
    """Yield the unpickled inner object from each row of a dynamic batch stream."""
    for batch in batches:
        col = batch.column(0)
        for i in range(batch.num_rows):
            yield loads(col[i].as_py())


def _iter_unpickled_groups(batches: Iterator[pa.RecordBatch]) -> Iterator[list[Any]]:
    """Yield one list of unpickled objects per input batch.

    Grouping per input batch lets the downstream tabular conversion
    take the fast ``pa.Table.from_pylist`` path when the inner objects
    are homogeneous record-shaped (dicts, dataclasses) — much cheaper
    than a per-row trip through the cast registry.
    """
    for batch in batches:
        col = batch.column(0)
        n = batch.num_rows
        if n == 0:
            continue
        yield [loads(col[i].as_py()) for i in range(n)]


def outputs_map_partition(
    batches: Iterator[pa.RecordBatch],
    schema: Schema,
    *,
    byte_size: int = 128 * 1024 * 1024,
) -> Iterator[pa.RecordBatch]:
    schema = Schema.from_any(schema)

    def _tables() -> Iterator[pa.Table]:
        for group in _iter_unpickled_groups(batches):
            yield any_to_arrow_table(group)

    return any_to_arrow_batch_iterator(
        _tables(),
        options=CastOptions(
            target_field=schema,
            safe=False,
            byte_size=byte_size,
        ),
    )


@dataclass(frozen=True, slots=True)
class DynamicFrame:
    df: DataFrame

    @property
    def sparkSession(self) -> SparkSession:
        return self.df.sparkSession

    @property
    def schema(self):
        return self.df.schema

    def __iter__(self) -> Iterator[Any]:
        return self.to_local_iterator()

    def count(self) -> int:
        return self.df.count()

    @classmethod
    def from_iterable(
        cls,
        items: Iterable[Any],
        *,
        spark_session: SparkSession | None = None,
    ) -> "DynamicFrame":
        """Build a ``DynamicFrame`` directly from an in-memory iterable.

        Each element is pickled into its own row. Use this when no map
        function is needed up front — call :meth:`map` or :meth:`cast`
        downstream to apply transforms or impose a schema.
        """
        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )

        df = spark_session.createDataFrame(
            ((dumps(x),) for x in items),
            schema=_spark_dynamic_schema(),
        )
        return cls(df=df)

    @classmethod
    def parallelize(
        cls,
        function: Callable[[Any], Any],
        inputs: Iterable[Any],
        *,
        spark_session: SparkSession | None = None,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        if spark_session is None:
            spark_session = PyEnv.spark_session(
                create=True,
                install_spark=False,
                import_error=True,
            )

        function_pickle = dumps(function)

        input_df = spark_session.createDataFrame(
            ((dumps(x),) for x in inputs),
            schema=_spark_dynamic_schema(),
        )

        result_df = input_df.mapInArrow(
            lambda batches: inputs_map_partition(
                function_pickle=function_pickle,
                batches=batches,
                byte_size=byte_size,
            ),
            schema=_spark_dynamic_schema(),
        )

        return cls(df=result_df)

    def explode(
        self,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Explode a ``DynamicFrame[Iterable[T]]`` into a ``DynamicFrame[T]``.

        Each pickled row must be an iterable; every element is serialised into
        its own row in the returned DynamicFrame.  This is the gather phase
        counterpart to :meth:`parallelize` when the mapped function returns a
        collection rather than a single value.
        """
        def _explode_batches(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            out: list[dict[str, bytes]] = []
            out_bytes = 0
            for batch in batches:
                col = batch.column(0)
                for i in range(batch.num_rows):
                    for item in loads(col[i].as_py()):
                        ser = dumps(item)
                        if out and out_bytes + len(ser) > byte_size:
                            yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa
                            out = []
                            out_bytes = 0
                        out.append({PICKLE_COLUMN_NAME: ser})
                        out_bytes += len(ser)
            if out:
                yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa

        result_df = self.df.mapInArrow(_explode_batches, schema=_spark_dynamic_schema())
        return type(self)(df=result_df)

    def map(
        self,
        function: Callable[[Any], Any],
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        function_pickle = dumps(function)

        result_df = self.df.mapInArrow(
            lambda batches: inputs_map_partition(
                function_pickle=function_pickle,
                batches=batches,
                byte_size=byte_size,
            ),
            schema=_spark_dynamic_schema(),
        )

        return type(self)(df=result_df)

    def apply(
        self,
        function: Callable[[Any], Any],
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Map ``function`` over each row, optionally casting outputs against ``schema``.

        Without a schema this is :meth:`map` — one function output, one
        re-pickled row.

        With a schema the function may return any tabular shape
        (dict, dataclass, list-of-rows, polars/pandas/arrow frame,
        ``pa.RecordBatch``). Outputs are streamed through
        :func:`any_to_arrow_batch_iterator` so the per-batch Arrow cast
        and ``byte_size`` rechunking run in one pass; each row of the
        cast batches becomes one re-pickled (``dict``) row in the
        returned DynamicFrame. Useful when the function fans out
        (single input → many output rows) or returns loosely-shaped
        data that needs normalisation before downstream stages.
        """
        if schema is None:
            return self.map(function, byte_size=byte_size)

        schema = Schema.from_any(schema)
        function_pickle = dumps(function)

        def _apply_batches(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            func = loads(function_pickle)
            options = CastOptions(
                target_field=schema,
                safe=False,
                byte_size=byte_size,
            )

            def _outputs() -> Iterator[Any]:
                for batch in batches:
                    col = batch.column(0)
                    n = batch.num_rows
                    if n == 0:
                        continue
                    yield [func(loads(col[i].as_py())) for i in range(n)]

            out: list[dict[str, bytes]] = []
            out_bytes = 0
            for cast_batch in any_to_arrow_batch_iterator(_outputs(), options=options):
                for row in cast_batch.to_pylist():
                    ser = dumps(row)
                    if out and out_bytes + len(ser) > byte_size:
                        yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa
                        out = []
                        out_bytes = 0
                    out.append({PICKLE_COLUMN_NAME: ser})
                    out_bytes += len(ser)
            if out:
                yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa

        result_df = self.df.mapInArrow(_apply_batches, schema=_spark_dynamic_schema())
        return type(self)(df=result_df)

    def filter(
        self,
        predicate: Callable[[Any], bool],
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> "DynamicFrame":
        """Drop rows where ``predicate(unpickled_obj)`` is false.

        The predicate is evaluated against the unpickled inner object,
        not the binary payload — so callers can write schemaless filters
        ("keep dicts where status == 'ok'") without first casting.
        """
        predicate_pickle = dumps(predicate)

        def _filter_batches(
            batches: Iterator[pa.RecordBatch],
        ) -> Iterator[pa.RecordBatch]:
            pred = loads(predicate_pickle)
            out: list[dict[str, bytes]] = []
            out_bytes = 0
            for batch in batches:
                col = batch.column(0)
                for i in range(batch.num_rows):
                    ser = col[i].as_py()
                    if not pred(loads(ser)):
                        continue
                    if out and out_bytes + len(ser) > byte_size:
                        yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa
                        out = []
                        out_bytes = 0
                    out.append({PICKLE_COLUMN_NAME: ser})
                    out_bytes += len(ser)
            if out:
                yield pa.RecordBatch.from_pylist(out, schema=_ARROW_DYNAMIC_SCHEMA)  # noqa

        result_df = self.df.mapInArrow(_filter_batches, schema=_spark_dynamic_schema())
        return type(self)(df=result_df)

    def collect(self) -> list[Any]:
        return [loads(row[PICKLE_COLUMN_NAME]) for row in self.df.collect()]

    def to_local_iterator(self) -> Iterator[Any]:
        for row in self.df.toLocalIterator():
            yield loads(row[PICKLE_COLUMN_NAME])

    def cast(
        self,
        schema: Schema,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> DataFrame:
        """Materialise rows against ``schema`` as a typed Spark DataFrame.

        Each pickled inner object is unpickled inside the executor and
        streamed through :func:`any_to_arrow_batch_iterator`, which runs
        the per-batch Arrow cast and ``byte_size`` rechunking in one pass.
        """
        schema = Schema.from_any(schema)

        return self.df.mapInArrow(
            lambda batches: outputs_map_partition(
                batches=batches,
                schema=schema,
                byte_size=byte_size,
            ),
            schema=schema.to_spark_schema(),
        )

    def toArrow(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ) -> pa.Table:
        """Return a ``pa.Table``. With ``schema=None`` the table is built by
        inferring shape from the unpickled rows on the driver."""
        if schema is None:
            return any_to_arrow_table(
                self.to_local_iterator(),
                options=CastOptions(byte_size=byte_size, safe=False),
            )
        return self.cast(schema=schema, byte_size=byte_size).toArrow()

    def toPandas(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        """Return a ``pandas.DataFrame``. Schemaless when ``schema is None``."""
        if schema is None:
            return self.toArrow(byte_size=byte_size).to_pandas()
        return self.cast(schema=schema, byte_size=byte_size).toPandas()

    def toPolars(
        self,
        schema: Schema | None = None,
        *,
        byte_size: int = 128 * 1024 * 1024,
    ):
        """Return a ``polars.DataFrame``. Schemaless when ``schema is None``."""
        from yggdrasil.polars.lib import polars as pl
        return pl.from_arrow(self.toArrow(schema=schema, byte_size=byte_size))