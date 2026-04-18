from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pyarrow as pa
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import BinaryType, StructField, StructType

from yggdrasil.data import schema as schema_builder, field as field_builder, Schema
from yggdrasil.data.cast import CastOptions, convert
from yggdrasil.environ import PyEnv
from yggdrasil.pickle.ser.serde import loads, dumps

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


def outputs_map_partition(
    batches: Iterator[pa.RecordBatch],
    schema: Schema,
    *,
    byte_size: int = 128 * 1024 * 1024,
) -> Iterator[pa.RecordBatch]:
    schema = Schema.from_any(schema)

    out_batches: list[pa.RecordBatch] = []
    out_bytes = 0

    def flush() -> Iterator[pa.RecordBatch]:
        nonlocal out_batches, out_bytes
        if out_batches:
            yield from out_batches
            out_batches = []
            out_bytes = 0

    for batch in batches:
        col = batch.column(0)

        for i in range(batch.num_rows):
            obj = loads(col[i].as_py())

            rb = convert(
                obj,
                target_hint=pa.RecordBatch,
                options=CastOptions(target_field=schema.to_arrow_schema(), safe=False),
            )

            if rb.num_rows == 0:
                continue

            rb_size = rb.nbytes

            if out_batches and (out_bytes + rb_size > byte_size):
                yield from flush()

            out_batches.append(rb)
            out_bytes += rb_size

    yield from flush()


@dataclass(frozen=True, slots=True)
class DynamicFrame:
    df: DataFrame

    @property
    def sparkSession(self):
        return self.df

    @property
    def schema(self):
        return self.df.schema

    @classmethod
    def parallelize(
        cls,
        function: Callable[[Any], Any],
        inputs: Iterator[Any],
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
        schema = Schema.from_any(schema)

        return self.df.mapInArrow(
            lambda batches: outputs_map_partition(
                batches=batches,
                schema=schema,
                byte_size=byte_size,
            ),
            schema=schema.to_spark_schema(),
        )

    def toArrow(self, schema: Schema | None = None):
        return self.cast(schema=schema).toArrow()