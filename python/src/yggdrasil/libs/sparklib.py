import functools

import pyarrow as pa

try:
    import pyspark  # type: ignore
    from pyspark.sql import SparkSession
    import pyspark.sql.types as T

    pyspark = pyspark
    SparkSession = SparkSession

    ARROW_TO_SPARK = {
        pa.bool_(): T.BooleanType(),
        pa.string(): T.StringType()
    }
except ImportError:
    pyspark = None
    SparkSession = None

    ARROW_TO_SPARK = {}

SPARK_TO_ARROW = {
    v: k
    for k, v in ARROW_TO_SPARK.items()
}


def require_pyspark(_func=None, *, active_session: bool = False):
    """
    Can be used as:

    @require_pyspark
    def f(...): ...

    or

    @require_pyspark(active_session=True)
    def f(...): ...
    """

    def decorator_require_pyspark(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1) pyspark must be importable
            if pyspark is None:
                raise ImportError(
                    "pyspark is required to use this function. "
                    "Install it or run inside a Spark/Databricks environment."
                )

            # 2) Optionally require an active SparkSession
            if active_session:
                if SparkSession is None:
                    raise ImportError(
                        "pyspark.sql.SparkSession is required to check for an active session."
                    )
                if SparkSession.getActiveSession() is None:
                    raise RuntimeError(
                        "An active SparkSession is required to use this function. "
                        "Create one with SparkSession.builder.getOrCreate()."
                    )

            return func(*args, **kwargs)

        return wrapper

    # Used as @require_pyspark(active_session=True)
    if _func is None:
        return decorator_require_pyspark

    # Used as @require_pyspark
    return decorator_require_pyspark(_func)


__all__ = [
    "pyspark",
    "require_pyspark",
    "SparkSession",
    "arrow_type_to_spark",
    "spark_type_to_arrow",
    "arrow_field_to_spark",
    "spark_field_to_arrow"
]


@require_pyspark
def arrow_type_to_spark(arrow_type: pa.DataType) -> "pyspark.sql.types.DataType":
    existing = ARROW_TO_SPARK.get(arrow_type)

    if existing is not None:
        return existing

    raise ValueError()


def spark_type_to_arrow(spark_type: "pyspark.sql.types.DataType") -> pa.DataType:
    existing = SPARK_TO_ARROW.get(spark_type)

    if existing is not None:
        return existing

    raise ValueError()


def arrow_field_to_spark(arrow_field: pa.Field) -> "pyspark.sql.types.StructField":
    spark_type = arrow_type_to_spark(arrow_field.type)

    return pyspark.sql.types.StructField(
        arrow_field.name,
        spark_type,
        arrow_field.nullable,
        metadata={
            k.decode(): v.decode()
            for k, v in arrow_field.metadata.items()
        } if arrow_field.metadata else {}
    )


def spark_field_to_arrow(spark_field: "pyspark.sql.types.StructField") -> pa.Field:
    arrow_type = spark_type_to_arrow(spark_field.dataType)

    return pa.field(
        spark_field.name,
        arrow_type,
        spark_field.nullable,
        metadata={
            k.encode(): v.encode() if isinstance(v, str) else str(v).encode()
            for k, v in spark_field.metadata.items()
            if k and v
        } if spark_field.metadata else {}
    )