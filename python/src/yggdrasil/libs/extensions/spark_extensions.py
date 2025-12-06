import datetime
import re
from typing import List, Union, Optional, Iterable, Callable, TYPE_CHECKING

import pyarrow as pa

from ..pandaslib import pandas
from ..sparklib import (
    pyspark,
    SparkDataFrame,
    SparkColumn,
    SparkDataType,
    spark_type_to_arrow_type,
    arrow_field_to_spark_field,
)
from ...types.cast.registry import convert

if TYPE_CHECKING:  # pragma: no cover
    from ...types.cast.arrow_cast import CastOptions

# Try to import pyspark.sql stuff if pyspark is actually there
try:
    if pyspark is not None:
        import pyspark.sql
        import pyspark.sql.types as T
        import pyspark.sql.functions as F
    else:
        T = None
        F = None
except ImportError:  # safety net if env is weird
    pyspark = None
    T = None
    F = None

__all__ = []

_COL_RE = re.compile(r"Column<\s*['\"]?`?(.+?)`?['\"]?\s*>")


def _require_pyspark(fn_name: str) -> None:
    if pyspark is None or F is None or T is None:
        raise RuntimeError(
            f"{fn_name} requires PySpark to be available. "
            "pyspark is None or not importable in this environment."
        )


def getAliases(
    obj: Union[SparkDataFrame, SparkColumn, str, Iterable[Union[SparkDataFrame, SparkColumn, str]]],
    full: bool = True,
) -> list[str]:
    if obj is None:
        return []

    if not isinstance(obj, (list, tuple, set)):
        return [getAlias(obj, full)]

    return [getAlias(_, full) for _ in obj]


def getAlias(
    obj: Union[SparkDataFrame, SparkColumn, str],
    full: bool = True,
) -> str:
    """
    Parse a column name out of a PySpark Column repr string.

    Examples it will handle:
      "Column<'curve_name'>"        -> "curve_name"
      'Column<"`weird-name`">'      -> "weird-name"
      'Column<"table.col">'         -> "table.col"
      " Column<'  x  '> "           -> "  x  " (keeps whitespace inside quotes)

    Returns the extracted string, or the original string if it's already a str.
    """
    if isinstance(obj, str):
        return obj

    result = str(obj)

    if isinstance(obj, SparkDataFrame):
        _require_pyspark("getAlias")
        jdf = getattr(obj, "_jdf", None)

        if not jdf:
            return None

        plan = jdf.queryExecution().analyzed().toString()
        # Logical plan contains something like:
        # SubqueryAlias my_alias, ...
        for line in plan.split("\n"):
            line = line.strip()
            if line.startswith("SubqueryAlias "):
                result = line.split("SubqueryAlias ")[1].split(",")[0].strip()
                break
    elif isinstance(obj, SparkColumn):
        m = _COL_RE.search(result)
        if m:
            result = m.group(1)
            if not full:
                result = result.split(".")[-1]
    else:
        raise ValueError(f"Cannot get alias for spark {type(obj)}")

    return result


def safe_spark_column(obj: Union[str, SparkColumn], holder: SparkDataFrame) -> SparkColumn:
    """Convert string or Spark Column into a Spark Column safely."""
    _require_pyspark("safe_spark_column")

    if isinstance(obj, SparkColumn):
        return obj
    elif isinstance(obj, str):
        names = holder.schema.fieldNames()
        if names and obj in names:
            return holder[obj]
        return F.col(obj)
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return F.lit(obj)
    else:
        raise ValueError(f"Invalid type for obj: {type(obj)}")


def truncate(
    column: SparkColumn,
    dataType: SparkDataType,
    value: int,
    name: str = None,
) -> SparkColumn:
    _require_pyspark("truncate")

    name = getAlias(column, full=False) if name is None else name

    if isinstance(dataType, T.IntegerType) or isinstance(dataType, T.LongType):
        value = int(value)
        truncated = (column.cast(dataType) / F.lit(value)).cast(dataType) * F.lit(value)
        return truncated

    if isinstance(dataType, T.TimestampType):
        if isinstance(value, datetime.timedelta):
            value = value // datetime.timedelta(seconds=1)

        truncated = truncate(
            column=column.cast(T.LongType()),
            dataType=T.LongType(),
            value=value,
            name=name,
        ).cast(dataType)

        return truncated

    raise ValueError(f"Cannot truncate {dataType} with {type(value)}")


def latest(
    df: SparkDataFrame,
    partitionBy: List[Union[str, SparkColumn]],
    orderBy: List[Union[str, SparkColumn]],
) -> SparkDataFrame:
    """
    Return latest rows from Spark DataFrame based on grouping and ordering.
    Preserves original column case.
    """
    _require_pyspark("latest")

    partition_col_names = getAliases(partitionBy)
    order_col_names = getAliases(orderBy)

    window_spec = (
        pyspark.sql.Window
        .partitionBy(*partition_col_names)
        .orderBy(*[df[_].desc() for _ in order_col_names])
    )

    return (
        df.withColumn("__rn", F.row_number().over(window_spec))
        .filter(F.col("__rn") == 1)
        .drop("__rn")
    )


def withNextValue(
    df: SparkDataFrame,
    orderBy: Union[str, SparkColumn],
    name: str,
    partitionBy: Optional[List[Union[str, SparkColumn]]] = None,
) -> SparkDataFrame:
    """
    Add a column with the next value of a given column for each row.
    """
    _require_pyspark("withNextValue")

    partition_col_names = getAliases(partitionBy)
    order_col_names = getAliases(orderBy)

    window_spec = (
        pyspark.sql.Window
        .partitionBy(*partition_col_names)
        .orderBy(*order_col_names)
    )

    return df.withColumn(name, F.lead(order_col_names[0]).over(window_spec))


def upsample(
    df: SparkDataFrame,
    time: Union[str, SparkColumn],
    interval: Union[str, datetime.timedelta],
    partitionBy: Optional[List[Union[str, SparkColumn]]] = None,
    fill: Optional[str] = "forward",
) -> SparkDataFrame:
    """
    Upsample using Polars via Arrow inside mapInArrow-style group apply.
    """
    _require_pyspark("upsample")

    from ...types.cast.polars_cast import (
        arrow_table_to_polars_dataframe,
        polars_dataframe_to_arrow_table,
    )
    from ...types.cast.arrow_cast import CastOptions

    df: pyspark.sql.DataFrame = df

    time_col_name = getAlias(time, full=False)
    partition_col_names = getAliases(partitionBy) or []

    if not partition_col_names:
        drop_col = "__repart"
        df = df.withColumn(drop_col, F.lit(1))
        partition_col_names = [drop_col]
    else:
        drop_col = None

    options = CastOptions.check_arg(spark_type_to_arrow_type(df.schema))
    spark_schema = arrow_field_to_spark_field(options.target_field)

    def within_group(tb: pa.Table) -> pa.Table:
        res = (
            arrow_table_to_polars_dataframe(tb, options)
            .sort(time_col_name)
            .upsample(time_col_name, every=interval, group_by=partition_col_names)
        )

        if fill:
            res = res.fill_null(strategy=fill)

        return polars_dataframe_to_arrow_table(res, options)

    result = (
        df
        .groupBy(*partition_col_names)
        .applyInArrow(within_group, schema=spark_schema.dataType)
    )

    if drop_col:
        result = result.drop(drop_col)

    return result


def checkJoin(
    df: "pyspark.sql.DataFrame",
    other: "pyspark.sql.DataFrame",
    on: Optional[Union[str, List[str], SparkColumn, List[SparkColumn]]] = None,
    *args,
    **kwargs,
):
    _require_pyspark("checkJoin")

    other = convert(other, SparkDataFrame)

    if isinstance(on, str):
        on = [on]
    elif isinstance(on, dict):
        on = list(on.items())

    if isinstance(on, list):
        checked = []

        for item in on:
            if isinstance(item, str):
                item = (item, item)

            if isinstance(item, tuple) and len(item) == 2:
                self_field = df.schema[item[0]]
                other_field = other.schema[item[1]]

                if self_field.dataType != other_field.dataType:
                    other = other.withColumn(
                        self_field.name,
                        other[other_field.name].cast(self_field.dataType),
                    )

                item = self_field.name

            checked.append(item)

        on = checked

    return df.join(other, on, *args, **kwargs)


def checkMapInArrow(
    df: "pyspark.sql.DataFrame",
    func: Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]],
    schema: Union["T.StructType", str],
    *args,
    **kwargs,
):
    _require_pyspark("mapInArrow")

    spark_schema = convert(schema, T.StructType)
    arrow_schema = convert(schema, pa.Field)

    def patched(batches: Iterable[pa.RecordBatch]):
        for src in func(batches):
            yield convert(src, pa.RecordBatch, arrow_schema)

    return df.mapInArrow(
        patched,
        spark_schema,
        *args,
        **kwargs,
    )


def checkMapInPandas(
    df: "pyspark.sql.DataFrame",
    func: Callable[[Iterable["pandas.DataFrame"]], Iterable["pandas.DataFrame"]],
    schema: Union["T.StructType", str],
    *args,
    **kwargs,
):
    _require_pyspark("mapInPandas")

    import pandas as _pd  # local import so we don't shadow the ..pandas module

    spark_schema = convert(schema, T.StructType)
    arrow_schema = convert(schema, pa.Field)

    def patched(batches: Iterable[_pd.DataFrame]):
        for src in func(batches):
            yield convert(src, _pd.DataFrame, arrow_schema)

    return df.mapInPandas(
        patched,
        spark_schema,
        *args,
        **kwargs,
    )


# Monkey-patch only when PySpark is actually there
if pyspark is not None:
    for method in [
        latest,
        withNextValue,
        upsample,
        checkJoin,
        getAlias,
        checkMapInArrow,
        checkMapInPandas,
    ]:
        setattr(SparkDataFrame, method.__name__, method)

    for method in [
        truncate,
        getAlias,
    ]:
        setattr(SparkColumn, method.__name__, method)
