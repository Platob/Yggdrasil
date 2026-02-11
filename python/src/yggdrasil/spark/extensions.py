"""Spark DataFrame extension helpers for aliases and resampling."""

import datetime
import inspect
import re
from typing import List, Union, Optional, Iterable, Callable, Mapping, Any

import pyarrow as pa
import pyspark.sql as SparkSQL
import pyspark.sql.functions as F
import pyspark.sql.types as T

from .cast import spark_type_to_arrow_type, arrow_field_to_spark_field
from ..pandas.lib import pandas
from ..polars.cast import arrow_table_to_polars_dataframe
from ..types.cast.cast_options import CastOptions
from ..types.cast.registry import convert

__all__ = []

_COL_RE = re.compile(r"Column<\s*['\"]?`?(.+?)`?['\"]?\s*>")

def getAliases(
    obj: Union[SparkSQL.DataFrame, SparkSQL.Column, str, Iterable[Union[SparkSQL.DataFrame, SparkSQL.Column, str]]],
    full: bool = True,
) -> list[str]:
    """Return aliases for Spark columns/dataframes or collections.

    Args:
        obj: Spark DataFrame/Column, string, or iterable of these.
        full: Whether to return full qualified names.

    Returns:
        List of alias strings.
    """
    if obj is None:
        return []

    if not isinstance(obj, (list, tuple, set)):
        return [getAlias(obj, full)]

    return [getAlias(_, full) for _ in obj]


def getAlias(
    obj: Union[SparkSQL.DataFrame, SparkSQL.Column, str],
    full: bool = True,
) -> str:
    """
    Parse a column name out of a PySpark Column repr string.
    """
    if isinstance(obj, str):
        return obj

    result = str(obj)

    if isinstance(obj, SparkSQL.DataFrame):
        jdf = getattr(obj, "_jdf", None)

        if not jdf:
            return None

        plan = jdf.queryExecution().analyzed().toString()
        for line in plan.split("\n"):
            line = line.strip()
            if line.startswith("SubqueryAlias "):
                result = line.split("SubqueryAlias ")[1].split(",")[0].strip()
                break
    elif isinstance(obj, SparkSQL.Column):
        m = _COL_RE.search(result)
        if m:
            result = m.group(1)
            if not full:
                result = result.split(".")[-1]
    else:
        raise ValueError(f"Cannot get alias for spark {type(obj)}")

    return result


def latest(
    df: SparkSQL.DataFrame,
    partitionBy: List[Union[str, SparkSQL.Column]],
    orderBy: List[Union[str, SparkSQL.Column]],
) -> SparkSQL.DataFrame:
    """Return the latest rows per partition based on ordering.

    Args:
        df: Spark DataFrame.
        partitionBy: Columns to partition by.
        orderBy: Columns to order by.

    Returns:
        Spark DataFrame with latest rows per partition.
    """
    partition_col_names = getAliases(partitionBy)
    order_col_names = getAliases(orderBy)

    window_spec = (
        SparkSQL.Window
        .partitionBy(*partition_col_names)
        .orderBy(*[df[_].desc() for _ in order_col_names])
    )

    return (
        df.withColumn("__rn", F.row_number().over(window_spec))
        .filter(F.col("__rn") == 1)
        .drop("__rn")
    )


def _infer_time_col_spark(df: SparkSQL.DataFrame) -> str:
    """
    Match the Polars extension behavior: if time not provided, pick the first TimestampType column.
    (Datetime-only inference; DateType does NOT count.)
    """
    for f in df.schema.fields:
        if isinstance(f.dataType, T.TimestampType):
            return f.name
    raise ValueError("resample: time not provided and no TimestampType column found in Spark schema.")


def _filter_kwargs_for_callable(fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter kwargs to only those accepted by the callable.

    Args:
        fn: Callable to inspect.
        kwargs: Candidate keyword arguments.

    Returns:
        Filtered keyword arguments.
    """
    sig = inspect.signature(fn)  # type: ignore[arg-type]
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if (k in allowed and v is not None)}


def _append_drop_col_to_spark_schema(schema: "T.StructType", drop_col: str) -> "T.StructType":
    """Ensure the drop column exists in the Spark schema.

    Args:
        schema: Spark schema to augment.
        drop_col: Column name to add if missing.

    Returns:
        Updated Spark schema.
    """
    if drop_col in schema.fieldNames():
        return schema
    return T.StructType(list(schema.fields) + [T.StructField(drop_col, T.IntegerType(), True)])


def upsample(
    df: SparkSQL.DataFrame,
    time: Union[str, SparkSQL.Column],
    interval: Union[str, datetime.timedelta],
    partitionBy: Optional[List[Union[str, SparkSQL.Column]]] = None,
    fill: Optional[str] = "forward",
) -> SparkSQL.DataFrame:
    """
    Upsample using Polars via Arrow inside mapInArrow-style group apply.
    """
    df: SparkSQL.DataFrame = df

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
        """Apply upsample logic to a grouped Arrow table.

        Args:
            tb: Arrow table for a grouped partition.

        Returns:
            Arrow table with upsampled data.
        """
        from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

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


def resample(
    df: SparkSQL.DataFrame,
    every: Union[str, datetime.timedelta],
    time: Optional[Union[str, SparkSQL.Column]] = None,
    partitionBy: Optional[List[Union[str, SparkSQL.Column]]] = None,
    agg: Optional[Mapping[str, Any]] = None,
    fill: Optional[str] = "forward",
    period: Optional[Union[str, datetime.timedelta]] = None,
    offset: Optional[Union[str, datetime.timedelta]] = None,
    closed: str = "left",
    label: str = "left",
    start_by: str = "window",
    schema: Optional[Union["T.StructType", str]] = None,
) -> SparkSQL.DataFrame:
    """
    Spark DataFrame .resample(...) implemented via Polars inside applyInArrow.

    Behavior:
      - If agg is None: UPSAMPLE mode (insert missing timestamps), then fill (forward/backward/zero/none)
      - If agg is provided: DOWNSAMPLE mode using polars group_by_dynamic + aggregations

    Notes / constraints:
      - time can be omitted: we infer the first Spark TimestampType column (Datetime-only)
      - For agg != None, you SHOULD pass `schema` (Spark StructType or schema string).
        Spark requires the output schema for applyInArrow.
        If schema is not provided, we raise.
      - agg should be a dict mapping column -> aggregator, e.g. {"qty":"sum","px":"last"}.
        (Keep it picklable; don't pass Polars Exprs here.)

    This uses the Polars DataFrame `.resample(...)` extension you added, so the
    upsample-with-group keys stays correct even on older Polars versions.
    """
    df: SparkSQL.DataFrame = df

    partition_col_names = getAliases(partitionBy) or []

    # Infer time column if not given (Datetime-only: TimestampType)
    if time is None:
        time_col_name = _infer_time_col_spark(df)
    else:
        time_col_name = getAlias(time, full=False)

    # If no partition keys, force a single group like upsample()
    if not partition_col_names:
        drop_col = "__repart"
        df = df.withColumn(drop_col, F.lit(1))
        partition_col_names = [drop_col]
    else:
        drop_col = None

    # Input conversion options always based on the (possibly augmented) input schema
    in_options = CastOptions.check_arg(spark_type_to_arrow_type(df.schema))

    # Output schema/options:
    # - upsample mode defaults to input schema
    # - downsample mode requires explicit schema
    if agg is None:
        out_options = in_options
        spark_schema_for_apply = arrow_field_to_spark_field(out_options.target_field).dataType
    else:
        if schema is None:
            raise ValueError(
                "resample: agg provided but schema is None. "
                "Spark applyInArrow requires the output schema for aggregated resample."
            )
        spark_schema_for_apply = convert(schema, T.StructType)

        # If we injected drop_col, it will be present in Polars output (as a group key).
        # So the applyInArrow schema must include it too, then we drop it after.
        if drop_col is not None:
            spark_schema_for_apply = _append_drop_col_to_spark_schema(spark_schema_for_apply, drop_col)

        # Build output cast options from the declared output schema
        out_arrow_field = convert(spark_schema_for_apply, pa.Field)
        out_options = CastOptions.check_arg(out_arrow_field)

    def within_group(tb: pa.Table) -> pa.Table:
        """Apply resample logic to a grouped Arrow table.

        Args:
            tb: Arrow table for a grouped partition.

        Returns:
            Arrow table with resampled data.
        """
        from yggdrasil.polars.cast import polars_dataframe_to_arrow_table
        from yggdrasil.polars.extensions import resample

        pdf = arrow_table_to_polars_dataframe(tb, in_options)

        # Call your Polars extension resample
        if agg is None:
            res = resample(
                pdf,
                time_col=time_col_name,
                every=every,
                group_by=partition_col_names,
                fill=(fill or "none"),
                period=period,
                offset=offset,
                closed=closed,
                label=label,
                start_by=start_by,
            )
        else:
            res = resample(
                pdf,
                time_col=time_col_name,
                every=every,
                group_by=partition_col_names,
                agg=dict(agg),
                period=period,
                offset=offset,
                closed=closed,
                label=label,
                start_by=start_by,
            )

        return polars_dataframe_to_arrow_table(res, out_options)

    result = (
        df.groupBy(*partition_col_names)
        .applyInArrow(within_group, schema=spark_schema_for_apply)
    )

    if drop_col:
        result = result.drop(drop_col)

    return result


def checkJoin(
    df: SparkSQL.DataFrame,
    other: SparkSQL.DataFrame,
    on: Optional[Union[str, List[str], SparkSQL.Column, List[SparkSQL.Column]]] = None,
    *args,
    **kwargs,
):
    """Join two DataFrames with schema-aware column casting.

    Args:
        df: Left Spark DataFrame.
        other: Right Spark DataFrame.
        on: Join keys or mapping.
        *args: Positional args passed to join.
        **kwargs: Keyword args passed to join.

    Returns:
        Joined Spark DataFrame.
    """
    other = convert(other, SparkSQL.DataFrame)

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
    df: SparkSQL.DataFrame,
    func: Callable[[Iterable[pa.RecordBatch]], Iterable[pa.RecordBatch]],
    schema: Union["T.StructType", str],
    *args,
    **kwargs,
):
    """Wrap mapInArrow to enforce output schema conversion.

    Args:
        df: Spark DataFrame.
        func: Generator function yielding RecordBatches.
        schema: Output schema (Spark StructType or DDL string).
        *args: Positional args passed to mapInArrow.
        **kwargs: Keyword args passed to mapInArrow.

    Returns:
        Spark DataFrame with enforced schema.
    """
    spark_schema = convert(schema, T.StructType)
    arrow_schema = convert(schema, pa.Field)

    def patched(batches: Iterable[pa.RecordBatch]):
        """Convert batches yielded by user function to the target schema.

        Args:
            batches: Input RecordBatch iterable.

        Yields:
            RecordBatch instances conforming to the output schema.
        """
        for src in func(batches):
            yield convert(src, pa.RecordBatch, arrow_schema)

    return df.mapInArrow(
        patched,
        spark_schema,
        *args,
        **kwargs,
    )


def checkMapInPandas(
    df: SparkSQL.DataFrame,
    func: Callable[[Iterable["pandas.DataFrame"]], Iterable["pandas.DataFrame"]],
    schema: Union["T.StructType", str],
    *args,
    **kwargs,
):
    """Wrap mapInPandas to enforce output schema conversion.

    Args:
        df: Spark DataFrame.
        func: Generator function yielding pandas DataFrames.
        schema: Output schema (Spark StructType or DDL string).
        *args: Positional args passed to mapInPandas.
        **kwargs: Keyword args passed to mapInPandas.

    Returns:
        Spark DataFrame with enforced schema.
    """
    import pandas as _pd  # local import so we don't shadow the ..pandas module

    spark_schema = convert(schema, T.StructType)
    arrow_schema = convert(schema, pa.Field)

    def patched(batches: Iterable[_pd.DataFrame]):
        """Convert pandas batches yielded by user function to the target schema.

        Args:
            batches: Input pandas DataFrame iterable.

        Yields:
            pandas DataFrames conforming to the output schema.
        """
        for src in func(batches):
            yield convert(src, _pd.DataFrame, arrow_schema)

    return df.mapInPandas(
        patched,
        spark_schema,
        *args,
        **kwargs,
    )


# Monkey-patch only when PySpark is actually there
for method in [
    latest,
    upsample,
    resample,
    checkJoin,
    getAlias,
    checkMapInArrow,
    checkMapInPandas,
]:
    setattr(SparkSQL.DataFrame, method.__name__, method)

for method in [
    getAlias,
]:
    setattr(SparkSQL.Column, method.__name__, method)
