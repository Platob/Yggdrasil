from typing import Optional

import pyarrow as pa

from ...libs.sparklib import (
    SparkSession,
    arrow_field_to_spark,
    pyspark,
    require_pyspark,
)
from .arrow import ArrowCastOptions, cast_arrow_table

__all__ = ["cast_spark_dataframe"]


@require_pyspark(active_session=True)
def cast_spark_dataframe(
    dataframe: "pyspark.sql.DataFrame",
    arrow_schema: pa.Schema,
    options: Optional[ArrowCastOptions] = None,
) -> "pyspark.sql.DataFrame":
    """Cast a Spark DataFrame to the provided Arrow schema.

    The DataFrame is materialized locally for casting consistency with the Arrow
    implementation and then reconstructed using the active Spark session.
    """

    options = options or ArrowCastOptions()

    pandas_df = dataframe.toPandas()
    arrow_table = pa.Table.from_pandas(pandas_df, preserve_index=False)
    casted = cast_arrow_table(arrow_table, arrow_schema, options)

    spark_schema = pyspark.sql.types.StructType(
        [arrow_field_to_spark(field) for field in casted.schema]
    )

    session = dataframe.sparkSession or SparkSession.getActiveSession()
    return session.createDataFrame(casted.to_pandas(), schema=spark_schema)
