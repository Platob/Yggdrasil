# Engine-specific cast extensions

Per-engine type conversion helpers that extend the Arrow casting layer.

Each engine module exposes conversion utilities for moving data between Arrow and its native types.

## Polars

```python
from yggdrasil.polars.cast import (
    arrow_type_to_polars_type,    # pa.DataType → polars.DataType
    polars_type_to_arrow_type,    # polars.DataType → pa.DataType
    arrow_field_to_polars_field,  # pa.Field → polars.Field
    polars_field_to_arrow_field,  # polars.Field → pa.Field
    polars_dataframe_to_arrow_table,   # polars.DataFrame → pa.Table
    arrow_table_to_polars_dataframe,   # pa.Table → polars.DataFrame
    any_to_polars_dataframe,           # any supported type → polars.DataFrame
)
```

### Bootstrap: convert Arrow to Polars

```python
import pyarrow as pa
from yggdrasil.polars.cast import arrow_table_to_polars_dataframe

table = pa.table({"id": [1, 2], "val": ["a", "b"]})
df = arrow_table_to_polars_dataframe(table)
```

### Bootstrap: convert Polars to Arrow

```python
import polars as pl
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
table = polars_dataframe_to_arrow_table(df)
```

---

## Pandas

```python
from yggdrasil.pandas.cast import (
    pandas_dataframe_to_arrow_table,   # pandas.DataFrame → pa.Table
    arrow_table_to_pandas_dataframe,   # pa.Table → pandas.DataFrame
    pandas_series_to_arrow_array,      # pandas.Series → pa.Array
    arrow_array_to_pandas_series,      # pa.Array → pandas.Series
)
```

### Bootstrap: round-trip through Arrow

```python
import pandas as pd
import pyarrow as pa
from yggdrasil.pandas.cast import (
    pandas_dataframe_to_arrow_table,
    arrow_table_to_pandas_dataframe,
)

df = pd.DataFrame({"id": [1, 2], "score": [0.9, 0.8]})
table = pandas_dataframe_to_arrow_table(df)

# cast / transform using Arrow ...

df_out = arrow_table_to_pandas_dataframe(table)
```

---

## Spark

```python
from yggdrasil.spark.cast import (
    arrow_schema_to_spark_schema,   # pa.Schema → pyspark.sql.types.StructType
    cast_spark_dataframe,           # enforce Arrow schema on Spark DF
)
```

### Bootstrap: Spark schema from Arrow

```python
import pyarrow as pa
from yggdrasil.spark.cast import arrow_schema_to_spark_schema

arrow_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("name", pa.string()),
])

spark_schema = arrow_schema_to_spark_schema(arrow_schema)
# StructType([StructField('id', LongType(), True), ...])
```
