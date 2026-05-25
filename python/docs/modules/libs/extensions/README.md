# Engine cast extension helpers

Engine modules expose Arrow↔engine conversion helpers in their `cast.py` modules. Import the module once to register its converters into the shared registry, then use `convert()` directly or call the helpers explicitly.

## Register converters

```python
import yggdrasil.polars   # registers pl.DataFrame, pl.LazyFrame, pl.Series converters
import yggdrasil.pandas   # registers pd.DataFrame, pd.Series converters
import yggdrasil.spark    # registers Spark DataFrame converters
```

---

## Arrow helpers (`yggdrasil.arrow.cast`)

```python
import pyarrow as pa
from yggdrasil.arrow.cast import (
    cast_arrow_array,
    cast_arrow_tabular,
    cast_arrow_record_batch_reader,
    any_to_arrow_table,
)
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
opts = CastOptions(target_field=target)

# Cast table / RecordBatch / RecordBatchReader
out = cast_arrow_tabular(raw, opts)

# Cast a single array
arr = cast_arrow_array(pa.array(["1", "2"]), opts)

# From anything (dict, list of dicts, Polars, pandas, Spark, ...)
tbl = any_to_arrow_table({"id": [1, 2], "score": [9.1, 8.7]})

# Streaming (yields RecordBatch)
reader = pa.RecordBatchReader.from_batches(raw.schema, raw.to_batches())
for batch in cast_arrow_record_batch_reader(reader, opts):
    print(batch.num_rows)
```

---

## Polars helpers (`yggdrasil.polars.cast`)

```python
import pyarrow as pa
import polars as pl
from yggdrasil.polars.cast import (
    cast_polars_dataframe,
    cast_polars_lazyframe,
    any_to_polars_dataframe,
    polars_dataframe_to_arrow_table,
)
from yggdrasil.data.cast.options import CastOptions

target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
opts   = CastOptions(target_field=target)

df = pl.DataFrame({"id": ["1", "2"], "score": ["9.1", "8.7"]})

# Cast DataFrame to target schema
out: pl.DataFrame = cast_polars_dataframe(df, opts)

# Cast LazyFrame (stays lazy)
lf: pl.LazyFrame = cast_polars_lazyframe(df.lazy(), opts)

# Convert from Arrow / pandas / Spark
arrow_tbl = pa.table({"x": [1, 2, 3]})
out = any_to_polars_dataframe(arrow_tbl)   # from Arrow
out = any_to_polars_dataframe(df.to_pandas())  # from pandas

# Zero-copy Polars → Arrow
arrow = polars_dataframe_to_arrow_table(df)
```

---

## pandas helpers (`yggdrasil.pandas.cast`)

```python
import pyarrow as pa
import pandas as pd
from yggdrasil.pandas.cast import (
    cast_pandas_dataframe,
    cast_pandas_series,
    any_to_pandas_dataframe,
    arrow_table_to_pandas_dataframe,
    pandas_dataframe_to_arrow_table,
)
from yggdrasil.data.cast.options import CastOptions

target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
opts   = CastOptions(target_field=target)

df = pd.DataFrame({"id": ["1", "2"], "score": ["9.1", "8.7"]})
out = cast_pandas_dataframe(df, opts)

# Series-level cast
s = pd.Series(["1.1", "2.2"], name="score")
out_s = cast_pandas_series(s, opts)

# From Arrow
tbl = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
df  = arrow_table_to_pandas_dataframe(tbl)

# Pandas → Arrow
arrow = pandas_dataframe_to_arrow_table(df)

# From anything
out = any_to_pandas_dataframe(tbl)      # from Arrow
out = any_to_pandas_dataframe([{"a": 1}])  # from records
```

---

## Spark helpers (`yggdrasil.spark.cast`)

```python
import pyarrow as pa
from pyspark.sql import SparkSession
from yggdrasil.spark.cast import (
    cast_spark_dataframe,
    any_to_spark_dataframe,
    spark_dataframe_to_arrow,
)
from yggdrasil.data.cast.options import CastOptions

spark  = SparkSession.builder.getOrCreate()
target = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.float64())])
opts   = CastOptions(target_field=target)

sdf = spark.createDataFrame([{"id": "1", "v": "9.1"}])

# Cast Spark DataFrame to target schema
out = cast_spark_dataframe(sdf, opts)

# From Arrow / pandas / Polars
arrow_tbl = pa.table({"id": [1, 2], "v": [3.0, 4.0]})
sdf = any_to_spark_dataframe(arrow_tbl, spark=spark)

# Collect to Arrow (zero-copy via Arrow-enabled config)
tbl = spark_dataframe_to_arrow(sdf)
```

---

## Full cross-engine conversion chain

```python
import pyarrow as pa
import yggdrasil.polars
import yggdrasil.pandas
import yggdrasil.spark
from yggdrasil.data.cast.registry import convert
import polars as pl
import pandas as pd

arrow = pa.table({"id": [1, 2, 3], "score": [9.1, 8.7, 7.4]})

polars_df  = convert(arrow, pl.DataFrame)      # Arrow → Polars
pandas_df  = convert(polars_df, pd.DataFrame)  # Polars → pandas
arrow_back = convert(pandas_df, pa.Table)      # pandas → Arrow
```
