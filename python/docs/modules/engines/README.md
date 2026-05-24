# Engine bridges — Polars, pandas, Spark

`yggdrasil.polars`, `yggdrasil.pandas`, and `yggdrasil.spark` each register their engine-specific converters into the shared cast registry when imported. After import, `convert(obj, pl.DataFrame)` / `convert(obj, pd.DataFrame)` / `convert(obj, SparkDataFrame)` dispatch through those converters automatically.

None of the three bridge packages are hard dependencies. Import via `yggdrasil.lazy_imports` so base installs that don't have the engine still get a clean "install extra X" error instead of an `ImportError`.

## One-liners

```python
from yggdrasil.polars.cast import cast_polars_dataframe   # Polars
from yggdrasil.pandas.cast import cast_pandas_dataframe   # pandas
import yggdrasil.spark                                     # register Spark converters
```

## Guard imports (base-install safe)

```python
# Use these in library / shared code instead of bare `import polars`
from yggdrasil.lazy_imports import polars   # raises helpful error if missing
from yggdrasil.lazy_imports import pandas
```

## Polars bridge (`yggdrasil.polars`)

Import once at startup to activate Polars converters:

```python
import yggdrasil.polars   # registers pl.DataFrame, pl.LazyFrame, pl.Series converters
```

Cast a Polars DataFrame to a target Arrow schema:

```python
import pyarrow as pa
import polars as pl
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.data.cast.options import CastOptions

df = pl.DataFrame({"id": ["1", "2", "3"], "score": ["9.1", "8.7", "7.4"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
out: pl.DataFrame = cast_polars_dataframe(df, CastOptions(target_field=target))
```

Cast a LazyFrame (stays lazy until `.collect()`):

```python
from yggdrasil.polars.cast import cast_polars_lazyframe

lf = pl.LazyFrame({"id": ["1", "2"], "value": ["4.2", "5.8"]})
casted_lf = cast_polars_lazyframe(lf, CastOptions(target_field=target))
result = casted_lf.collect()
```

Convert any supported object to Polars DataFrame:

```python
from yggdrasil.polars.cast import any_to_polars_dataframe
import pyarrow as pa

# from Arrow Table
arrow_table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
df = any_to_polars_dataframe(arrow_table)

# from pandas DataFrame
import pandas as pd
pandas_df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
df = any_to_polars_dataframe(pandas_df)

# from Spark DataFrame (materialises via Arrow)
# df = any_to_polars_dataframe(spark_df)
```

Arrow ↔ Polars (zero-copy):

```python
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

arrow = polars_dataframe_to_arrow_table(df)   # zero-copy
back  = pl.from_arrow(arrow)                  # zero-copy
```

TestCase base class (use in tests — handles optional-dependency skipping):

```python
from yggdrasil.polars.tests import PolarsTestCase

class MyTest(PolarsTestCase):
    def test_frame(self):
        df = self.df({"a": [1, 2], "b": ["x", "y"]})
        result = my_transform(df)
        self.assertFrameEqual(result, self.df({"a": [1, 2], "b": ["X", "Y"]}))
```

## pandas bridge (`yggdrasil.pandas`)

Import once to activate pandas converters:

```python
import yggdrasil.pandas   # registers pd.DataFrame, pd.Series converters
```

Cast a pandas Series to a target Arrow type:

```python
import pandas as pd
import pyarrow as pa
from yggdrasil.pandas.cast import cast_pandas_series
from yggdrasil.data.cast.options import CastOptions

s = pd.Series(["1.1", "2.2", "3.3"], name="value")
target = pa.schema([pa.field("value", pa.float64())])
out = cast_pandas_series(s, CastOptions(target_field=target))
```

Cast a pandas DataFrame:

```python
from yggdrasil.pandas.cast import cast_pandas_dataframe

df = pd.DataFrame({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
out = cast_pandas_dataframe(df, CastOptions(target_field=target))
```

Arrow ↔ pandas:

```python
from yggdrasil.pandas.cast import (
    arrow_table_to_pandas_dataframe,
    pandas_dataframe_to_arrow_table,
)

arrow_tbl = pa.table({"x": [1, 2, 3]})
df        = arrow_table_to_pandas_dataframe(arrow_tbl)
arrow_tbl = pandas_dataframe_to_arrow_table(df)
```

Convert any supported object to pandas:

```python
from yggdrasil.pandas.cast import any_to_pandas_dataframe

df = any_to_pandas_dataframe(arrow_table)   # from Arrow
df = any_to_pandas_dataframe(polars_df)     # from Polars
df = any_to_pandas_dataframe([{"a": 1}])    # from records list
```

TestCase base class:

```python
from yggdrasil.pandas.tests import PandasTestCase

class MyTest(PandasTestCase):
    def test_frame(self):
        df = self.df({"a": [1, 2], "b": [3, 4]})
        self.assertFrameEqual(result, expected)
```

## Spark bridge (`yggdrasil.spark`)

Import once to activate Spark converters and configure the SparkSession defaults:

```python
import yggdrasil.spark   # registers Spark converters + applies Arrow optimizations
```

TestCase base class (shares a single process-wide SparkSession):

```python
from yggdrasil.spark.tests import SparkTestCase

class MyTest(SparkTestCase):
    def test_spark(self):
        sdf = self.spark.createDataFrame([{"id": 1, "v": 2.5}])
        result = my_transform(sdf)
        self.assertSparkEqual(result, expected)
```

## Full cross-engine conversion chain

```python
import pyarrow as pa
import yggdrasil.polars
import yggdrasil.pandas
from yggdrasil.data.cast.registry import convert
import polars as pl
import pandas as pd

arrow_src = pa.table({"id": [1, 2, 3], "score": [9.1, 8.7, 7.4]})

# Arrow → Polars → pandas → Arrow
polars_df  = convert(arrow_src, pl.DataFrame)
pandas_df  = convert(polars_df, pd.DataFrame)
arrow_back = convert(pandas_df, pa.Table)
```

---

## Polars — vectorised expressions (no row loops)

```python
import polars as pl
import pyarrow as pa
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

df = pl.DataFrame({
    "raw_price": ["1,234.56", "2,345.67", "not-a-number", "789.00"],
    "currency":  ["USD", "EUR", None, "GBP"],
})

cleaned = (
    df
    .with_columns([
        pl.col("raw_price")
          .str.replace_all(",", "")
          .cast(pl.Float64, strict=False)
          .alias("price"),
        pl.col("currency").fill_null("USD"),
    ])
    .drop("raw_price")
    .filter(pl.col("price").is_not_null())
)

# Back to Arrow for downstream — zero copy
arrow_out = polars_dataframe_to_arrow_table(cleaned)
print(arrow_out)
```

---

## Polars — JSON decoding without Python loops

```python
import polars as pl

# Decode a column of JSON strings vectorially
df = pl.DataFrame({"payload": ['{"id":1,"v":2.5}', '{"id":2,"v":3.1}']})
decoded = df.with_columns(
    pl.col("payload").str.json_decode(pl.Struct({"id": pl.Int64, "v": pl.Float64}))
      .alias("parsed")
).unnest("parsed")
print(decoded)
# ┌─────┬─────┐
# │ id  │ v   │
```

---

## pandas — vectorised string normalization

```python
import pandas as pd
import pyarrow as pa
from yggdrasil.pandas.cast import pandas_dataframe_to_arrow_table

df = pd.DataFrame({
    "name":  ["  Alice ", "BOB", None, "carol"],
    "score": ["9.1", "bad", "8.7", "7.4"],
})

cleaned = df.copy()
cleaned["name"]  = df["name"].str.strip().str.title()
cleaned["score"] = pd.to_numeric(df["score"], errors="coerce")

arrow = pandas_dataframe_to_arrow_table(cleaned.dropna(subset=["score"]))
print(arrow)
```

---

## Spark — Dataset API for distributed transforms

```python
from yggdrasil.spark.tabular import Dataset
from yggdrasil.data import field, Schema
import pyarrow as pa

output_schema = Schema([
    field("event_id",   "string"),
    field("user_id",    "int64"),
    field("revenue",    "float64"),
    field("currency",   "string"),
], name="events")

def transform(batch: dict) -> list[dict]:
    return [
        {
            "event_id": row["id"],
            "user_id":  int(row["user"]),
            "revenue":  float(row["amount"]),
            "currency": row.get("ccy", "USD"),
        }
        for row in batch
    ]

# Distribute over Spark executors with Arrow bridge
ds = Dataset.from_iterable(
    [{"id": f"e{i}", "user": i % 100, "amount": i * 1.5} for i in range(10_000)],
    schema=output_schema,
    spark_session=spark,
)
result = ds.map(transform, schema=output_schema).toArrow()
print(result.num_rows)
```

---

## Spark — Spark TestCase shared session

```python
from yggdrasil.spark.tests import SparkTestCase
import pyarrow as pa

class TestEventTransform(SparkTestCase):
    def test_revenue_aggregation(self):
        raw = self.spark.createDataFrame([
            {"event": "purchase", "amount": 10.0},
            {"event": "purchase", "amount": 20.0},
            {"event": "refund",   "amount": -5.0},
        ])

        from pyspark.sql import functions as F
        result = (
            raw.groupBy("event")
               .agg(F.sum("amount").alias("total"))
               .orderBy("event")
        )
        self.assertSparkEqual(
            result,
            self.spark.createDataFrame([
                {"event": "purchase", "total": 30.0},
                {"event": "refund",   "total": -5.0},
            ]),
        )
```
