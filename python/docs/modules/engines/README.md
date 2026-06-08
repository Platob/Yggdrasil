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
from yggdrasil.data.options import CastOptions

df = pl.DataFrame({"id": ["1", "2", "3"], "score": ["9.1", "8.7", "7.4"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
out: pl.DataFrame = cast_polars_dataframe(df, CastOptions(target=target))
```

Cast a LazyFrame (stays lazy until `.collect()`):

```python
from yggdrasil.polars.cast import cast_polars_lazyframe

lf = pl.LazyFrame({"id": ["1", "2"], "value": ["4.2", "5.8"]})
casted_lf = cast_polars_lazyframe(lf, CastOptions(target=target))
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
from yggdrasil.data.options import CastOptions

s = pd.Series(["1.1", "2.2", "3.3"], name="value")
target = pa.schema([pa.field("value", pa.float64())])
out = cast_pandas_series(s, CastOptions(target=target))
```

Cast a pandas DataFrame:

```python
from yggdrasil.pandas.cast import cast_pandas_dataframe

df = pd.DataFrame({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
out = cast_pandas_dataframe(df, CastOptions(target=target))
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
