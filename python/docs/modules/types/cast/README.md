# yggdrasil.arrow.cast — engine casting

Apply `CastOptions` to Arrow, pandas, Polars, and Spark data.

## API surface

| Function | Import from | Input |
|---|---|---|
| `cast_arrow_tabular` | `yggdrasil.arrow.cast` | `pa.Table` / `pa.RecordBatch` |
| `cast_arrow_record_batch_reader` | `yggdrasil.arrow.cast` | `pa.RecordBatchReader` |
| `cast_pandas_dataframe` | `yggdrasil.pandas.cast` | `pandas.DataFrame` |
| `cast_polars_dataframe` | `yggdrasil.polars.cast` | `polars.DataFrame` |
| `cast_polars_lazyframe` | `yggdrasil.polars.cast` | `polars.LazyFrame` |
| `cast_spark_dataframe` | `yggdrasil.spark.cast` | `pyspark.sql.DataFrame` |

All functions take `(data, options: CastOptions)`.

---

## Bootstrap: Arrow table

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

target = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("score", pa.float64()),
    pa.field("ts", pa.timestamp("us", tz="UTC")),
])

raw = pa.table({
    "id":    ["1", "2", "3"],
    "score": ["9.5", "8.0", "7.2"],
    "ts":    ["2024-01-01", "2024-01-02", "2024-01-03"],
})

out = cast_arrow_tabular(raw, CastOptions(target_field=target))
```

---

## Bootstrap: pandas DataFrame

```python
import pandas as pd
import pyarrow as pa
from yggdrasil.pandas.cast import cast_pandas_dataframe
from yggdrasil.data.cast import CastOptions

target = pa.schema([
    pa.field("user_id", pa.int64()),
    pa.field("revenue", pa.float64()),
])

df = pd.DataFrame({"user_id": ["10", "20"], "revenue": ["100.5", "200.0"]})
out = cast_pandas_dataframe(df, CastOptions(target_field=target))
```

---

## Bootstrap: Polars DataFrame

```python
import polars as pl
import pyarrow as pa
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.data.cast import CastOptions

target = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("value", pa.float64()),
])

df = pl.DataFrame({"id": ["1", "2"], "value": ["3.14", "2.71"]})
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

---

## Bootstrap: fill missing columns with defaults

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

target = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("status", pa.string()),   # not in source → ""
    pa.field("score", pa.float64()),   # not in source → 0.0
])

raw = pa.table({"id": [1, 2, 3]})
out = cast_arrow_tabular(raw, CastOptions(
    target_field=target,
    add_missing_columns=True,
))
```

---

## Bootstrap: streaming cast

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_record_batch_reader
from yggdrasil.data.cast import CastOptions

target = pa.schema([pa.field("id", pa.int64()), pa.field("val", pa.float64())])
opts = CastOptions(target_field=target)

reader = pa.ipc.open_stream(source_bytes)
casted = cast_arrow_record_batch_reader(reader, opts)

for batch in casted:
    process(batch)
```

---

## CastOptions quick reference

```python
CastOptions(
    target_field=schema,         # pa.Schema | pa.Field | pa.DataType  ← required
    safe=False,                  # True: only safe Arrow casts
    add_missing_columns=True,    # fill missing with type defaults
    strict_match_names=False,    # True: exact case-sensitive match only
    allow_add_columns=False,     # True: preserve extra source columns
    datetime_patterns=None,      # custom strptime patterns, e.g. ["%Y/%m/%d"]
)
```
