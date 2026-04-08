# Engine casting APIs

Use these helpers to apply `CastOptions` to tabular data across Arrow and dataframe engines.

## Main functions

- `yggdrasil.arrow.cast.cast_arrow_tabular`
- `yggdrasil.arrow.cast.cast_arrow_record_batch_reader`
- `yggdrasil.pandas.cast.cast_pandas_dataframe`
- `yggdrasil.polars.cast.cast_polars_dataframe`
- `yggdrasil.polars.cast.cast_polars_lazyframe`
- `yggdrasil.spark.cast.cast_spark_dataframe`

## Arrow table cast

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1", "2"], "value": ["10.1", "20.5"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("value", pa.float64())])
out = cast_arrow_tabular(raw, CastOptions(target_field=target))
```

## Polars cast

```python
import yggdrasil.arrow as pa
from yggdrasil.data.cast import CastOptions
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.polars.lib import polars

df = polars.DataFrame({"id": ["1"], "score": ["4.5"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

## Streaming cast

```python
from yggdrasil.arrow.cast import cast_arrow_record_batch_reader

reader = ...  # pyarrow.RecordBatchReader
opts = ...    # CastOptions
for batch in cast_arrow_record_batch_reader(reader, opts):
    process(batch)
```
