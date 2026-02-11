# yggdrasil.types.cast

`yggdrasil.types.cast` provides engine-specific casting functions that enforce schema contracts consistently.

It is the practical layer you use once a target schema is known.

---

## Core APIs

- `cast_arrow_tabular`
- `cast_pandas_dataframe`
- `cast_polars_dataframe`
- `cast_spark_dataframe`
- `CastOptions`

---

## Bootstrap: Arrow table casting

```python
import pyarrow as pa
from yggdrasil.types.cast import cast_arrow_tabular

input_table = pa.table({"id": ["1", "2"], "score": ["10.5", "20.0"]})
target_schema = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("score", pa.float64()),
])

output_table = cast_arrow_tabular(input_table, target_schema)
```

---

## Bootstrap: Pandas dataframe casting

```python
from yggdrasil.types.cast import cast_pandas_dataframe, CastOptions

options = CastOptions(strict=False)
casted_df = cast_pandas_dataframe(input_df, target_schema, options=options)
```

---

## Bootstrap: Polars dataframe casting

```python
from yggdrasil.types.cast import cast_polars_dataframe

casted_pl = cast_polars_dataframe(input_polars_df, target_schema)
```

---

## Bootstrap: Spark dataframe casting

```python
from yggdrasil.types.cast import cast_spark_dataframe

casted_spark_df = cast_spark_dataframe(input_spark_df, target_schema)
```

---

## CastOptions guidance

Use `CastOptions` to control:
- strictness
- missing column behavior
- compatible coercion strategy

For ingestion jobs, start permissive (`strict=False`) and tighten incrementally with data quality checks.
