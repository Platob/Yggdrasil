# yggdrasil.types.cast

Casting helpers that normalize Arrow, Pandas, Polars, and Spark dataframes based on Python type hints.

## When to use
- You need consistent schema enforcement across dataframe engines.
- You want to cast Arrow tables/record batches to match a target schema.

## Core APIs
- `cast_arrow_tabular` casts Arrow tables/record batches to a target schema.
- `cast_pandas_dataframe`, `cast_polars_dataframe`, `cast_spark_dataframe` cast dataframe engines.
- `cast_*_schema` helpers return engine-specific schemas from Arrow schemas.
- `CastOptions` configures column matching, missing columns, and strictness.

```python
from yggdrasil.types.cast import cast_pandas_dataframe, CastOptions

options = CastOptions(strict=False)
output = cast_pandas_dataframe(input_df, target_schema, options=options)
```

## Related modules
- [yggdrasil.types](../README.md) for type conversions and registry APIs.
- [yggdrasil.libs](../../libs/README.md) for dependency guards used by the cast helpers.
