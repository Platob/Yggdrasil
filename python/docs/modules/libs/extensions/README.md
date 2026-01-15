# yggdrasil.libs.extensions

Optional dataframe extensions for Polars and Spark.

## When to use
- You want convenience helpers for joining or resampling Polars dataframes.
- You need Spark dataframe helpers such as latest-row selection or alias discovery.

## Polars helpers
Provided by `polars_extensions`:
- `join_coalesced` joins two dataframes and coalesces overlapping columns.
- `resample` groups by a datetime column and aggregates at a regular cadence.

```python
from yggdrasil.libs.extensions import join_coalesced

result = join_coalesced(left_df, right_df, on="id")
```

## Spark helpers
Provided by `spark_extensions`:
- Utilities for working with Spark columns and aliases.
- Convenience resampling helpers backed by Spark + Arrow/Polars integration.

## Related modules
- [yggdrasil.libs](../README.md) for dependency guards and Spark/Arrow conversions.
