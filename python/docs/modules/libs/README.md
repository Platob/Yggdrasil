# yggdrasil.libs

Optional dependency helpers and small extensions for Spark and Polars workflows.

## When to use
- Guard code paths that require optional dependencies like PySpark, Polars, or pandas.
- Convert Arrow types to Spark SQL types and back.
- Use helper extensions for Polars/Spark dataframes when those libraries are installed.

## Dependency guards
- `require_pandas()`, `require_polars()`, `require_pyspark()` raise informative errors when a dependency is missing.
- `databrickslib.require_databricks_sdk()` checks for the Databricks SDK.

## Spark/Arrow type mappings
Located in `yggdrasil.libs.sparklib`:
- `arrow_type_to_spark_type` / `spark_type_to_arrow_type` for single types.
- `arrow_field_to_spark_field` / `spark_field_to_arrow_field` for schema fields.

## Extensions
`yggdrasil.libs.extensions` exposes helper functions when the corresponding engine is installed:
- **Polars**: `join_coalesced`, `resample` for dataframe joins and time-based aggregation.
- **Spark**: helpers for alias discovery, `latest` row selection, and Arrow/Polars-backed resampling routines.

## Related modules
- [yggdrasil.types](../types/README.md) for casting helpers built on these dependencies.
- [yggdrasil.databricks](../databricks/README.md) for Databricks-specific integrations.
