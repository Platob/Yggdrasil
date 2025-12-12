# yggdrasil.libs

Optional bridges to external dataframe libraries and runtime dependency checks.

## When to use
- Convert Arrow schemas to Polars/pandas/Spark dtypes or vice versa.
- Gate code paths on optional dependencies and provide clear installation guidance.
- Access engine-specific conveniences (e.g., Polars and Spark extensions) only when those engines are present.

## Modules
- `polarslib`, `pandaslib`, `sparklib` – thin wrappers exposing imported libraries (if installed) and conversions between Arrow types and each library's schema types. Helpers like `arrow_type_to_polars_type`, `arrow_field_to_pandas_dtype`, and `cast_spark_dataframe` live here.
- `databrickslib` – lazily imports the Databricks SDK and exposes `databricks_sdk` plus `require_databricks_sdk()` guards.
- `extensions` – mixins for Polars and Spark (e.g., `.with_fields()` helpers) registered when the respective backends are available.
- `modules` – dependency guard/auto-install utilities such as `check_modules`.

## Usage
```python
from yggdrasil.libs.polarslib import arrow_type_to_polars_type, require_polars

require_polars()
pl_type = arrow_type_to_polars_type(pa.int64())
```

## Notes and pitfalls
- Import guards raise informative `ImportError`s when optional dependencies are missing; catch them to provide actionable user messaging.
- Some conversions require Arrow metadata (e.g., timestamps with timezones). Pass `CastOptions` when casting Spark/Polars dataframes via `yggdrasil.types.cast.convert`.
- The auto-install helpers assume the environment can install packages at runtime—avoid in locked-down production environments.

## Related modules
- [yggdrasil.types](../types/README.md) for casting logic and Arrow inference.
- [yggdrasil.databricks](../databricks/README.md) for Databricks-specific workflows.
