# yggdrasil.libs

Optional bridges to external dataframe libraries and runtime dependency checks.

## Modules
- `polarslib`, `pandaslib`, `sparklib` – thin wrappers exposing imported libraries (if installed) and conversions between Arrow types and each library's schema types. Helpers like `arrow_type_to_polars_type`, `arrow_field_to_pandas_dtype`, and `cast_spark_dataframe` live here.
- `databrickslib` – lazily imports the Databricks SDK and exposes `databricks_sdk` plus `require_databricks_sdk()` guards.
- `extensions` – small mixins for Polars and Spark (e.g., `.with_fields()` helpers) registered when the respective backends are available.

## Usage
```python
from yggdrasil.libs.polarslib import arrow_type_to_polars_type, require_polars

require_polars()
pl_type = arrow_type_to_polars_type(pa.int64())
```

These helpers raise informative `ImportError`s when optional dependencies are missing so callers can gate functionality appropriately.

## Navigation
- [Module overview](../../modules.md)
- [Dataclasses](../dataclasses/README.md)
- [Libs](./README.md)
- [Requests](../requests/README.md)
- [Types](../types/README.md)
- [Databricks](../databricks/README.md)
