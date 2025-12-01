# Library integration helpers

Utilities that manage optional dependencies and translate between PyArrow and
popular data engines.

## Dependency guards (`modules.py`)
- `check_modules` decorator auto-installs missing packages (with optional
  retries) based on module import errors or an explicit `package` hint.
- `install_package` performs runtime `pip install` with optional `--upgrade`
  or `--quiet` flags; raises `ModuleInstallError` on failure.
- Use to wrap lightweight importers so downstream helpers can run even when
  optional packages are absent initially.

## Arrow-to-engine converters
- `pandaslib`: maps PyArrow types to pandas dtypes and handles pandas-specific
  quirks such as timezone handling.
- `polarslib`: returns Polars `DataType` instances corresponding to Arrow
  schemas.
- `sparklib`: produces PySpark `DataType` objects and schema fields from Arrow
  definitions.
- `databrickslib`: thin wrappers around Databricks SDK types plus convenience
  imports guarded by `require_databricks_sdk`.

## Extension hooks
- `extensions/` contains optional helpers, such as Spark-specific shims, that
  are loaded only when the corresponding dependency is present.
