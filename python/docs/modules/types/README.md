# Type-driven conversions

Helpers that infer Arrow schemas from Python annotations and cast values between
PyArrow and engine-specific representations.

## Core hint utilities
- `python_defaults`: normalizes Python type hints and produces default values
  used by the enhanced dataclass decorator.
- `python_arrow`: maps Python type hints to `pyarrow.DataType` objects.
- `libs`: shared helpers used across casting backends.

## Casting registry (`yggdrasil.types.cast`)
- `registry`: dispatch table that routes `convert(value, target_type)` requests
  to the correct backend.
- `arrow`: Arrow-native conversions and utilities.
- `pandas`, `polars`, `spark`: backend-specific casting helpers for pandas,
  Polars, and PySpark types, respectively.

## Usage patterns
- Call `yggdrasil.types.arrow_field_from_hint(cls)` or
  `arrow_schema_from_hint(cls)` to emit Arrow schemas derived from dataclass
  annotations.
- Use `yggdrasil.types.cast.convert(value, target_type)` to coerce incoming
  values inside data loaders, API adapters, or dataclass constructors.
