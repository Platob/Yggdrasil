# yggdrasil.types

Type conversion utilities and Arrow interoperability helpers, including a registry-driven casting system and schema inference from Python type hints.

## When to use
- Normalize runtime values into annotated Python types (including nested containers and dataclasses).
- Generate Arrow schemas directly from Python type hints.
- Enforce schema consistency across Arrow, Pandas, Polars, and Spark dataframes.

## Core APIs
### `convert(value, target_hint, options=None, **kwargs)`
Central entry point for type-aware casting. Handles optionals, collections, enums, dataclasses, and engine-specific converters registered under `yggdrasil.types.cast`.

```python
from yggdrasil.types import convert

convert("3.14", float)
convert(["1", "2"], list[int])
```

### `register_converter(from_hint, to_hint)`
Decorator for custom converters. Functions accept `(value, options)` and should return the converted value.

```python
from yggdrasil.types import register_converter

@register_converter(str, complex)
def parse_complex(value, options):
    return complex(value)
```

### `CastOptions`
Configuration object used by the casting helpers to control schema matching, missing columns, and strictness. Use `CastOptions.check_arg(...)` to normalize option inputs.

## Arrow helpers
- `arrow_field_from_hint(hint, name=None, index=None)` – build a `pyarrow.Field` from a Python type hint.
- `is_arrow_type_string_like`, `is_arrow_type_binary_like`, `is_arrow_type_list_like` – Arrow type classifiers.

## Defaults
- `default_scalar(hint)` and `default_python_scalar(hint)` – produce sensible defaults for Python/typing hints.
- `default_arrow_scalar(dtype, nullable)` and `default_arrow_array(dtype, nullable)` – build default Arrow scalars/arrays.

## Dataframe casting helpers
Available under `yggdrasil.types.cast`:
- `cast_arrow_tabular` for Arrow tables/record batches.
- `cast_pandas_dataframe`, `cast_polars_dataframe`, `cast_spark_dataframe` for dataframe engines.

These helpers rely on optional dependencies (PySpark, Polars, Pandas) when relevant.

For details, see [yggdrasil.types.cast](cast/README.md).

## Related modules
- [yggdrasil.dataclasses](../dataclasses/README.md) for Arrow-aware dataclass helpers.
- [yggdrasil.libs](../libs/README.md) for dependency guards and Spark type mappings.
