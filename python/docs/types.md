# yggdrasil.types

Type conversion and Arrow interoperability utilities.

## Conversion registry

### `convert(value, target_hint, options=None, **kwargs)`
Central entry point to cast runtime values to annotated types. It understands optionals, enums, dataclasses, iterables, mappings, and Arrow/Polars/Pandas/Spark-specific converters registered in `yggdrasil.types.cast.registry`.

Example:
```python
from yggdrasil.types.cast import convert

convert("3.14", float)            # 3.14
convert(["1", "2", "3"], list[int])  # [1, 2, 3]
```

### `register_converter(from_hint, to_hint)`
Decorator to register new converters. Functions accept `(value, options)` and should return the converted value.

```python
from yggdrasil.types.cast import register_converter

@register_converter(str, complex)
def parse_complex(value, options):
    return complex(value)
```

### `CastOptions`
Options object (validated via `CastOptions.check_arg`) passed into converters to control defaults or Arrow metadata.

## Arrow helpers

### `arrow_field_from_hint(hint, name=None, index=None)`
Builds a `pyarrow.Field` from a Python type hint, handling optionals, containers, dataclasses, tuples, and Annotated metadata for custom Arrow details.

## Defaults

### `default_scalar(hint)` / `default_python_scalar(hint)` / `default_arrow_scalar(dtype, nullable)`
Produce sensible default values for Python and Arrow types, including nested structs, lists, maps, and dataclasses.

## Notes
- Primitive string, numeric, boolean, datetime, and timedelta parsers are included out of the box.
- Iterable conversion supports lists, sets, tuples (including variadic tuples), and mappings with per-element casting.
- Arrow/Polars/Spark conversions rely on optional dependencies; install the relevant libraries to enable those paths.
