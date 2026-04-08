# yggdrasil.data.cast

Registry-driven conversion system used across the package.

## Key symbols

- `convert` — convert any value into a target type.
- `register_converter` — register custom converters.
- `CastOptions` — option object reused by Arrow/engine cast paths.

## Scalar + dataclass conversion

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    active: bool

print(convert("3", int))
print(convert({"id": "7", "active": "true"}, User))
```

## Register a converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import register_converter, convert

@register_converter(str, Decimal)
def str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value)

print(convert("19.95", Decimal))
```

## `CastOptions.check_arg` normalization

```python
import yggdrasil.arrow as pa
from yggdrasil.data.cast import CastOptions

schema = pa.schema([pa.field("id", pa.int64())])
opts = CastOptions.check_arg(schema, strict_match_names=True)
print(opts.target_field)
```
