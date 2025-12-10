# yggdrasil.dataclasses

Decorator and helpers to build dataclasses that understand defaults, safe construction, and Arrow schemas.

## Key exports

### `yggdataclass`
Wraps `dataclasses.dataclass` to add convenience methods:
- `__safe_init__(*args, **kwargs)` — constructs instances with type conversion and default fallbacks using the casting registry.
- `from_dict(mapping, safe=True)` / `from_tuple(iterable, safe=True)` — build instances from structured inputs with optional type-safe casting.
- `to_dict()` / `to_tuple()` — serialize instances.
- `default_instance()` — lazily builds a default instance using type defaults.
- `__arrow_field__(name: str | None = None)` — derive a `pyarrow.Field` from annotations.

Usage:
```python
from yggdrasil.dataclasses import yggdataclass
from yggdrasil.types import convert

@yggdataclass
class Item:
    id: int
    quantity: int = 0

item = Item.__safe_init__("42")
assert item.id == 42 and item.quantity == 0
```

### `is_yggdataclass(obj)`
Returns True when the class or instance was decorated with `@yggdataclass`.

### `get_dataclass_arrow_field(obj)`
Returns the cached `pyarrow.Field` for a yggdrasil dataclass or native dataclass by inspecting annotations.
