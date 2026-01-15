# yggdrasil.dataclasses

Arrow-aware dataclass helpers that add safe construction and schema inspection on top of `dataclasses.dataclass`.

## When to use
- You want to coerce inbound values into annotated types without hand-writing parsing logic.
- You need a `pyarrow.Field` representation of a dataclass for schemas or table casting.
- You want consistent default-instance behavior for dataclasses with nested type hints.

## Key exports
### `yggdataclass`
Decorator that wraps `dataclasses.dataclass` and injects helper methods:
- `__safe_init__(*args, **kwargs)` – builds an instance while casting values via `yggdrasil.types.convert` and falling back to defaults.
- `default_instance()` – builds a cached default instance using `yggdrasil.types.default_scalar`.
- `__arrow_field__(name: str | None = None)` – returns a `pyarrow.Field` derived from the dataclass annotations.

```python
from yggdrasil.dataclasses import yggdataclass

@yggdataclass
class Item:
    id: int
    count: int = 0

item = Item.__safe_init__("42")
assert item.id == 42
```

### `is_yggdataclass(obj)`
Returns `True` when a class or instance has been wrapped by `@yggdataclass`.

### `get_dataclass_arrow_field(obj)`
Returns a cached Arrow field for a dataclass by inspecting type annotations.

> Note: `is_yggdataclass` and `get_dataclass_arrow_field` live in
> `yggdrasil.dataclasses.dataclass` (they are not re-exported from the package root).

## Notes
- Arrow schema helpers require `pyarrow` (installed by default for this package).
- Casting relies on the global `convert` registry; register custom converters if you need non-standard types.

## Related modules
- [yggdrasil.types](../types/README.md) for casting logic and Arrow inference.
