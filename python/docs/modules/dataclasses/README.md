# Dataclasses utilities

Yggdrasil ships an enhanced `@dataclass` decorator that layers convenience
methods and safe type conversion on top of the standard library while preserving
familiar semantics.

## Key capabilities
- Drop-in replacement for `dataclasses.dataclass` that still supports the full
  decorator signature (`init`, `repr`, `eq`, `order`, `frozen`, etc.).
- Auto-attached helpers for converting instances: `to_dict`, `from_dict`,
  `to_tuple`, `from_tuple`, and `copy`.
- `default_instance()` caches and returns an instance populated with type-aware
  defaults based on field hints.
- `__safe_init__` builds instances using defaults and `yggdrasil.types.cast`
  converters to coerce incoming data.
- Arrow interoperability through `arrow_field(name=None)` for emitting
  `pyarrow.Field` objects directly from dataclass definitions.

## Quick start
```python
from yggdrasil.dataclasses import dataclass

@dataclass
class Event:
    id: int
    payload: dict[str, str] | None = None

record = Event.from_dict({"id": "42", "payload": {"k": "v"}})
assert record.to_tuple() == (42, {"k": "v"})

# Safe initialization with positional overrides
copy = Event.copy(100)
assert copy.id == 100
```

## Related modules
- `yggdrasil.types.python_defaults` infers default values from type hints.
- `yggdrasil.types.cast` powers the `from_dict`, `from_tuple`, and
  `__safe_init__` conversions.
