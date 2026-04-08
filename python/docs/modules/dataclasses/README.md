# yggdrasil.dataclasses

Dataclass-focused helpers used by casting and execution layers.

## Common exports

- `dataclass_to_arrow_field`
- `WaitingConfig`
- `Expiring` and `ExpiringDict`

## Dataclass → Arrow field

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Position:
    symbol: str
    quantity: float

field = dataclass_to_arrow_field(Position)
print(field)
```

## Waiting config for polling flows

```python
from yggdrasil.dataclasses import WaitingConfig

wait = WaitingConfig.check_arg(True)
print(wait)
```
