# yggdrasil.arrow

Infer Arrow fields and schemas directly from Python type hints and dataclasses.

## Type map

| Python | Arrow |
|---|---|
| `str` | `pa.string()` |
| `int` | `pa.int64()` |
| `float` | `pa.float64()` |
| `bool` | `pa.bool_()` |
| `bytes` | `pa.binary()` |
| `datetime.datetime` | `pa.timestamp('us', tz='UTC')` |
| `datetime.date` | `pa.date32()` |
| `datetime.time` | `pa.time64('us')` |
| `datetime.timedelta` | `pa.duration('us')` |
| `uuid.UUID` | `pa.binary(16)` |
| `decimal.Decimal` | `pa.decimal128(38, 18)` |
| `Optional[T]` | nullable version of `T` |
| `list[T]` | `pa.list_(T)` |
| `dict[K, V]` | `pa.map_(K, V)` |
| dataclass | `pa.struct(...)` |

---

## Bootstrap: single field from a hint

```python
from yggdrasil.arrow import arrow_field_from_hint
import datetime

field = arrow_field_from_hint(int, name="user_id")
# user_id: int64 not null

field = arrow_field_from_hint(datetime.datetime, name="ts")
# ts: timestamp[us, tz=UTC] not null

field = arrow_field_from_hint(str | None, name="note")
# note: string
```

---

## Bootstrap: schema from a dataclass

```python
from dataclasses import dataclass
from typing import Optional
from yggdrasil.arrow import arrow_field_from_hint

@dataclass
class Trade:
    id: int
    symbol: str
    price: float
    quantity: float
    note: Optional[str] = None

field = arrow_field_from_hint(Trade)
schema = field.type.to_schema()
print(schema)
# id: int64 not null
# symbol: string not null
# price: double not null
# quantity: double not null
# note: string
```

---

## Bootstrap: nested structs

```python
from dataclasses import dataclass
from yggdrasil.arrow import arrow_field_from_hint

@dataclass
class Address:
    city: str
    country: str

@dataclass
class Customer:
    id: int
    name: str
    address: Address

field = arrow_field_from_hint(Customer)
# struct<id: int64, name: string, address: struct<city: string, country: string>>
```

---

## Bootstrap: generic containers

```python
from yggdrasil.arrow import arrow_field_from_hint

# list
field = arrow_field_from_hint(list[int], name="scores")
# scores: list<item: int64> not null

# dict (map)
field = arrow_field_from_hint(dict[str, float], name="metrics")
# metrics: map<string, double> not null
```

---

## API

```python
from yggdrasil.arrow import arrow_field_from_hint

arrow_field_from_hint(hint, name=None, index=None) -> pa.Field
```

- `hint` — any Python type annotation
- `name` — override the field name (default: derived from the type)
- `index` — positional index used when no name is available
