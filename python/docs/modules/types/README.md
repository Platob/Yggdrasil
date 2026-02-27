# yggdrasil.data.cast

The casting registry: convert Python values between types, configure how tables are cast to schemas.

## Key exports

| Symbol | Module | Purpose |
|---|---|---|
| `CastOptions` | `yggdrasil.data.cast` | Configure casting behavior |
| `convert` | `yggdrasil.data.cast.registry` | Convert a value to a target type |
| `register_converter` | `yggdrasil.data.cast.registry` | Register a custom converter |

---

## Bootstrap: dict → dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class Order:
    id: int
    amount: float
    active: bool

obj = convert({"id": "42", "amount": "19.99", "active": "true"}, Order)
print(obj)  # Order(id=42, amount=19.99, active=True)
```

---

## Bootstrap: scalar conversions

```python
from yggdrasil.data.cast.registry import convert
import datetime

convert("2024-01-15", datetime.date)   # datetime.date(2024, 1, 15)
convert("3.14", float)                 # 3.14
convert("true", bool)                  # True
convert(42, str)                       # "42"
```

---

## Bootstrap: register a custom converter

```python
from yggdrasil.data.cast.registry import register_converter, convert
from decimal import Decimal

@register_converter(str, Decimal)
def str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value).quantize(Decimal("0.01"))

convert("3.14159", Decimal)   # Decimal('3.14')
```

---

## CastOptions reference

```python
from yggdrasil.data.cast import CastOptions
import pyarrow as pa

# Minimal — just set target schema
opts = CastOptions(target_field=pa.schema([
    pa.field("id", pa.int64()),
    pa.field("name", pa.string()),
]))

# Full options (with defaults)
opts = CastOptions(
    safe=False,                  # True: only safe Arrow casts
    add_missing_columns=True,    # fill missing columns with type defaults
    strict_match_names=False,    # False: case-insensitive + positional match
    allow_add_columns=False,     # True: keep extra source columns
    eager=False,
    datetime_patterns=None,      # e.g. ["%Y/%m/%d", "%d-%m-%Y"]
    merge=False,
    target_field=None,           # pa.Schema | pa.Field | pa.DataType
)
```

`target_field` accepts `pa.Schema`, `pa.Field`, or `pa.DataType` — normalized internally.

---

## Bootstrap: strict schema enforcement

```python
import pyarrow as pa
from yggdrasil.data.cast import CastOptions
from yggdrasil.arrow.cast import cast_arrow_tabular

target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
])

opts = CastOptions(
    target_field=target,
    strict_match_names=True,    # exact column name match
    add_missing_columns=False,  # raise if column missing
)

out = cast_arrow_tabular(raw_table, opts)
```

---

## Related

- [arrow.cast](cast/README.md) — apply `CastOptions` to tables and dataframes
- [arrow](../arrow/README.md) — infer Arrow fields from Python type hints
