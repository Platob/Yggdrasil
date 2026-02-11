# yggdrasil.types

`yggdrasil.types` is the schema and type normalization layer for cross-engine dataframe workflows.

It helps ensure that your expected contract is preserved when data moves across:
- Python objects
- PyArrow
- Pandas
- Polars
- Spark

---

## Core capabilities

- Infer Arrow schemas from Python type hints.
- Provide default/null-safe type behaviors.
- Convert values and structures through registry-based casting.
- Support strict and permissive casting strategies.

---

## Bootstrap: infer Arrow schema from dataclass

```python
from dataclasses import dataclass
from yggdrasil.types import python_type_to_arrow_schema

@dataclass
class Transaction:
    id: int
    amount: float
    country: str

schema = python_type_to_arrow_schema(Transaction)
print(schema)
```

---

## Bootstrap: normalize records to typed object model

```python
from dataclasses import dataclass
from yggdrasil.types.cast.registry import convert

@dataclass
class User:
    id: int
    active: bool

obj = convert({"id": "42", "active": "true"}, User)
print(obj)
```

---

## Bootstrap: enforce schema on dataframe-like content

```python
from yggdrasil.types.cast import cast_arrow_tabular

# table is a pyarrow.Table and target_schema is pyarrow.Schema
# casted = cast_arrow_tabular(table, target_schema)
```

---

## Related module

- [types.cast](cast/README.md): engine-specific cast utilities and options.
