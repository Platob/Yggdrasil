# Getting Started

## Install

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Optional extras:

```bash
uv pip install -e .[polars]
uv pip install -e .[pandas]
uv pip install -e .[spark]
uv pip install -e .[databricks]
uv pip install -e .[api]
```

## First conversion

```python
from yggdrasil.data.cast.registry import convert

print(convert("42", int))
print(convert("true", bool))
```

## Dict to dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str

print(convert({"id": "1", "email": "ada@example.com"}, User))
```

## Arrow tabular cast

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1"], "score": ["3.14"]})
opts = CastOptions(target_field=pa.schema([
    pa.field("id", pa.int64()),
    pa.field("score", pa.float64()),
]))

print(cast_arrow_tabular(raw, opts).schema)
```
