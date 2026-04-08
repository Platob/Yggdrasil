# Yggdrasil

Yggdrasil is a schema-aware data interchange toolkit. The primary deliverable is the Python package in [`python/`](python/), published on PyPI as **`ygg`** and imported as **`yggdrasil`**.

## What it does

- Registry-driven casting across Python values, dataclasses, Arrow, Polars, pandas, Spark, and Databricks flows.
- Arrow schema inference from Python hints.
- Databricks workspace / SQL / compute helpers.
- IO utilities (`BytesIO`, URL/HTTP abstractions) and operational helpers (`retry`, `parallelize`, bounded job execution).

## Quick start

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Examples (easy → advanced)

### 1) Scalar conversion

```python
from yggdrasil.data.cast.registry import convert

print(convert("42", int))
print(convert("true", bool))
```

### 2) Dict to dataclass conversion

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str

print(convert({"id": "1", "email": "ada@example.com"}, User))
```

### 3) Arrow cast with `CastOptions`

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

### 4) Databricks SQL

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

client = Workspace(host="https://<workspace>", token="<token>")
engine = SQLEngine(client=client)
print(engine.execute("SELECT 1 AS value").to_arrow_table())
```

## Repository map

- [`python/README.md`](python/README.md) — package-level guide.
- [`python/docs/README.md`](python/docs/README.md) — progressive docs with snippets.
- [`python/docs/modules.md`](python/docs/modules.md) — module index.

## License

See [LICENSE](LICENSE).
