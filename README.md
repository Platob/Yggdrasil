# Yggdrasil

Yggdrasil is a collection of schema-aware utilities for data interchange across Python types, Apache Arrow, pandas, Polars, Spark, and Databricks. The main deliverable today is the Python package in [`python/`](python/), published to PyPI as **`ygg`** with the import namespace **`yggdrasil`**.

## What you can do with it
- **Type-safe casting** between primitives, containers, enums, and dataclasses via a registry of converters.
- **Arrow schema inference** from Python type hints (including dataclasses and `typing.Annotated`).
- **Databricks helpers** for workspace config, SQL execution, and Arrow/Spark conversions.
- **Operational utilities** such as retry-enabled HTTP sessions and simple parallelization helpers.

## Install
Quick install from PyPI:

```bash
pip install ygg
```

For development, create a local environment and install the package in editable mode:

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

## Code examples

### 1) Convert values and dataclasses

```python
from dataclasses import dataclass

from yggdrasil.types import convert

@dataclass
class User:
    id: int
    email: str
    active: bool = True

raw = {"id": "42", "email": "ada@example.com", "active": "false"}
user = convert(raw, User)
print(user)
```

### 2) Build Arrow schema fields from type hints

```python
from dataclasses import dataclass

from yggdrasil.dataclasses import get_dataclass_arrow_field
from yggdrasil.types import arrow_field_from_hint

@dataclass
class Event:
    ts: str
    value: float

print(arrow_field_from_hint(list[int], name="counts"))
print(get_dataclass_arrow_field(Event))
```

### 3) Run Databricks SQL and retrieve Arrow results

```python
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.workspaces import Workspace

engine = SQLEngine(Workspace(host="https://<workspace>", token="<token>"))
result = engine.execute("SELECT 1 AS value")
print(result.to_arrow_table().to_pandas())
```

### 4) Retryable HTTP sessions and simple parallel work

```python
from yggdrasil.requests import YGGSession
from yggdrasil.pyutils import parallelize, retry

session = YGGSession(num_retry=3)

@parallelize(max_workers=4)
def square(x: int) -> int:
    return x * x

@retry(tries=3, delay=0.1, backoff=2)
def flaky(value: int) -> int:
    return value

print(list(square(range(4))))
```

## Repository map
- [`python/`](python/) — Python package source, docs, and tests.
- [`python/README.md`](python/README.md) — Package-specific setup and usage.
- [`python/docs/`](python/docs/README.md) — Detailed documentation and module references.

## Development notes
- The package requires Python **3.10+**.
- Entry point `yggenv` is available after installation for environment helpers.
- Run tests from the `python/` directory with `pytest`.

## License
Licensed under the terms in [LICENSE](LICENSE).
