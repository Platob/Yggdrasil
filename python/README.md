# Yggdrasil (Python)

Type-friendly utilities for moving data between Python objects, Arrow, Polars, pandas, Spark, and Databricks. The package bundles enhanced dataclasses, casting utilities, and lightweight wrappers around Databricks and HTTP clients so Python/data engineers can focus on schemas instead of plumbing.

## When to use this package
Use Yggdrasil when you need to:
- Convert payloads across dataframe engines without rewriting type logic for each backend.
- Define dataclasses that auto-coerce inputs, expose defaults, and surface Arrow schemas.
- Run Databricks SQL jobs or manage clusters with minimal boilerplate.
- Add resilient retries, concurrency helpers, and dependency guards to data pipelines.

## Prerequisites
- Python **3.10+**
- [uv](https://docs.astral.sh/uv/) for virtualenv and dependency management.

Optional extras:
- `polars`, `pandas`, `pyarrow`, and `pyspark` for engine-specific conversions.
- `databricks-sdk` for workspace, SQL, jobs, and compute helpers.
- `msal` for Azure AD authentication when using `MSALSession`.

## Installation
From the `python/` directory:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Extras are grouped by engine:
- `.[polars]`, `.[pandas]`, `.[spark]`, `.[databricks]` – install only the integrations you need.
- `.[dev]` – adds testing, linting, and typing tools (`pytest`, `ruff`, `black`, `mypy`).

## Quickstart
Define an Arrow-aware dataclass, coerce inputs, and cast across containers:

```python
from yggdrasil import yggdataclass
from yggdrasil.types.cast import convert
from yggdrasil.types import arrow_field_from_hint

@yggdataclass
class User:
    id: int
    email: str
    active: bool = True

user = User.__safe_init__("123", email="alice@example.com")
assert user.id == 123 and user.active is True

payload = {"id": "45", "email": "bob@example.com", "active": "false"}
clean = User.from_dict(payload)
print(clean.to_dict())

field = arrow_field_from_hint(User, name="user")
print(field)  # user: struct<id: int64, email: string, active: bool>

numbers = convert(["1", "2", "3"], list[int])
print(numbers)
```

### Databricks example
Install the `databricks` extra and run SQL with typed results:

```python
from yggdrasil.databricks.workspaces import Workspace
from yggdrasil.databricks.sql import SQLEngine

ws = Workspace(host="https://<workspace-url>", token="<token>")
engine = SQLEngine(workspace=ws)

stmt = engine.execute("SELECT 1 AS value")
result = stmt.wait(engine)
tbl = result.arrow_table()
print(tbl.to_pandas())
```

### Parallel processing and retries

```python
from yggdrasil.pyutils import parallelize, retry

@parallelize(max_workers=4)
def square(x):
    return x * x

@retry(tries=5, delay=0.2, backoff=2)
def sometimes_fails(value: int) -> int:
    ...

print(list(square(range(5))))
```

## Project layout
- `yggdrasil/dataclasses` – `yggdataclass` decorator plus Arrow schema helpers.
- `yggdrasil/types` – casting registry (`convert`, `register_converter`), Arrow inference, and default generators.
- `yggdrasil/libs` – optional bridges to Polars, pandas, Spark, and Databricks SDK types.
- `yggdrasil/databricks` – workspace, SQL, jobs, and compute helpers built on the Databricks SDK.
- `yggdrasil/requests` – retry-capable HTTP sessions and Azure MSAL auth helpers.
- `yggdrasil/pyutils` – concurrency and retry decorators.
- `yggdrasil/ser` – serialization helpers and dependency inspection utilities.
- `tests/` – pytest-based coverage for conversions, dataclasses, requests, and platform helpers.

## Testing
From `python/`:

```bash
pytest
```

Optional checks when developing:

```bash
ruff check
black .
mypy
```

## Troubleshooting and common pitfalls
- **Missing optional dependency**: Install the matching extra (e.g., `uv pip install -e .[polars]`) or wrap calls with `require_polars`/`require_pyspark` from `yggdrasil.libs`.
- **Schema mismatches**: Use `arrow_field_from_hint` and `CastOptions` to enforce expected Arrow metadata when casting.
- **Databricks auth**: Provide `host` and `token` to `Workspace`. For Azure, ensure environment variables align with your workspace deployment.

## Contributing
1. Fork and branch.
2. Install with `uv pip install -e .[dev]`.
3. Run tests and linters.
4. Submit a PR describing the change and any new examples added to the docs.
