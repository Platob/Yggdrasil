# Yggdrasil (Python)

Type-friendly utilities for moving data between Python objects, Arrow, Polars, Pandas, Spark, and Databricks. The package bundles dataclass helpers, casting utilities, and light wrappers around Databricks and HTTP clients so Python/data engineers can focus on schemas instead of plumbing.

## Features
- `@yggdataclass` decorator that adds safe init/from/to helpers and Arrow schema awareness.
- Rich conversion registry to cast between Python types, Arrow, Polars, Pandas, and Spark objects.
- Arrow type inference from Python type hints and sensible default values for common dtypes.
- Parallelization and retry utilities for robust data pipelines.
- Databricks helpers for SQL execution, workspace file management, jobs, and compute interactions.
- HTTP sessions with built-in retries plus optional Azure MSAL authentication.

## Installation
Requirements: Python **3.10+** and [uv](https://docs.astral.sh/uv/).

```bash
# from the python/ directory
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

The editable install makes it easy to iterate locally. Add `.[dev]` to include pytest, black, ruff, and mypy for development.

## Quickstart
Import the package and use the provided helpers to define dataclasses and perform typed conversions.

```python
from yggdrasil import yggdataclass, convert
from yggdrasil.types import arrow_field_from_hint

@yggdataclass
class User:
    id: int
    email: str
    active: bool = True

# Safe construction with type conversion and defaults
user = User.__safe_init__("123", email="alice@example.com")
assert user.id == 123

# Convert incoming payloads to typed instances
payload = {"id": "45", "email": "bob@example.com", "active": "false"}
clean = User.from_dict(payload)

# Arrow schema from type hints
field = User.__arrow_field__(name="user")
print(field)  # user: struct<id: int64, email: string, active: bool>

# Cast between types
from yggdrasil.types.cast import convert
converted = convert(["1", "2", "3"], list[int])

# Parallelize a function over an iterable
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=4)
def square(x):
    return x * x

results = list(square(range(5)))  # [0, 1, 4, 9, 16]
```

### Databricks example
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

## Configuration
- `MSALAuth` and `MSALSession` pull Azure credentials from environment variables such as `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`, and `AZURE_SCOPES`.
- Databricks helpers accept host/token or workspace configuration arguments; see `yggdrasil.databricks.workspaces.Workspace` for details.
- Casting utilities accept `CastOptions` for defaults and Arrow metadata when converting.

## Project structure
- `yggdrasil/dataclasses` – `yggdataclass` decorator with safe init/from/to helpers and Arrow schema support.
- `yggdrasil/types` – Conversion registry (`convert`, `register_converter`), Arrow type inference, and default value helpers.
- `yggdrasil/libs` – Optional bridges to Polars, Pandas, Spark, and Databricks SDK types.
- `yggdrasil/databricks` – Workspace, SQL, jobs, and compute helpers built on the Databricks SDK.
- `yggdrasil/requests` – Retry-capable HTTP sessions and Azure MSAL auth helpers.
- `yggdrasil/pyutils` – Utility decorators for parallelism and retries.
- `yggdrasil/ser` – Serialization helpers and dependency inspection utilities.
- `tests/` – Pytest-based tests for the above modules.

## Testing
Run the test suite from the `python/` directory:

```bash
pytest
```

## Contributing
1. Fork the repo and create a feature branch.
2. Install with `uv pip install -e .[dev]` to pull in linting/type-checking tools.
3. Run `pytest` (and optionally `ruff`, `black`, `mypy`) before opening a PR.
4. Submit a PR describing your changes.
