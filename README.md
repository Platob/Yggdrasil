# Yggdrasil

Utilities for schema-aware data interchange, conversions, and platform integrations. The repository currently focuses on the Python package in [`python/`](python/), which ships dataclass helpers, casting utilities, Databricks wrappers, and resilient HTTP sessions so you can move data between engines without re-learning their type systems.

## Why Yggdrasil?
- **Type-first conversions** between Python objects, Arrow, Polars, pandas, PySpark, and Databricks SQL results.
- **Safer dataclasses** with automatic defaults, Arrow schema generation, and coercion on construction.
- **Platform helpers** for Databricks workspaces, jobs, compute, and SQL execution.
- **Operational utilities** including retries, concurrency decorators, and dependency guards for optional libraries.

## Quickstart
Install directly from PyPI for the quickest trial:

```bash
pip install ygg
```

Or clone the repo, create a virtual environment with [uv](https://docs.astral.sh/uv/), and install the Python package in editable mode:

```bash
cd python
uv venv .venv
source .venv/bin/activate
uv pip install -e .[dev]
```

Run a minimal example that defines a dataclass, converts values, and generates an Arrow schema:

```python
from yggdrasil import yggdataclass
from yggdrasil.types.cast import convert
from yggdrasil.types import arrow_field_from_hint

@yggdataclass
class User:
    id: int
    email: str
    active: bool = True

user = User.__safe_init__("1", email="ada@example.com")
print(user)  # User(id=1, email='ada@example.com', active=True)

payload = {"id": "2", "email": "bob@example.com", "active": "false"}
print(User.from_dict(payload))

arrow_field = arrow_field_from_hint(User, name="user")
print(arrow_field)

values = convert(["1", "2", "3"], list[int])
print(values)
```

### Databricks hello world
With the Databricks extras installed, execute SQL and fetch results as Arrow:

```python
from yggdrasil.databricks.sql import SQLEngine
from yggdrasil.databricks.workspaces import Workspace

engine = SQLEngine(Workspace(host="https://<workspace>", token="<token>"))
result = engine.execute("SELECT 1 AS value").wait()
print(result.arrow_table().to_pandas())
```

## Documentation map
- [`python/README.md`](python/README.md) – installation, quickstart, and feature overview for the Python package.
- [`python/docs/`](python/docs/README.md) – structured guides, module reference, and developer templates.
- [`python/docs/modules/`](python/docs/modules/README.md) – per-module documentation for dataclasses, types, libs, requests, Databricks, and more.

## Publishing
The GitHub Actions workflow at [`.github/workflows/publish.yml`](.github/workflows/publish.yml) builds and publishes the Python package when pushing to `main`. To authorize uploads:
1. Create a PyPI API token (Account settings → **API tokens** → **Add API token**).
2. Add a repository Actions secret named `PYPI_API_TOKEN` containing the token. The username is preset to `__token__` in the workflow.
3. Push to `main` or trigger the workflow manually; the package is built from `python/` and uploaded using the stored secret.

## Support matrix and extras
Many conversions rely on optional dependencies. Install only what you need:
- `.[dev]` – development, linting, and testing tools.
- `.[databricks]` – Databricks SDK and PySpark dependencies.
- `.[pandas]`, `.[polars]`, `.[spark]` – engine-specific extras for conversions.

## Contributing
1. Create a branch and install with `uv pip install -e .[dev]` inside `python/`.
2. Run the pytest suite (and optionally `ruff`, `black`, `mypy`) before opening a pull request.
3. Keep examples copy-pasteable and prefer type-hinted signatures in new utilities.
