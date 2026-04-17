# Yggdrasil

Yggdrasil is a schema-aware data interchange library for Python teams that are tired of writing one-off conversion glue.

It helps you move cleanly between Python types, dataclasses, Apache Arrow, Polars, pandas, Spark, and Databricks without losing control of schema, nullability, or metadata.

The main package lives in [`python/`](python/), is published on PyPI as `ygg`, and is imported as `yggdrasil`.

## Why people adopt it

- Stop hand-writing brittle casting code between app models, dataframes, and warehouse-facing schemas.
- Keep Arrow schema as the contract surface instead of letting every tool infer something different.
- Use one conversion registry instead of separate ad hoc utilities for Python, Polars, pandas, Spark, and Databricks.
- Install only what you need. `pyarrow` is the only hard runtime dependency; most integrations stay optional.

## What Yggdrasil gives you

- Registry-driven conversion across Python values, dataclasses, Arrow, Polars, pandas, Spark, and Databricks workflows.
- Arrow schema inference from Python type hints.
- Schema-aware casting with `CastOptions` for explicit, repeatable behavior.
- Databricks helpers for SQL, workspaces, jobs, compute, IAM, and secrets.
- IO and HTTP utilities for buffers, sessions, URLs, retries, batching, and concurrency.

## Quick start

Install from PyPI:

```bash
pip install ygg
```

Or work from source:

```bash
cd python
uv venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux / macOS
uv pip install -e .
```

Add optional integrations only when you need them:

```bash
uv pip install -e .[data]
uv pip install -e .[bigdata]
uv pip install -e .[databricks]
```

## First win in 30 seconds

Turn loosely typed payloads into strongly typed objects:

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert


@dataclass
class User:
    id: int
    email: str
    active: bool = True


payload = {"id": "7", "email": "ada@example.com", "active": "false"}
user = convert(payload, User)

print(user)
# User(id=7, email='ada@example.com', active=False)
```

Make Arrow schema explicit before data moves downstream:

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

out = cast_arrow_tabular(raw, CastOptions(target_field=target))
print(out.schema)
```

## Common use cases

- Normalize API payloads into typed Python models.
- Convert application schemas into Arrow for storage or transport.
- Bridge Arrow, Polars, pandas, and Spark in mixed analytics stacks.
- Build Databricks-facing workflows that keep schema handling explicit.
- Reuse one set of casting rules across local development and production pipelines.

## Why it is different

Most libraries solve one layer of the stack. Yggdrasil is designed to connect them.

Instead of treating Python objects, dataframe engines, and warehouse tooling as separate worlds, it uses a central converter registry so the same casting model can apply across all of them. That reduces duplicate logic, reduces silent schema drift, and makes data movement easier to reason about.

## Repository guide

- [`python/README.md`](python/README.md): package-level guide and richer examples.
- [`python/docs/README.md`](python/docs/README.md): progressive documentation.
- [`python/docs/modules.md`](python/docs/modules.md): module index.
- [`powerquery/`](powerquery/): Power Query connector.
- [`rust/`](rust/): optional native acceleration package (`yggrs`).

## Development

```bash
cd python
uv pip install -e .[dev]
pytest
ruff check
black .
```

## License

See [LICENSE](LICENSE).
