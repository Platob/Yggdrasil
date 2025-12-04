# Yggdrasil Python Library

Yggdrasil provides helper utilities for data applications, including

auto-enhanced dataclasses, runtime-managed dependencies, Arrow-compatible
schema translation helpers, and HTTP session helpers for resilient,
authenticated calls.

## Package highlights

- **Enhanced dataclasses**: the custom `@dataclass` decorator adds
  serialization helpers (`to_dict`, `from_dict`, `to_tuple`, `from_tuple`),
  default instances, safe initialization, and Arrow schema generation on top
  of the standard library behavior.
- **Dependency guards**: helper decorators ensure optional ecosystems such as
  Polars, pandas, PySpark, or Databricks SDK are available before executing
  integration code.
- **Data interchange**: conversion helpers map PyArrow types to engine-specific
  schema classes for Polars and Spark to keep data pipelines consistent.
- **HTTP and auth utilities**: HTTP sessions with retry behavior and optional
  Microsoft identity (MSAL) token handling simplify authenticated API access.

## Repository layout

- `src/yggdrasil/` – library source code.
- `tests/` – automated tests (if present) for the Python package.
- `docs/` – documentation and developer templates.

## Installation

Install the package into an existing environment:

```bash
pip install -e .[dev]
```

The optional `dev` extras supply linting and Databricks dependencies if you are
working in that environment.

## Quick start

Here is a minimal example that uses the enhanced dataclass decorator together
with the conversion helpers:

```python
from yggdrasil.dataclasses import yggdataclass
from yggdrasil.types import arrow_field_from_hint
import pyarrow as pa


@yggdataclass
class Example:
    id: int
    name: str


example = Example.default_instance()
print(example.to_dict())

schema_field = Example.arrow_field("id")
assert isinstance(schema_field.type, pa.DataType)
```

For a map of every submodule and its responsibilities, see
`modules.md`. Detailed pages for each module live under `modules/`, and
additional starter snippets are available in `developer-templates.md` in this
directory.
