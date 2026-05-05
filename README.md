# Yggdrasil

Yggdrasil is a schema-aware data interchange library for Python teams that are tired of writing one-off conversion glue.

It helps you move cleanly between Python types, dataclasses, Apache Arrow, Polars, pandas, Spark, and Databricks without losing control of schema, nullability, or metadata.

The main package lives in [`python/`](python/), is published on PyPI as `ygg`, and is imported as `yggdrasil`. The Rust acceleration kernels live in [`rust/`](rust/) and ship as `yggrs`; `ygg` declares `yggrs` as a hard dependency, so installing one pulls the other.

## Why people adopt it

- Stop hand-writing brittle casting code between app models, dataframes, and warehouse-facing schemas.
- Keep Arrow schema as the contract surface instead of letting every tool infer something different.
- Use one conversion registry instead of separate ad hoc utilities for Python, Polars, pandas, Spark, and Databricks.
- Install only what you need beyond the core. Most integrations are optional extras.

## What Yggdrasil gives you

- Registry-driven conversion across Python values, dataclasses, Arrow, Polars, pandas, Spark, and Databricks workflows.
- Arrow schema inference from Python type hints.
- Schema-aware casting with `CastOptions` for explicit, repeatable behavior.
- Databricks helpers for SQL, workspaces, jobs, compute, IAM, and secrets.
- IO and HTTP utilities for buffers, sessions, URLs, retries, batching, and concurrency.
- Native acceleration for hot paths (URL parsing, percent-encoding, query normalization) via `yggrs`.

## Quick start

Install from PyPI — `pip` will pull both `ygg` and the matching `yggrs` wheel for your platform:

```bash
pip install ygg
```

A first sample:

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

And making Arrow schema explicit before data moves downstream:

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

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

- [`python/`](python/) — `ygg` source, tests, and MkDocs site.
  - [`python/README.md`](python/README.md) — package-level guide and richer examples.
  - [`python/docs/README.md`](python/docs/README.md) — progressive documentation.
  - [`python/docs/modules.md`](python/docs/modules.md) — module index.
- [`rust/`](rust/) — `yggrs` Rust acceleration crate (PyO3 + maturin, `abi3-py310`).
- [`powerquery/`](powerquery/) — Power Query / Power BI connectors over the FastAPI service.
- [`AGENTS.md`](AGENTS.md) — house style: tone, error messages, comment voice, API ergonomics.
- [`CLAUDE.md`](CLAUDE.md) — agent-facing notes that mirror this layout.

## Development

The dev loop is two installs in one venv: `ygg` editable from `python/`, and `yggrs` editable from `rust/` via `maturin develop`. The Python side imports the compiled module via the `yggdrasil/rs.py` bridge, so the editable install lets you iterate on either side without reinstalling.

### Prerequisites

- **Python 3.10+** (`abi3-py310` is the floor — Rust wheels work on every newer 3.x without a rebuild).
- **Rust toolchain** via [`rustup`](https://rustup.rs).
- **`uv`** (recommended) or `pip` for the Python venv.
- **`maturin>=1.7`** for compiling the Rust extension. Installed automatically by `pip install -e .[dev]`.

Linux only: building the manylinux-style wheels in CI uses QEMU for `aarch64`. Local builds compile for the host arch only.

### One-time setup

```bash
git clone https://github.com/Platob/Yggdrasil.git
cd Yggdrasil

# 1. Python venv + editable ygg install (with dev tooling)
cd python
uv venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e .[dev]

# 2. Rust extension, editable, into the same venv
cd ../rust
maturin develop --release          # use --release; debug builds are very slow
```

`maturin develop` compiles the cdylib, packages it as a wheel, and installs it as `yggrs` in the active venv. The compiled module lands at `_yggrs.abi3.so` (Linux/macOS) or `_yggrs.pyd` (Windows). `yggdrasil.rs` is the Python bridge that re-publishes its submodules under the `yggdrasil.rust.<name>` namespace, so call sites can keep writing `from yggdrasil.rust.io import url`.

> Why a top-level `_yggrs` instead of `yggdrasil.rust`? Two wheels can't both own the `yggdrasil/` package directory — `ygg` already ships `yggdrasil/__init__.py` as a regular package. Shipping the extension as `_yggrs` and rebinding it through `yggdrasil/rs.py` keeps both wheels installable side by side.

### Daily loop

| Editing… | Command |
|---|---|
| Python only (`python/src/**`) | nothing — editable install picks up changes on next `import` |
| Rust (`rust/src/**`) | `cd rust && maturin develop --release` (re-imports pick it up after Python restart) |
| Cargo dependencies | `cd rust && cargo update`, then `maturin develop --release` |
| Switching Python versions | rebuild once: `maturin develop --release` (the `abi3-py310` wheel works across 3.10+, but the editable install path is per-venv) |

Inside a Python REPL, `importlib.reload` does **not** swap a compiled extension — restart the interpreter after `maturin develop`.

### Tests

```bash
# Python tests (always available, work without yggrs as a fallback)
cd python
pytest                                          # full suite
pytest tests/test_yggdrasil/test_io/test_url.py # single file

# Rust unit tests (none today; add under rust/src/<mod>/tests when needed)
cd ../rust
cargo test
```

Rust delegation is verified end-to-end through the Python tests — `yggdrasil.io.url.URL.from_str` calls into `_yggrs.io.url.parse_url`, and the URL test suite exercises that path.

### Optional dependencies

The package keeps `pyarrow` and `yggrs` as the only hard runtime deps. Pull integrations in as needed:

```bash
uv pip install -e .[data]         # pandas + numpy + sqlglot
uv pip install -e .[bigdata]      # pyspark + delta-spark
uv pip install -e .[databricks]   # databricks-sdk
uv pip install -e .[api]          # fastapi + uvicorn + pydantic
uv pip install -e .[pickle]       # cloudpickle/dill/zstandard/xxhash/blake3
uv pip install -e .[http]         # urllib3 + xxhash
uv pip install -e .[mongo]        # mongoengine
```

The Databricks live-integration tests are gated by the `integration` marker and skipped unless `DATABRICKS_HOST` is set.

### Lint, format, docs

```bash
cd python
ruff check
black .
mkdocs serve         # docs at http://127.0.0.1:8000
```

### Releasing

The version in [`python/pyproject.toml`](python/pyproject.toml) is the single source of truth.

- [`.github/workflows/publish.yml`](.github/workflows/publish.yml) builds and publishes the pure-Python `ygg` wheel + sdist.
- [`.github/workflows/publish-native.yml`](.github/workflows/publish-native.yml) builds and publishes `yggrs` wheels for `linux-{x86_64,aarch64}`, `windows-x86_64`, `macos-{x86_64,arm64}` plus an sdist, stamping the same version into `rust/pyproject.toml` and `rust/Cargo.toml` before building.

Both workflows trigger on the same events (push to `main` touching `python/src/**`, `rust/**`, the workflows, or any `v*` tag). They run in parallel; `ygg` declares `yggrs==X.Y.Z` so a release is only fully usable once both have landed.

Do not push to `main` from an agent session — develop on a branch and open a PR.

## License

See [LICENSE](LICENSE).
