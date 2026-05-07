# Yggdrasil

**Schema-aware data interchange for Python.** One conversion registry that moves values cleanly between Python types, dataclasses, Arrow, Polars, pandas, Spark, Databricks, and the wire — without losing schema, nullability, or metadata along the way.

| Package | What it is | Where it lives |
|---|---|---|
| `ygg` (PyPI) / `yggdrasil` (import) | Pure-Python core: cast registry, Arrow schema, engine bridges, IO/HTTP, Databricks, FastAPI | [`python/`](python/) |
| `yggrs` (PyPI) | Optional Rust acceleration kernels (PyO3, `abi3-py310`) | [`rust/`](rust/) |
| Power Query connector | Excel `.pq` and Power BI `.mez` connectors that call the FastAPI service | [`powerquery/`](powerquery/) |

`pip install ygg` pulls both `ygg` and the matching `yggrs` wheel for your platform.

📚 **Docs site:** https://platob.github.io/Yggdrasil/

---

## Install

```bash
pip install ygg                   # core
pip install "ygg[data]"           # + pandas, numpy, sqlglot
pip install "ygg[bigdata]"        # + pyspark, delta-spark
pip install "ygg[databricks]"     # + databricks-sdk
pip install "ygg[api]"            # + fastapi, uvicorn, pydantic
pip install "ygg[http]"           # + urllib3, xxhash
pip install "ygg[pickle]"         # + cloudpickle, dill, zstandard, blake3
pip install "ygg[mongo]"          # + mongoengine
pip install "ygg[postgres]"       # + psycopg, adbc-driver-postgresql
pip install "ygg[kafka]"          # + confluent-kafka
pip install "ygg[delta]"          # + deltalake
```

The only hard runtime deps are `pyarrow>=20`, `polars>=1.3`, and the matching `yggrs` wheel. Everything else is opt-in.

---

## 60-second tour

### Cast anything into anything

```python
from yggdrasil.data.cast.registry import convert

convert("42", int)              # 42
convert("true", bool)           # True
convert("2024-01-15", "date")   # datetime.date(2024, 1, 15)
```

### Dict → typed dataclass (forgiving on input, strict on meaning)

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class Order:
    id: int
    amount: float
    paid: bool = False

convert({"id": "7", "amount": "99.50", "paid": "yes"}, Order)
# Order(id=7, amount=99.5, paid=True)
```

### Arrow schema as the contract surface

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])

out = cast_arrow_tabular(raw, CastOptions(target_field=target))
print(out.schema)
```

### Cross-engine in one move

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient().sql.execute("SELECT * FROM main.default.orders LIMIT 100")

stmt.to_arrow_table()   # pyarrow.Table
stmt.to_pandas()        # pandas.DataFrame
stmt.to_polars()        # polars.DataFrame
stmt.to_spark()         # pyspark.sql.DataFrame
stmt.to_pylist()        # list[dict]
```

---

## What you get

- **One conversion registry.** Register a converter once, dispatch from anywhere. Order: exact match → identity → `Any` wildcard → MRO fallback → one-hop composition.
- **Arrow schema as the contract.** Field names, order, nullability, metadata, nested structure, timezone intent are preserved across boundaries.
- **Engines bridge into Arrow.** Polars, pandas, Spark each register on import — `from yggdrasil.polars.cast import cast_polars_dataframe` etc.
- **Production HTTP stack.** `HTTPSession`, prepared requests, batch dispatch, typed response → Arrow/pandas/Polars/Spark.
- **Databricks toolkit.** `DatabricksClient` covers SQL, Unity Catalog, Compute, DBFS/Volumes, Secrets, IAM, Genie, Spark Connect.
- **Optional dep guards.** Base installs stay light. `from yggdrasil.polars.lib import polars` is the safe import.
- **Rust fast path, Python canonical.** `yggdrasil.rs` exposes accelerated functions; tests pass with and without `yggrs` installed.

---

## Use cases at a glance

| You want to… | Reach for |
|---|---|
| Normalize dicts/JSON into typed dataclasses | `convert(payload, MyDataclass)` |
| Pin a downstream Arrow schema | `cast_arrow_tabular(t, CastOptions(target_field=schema))` |
| Convert Polars ↔ Arrow ↔ pandas ↔ Spark | `yggdrasil.{polars,pandas,spark}.cast` |
| Fan out HTTP requests with retries | `HTTPSession().send_many(reqs, SendManyConfig(...))` |
| Run SQL on Databricks and get a DataFrame | `DatabricksClient().sql.execute(q).to_polars()` |
| Read/write DBFS or Volume files | `DatabricksClient().dbfs_path("...").write_text(...)` |
| Type-check job widget params | `MyConfig.from_environment()` (subclass `NotebookConfig`) |
| Talk to Databricks from Excel/Power BI | Power Query connector via FastAPI service |

---

## Repository guide

- [`python/`](python/) — `ygg` source, tests, MkDocs site.
  - [`python/README.md`](python/README.md) — package guide with progressive examples (scalars → schema → engines → HTTP → Databricks).
  - [`python/docs/`](python/docs/) — published documentation source (https://platob.github.io/Yggdrasil/).
- [`rust/`](rust/) — `yggrs` crate (maturin + PyO3, `abi3-py310`).
- [`powerquery/`](powerquery/) — Excel `.pq` and Power BI `.mez` connectors over the FastAPI service.
- [`vercel/`](vercel/) — Next.js frontends and Vercel-hosted apps. One subfolder per app, each with its own Vercel project.
- [`AGENTS.md`](AGENTS.md) — house style, error-message tone, comment voice, API ergonomics.
- [`CLAUDE.md`](CLAUDE.md) — agent-facing notes for AI contributors.

---

## Develop locally

```bash
git clone https://github.com/Platob/Yggdrasil.git
cd Yggdrasil/python

uv venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
uv pip install -e .[dev]                       # core + dev tooling

# Optional: native acceleration (editable, into the same venv)
cd ../rust && maturin develop --release
```

`yggdrasil/rs.py` is the only place that imports from `yggdrasil.rust.*`. With `yggrs` installed it dispatches to native; without it, the pure-Python fallback runs. Tests must pass either way.

```bash
cd python
pytest                          # full suite
pytest tests/test_yggdrasil/test_io/test_url.py   # one file
ruff check
black .
mkdocs serve                    # docs at http://127.0.0.1:8000
```

Databricks live-integration tests are gated by the `integration` marker and skipped unless `DATABRICKS_HOST` is set.

---

## Release pipeline

The version in [`python/pyproject.toml`](python/pyproject.toml) is the single source of truth.

| Workflow | Builds | Triggers |
|---|---|---|
| [`publish.yml`](.github/workflows/publish.yml) | `ygg` sdist + pure-Python wheel → PyPI, then tags `vX.Y.Z` and cuts a GitHub Release | push to `main` touching `python/src/**`, `pyproject.toml`, README, LICENSE, `rust/**`, or workflow itself |
| [`publish-native.yml`](.github/workflows/publish-native.yml) | `yggrs` wheels for `linux-{x86_64,aarch64}`, `windows-x86_64`, `macos-{x86_64,arm64}` + sdist → PyPI | same triggers |
| [`docs.yml`](.github/workflows/docs.yml) | MkDocs Material site → GitHub Pages (https://platob.github.io/Yggdrasil/) | push to `main` touching `python/docs/**`, `python/src/**`, `mkdocs.yml`, or workflow itself |

`ygg` declares `yggrs==X.Y.Z` so a release is fully usable once both PyPI uploads land.

Do not push to `main` from an agent session — develop on a branch and open a PR.

---

## License

[Apache-2.0](LICENSE).
