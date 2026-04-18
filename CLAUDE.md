# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project layout

Three sibling packages live at the repo root:

- `python/` — the main library. PyPI package `ygg`, import package `yggdrasil`. Sources in `python/src/yggdrasil/`, tests in `python/tests/test_yggdrasil/`, docs in `python/docs/` (MkDocs).
- `rust/` — optional native acceleration published separately as `yggrs`. Built with `maturin` + PyO3 `abi3-py310`. Slots into the `yggdrasil` namespace as `yggdrasil.rust.*`; the library reaches it only through `yggdrasil/rs.py` (the Python-fallback bridge).
- `powerquery/` — Excel (`.pq`) and Power BI (`.mez`) connectors that talk to the FastAPI service in `yggdrasil.fastapi`. Build/install via `build.ps1` / `install.ps1` (PowerShell).

`AGENTS.md` at the root is the canonical style, tone, and design-rule guide — read it before non-trivial changes. It supersedes generic instincts on error messages, comment voice, and API ergonomics.

## Common commands

All Python commands run from `python/`.

```bash
cd python
uv venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
uv pip install -e .[dev]                     # core dev tooling

# Optional extras are additive — install only what a task needs:
#   [data] pandas+polars+numpy   [bigdata] pyspark+delta
#   [databricks] databricks-sdk  [api] fastapi+uvicorn+pydantic
#   [pickle] cloudpickle/dill/zstandard/xxhash/blake3
#   [http] urllib3   [mongo] mongoengine

pytest                                        # full suite
pytest tests/test_yggdrasil/test_data/        # scope to one area
pytest tests/test_yggdrasil/test_data/test_registry.py::test_name  # single test
ruff check
black .
```

- Databricks live-integration tests are gated by the `integration` marker and skipped unless `DATABRICKS_HOST` is set (see `[tool.pytest.ini_options]` in `python/pyproject.toml`).
- `pytest-asyncio` runs in `strict` mode — async tests need the explicit marker.
- `python/tests/conftest.py` injects `python/src` onto `sys.path` and wires a `yggdrasil` DEBUG logger, so tests run without installing the package.

### Data tests use the engine TestCase base classes

Any test that touches a dataframe, Arrow object, or engine-side type **must** subclass the matching base class from `yggdrasil.*.tests` instead of importing the engine module at the top of the file. These bases handle optional-dependency skipping, per-test tmp dirs, Arrow interop, and frame/schema assertions — you get a consistent skip story and less boilerplate for free.

| Engine | Base class | Module |
| --- | --- | --- |
| Arrow | `ArrowTestCase` | `yggdrasil.arrow.tests` |
| Polars | `PolarsTestCase` | `yggdrasil.polars.tests` |
| pandas | `PandasTestCase` | `yggdrasil.pandas.tests` |
| Spark | `SparkTestCase` | `yggdrasil.spark.tests` |

Rules:

- Do `from yggdrasil.arrow.tests import ArrowTestCase` (etc.) and subclass it. Use `self.pa` / `self.pl` / `self.pd` / `self.spark` rather than a top-level `import pyarrow as pa` — a bare import breaks base installs and defeats the skip-on-missing behavior.
- Use the provided helpers (`self.table(...)`, `self.df(...)`, `self.lazy(...)`, `self.record_batch(...)`, `self.arrow_to_polars(...)`, `self.write_parquet(...)`, `self.tmp_path`, etc.) instead of reimplementing fixtures.
- Prefer the built-in assertions (`assertFrameEqual`, `assertSchemaEqual`, `assertSeriesEqual`, `SparkTestCase.assertDataFrameEqual` / `assertSparkEqual`) — they give readable diffs and handle dtype/order/index knobs consistently.
- Cross-engine tests can multi-inherit (e.g. `class TestX(PolarsTestCase, ArrowTestCase):`) or split into sibling classes in the same file — both work; pick whichever keeps the test body clean.
- `SparkTestCase` shares a single process-wide `SparkSession`; don't call `SparkSession.builder` yourself and don't stop the session in `tearDown`.
- For Databricks integration tests, keep the `integration` marker *and* subclass `ArrowTestCase` (or whichever engine is actually exercised) so the local run still skips cleanly without `DATABRICKS_HOST`.

Rust extension (only when touching `rust/`):

```bash
cd rust
maturin develop --extras dev   # editable build, compiles the extension
maturin build --release        # wheels → rust/dist/
```

Docs site (MkDocs Material):

```bash
cd python
mkdocs serve    # or `mkdocs build`
```

FastAPI service (used by the Power Query connector):

```bash
python -m yggdrasil.fastapi.main   # entry point, also exposed as `ygg-api`
```

## Architecture

### Reach for `yggdrasil.data` before raw engine APIs

`yggdrasil.data` is the canonical surface for describing and moving frames. When a task involves a dataframe, schema, field, type, or a cross-engine conversion, start from `yggdrasil.data` and only drop down to `polars` / `pandas` / `pyspark` / `pyarrow` when you actually need something the abstraction does not cover.

Prefer these primitives in this order:

- `yggdrasil.data.DataField` / `Field` and `yggdrasil.data.Schema` for describing columns — they carry names, nullability, metadata, tags, nested structure, and engine-side dtype intent in one place. Build them with `Field.from_pandas`, `Field.from_polars`, `Field.from_arrow`, `Schema.from_any_fields`, etc. instead of re-building per-engine schemas by hand.
- `yggdrasil.data.DataType` / `DataTypeId` (and the `types/` submodules: `primitive`, `nested`, `iso`, `extensions`) for type hints — don't hand-roll `pa.int64()` / `pl.Int64` / `"bigint"` strings when a `DataType` can produce all of them.
- `yggdrasil.data.DataTable` and `yggdrasil.data.statement_result.StatementResult` for anything that looks like "execute a query, then move rows somewhere" — new integrations should implement these interfaces rather than invent a parallel one.
- `yggdrasil.data.cast.convert(value, target, options=...)` with `CastOptions` for value conversion. Do not call `df.to_pandas()`, `pl.from_arrow(...)`, or `spark.createDataFrame(...)` directly from feature code if a registered converter already handles it — that's what the registry is for.
- `yggdrasil.data.enums` (currency, geozone, timezone) when you need normalized domain values; do not re-parse those strings inline.

Only reach past this layer when adding a new converter, writing engine-internal glue in `arrow/` / `polars/` / `pandas/` / `spark/`, or implementing performance-critical code that the abstraction intentionally doesn't cover. When you do, register the new behavior back into `yggdrasil.data` (converter + `DataType` + `DataField`/`Schema` support) so the next caller gets it for free.

### Converter registry is the core

`yggdrasil/data/cast/registry.py` holds a single registry that every engine plugs into. Register converters with `@register_converter(from_hint, to_hint)` and dispatch them via `convert(value, target)`. Dispatch order is: **exact match → identity → `Any` wildcard → MRO fallback → one-hop composition.** New conversions belong in the registry (or an engine-specific converter module or shared normalization helper) — not as ad-hoc one-offs at call sites.

Engine modules register their converters **on import**: `arrow/cast.py`, `polars/cast.py`, `pandas/cast.py`, `spark/cast.py`. If an expected conversion does not fire, check whether the engine module has been imported.

### `CastOptions` is the shared options object

`yggdrasil/data/cast/options.py` defines `CastOptions` — the single normalized carrier for source hints, target fields/schemas, safety/memory/nullability behavior, and strictness flags. Do not invent parallel per-call option objects; extend `CastOptions` or pass it through.

### Optional-dependency pattern (`lib.py`)

Subsystems that depend on optional packages (polars, pandas, spark, databricks, blake3, xxhash, etc.) expose a `lib.py` guard that does the import once and raises a helpful "install extra X" error on failure. Always go through the guard:

```python
from yggdrasil.polars.lib import polars   # correct
import polars                             # wrong — breaks base installs
```

The only hard runtime dependency of `ygg` is `pyarrow` (`>=20`). Base installs must keep working without any other engine.

### `io/` is a real integration boundary

`yggdrasil/io/` is not "utilities" — it's the transport surface: buffers (`buffer/` → `BytesIO`, `MediaIO`), URLs, requests, responses, codecs, media types, sessions, pagination, caching. For new HTTP work prefer the modern stack in `io/http_/`: `HTTPSession`, `PreparedRequest`, `Response`, `SendConfig` / `SendManyConfig`. Preserve observability fields (normalized URL parts, promoted/remaining headers, body bytes, payload hashes, timestamps, status/timing) — tooling downstream relies on them. Buffer changes must preserve spill-to-disk behavior, codec handling, cursor safety, and Arrow/Parquet/JSON/IPC compatibility.

### Rust fast path, Python is canonical

`yggdrasil/rs.py` is the **only** place that imports from `yggdrasil.rust.*`. It exposes `HAS_RS` plus the fallback-capable entry points (e.g. `utf8_len`). Rules:

- Python behavior is the source of truth; Rust must match it, not diverge.
- Pure-Python fallback must stay correct on its own — tests must pass with and without `yggrs` installed.
- Only add Rust to a path that is actually hot and semantically stable.

### Module map (top level of `yggdrasil/`)

- **Schema & conversion (start here):** `data/` is the canonical surface — cast registry + `CastOptions`, `DataType`/`DataTypeId`, `DataField`/`Schema`/`DataTable`, `StatementResult`, normalized enums (currency, geozone, timezone). `arrow/` holds type inference and Arrow cast helpers; `dataclasses/` has dataclass→Arrow field helpers and waiting/expiring utilities.
- **Dataframe engines:** `polars/`, `pandas/`, `spark/` — each registers on import and provides a `lib.py` guard *and* a `tests.py` TestCase base. Prefer the `yggdrasil.data` abstractions over calling these engines directly from feature code.
- **Platform integrations:** `databricks/` (sub-packages `sql/`, `jobs/`, `compute/`, `iam/`, `secrets/`, `workspaces/`, `fs/`, `account/`, `ai/`; entry via `DatabricksClient`), `mongo/`, `mongoengine/`, `fastapi/` (routers power the Power Query connector).
- **Serialization & hashing:** `pickle/` (custom serialization with optional cloudpickle/dill/zstandard), `blake3/`, `xxhash/` — all guarded.
- **Utilities:** `pyutils/` (retry, parallelize, helpers), `concurrent/`, `environ/` (runtime import/install logic), `fxrates/`, `requests/`, `exceptions.py`, `version.py`.

## Conventions that aren't obvious

- **Python 3.10 minimum.** Don't use 3.11+ syntax (e.g. `typing.Self`) without a fallback. The Rust `abi3-py310` wheel is the compatibility floor.
- **Be forgiving on input, strict on meaning.** Accept the shapes a real caller has (type hints, strings, dict payloads, framework objects, things with `.schema`/`.type`/`.arrow_schema`). But fail loudly on *conflicting* arguments — don't silently pick a side.
- **Preserve schema intent across boundaries.** Field names, order, nullability, metadata, nested structure, precision/scale, timezone intent are part of the user contract. Don't drop them unless the API documents the loss.
- **Error messages must answer: what you passed, what was expected, valid values, what to try next.** Suggest near matches on typos when practical. See `AGENTS.md` for worked examples and tone (blunt, helpful, lightly conversational — no slang/emoji/meme formatting).
- **Comments describe weirdness, not syntax.** Version quirks, engine edge cases, schema invariants, compatibility hacks — yes. `# loop through fields` — no.
- **Type hints must match runtime.** If a method can return `None`, annotate `| None`.
- **Keyword-only arguments** are preferred for ambiguous options.

## Release & CI

- Version is the single source of truth in `python/pyproject.toml`. `.github/workflows/publish.yml` reads it, checks PyPI, and on `main` pushes that touch `python/src/**`, `pyproject.toml`, README, or LICENSE, it builds the sdist + pure-Python wheel and publishes `ygg`, then tags `vX.Y.Z` and cuts a GitHub Release. Native wheels for `yggrs` are built separately by a maturin-based workflow across linux (x86_64 + aarch64/QEMU), windows-x86_64, macos-arm64, and macos-x86_64.
- `.github/workflows/docs.yml` publishes the MkDocs site to GitHub Pages (`https://platob.github.io/Yggdrasil/`).
- Do not push to `main` directly from an agent session — develop on the branch you were given and open a PR only when the user asks.
