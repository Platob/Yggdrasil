# AGENTS.md

## Project Overview
Yggdrasil is a **schema-aware data interchange library** (PyPI: `ygg`, import: `yggdrasil`). It converts values/schemas between Python types, Apache Arrow, Polars, pandas, Spark, and Databricks using a central **converter registry**. A secondary Power Query connector lives in `powerquery/` but is rarely changed.

## Architecture

### Converter Registry — the core pattern
Everything flows through `yggdrasil/data/cast/registry.py`. Converters are registered with `@register_converter(from_hint, to_hint)` and dispatched via `convert(value, target_type)`. Dispatch order: exact match → identity → Any-wildcard → MRO fallback → one-hop composition. Each engine module (`arrow/cast.py`, `polars/cast.py`, `pandas/cast.py`, `spark/cast.py`) registers its own converters on import — understand this pattern before adding new conversions.

### `lib.py` — lazy dependency guard
Every optional-dependency subpackage has a `lib.py` that tries to import the library and falls back to `PyEnv.runtime_import_module()` for auto-install. **Always import the external library through its `lib.py`**, never `import polars` directly:
```python
# correct
from yggdrasil.polars.lib import polars
# wrong — breaks when polars isn't installed
import polars
```

### `CastOptions` — the options object
`CastOptions` (in `data/cast/options.py`) threads through nearly every conversion. Use `CastOptions.check_arg()` to normalize the many accepted input types (`dict`, `pa.Field`, `pa.Schema`, `None`).

### `io/` — HTTP/IO subsystem
The `io/` package is the primary HTTP and binary-buffer layer:
- `io/session.py` defines the abstract `Session` base (Delta-table caching, batch dispatch via `send_many`, Spark scatter via `spark_send`).
- `io/http_/session.py` provides `HTTPSession`, the concrete urllib3-backed implementation. This is the **preferred HTTP client** for new code.
- `io/url.py` has `URL` (immutable parsed URL) and `URLResource` (base class for resources addressable by URL; register via `@register_url_resource`).
- `io/buffer/` has `BytesIO` (spill-to-disk buffer with compression/media-type detection) and format readers (`arrow_ipc_io.py`, `parquet_io.py`, `json_io.py`, `zip_io.py`).
- `io/send_config.py` has `SendConfig` / `SendManyConfig` for request-level options including cache control.
- `requests/session.py` has the older `YGGSession` (built on `requests.Session`), still used for simple retry-only HTTP. Prefer `io/http_/HTTPSession` for new features (caching, pagination, concurrency).

### `ai/` — AI session
`AISession` (in `ai/session.py`) wraps OpenAI with retry, caching (`Expiring`), and structured response parsing. `SQLSession` (in `ai/sql_session.py`) extends it for multi-dialect SQL generation (`SQLFlavor`: Databricks, PostgreSQL, DuckDB, Polars SQL, MongoDB). Both import OpenAI through a try/except + `PyEnv.runtime_import_module` guard (no `lib.py` file; the guard is inline).

### Module layout
- `data/cast/` — registry, `CastOptions`, datetime converters (core, no optional deps)
- `data/enums/` — `Timezone` normalisation; `geozone/` sub-package with `GeoZone` frozen dataclass (WKB-cached geospatial zones, Polars expression builders for batch parsing)
- `arrow/` — Arrow schema inference from type hints (`python_arrow.py`), Arrow↔Arrow casting (`cast.py`)
- `polars/`, `pandas/`, `spark/` — engine-specific converters, each with `lib.py` + `cast.py` + `extensions.py`
- `io/` — HTTP/IO subsystem: `BytesIO` spill-to-disk buffer, `URL`/`URLResource` registry, `PreparedRequest`/`Response`, `HTTPSession` (urllib3-backed with Delta-table caching), `SendConfig`/`SendManyConfig`, media-type detection, codec enums, format-specific buffer readers (`buffer/arrow_ipc_io.py`, `parquet_io.py`, `json_io.py`, `zip_io.py`)
- `databricks/` — workspace config, SQL engine, jobs, compute/cluster management, IAM (users/groups), secrets, account services
- `dataclasses/` — `dataclass_to_arrow_field`, `WaitingConfig`, `Expiring[T]` cache, `ExpiringDict[K,V]` TTL dict
- `pickle/` — custom serialization with wire-tag ranges (`ser/`), plus `json.py`, `dill.py`, `serde.py`
- `requests/` — `YGGSession` (retry-enabled HTTP via `requests` library), MSAL auth
- `ai/` — `AISession` (OpenAI-backed chat with retry/caching), `SQLSession` (multi-dialect SQL generation: Databricks, PostgreSQL, DuckDB, Polars SQL, MongoDB)
- `mongoengine/` — MongoEngine integration with `lib.py` guard, connection management, custom extensions (document/fields/queryset), `@with_mongo_connection` decorator
- `concurrent/` — `JobPoolExecutor` bounded thread pool for large/infinite job streams, Spark placeholder
- `web/` — async proxy servers (`proxy/http_.py` HTTP proxy, `proxy/mongo.py` MongoDB TLS proxy)
- `pyutils/` — `@retry`, `@parallelize` decorators, plus `equality`, `exceptions`, `dummy` helpers
- `environ/` — `PyEnv` for runtime module import/install, dependency metadata; also exports `runtime_import_module()` as a standalone function
- `blake3/`, `xxhash/` — hash library wrappers following the `lib.py` guard pattern
- `fastapi/` — optional REST API layer (entry point: `ygg-api`), with routers/services/schemas for Python, system, Databricks, and Excel execution

## Development Setup
```bash
cd python
uv venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac
uv pip install -e .[dev]
```
Install only the extras you need: `.[polars]`, `.[databricks]`, `.[bigdata]`, `.[pickle]`, `.[api]`, `.[http]`, `.[data]`.

## Testing
```bash
cd python
pytest                  # run all tests
pytest tests/test_yggdrasil/test_data/   # run a specific submodule
ruff check              # lint
black .                 # format
```
- Tests live in `python/tests/test_yggdrasil/`, mirroring the `src/yggdrasil/` layout.
- `conftest.py` injects `python/src` into `sys.path` and sets `DATABRICKS_HOST` env var for offline test stubs.
- Python **3.10+** required; CI runs 3.10–3.13.

## Conventions
- **`__all__` in every module** — public API is explicit; follow this when adding exports.
- **`from __future__ import annotations`** — used in most modules for deferred evaluation; new modules should include it.
- **Frozen/slotted dataclasses** for immutable config objects (`@dataclass(frozen=True, slots=True)`).
- **Converter signature**: `func(value, options=None) -> converted_value` — `options` is always `Optional[CastOptions]`.
- **`*` re-exports in `__init__.py`** — subpackages barrel-export via `from .module import *`.
- **Version** is in `src/yggdrasil/version.py` as a `VersionInfo` named tuple; bump there and in `pyproject.toml`.
- **pyarrow is the only hard runtime dependency** — everything else is optional and guarded.

