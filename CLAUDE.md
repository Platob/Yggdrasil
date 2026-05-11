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
- **Integrate, don't invent.** Before writing anything new, find where the behavior already lives and extend it. Modify the function/class that already does most of the job, register into `data/cast/registry.py` instead of writing a one-off conversion at a call site, add a flag to `CastOptions` instead of a new options object, add a method to `DataField`/`Schema`/`DataTable`/`StatementResult` instead of a sibling helper module, use `HTTPSession`/`PreparedRequest`/`Response` instead of a parallel HTTP path. Only create a new module/class/abstraction when the existing surface genuinely cannot host the behavior — and wire it back through the canonical path so the next caller finds it.
- **No Python `for` loops over data.** Row-by-row Python iteration over Arrow / Polars / pandas / Spark values is the single biggest performance trap in this codebase. Reach for vectorised primitives in this order before considering a loop:
  1. **pyarrow.compute** kernels (`pc.cast`, `pc.if_else`, `pc.list_element`, `pc.binary_join_element_wise`, `pc.replace_substring_regex`, `pc.fill_null`, `pc.equal`, etc.) — these stay inside the C++ runtime and don't cross a Python frame per row.
  2. **pyarrow.json / pyarrow.csv** vectorised readers when the workload is "decode N rows of `<format>` into a typed Arrow array" (see `_cast_json.py:_vectorized_parse_json`).
  3. **Polars expressions** (`pl.col(...).str.json_decode`, `.str.contains`, `.cast`, `.list.eval`, `pl.when(...).then(...).otherwise(...)`) as the next-best vectorised fallback when pyarrow.compute doesn't cover the operation. Build a polars Series / LazyFrame, run the expression, hand the result back to Arrow via `.to_arrow()`.
  4. **Numpy ufuncs** for numerical work where the data is already in a numpy-backed buffer.
  - Only after those exhaust their coverage is a Python loop acceptable, and even then **only as a documented fallback path** (per-row-on-failure permissive decode, per-row-because-pyarrow-can't-emit-maps, etc.) — comment why the vectorised path doesn't cover the case and what the fallback's cost is.
  - `array.to_pylist()` followed by a Python comprehension is the same trap with one less line — same rule applies. Look for a vectorised expression first; reach for `to_pylist` only when materialising into Python objects is the genuine endpoint (e.g. JSON encoding via `json.dumps`).
  - The shape of an acceptable per-row fallback already exists at `_cast_json.py:_parse_via_python` (vectorised C++ NDJSON first, per-row only when that raises in permissive mode). Mirror that pattern; don't reinvent it.
- **Never `to_pylist` / `to_list` / `tolist` heavy data.** `pa.Array.to_pylist`, `pl.Series.to_list`, `pd.Series.tolist`, and `np.ndarray.tolist` all walk every cell into a Python object — same per-row cost as a Python loop, just hidden in C. **Do not use them in cast helpers, frame converters, batch readers, type inference, or any hot transform path.** When you need to leave Arrow/Polars/pandas, use the engine's own zero-copy bridge:
  - Arrow → pandas: `Array.to_pandas()` (struct cells surface as dicts, list cells as numpy arrays — the standard pyarrow → pandas mapping; no per-row hop).
  - Arrow → polars: `pl.from_arrow(arr)` (zero-copy).
  - Arrow → numpy (numeric only): `Array.to_numpy(zero_copy_only=True)`.
  - Pandas → Arrow: `pa.array(series, from_pandas=True, type=...)` — pass the type so the C bridge skips inference.
  - Polars → Arrow: `series.to_arrow()` (zero-copy).
  - Pandas → Polars: `pl.from_pandas(df)`.
  - Build a per-row Series of dicts from per-child arrays: `pa.StructArray.from_arrays(arrs, names=..., mask=...).to_pandas()` — one C-bridge pass, no row loop.
  Three exemptions, *only* these, keep a `to_pylist` / `to_list` call: (1) documented per-row fallback for shapes vectorised engines genuinely can't emit (e.g. permissive `_cast_json` paths); (2) genuine row endpoints where the workload IS "yield Python rows to a downstream sink" — ndjson / xlsx writers, kafka producers, mongo inserts, JSON HTTP responses, `Tabular.to_pylist` / `iter_pylist` API methods, pickle one-shot serialization; (3) diagnostics — test assertion formatters, debug logging, `repr`. Anywhere else, the answer is "use the zero-copy bridge above." See `AGENTS.md` → "Never materialise heavy data via `to_pylist` / `to_list` / `tolist`" for the worked rules.
- **Use `yggdrasil.pickle.json` for JSON, not stdlib `json`.** `orjson` is a hard dependency; `yggdrasil.pickle.json` wraps it with the type coverage we use across the codebase (datetime/date/time/UUID/Path/Enum/dataclass/namedtuple/set/Mapping/bytes/Decimal). Same `loads / dumps / load / dump` surface, faster, fewer surprises with rich types. Drop down to `import orjson` directly only inside hot inner loops where you don't need any of the type coercions. Stdlib `import json` is reserved for: parsing third-party config where stdlib semantics are part of the contract, debug `print` formatting, and inside `yggdrasil.pickle.json` itself (which keeps stdlib as a fallback for option combos orjson can't express).
- **Reach for the enums in `yggdrasil.data.enums` first.** Whenever a value belongs to a fixed token set — currencies, time units, byte sizes (`ByteUnit`), media types, MIME types, modes, codecs, geozones, timezones — use the enum's member or its `Enum.from_(...)` / `Enum.parse_size(...)` / `Enum.parse(...)` coercion at the API boundary instead of hand-rolling a dict, regex, alias table, or expression like `int(x) * 1024 * 1024` at the call site. Centralizing these tokens in an enum is how aliases, validation, formatting, and cross-engine mapping stay consistent across the codebase. If an enum is missing a member or alias the code legitimately needs, **add it to the enum** and have the caller route through it; do not branch around the enum with a one-off lookup. When introducing a new fixed-vocabulary concept, add a new enum to `yggdrasil.data.enums` (matching the `from_` / `parse*` / `is_valid` shape of the existing ones) instead of scattering string constants.
- **Keep diffs small and reuse-heavy.** Prefer a small edit to an existing function over a new module. Don't add parameters, branches, classes, or files "in case we need them later" — add them when a real caller needs them. Three nearly-identical lines is fine; a premature base class is not.
- **No dead or isolated code.** Every new function, class, option, file, or test must be reachable from a real caller in the same change. No helpers defined but never called, no flags never read, no new files that only re-export existing names, no speculative branches for inputs the public API doesn't accept. If nothing uses it yet, delete it or wait for the caller.
- **Be forgiving on input, strict on meaning.** Accept the shapes a real caller has (type hints, strings, dict payloads, framework objects, things with `.schema`/`.type`/`.arrow_schema`). But fail loudly on *conflicting* arguments — don't silently pick a side.
- **Fail fast on remote resources; retry the real call, don't pre-check it.** Every call to a remote system (HTTP, Databricks Files / SQL / Workspace, S3, MongoDB, Spark cluster, …) costs latency and quota. Do the operation and handle the error — don't gate `download` / `upload` / `delete` / `read_bytes` / `iterdir` on a preceding `exists()` / `stat()` / `get_metadata()` / `HEAD` probe. The probe doubles the round trip, races concurrent writers, and lies under eventual consistency; catch `NotFound` / `404` / `FileNotFoundError` from the real call instead. Wrap the operation in the existing retry policy (`retry_sdk_call`, `_call`, `_call_ensuring_parents`, the HTTP send retry config) to absorb transient 5xx / throttling / connect timeouts; let deterministic errors (`NotFound`, `AlreadyExists`, `PermissionDenied`, stable `BadRequest`) propagate immediately. Reuse the existing stat / metadata caches (`RemotePath._STAT_CACHE`) instead of re-issuing the call, and invalidate them after mutations (`_invalidate_stat_cache`). Default round-trip budget for a new op: **one** for the intended action plus at most one parent-recovery retry — count and justify any more. See `AGENTS.md` → "Fail fast on remote resources" for the worked rules.
- **Preserve schema intent across boundaries.** Field names, order, nullability, metadata, nested structure, precision/scale, timezone intent are part of the user contract. Don't drop them unless the API documents the loss.
- **Error messages must answer: what you passed, what was expected, valid values, what to try next.** Suggest near matches on typos when practical. See `AGENTS.md` for worked examples and tone (blunt, helpful, lightly conversational — no slang/emoji/meme formatting).
- **Comments describe weirdness, not syntax.** Version quirks, engine edge cases, schema invariants, compatibility hacks — yes. `# loop through fields` — no.
- **Type hints must match runtime.** If a method can return `None`, annotate `| None`.
- **Keyword-only arguments** are preferred for ambiguous options.
- **Fetch online documentation when third-party behaviour is in question.** When a fix or integration turns on the contract of an external SDK / HTTP API / cloud service (Databricks, Spark, MongoDB, Polars, pandas, FastAPI, Power Query, etc.), reach for `WebSearch` / `WebFetch` (or the relevant MCP doc-fetch tool) and read the vendor docs / SDK reference / GitHub source before guessing. Treat the official docs and the SDK's own source as the source of truth — quote the relevant signature, default, or error string in the commit message or PR body so the next reader can verify. This is explicitly allowed and encouraged; do not hand-wave around an unfamiliar API surface.
- **Use `...` (Ellipsis) as the unset / missing sentinel.** When you need to distinguish "caller didn't pass this" from "caller passed `None`" — keyword defaults, `dict.get(key, ...)` to tell missing from `None`, lazy-init cache slots — reach for the built-in `...` singleton instead of a private `_UNSET = object()` / `_MISSING = object()` per module. Reads cleanly (`if cached is not ...:`), avoids per-module sentinel proliferation, and is a real singleton across pickle boundaries. Only allocate a private sentinel when `...` is itself a legitimate value in the domain.
- **Make every long-lived object picklable + hashable + singleton-by-config.** Configs / clients / sessions / services / paths / schemas all cross pickle boundaries (Spark workers, multiprocessing, FastAPI forks, Power Query bridges) and end up as dict keys / set members / `groupBy` keys. The standard pattern, used by `yggdrasil.io.session.Session` and `yggdrasil.aws.AWSClient`:
  - Pure-data configs are `@dataclass(unsafe_hash=True)` with `compare=False, hash=False, repr=False` on unhashable members (callables, live handles).
  - Live "client/session/service" classes cache instances per `(cls, config)` in a class-level `_INSTANCES` dict via `__new__`; `__init__` is idempotent (`if getattr(self, "_initialized", False): return`); `__getnewargs__` returns the cache key so in-process unpickle collapses to the live singleton.
  - Generic `__getstate__`/`__setstate__`: list non-picklable handles in `_TRANSIENT_STATE_ATTRS: ClassVar[frozenset[str]]`; `__getstate__` filters them out of `__dict__`; `__setstate__` short-circuits when the singleton is already initialized, otherwise restores `__dict__` and re-inits the transient slots to their fresh defaults. Subclasses extend `_TRANSIENT_STATE_ATTRS` with `Base._TRANSIENT_STATE_ATTRS | frozenset({...})` and chain `__setstate__` to super.
  - Hash + equality follow `(type(self), config)`, not `id(self)`.
  - Every new such class ships at least: same-config-same-instance, idempotent `__init__`, in-process pickle preserves singleton, cross-process pickle (`Cls._INSTANCES.clear()` then unpickle) rebuilds transients, `_TRANSIENT_STATE_ATTRS` actually keeps live handles out of the payload. See `AGENTS.md` → "Make objects picklable and hashable by default" for the full pattern and worked example.

## Release & CI

- Version is the single source of truth in `python/pyproject.toml`. `.github/workflows/publish.yml` reads it, checks PyPI, and on `main` pushes that touch `python/src/**`, `pyproject.toml`, README, or LICENSE, it builds the sdist + pure-Python wheel and publishes `ygg`, then tags `vX.Y.Z` and cuts a GitHub Release. Native wheels for `yggrs` are built separately by a maturin-based workflow across linux (x86_64 + aarch64/QEMU), windows-x86_64, macos-arm64, and macos-x86_64.
- `.github/workflows/docs.yml` publishes the MkDocs site to GitHub Pages (`https://platob.github.io/Yggdrasil/`).
- Do not push to `main` directly from an agent session — develop on the branch you were given and open a PR only when the user asks.
