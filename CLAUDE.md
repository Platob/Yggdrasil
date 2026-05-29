# Yggdrasil

Distributed node framework — Python backend, Next.js frontend, Nordic dark UI.

## Coding Style

1. **Inline over micro-functions** — prefer inlining logic directly where it's used. Don't extract one-liner helpers, don't create utility functions called from a single site, don't wrap stdlib calls in project-specific wrappers. A 15-line method is better than 5 three-line functions calling each other. Extract only when the same logic appears 3+ times or the extracted function has genuine standalone semantics (e.g. `make_id`, `serialize_result`).
2. **Flat call stacks** — avoid deep delegation chains like `router → service → helper → sub-helper → util`. Routers call services. Services do the work. Two levels max for the common path.
3. **No defensive wrappers** — don't wrap `json.loads` in a `safe_parse_json`, don't wrap `Path.mkdir` in an `ensure_dir`. Call stdlib directly. Handle errors at the boundary where they matter.
4. **No premature abstraction** — three similar blocks of code is fine. A `BaseService` with one subclass is not. Don't create interfaces, registries, or plugin systems until the third concrete user exists.
5. **Collocate related code** — keep schemas, service logic, and the router for one concept readable together. Jumping between 5 files to understand one endpoint is worse than a 200-line service file.
6. **Delete dead code** — no commented-out blocks, no `# TODO: maybe later`, no unused imports. If it's not called, it doesn't exist.
7. **Prefer data over code** — dicts and lists over class hierarchies. Pydantic models over hand-rolled validation. Enum values over if/elif chains.

## Communication & Transport

1. **Arrow IPC for tabular data** — all structured data between nodes uses `application/vnd.apache.arrow.stream`. Use `transport.serialize_result` / `deserialize_result` which auto-dispatches: Arrow for tables/dataframes, yggdrasil pickle for complex Python objects.
2. **Stream everything** — SSE for logs, metrics, run state. Chunked HTTP for file transfer (64KB). Never buffer an entire large response in memory. Use `StreamingResponse` + async generators.
3. **Lazy by default** — don't fetch what you don't need. Directory listings are lazy (click to expand). File contents only on demand. Peer metadata cached with TTL, refreshed on access. Cache small payloads (<4MB) in `ExpiringDict`, stream anything larger.
4. **Connection pooling** — reuse `httpx.AsyncClient` per peer node instead of creating one per request. The `NetworkService` owns the pool.
5. **Content-addressed assets** — every asset (PyFunc, PyEnv, DAG) has a `content_hash` (sha256 of defining content). Two assets with the same hash are identical regardless of which node created them. Use hash for fast replication diff: only replicate what changed.

## Asset Design

1. **Self-describing** — every asset carries all metadata to rebuild it anywhere: code, dependencies, python version, content hash. No external references that can't be resolved.
2. **Cross-references by hash** — when a PyFuncRun references a PyFunc and PyEnv, it stores their content hashes. A receiving node can verify it has the right versions or request them.
3. **Multi-Python via uv** — `PyEnv` specifies `python_version` (3.11, 3.12, 3.13). `uv venv --python X.Y` creates isolated envs. Code is replicable across nodes with different system Pythons because uv downloads the right interpreter.
4. **Dependency inference** — `@function` decorator infers dependencies from AST import analysis. Explicitly listed deps override inference. Dependencies are part of the content hash.

## Permissions & Security Posture

1. **User identity** — every request carries a user identity (sha256 hash of key+hostname). `UserService` auto-registers the local user and discovers peers.
2. **Open by default** — all operations are allowed for all users. Permission checks are a middleware concern, not service logic. The `user_hash` is logged on every mutation for audit trail, but never blocks.
3. **Audit log** — mutations (create, update, delete, replicate) log `(timestamp, user_hash, operation, asset_hash)`. Read-only operations are not logged.
4. **Node acts as user** — the node operates with the current user's full permissions. It can read/write files, install packages (`uv pip`, `npm install`), spawn subprocesses, execute arbitrary Python received over HTTP. There is no sandboxing, no syscall jail, no capability filter — the node IS the user's workstation, exposed as an HTTP API.
5. **Intentional, not accidental** — POC mode prioritizes velocity and ergonomics over isolation. Running the node bound to a public interface on a hostile network is unsupported. For multi-tenant isolation, run one node per tenant in a VM or container and federate them via the v2 network mesh.
6. **Secrets handling** — environment-variable secrets live in `~/.node/.env` (gitignored). The node never persists API keys to its asset store; replicated PyFuncs carry code only, not the env they execute in.

## Auto-configuration

1. **Zero-config decorators** — `@function` infers name, code, dependencies, python version. `@function(auto_env=True)` also creates a matching PyEnv with the right Python version and deps.
2. **Auto-env creation** — when a function is registered without an explicit environment, the system auto-creates one named `auto_{func_name}` with inferred deps.
3. **Multi-python** — `uv venv --python 3.11/3.12/3.13` creates isolated envs. Code is replicable across nodes with different system Pythons.
4. **Auto-install on start** — `ygg node start` auto-registers systemd/launchd boot services so the node survives reboots.

## Principles

1. **Exceptions** — derive from `YGGException`. API errors use `yggdrasil.exceptions.api`. No ad-hoc exception classes.
2. **Services own logic** — routers are thin. Business rules + state live in `services/`.
3. **Schemas are contracts** — `StrictModel` (extra="forbid") for all request/response types.
4. **Streaming first** — use SSE for logs, metrics, long-running operations.
5. **POC mode** — ship fast, iterate. No legacy compat.
6. **Int64 IDs** — prefer `int` over string UUIDs. Use xxhash-based composites: `xxh32(semantic_key) << 32 | timestamp_ms`. Two xxhash int64s are acceptable when coupling different concepts. Never use cryptographic hashes for IDs.
7. **Upsert by default** — POST endpoints create if name not found, update if it exists. ID is immutable once assigned.

## Layout

```
python/src/yggdrasil/
  node/                 Node server (FastAPI, default :8100)
    api/                v2 API — formalized PyEnv/PyFunc/PyFuncRun concepts
      schemas/          Pydantic models (PyEnv, PyFunc, PyFuncRun, DAG, Backend, Network)
      services/         Business logic (env mgmt, execution, metrics, peer mesh)
      routers/          HTTP handlers mounted at /api/v2/*
      app.py            Standalone v2 FastAPI app factory
      deps.py           Dependency injection
    routers/            v1 HTTP handlers (function, environment, run, monitor, ...)
    services/           v1 Business logic + state
    schemas/            v1 Pydantic models
    geo.py              IP geolocation
    ids.py              Int64 ID generation (xxhash)
    fn.py               @function decorator framework
    transport.py        Arrow IPC + pickle serialization for inter-node comms
    path.py             NodePath — pathlib-like local/remote filesystem
  cli/                  ygg CLI
  exceptions/api.py     APIError hierarchy
  databricks/           Databricks SDK integrations
nextjs/                 Frontend (React 19, Next.js 15, Tailwind v4)
  src/app/              Next.js app router pages
  src/components/       Shared React components (globe, brain mesh, sidebar)
```

## Node API

| Prefix | Description |
|--------|-------------|
| `/api/hello` | Discovery, peer registration, geolocation |
| `/api/function` | Functions CRUD (code, metadata, deps) |
| `/api/function/{id}/run` | Trigger function runs |
| `/api/environment` | Python venv management (uv-based) |
| `/api/run` | Run history, logs (SSE streaming) |
| `/api/monitor` | Resource + network IO metrics (SSE streaming) |
| `/api/dag` | DAG definitions — chain functions across nodes |
| `/api/dag/{id}/run` | Execute DAG with cross-node orchestration |
| `/api/python` | Direct Python code execution |
| `/api/cmd` | Shell command execution |
| `/api/job` | Job definitions + scheduled runs |
| `/api/messenger` | Real-time messaging channels |
| `/api/call` | @remote function registry + invocation |
| `/api/env` | Environment variable get/set |
| `/api/fs` | Filesystem operations (ls, read, write, mkdir, upload, streaming) |

## Node v2 API (PyEnv / PyFunc / PyFuncRun)

Core concepts — workstation as remote executor/driver:

- **PyEnv** — Python environment (uv venv). Has `execute_pyfunc()` for inner dispatch.
- **PyFunc** — Executable (code + deps + metadata). Upserted by name.
- **PyFuncRun** — Execution = PyEnv + PyFunc + args/kwargs + metadata.
- **DAG** — Composed of sub-PyFuncs with edges, cross-node orchestration.
- **Backend** — Node metadata: CPU, RAM, GPU, disk, network metrics (SSE streaming).
- **Network** — Peer mesh. Nodes swap roles (driver/executor/hybrid). Arrow IPC transport.

| Prefix | Description |
|--------|-------------|
| `/api/ping` | Fast health check (`{pong: true, node_id}`) |
| `/api/card` | Full node identity card (hardware, geo, counts) |
| `/api/v2/pyenv` | PyEnv CRUD (create, get, list, update, delete) |
| `/api/v2/pyfunc` | PyFunc CRUD |
| `/api/v2/pyfuncrun` | PyFuncRun CRUD + `/logs` SSE + `/result` Arrow IPC |
| `/api/v2/dag` | DAG CRUD + `/{id}/run` execution |
| `/api/v2/backend` | Node metrics snapshot + `/history` + `/stream` SSE |
| `/api/v2/network` | Self info + `/register` + `/peers` + `/role` + `/dispatch` + `/arrow` + `/ping` |
| `/api/v2/fs` | Filesystem CRUD (ls, stat, read, write, delete, move, mkdir, stream, upload, download zip, tree, du, search, grep). `/nodes` lists global-tree roots; every read/write takes `?node=` to proxy to a linked peer |
| `/api/v2/tabular` | LazyTabular inspect/preview/write — schema + metadata + bounded typed-row preview (JSON `/preview` or Arrow IPC `/preview.arrow`) + bounded in-place edit (`?node=` proxied). Drives the reusable `TabularModal` |
| `/api/v2/workbook` | ExcelFile (xlsx) surface — `/sheets` (dims), `/read` (windowed sheet → Arrow IPC), `/edit` (surgical cell/range edits preserving formulas + other sheets). `?node=` proxied |
| `/api/v2/analysis` | polars **lazy**-over-Arrow analytics (scan + projection/predicate/slice pushdown + streaming) — `/aggregate`, `/describe`, `/finance`, `/series` (adaptive downsample + x-zoom), `/ohlc` (candlesticks). All take `filters` (predicate pushdown). `/export` applies a `Transform` (filters + casts incl. timezone→UTC + projection) and downloads in any media type (csv/parquet/json/ndjson/arrow/xlsx). `?node=` proxied. Drives the `TabularModal` Analyze panel (pivot/series/candles + collapsible filters&casts + Download-as) and `Chart` (bar/line/area/candle, MA overlay, volume panel). Analyze fetches are client-cached |
| `/api/v2/saga` | **Saga** distributed catalog — `catalog`/`schema`/`table` CRUD (column + statistics metadata, `/refresh`, `/discover` folder scan, per-table `/log` op-history) + SQL editor: `/sql` (JSON grid, node-failover), `/sql.arrow` (Arrow IPC stream, tmp disk-spill when heavy), `/sql.stage` (run where data lives, write Arrow to a staging NodePath), `/explain`. Replication: `/replicate` (metadata\|data → peer) + `/import` (receive). Parses via `plan.sql_parser` (default postgres), executes via `plan.execute` over registered tables + node-home-rooted file URLs with pushdown. Managed store in `~/.saga/{node_id}` (off the network fs). `?node=` proxied; compute follows the data |
| `/api/v2/user` | User identity (`/me`, list, register from peers) |
| `/api/v2/messenger` | Chat channels + messages + SSE streaming |
| `/api/v2/replicate` | Export/import/push/pull node assets between nodes |

## Frontend Routes (nextjs/)

| Route | Description |
|-------|-------------|
| `/` | 3D globe welcome with node card |
| `/dashboard` | Cluster dashboard — Quick Actions, function/DAG mgmt |
| `/nodes` | Cluster overview — KPI aggregation + node grid + sidebar |
| `/nodes/[id]` | Per-node detail — resources, assets, replicated items |
| `/dags` | DAG builder — chain functions across nodes |
| `/chat` | Real-time messenger — channels, messages, SSE live updates |
| `/files` | Filesystem browser — lazy directory listing, file preview |
| `/saga` | Distributed catalog — lazy catalog/schema/table tree, per-table detail (columns + stats + op-history, data preview, replicate, refresh) + SQL editor (Run/Explain, dialect + default catalog/schema, results grid, plan view, node selector). Files page can register a file in Saga; Excel task pane has a Saga tab |
| `/metrics` | Per-function metrics + recent runs |
| `/topology` | Neural Mesh — layered brain network visualization |

## Python Decorator Framework

```python
from yggdrasil.node.fn import function, dag

@function
def process(data: list) -> dict:
    import pandas as pd
    return {"count": len(data)}

# Call like a normal function — returns FunctionRun (Future-like)
run = process([1, 2, 3])
result = run.wait()          # block until done
result = run.wait(wait=10)   # timeout after 10s (uses WaitingConfig)

# Target a specific environment or remote node
run = process.with_env("ml-env")([1, 2, 3])
run = process.on("http://node-2:8100")([1, 2, 3])

# DAG chaining with >> operator
pipeline = dag("etl", extract >> transform >> load)
run = pipeline()
```

The `@function` decorator infers: name, source code, dependencies (AST), python version. `FunctionRun` integrates with `WaitingConfig`, `State` enum, and polls the node API.

## CLI

`ygg node start` — background daemon (auto-installs boot service)
`ygg node stop` — stop the daemon
`ygg node create <name>` — create named node at `~/.node/<name>/`
`ygg node serve` — foreground server + frontend
`ygg node status` — show pid, port, boot service state
`ygg node run <func>` — call a @remote function
`ygg databricks` — Databricks management CLI

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `YGG_NODE_PORT` | 8100 | Node port |
| `YGG_NODE_FRONT_PORT` | 3000 | Frontend port |
| `BOT_API_URL` | `http://127.0.0.1:8100` | Frontend → node proxy |
| `YGG_NODE_TMP_TTL` | 86400 | Lifetime baked into a `tmp/` scratch name before the janitor reclaims it |
| `YGG_NODE_STG_TTL` | 604800 | Lifetime for `stg/` staging entries (persistent staging) |
| `YGG_NODE_SAGA_HOME` | `~/.saga/{node_id}` | Managed Saga store (metadata + replicas), off the network fs |
| `YGG_NODE_SAGA_DIALECT` | postgres | Default SQL dialect for the Saga editor |
