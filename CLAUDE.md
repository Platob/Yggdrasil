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
src/                    Frontend (React 19, Next.js 16, Tailwind v4)
  app/                  Next.js app router pages
  components/           Shared React components (globe, sidebar, logo)
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
| `/api/v2/pyenv` | PyEnv CRUD (create, get, list, update, delete) |
| `/api/v2/pyfunc` | PyFunc CRUD |
| `/api/v2/pyfuncrun` | PyFuncRun CRUD + `/logs` SSE + `/result` Arrow IPC |
| `/api/v2/dag` | DAG CRUD + `/{id}/run` execution |
| `/api/v2/backend` | Node metrics snapshot + `/history` + `/stream` SSE |
| `/api/v2/network` | Self info + `/register` + `/peers` + `/role` + `/dispatch` + `/arrow` |

## Frontend Routes

| Route | Description |
|-------|-------------|
| `/` | 3D globe welcome |
| `/bot` | Bot dashboard — node info, registry |
| `/bot/chat` | Messaging chat interface |
| `/bot/execute` | Direct Python/shell code execution |
| `/bot/network` | 3D network visualization |
| `/msg` | Messaging channels |

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
