# Yggdrasil

Distributed node framework — Python backend, Next.js frontend, Nordic dark UI.

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
    routers/            HTTP handlers (function, environment, run, monitor, ...)
    services/           Business logic + state
    schemas/            Pydantic models
    geo.py              IP geolocation
    ids.py              Int64 ID generation (xxhash)
    fn.py               @function decorator framework
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

`ygg node serve` — node + frontend | `ygg node front` — frontend only
`ygg node serve --no-front` — node only | `ygg node status/stop` — daemon
`ygg node install/uninstall` — boot service (systemd/launchd)
`ygg node run` — call a @remote function | `ygg node chat` — YGGCHAT terminal
`ygg databricks` — Databricks management CLI

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `YGG_NODE_PORT` | 8100 | Node port |
| `YGG_NODE_FRONT_PORT` | 3000 | Frontend port |
| `BOT_API_URL` | `http://127.0.0.1:8100` | Frontend → node proxy |
