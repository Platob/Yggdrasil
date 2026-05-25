# Yggdrasil

Distributed node framework — Python backend, Next.js frontend, Nordic dark UI.

## Principles

1. **Exceptions** — derive from `YGGException`. API errors use `yggdrasil.exceptions.api`. No ad-hoc exception classes.
2. **Services own logic** — routers are thin. Business rules + state live in `services/`.
3. **Schemas are contracts** — `StrictModel` (extra="forbid") for all request/response types.
4. **Streaming first** — use SSE for logs, metrics, long-running operations.
5. **POC mode** — ship fast, iterate. No legacy compat.

## Layout

```
python/src/yggdrasil/
  node/                 Node server (FastAPI, default :8100)
    routers/            HTTP handlers (function, environment, run, monitor, ...)
    services/           Business logic + state
    schemas/            Pydantic models
    geo.py              IP geolocation
  cli/                  ygg CLI
  exceptions/api.py     APIError hierarchy
  databricks/           Databricks SDK integrations
next/ygg/               Frontend (React 19, Next.js 16, Tailwind v4)
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
| `/api/python` | Direct Python code execution |
| `/api/cmd` | Shell command execution |
| `/api/job` | Job definitions + scheduled runs |
| `/api/messenger` | Real-time messaging channels |
| `/api/call` | @remote function registry + invocation |

## Frontend Routes

| Route | Description |
|-------|-------------|
| `/` | 3D globe welcome |
| `/node` | Network overview — all nodes, closest neighbors |
| `/node/{id}` | Node detail — resource graphs, processes |
| `/node/functions` | Functions CRUD — create, edit, run |
| `/node/functions/{id}` | Function detail — code, runs, SSE logs |
| `/node/environments` | Python venv management |
| `/node/network` | 3D network visualization |
| `/node/execute` | Direct code execution |
| `/msg` | Messaging channels |

## CLI

`ygg node serve` — node + frontend | `ygg node front` — frontend only
`ygg node serve --no-front` — node only | `ygg node status/stop` — daemon

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `YGG_NODE_PORT` | 8100 | Node port |
| `YGG_NODE_FRONT_PORT` | 3000 | Frontend port |
| `NODE_API_URL` | `http://127.0.0.1:8100` | Frontend → node proxy |
