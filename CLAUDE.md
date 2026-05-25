# Yggdrasil

Distributed node framework — Python backend, Next.js frontend, Nordic dark UI.

## Principles

1. **Exceptions** — derive from `YGGException`. API errors use `yggdrasil.exceptions.api` (`NotFoundError`, `ConflictError`, etc). No ad-hoc exception classes.
2. **Services own logic** — routers are thin (validate → call service → return). Business rules live in `services/`.
3. **Schemas are contracts** — all request/response types in `schemas/`. Use `StrictModel` (extra="forbid").
4. **POC mode** — ship fast, iterate. No legacy compat shims.

## Layout

```
python/src/yggdrasil/
  node/                 Node server (FastAPI, default :8100)
    routers/            Thin HTTP handlers
    services/           Business logic
    schemas/            Pydantic models (StrictModel)
    geo.py              IP geolocation (lat/lon)
  cli/                  ygg CLI (argparse)
  exceptions/
    api.py              APIError → NotFoundError, ConflictError, ...
    http.py             HTTP client errors (with Response object)
  databricks/           Databricks SDK integrations
next/ygg/               Frontend (React 19, Next.js 16, Tailwind v4)
  src/app/node/         Node dashboard pages
  src/app/node/[id]/    Single-node detail view
  src/app/msg/          Messaging
  src/lib/api.ts        API client (node.* + api.* namespaces)
```

## Node API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/hello` | GET | Self node info (NodeInfo with lat/lon) |
| `/api/hello` | POST | Register peer, returns peers list |
| `/api/hello/peers` | GET | All known peers |
| `/api/hello/discover` | POST | Discover peers from URL list |
| `/api/python` | POST | Execute Python code |
| `/api/cmd` | POST | Execute shell command |
| `/api/messenger` | POST | Send message |
| `/api/messenger/channels` | GET | List channels |
| `/api/call/registry` | GET | List @remote functions |
| `/api/call/{func}` | POST | Call a remote function |

## Frontend Routes

| Route | Description |
|-------|-------------|
| `/` | Welcome — interactive 3D globe |
| `/node` | Network overview — all nodes, closest neighbors |
| `/node/{id}` | Node detail — resource graphs, processes, system info |
| `/node/network` | 3D network map |
| `/node/execute` | Python/shell execution |
| `/msg` | Real-time messaging |

## Exceptions

```python
from yggdrasil.exceptions.api import NotFoundError, register_api_exception_handlers
raise NotFoundError(f"Channel {name!r} not found")
register_api_exception_handlers(app)
```

## CLI

`ygg node serve` — node + frontend | `ygg node front` — frontend only
`ygg node serve --no-front` — node only | `ygg node status/stop` — manage daemon

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `YGG_NODE_PORT` | 8100 | Node port |
| `YGG_NODE_FRONT_PORT` | 3000 | Frontend port |
| `NODE_API_URL` | `http://127.0.0.1:8100` | Frontend → node proxy |
