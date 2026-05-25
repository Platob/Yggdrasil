# Yggdrasil — Project Instructions

## Principles

1. **One exception hierarchy** — all exceptions derive from `YGGException`. API server errors use `yggdrasil.exceptions.api.APIError` and its subclasses. Never define ad-hoc exception classes in feature modules.
2. **Owner maintains Python backends** — do not rewrite FastAPI or bot service logic without being asked. Frontend and CLI are fair game.
3. **Dark Nordic aesthetic** — the brand color is `#f26b3a` (coral/orange). Dark backgrounds, subtle glows, monospace for data.
4. **Prefer existing utilities** — check `yggdrasil.exceptions`, `yggdrasil.pyutils`, `yggdrasil.cli.style` before writing new helpers.

## Repository Layout

```
python/src/yggdrasil/     Python library + services
  bot/                    Bot server (FastAPI on port 8100)
  fastapi/                Standalone FastAPI service (port 8000)
  cli/                    CLI entry point: ygg
  exceptions/             Centralized exception hierarchy
    base.py               YGGException root
    http.py               HTTP client errors (4xx/5xx with Response)
    api.py                API server errors (detail + status_code)
  databricks/             Databricks integrations
next/ygg/                 Next.js frontend (React 19, Tailwind v4)
```

## Exception System

```
YGGException
├── APIError              API server errors (bot + fastapi)
│   ├── BadRequestError       400
│   ├── UnauthorizedError     401
│   ├── ForbiddenError        403
│   ├── NotFoundError         404
│   ├── ConflictError         409
│   ├── TimeoutError          408
│   └── UnprocessableError    422
├── HTTPError             HTTP client errors (with Response object)
│   ├── ClientError           4xx (BadRequest, NotFoundError, ...)
│   └── ServerError           5xx (InternalServerError, ...)
└── CastError             Arrow/Pandas type cast failures
```

**Raising in services:**
```python
from yggdrasil.exceptions.api import NotFoundError, ConflictError
raise NotFoundError(f"Channel {name!r} not found")
raise ConflictError(f"Channel {name!r} already exists")
```

**Registering handlers in app:**
```python
from yggdrasil.exceptions.api import register_api_exception_handlers
register_api_exception_handlers(app)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ygg bot serve` | Start bot + frontend (foreground) |
| `ygg bot serve --no-front` | Bot only, no frontend |
| `ygg bot front` | Frontend dev server only |
| `ygg bot run <func>` | Call a @remote function |
| `ygg bot chat` | Terminal chat client |
| `ygg bot status` | Show bot status |
| `ygg bot stop` | Stop background bot |
| `ygg genie` | Databricks Genie CLI |
| `ygg databricks` | Databricks management CLI |

## Frontend (next/ygg/)

- **Bot API proxy**: `/api/bot/*` → FastAPI at `BOT_API_URL` (default `http://127.0.0.1:8100`)
- **Client API**: `import { bot } from "@/lib/api"` for bot calls, `import { api }` for Next.js routes
- **GlobalSidebar** in root layout — present on every page, handles dark/light mode
- **Design tokens** in `globals.css` via CSS custom properties (dark default, `.light` override)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YGG_BOT_PORT` | `8100` | Bot server port |
| `YGG_BOT_HOST` | `0.0.0.0` | Bot bind host |
| `YGG_BOT_FRONT_PORT` | `3000` | Frontend dev server port |
| `YGG_BOT_FRONT_HOME` | `<repo>/next/ygg` | Frontend directory |
| `BOT_API_URL` | `http://127.0.0.1:8100` | Frontend → bot proxy target |
| `YGG_FASTAPI_PORT` | `8000` | Standalone FastAPI port |
