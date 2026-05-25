# yggdrasil.fastapi

FastAPI service that backs the Power Query / Excel connector. Exposes Python environment management, code execution, and Databricks SQL as REST endpoints.

**Optional dependency:** `pip install "ygg[api]"` (fastapi + uvicorn + pydantic >= 2.10).

## Quick start

```bash
# Start the server (default: 127.0.0.1:8000, localhost-only)
python -m yggdrasil.fastapi.main

# Or via the installed CLI entry point
ygg-api
```

The interactive API docs are at `http://localhost:8000/docs`.

---

## Configuration

`Settings` is a frozen dataclass. Override via environment variables:

| Env var | Default | Description |
| --- | --- | --- |
| `YGG_FASTAPI_HOST` | `127.0.0.1` | Bind host |
| `YGG_FASTAPI_PORT` | `8000` | Bind port |
| `YGG_FASTAPI_ALLOW_REMOTE` | `0` | Allow non-localhost clients when `1` |
| `YGG_FASTAPI_API_PREFIX` | `/api` | URL prefix for all API routes |
| `YGG_FASTAPI_ENV_HOME` | `~/.local/yggdrasil/python/envs` | Root directory for managed Python venvs |
| `YGG_FASTAPI_CACHE_HOME` | `~/.cache/yggdrasil` | Root directory for caches |

```bash
# Allow remote clients and use a custom port
YGG_FASTAPI_ALLOW_REMOTE=1 YGG_FASTAPI_PORT=9000 ygg-api
```

---

## Create and mount the app programmatically

```python
from yggdrasil.fastapi import create_app, app
from yggdrasil.fastapi.config import Settings

# Default singleton
app   # FastAPI instance with all routers registered

# Custom settings
custom_app = create_app(Settings(port=9000, allow_remote=True))
```

Embed in an existing FastAPI application:

```python
from fastapi import FastAPI
from yggdrasil.fastapi import create_app
from yggdrasil.fastapi.config import Settings

parent = FastAPI()
ygg   = create_app(Settings(api_prefix="/ygg"))

parent.mount("/ygg", ygg)
```

---

## API surface

### System endpoints (`/api/system/`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/system/info` | Server info (version, Python, platform) |
| `GET` | `/api/system/healthz` | Health check (`{"status": "ok"}`) |

```bash
curl http://localhost:8000/api/system/healthz
# {"status": "ok"}

curl http://localhost:8000/api/system/info
```

### Python / environment endpoints (`/api/python/`)

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/python/envs/current` | Current Python environment |
| `GET` | `/api/python/envs` | List all managed environments |
| `POST` | `/api/python/envs/resolve` | Resolve an environment by ref |
| `POST` | `/api/python/envs` | Create a new virtual environment |
| `DELETE` | `/api/python/envs` | Delete a virtual environment |
| `GET` | `/api/python/requirements` | Installed packages in an environment |
| `POST` | `/api/python/packages/install` | Install packages into an environment |
| `POST` | `/api/python/packages/uninstall` | Uninstall packages |
| `POST` | `/api/python/execute` | Execute arbitrary Python code |

### Excel / Databricks endpoints (`/api/python/excel/`)

These power the Power Query Excel connector. They accept gzip-compressed request bodies.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/python/excel/execute` | Execute a Power Query / Python expression |
| `POST` | `/api/python/excel/databricks/query` | Execute a Databricks SQL query, return Arrow/JSON |

---

## Execute Python code

```python
import httpx

resp = httpx.post("http://localhost:8000/api/python/execute", json={
    "code": "result = [i**2 for i in range(5)]",
    "env": "current",
})
print(resp.json())
```

## Create a managed venv

```python
import httpx

resp = httpx.post("http://localhost:8000/api/python/envs", json={
    "name": "my-pipeline-env",
    "python": "3.12",
    "packages": ["polars", "pyarrow"],
})
print(resp.json())
```

## List installed packages

```python
import httpx

resp = httpx.get("http://localhost:8000/api/python/requirements",
                 params={"identifier": "current"})
print(resp.json())
```

---

## Running in production (behind a reverse proxy)

The service binds to `127.0.0.1` by default, which restricts it to localhost. To expose it securely behind nginx or a cloud load balancer:

```bash
# Bind to all interfaces; rely on the reverse proxy for TLS + auth
YGG_FASTAPI_HOST=0.0.0.0 YGG_FASTAPI_ALLOW_REMOTE=1 ygg-api
```

---

## Request compression

The Excel router accepts gzip-compressed request bodies. Clients that compress payloads must set `Content-Encoding: gzip`. The GZip response middleware compresses responses ≥ 1 KB automatically.

---

## Adding custom routes

```python
from fastapi import FastAPI
from yggdrasil.fastapi import create_app

app = create_app()

@app.get("/api/custom/ping")
async def ping():
    return {"pong": True}
```
