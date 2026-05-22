# yggdrasil.fastapi

FastAPI service that powers the Power Query Excel connector. Exposes Python
evaluation, Databricks SQL/catalog navigation, and system-info endpoints
over a local HTTP API.

## Surface map

| Symbol | Use for |
|---|---|
| `app` | Pre-built FastAPI application instance |
| `create_app(settings=None)` | Factory — build a fresh app with custom settings |
| `ygg-api` CLI | Entry point: `python -m yggdrasil.fastapi.main` |

---

## 1) One-liners

```bash
# Start the API server (listens on 127.0.0.1:8000 by default)
python -m yggdrasil.fastapi.main

# Or via the installed CLI entry point
ygg-api
```

```python
# Embed in a test or another ASGI app
from yggdrasil.fastapi import app
```

---

## 2) Start the server

```bash
# Default (local-only, port 8000)
python -m yggdrasil.fastapi.main

# Custom port
YGG_FASTAPI_PORT=9000 python -m yggdrasil.fastapi.main

# Allow non-local clients (Power BI Desktop on another host)
YGG_FASTAPI_ALLOW_REMOTE=1 python -m yggdrasil.fastapi.main

# Custom Databricks workspace (used by the Excel service)
DATABRICKS_HOST=https://<workspace>.azuredatabricks.net \
DATABRICKS_TOKEN=<token> \
python -m yggdrasil.fastapi.main
```

---

## 3) Configuration via environment variables

| Variable | Default | Description |
|---|---|---|
| `YGG_FASTAPI_HOST` | `127.0.0.1` | Bind address |
| `YGG_FASTAPI_PORT` | `8000` | Port |
| `YGG_FASTAPI_ALLOW_REMOTE` | `0` | Allow non-local clients (set `1` to enable) |
| `YGG_FASTAPI_API_PREFIX` | `/api` | URL prefix for all routes |
| `DATABRICKS_HOST` | — | Workspace URL for the Databricks Excel service |
| `DATABRICKS_TOKEN` | — | PAT token for the Databricks Excel service |

---

## 4) Routers

### System (`/api/system/`)

Health-check and version endpoints, used by the Power Query connector to
test connectivity.

```bash
curl http://localhost:8000/api/system/health
# {"status": "ok", "version": "0.7.x"}

curl http://localhost:8000/api/system/version
```

### Python (`/api/python/`)

Evaluate arbitrary Python expressions and return the result as JSON or
Arrow IPC.

```bash
# Evaluate a Python expression
curl -X POST http://localhost:8000/api/python/eval \
  -H "Content-Type: application/json" \
  -d '{"code": "[i**2 for i in range(5)]"}'
# {"result": [0, 1, 4, 9, 16]}
```

### Excel / Databricks (`/api/python/excel/`)

Databricks SQL + catalog navigation for the Power Query connector.

```bash
# Execute SQL and return as Arrow IPC (consumed by Power Query)
curl -X POST "http://localhost:8000/api/python/excel/sql" \
  -H "Content-Type: application/json" \
  -d '{"statement": "SELECT * FROM main.sales.orders LIMIT 100"}'
```

---

## 5) Custom app factory

```python
from yggdrasil.fastapi import create_app
from yggdrasil.fastapi.config import Settings

settings = Settings(
    api_prefix="/v1",
    allow_remote=True,
    port=9000,
)
app = create_app(settings)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=9000)
```

---

## 6) Embed in an existing FastAPI app

```python
from fastapi import FastAPI
from yggdrasil.fastapi import create_app as ygg_app

parent = FastAPI(title="My App")
ygg    = ygg_app()

# Mount yggdrasil API under /ygg
parent.mount("/ygg", ygg)
```

---

## 7) Use in tests

```python
from fastapi.testclient import TestClient
from yggdrasil.fastapi import app

client = TestClient(app)

def test_health():
    resp = client.get("/api/system/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

def test_python_eval():
    resp = client.post("/api/python/eval", json={"code": "1 + 1"})
    assert resp.json()["result"] == 2
```
