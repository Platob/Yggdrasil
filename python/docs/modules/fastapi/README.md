# yggdrasil.fastapi

FastAPI service that powers the Power Query / Excel connector. Exposes a REST API for Python execution, system introspection, and Databricks-backed Excel workbook generation.

**Install:** `pip install "ygg[api]"` (pulls FastAPI + uvicorn + pydantic).

---

## One-liner

```bash
ygg-api                             # starts on http://0.0.0.0:8000
python -m yggdrasil.fastapi.main    # alternative
```

---

## 1) Quick start

```python
from yggdrasil.fastapi import create_app

app = create_app()   # returns the FastAPI instance

# Or use the pre-built singleton (same instance)
from yggdrasil.fastapi import app
```

Run with uvicorn:

```bash
uvicorn yggdrasil.fastapi.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 2) Configuration

Settings are read from environment variables (prefix `YGG_FASTAPI_`):

| Variable | Default | Description |
|---|---|---|
| `YGG_FASTAPI_ALLOW_REMOTE` | `false` | Allow connections from non-localhost IPs |
| `YGG_FASTAPI_MAX_BODY_SIZE` | `10 * 1024 * 1024` | Max request body in bytes (10 MB) |
| `YGG_FASTAPI_GZIP_MIN_SIZE` | `1024` | Min response size to gzip (1 KB) |

```python
import os
os.environ["YGG_FASTAPI_ALLOW_REMOTE"] = "true"   # allow external access

from yggdrasil.fastapi import create_app
app = create_app()
```

---

## 3) Routers

The app mounts three routers:

### `system_router` — health & introspection

```
GET  /system/health          → {"status": "ok", "version": "..."}
GET  /system/info            → Python version, platform, installed packages
```

```python
import httpx

resp = httpx.get("http://localhost:8000/system/health")
print(resp.json())   # {"status": "ok", "version": "0.7.x"}
```

### `python_router` — execute Python expressions

```
POST /python/eval            → evaluate a Python expression, return result as JSON
POST /python/exec            → execute a Python statement block (no return value)
```

```python
import httpx

resp = httpx.post("http://localhost:8000/python/eval", json={
    "code": "2 + 2"
})
print(resp.json())   # {"result": 4}

resp = httpx.post("http://localhost:8000/python/exec", json={
    "code": "import yggdrasil; print(yggdrasil.__version__)"
})
```

> **Security note:** The Python execution endpoint is local-only by default (`YGG_FASTAPI_ALLOW_REMOTE=false`). Do not expose it over a public network.

### `excel_router` — Databricks → Excel workbook generation

```
POST /excel/workbook         → generate an Excel workbook from a SQL query / table
GET  /excel/workbook/{id}    → download a previously generated workbook
```

The Excel router is what the Power Query connector in `powerquery/` calls when fetching data from Databricks for an Excel / Power BI model.

```python
import httpx

resp = httpx.post("http://localhost:8000/excel/workbook", json={
    "host":   "https://<workspace>",
    "token":  "<token>",
    "query":  "SELECT * FROM main.default.orders LIMIT 1000",
})
workbook_id = resp.json()["id"]

xlsx_bytes = httpx.get(f"http://localhost:8000/excel/workbook/{workbook_id}").content
with open("orders.xlsx", "wb") as f:
    f.write(xlsx_bytes)
```

---

## 4) Extend with custom routers

```python
from fastapi import APIRouter
from yggdrasil.fastapi import create_app

my_router = APIRouter(prefix="/custom", tags=["custom"])

@my_router.get("/ping")
def ping() -> dict:
    return {"pong": True}

app = create_app()
app.include_router(my_router)
```

---

## 5) Power Query connector integration

The `powerquery/` directory at the repo root contains an Excel connector (`.mez`) and build scripts (`build.ps1`, `install.ps1`). The connector calls the FastAPI service at `http://localhost:8000` to execute queries and return Arrow-formatted results.

Setup:

```bash
# 1. Start the API service
ygg-api

# 2. In Excel / Power BI: Data → Get Data → From Other Sources → Yggdrasil
#    (after installing the .mez connector from powerquery/)
```

---

## 6) Production deployment

For production use behind a reverse proxy:

```bash
# Behind nginx (nginx handles TLS termination)
uvicorn yggdrasil.fastapi.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 4

# With environment config
YGG_FASTAPI_ALLOW_REMOTE=true uvicorn yggdrasil.fastapi.main:app \
    --host 0.0.0.0 --port 8000
```

Docker:

```dockerfile
FROM python:3.12-slim
RUN pip install "ygg[api,databricks]"
EXPOSE 8000
CMD ["ygg-api"]
```
