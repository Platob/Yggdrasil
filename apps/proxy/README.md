# Proxy Server

A FastAPI-based reverse proxy with a typed backend API.

| Layer | Path | Description |
|-------|------|-------------|
| Backend | `/api/v1/*` | Your application's REST API |
| Frontend proxy | `/*` | Forwards everything else to `FRONTEND_UPSTREAM` |

## Quick start

```bash
cd apps/proxy

# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env – at minimum set FRONTEND_UPSTREAM

# 3. Run
uvicorn main:app --reload
```

API docs are available at `http://localhost:8000/api/docs` once running.

## Project layout

```
apps/proxy/
├── main.py                          # FastAPI app factory & entry point
├── requirements.txt
├── .env.example
└── app/
    ├── config.py                    # Pydantic settings (reads .env)
    ├── api/
    │   └── v1/
    │       ├── router.py            # Aggregates all v1 endpoint routers
    │       └── endpoints/
    │           └── health.py        # GET /api/v1/health
    └── proxy/
        └── frontend.py              # Catch-all reverse proxy to FRONTEND_UPSTREAM
```

## Adding new API endpoints

1. Create `app/api/v1/endpoints/<name>.py` with an `APIRouter`.
2. Import and include it in `app/api/v1/router.py`.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Listen port |
| `DEBUG` | `false` | Enable uvicorn `--reload` |
| `API_V1_PREFIX` | `/api/v1` | Prefix for all backend routes |
| `FRONTEND_UPSTREAM` | `http://localhost:5173` | URL to proxy frontend requests to |
| `FRONTEND_STRIP_PREFIX` | *(empty)* | Path prefix to strip before forwarding |
