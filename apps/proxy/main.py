"""
Proxy server entry point.

Run with:
    uvicorn main:app --reload
or:
    python main.py
"""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as api_v1_router
from app.config import get_settings
from app.proxy.frontend import router as frontend_router

settings = get_settings()

app = FastAPI(
    title="Proxy Server",
    description=(
        "Backend API served at /api/v1. "
        "All other requests are forwarded to the configured frontend upstream."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers – order matters: API routes first, catch-all proxy last.
# ---------------------------------------------------------------------------
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)
app.include_router(frontend_router)  # catch-all, must be last


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
