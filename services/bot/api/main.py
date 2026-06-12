"""YGG Bot — FastAPI trading service.

Entry point::

    uvicorn services.bot.api.main:app --reload --port 8000

Or from this directory::

    python -m uvicorn main:app --reload
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routers import health, market, portfolio, signals, ai, ws

# ---------------------------------------------------------------------------
# lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: nothing blocking at startup
    yield


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="YGG Bot",
    description="Trading analytics + AI signals service powered by Yggdrasil",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(market.router)
app.include_router(portfolio.router)
app.include_router(signals.router)
app.include_router(ai.router)
app.include_router(ws.router)
