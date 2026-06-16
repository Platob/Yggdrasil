"""v2 API hot endpoints: ping, stats, backend, health, audit, pyfunc, pyenv.

State (start time, request counter, audit log, function registry) lives on
``app.state``, populated by :func:`yggdrasil.node.api.app.create_api`.
"""
from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request

router = APIRouter()


def _mem_mb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        # stdlib fallback: ru_maxrss is kB on Linux, bytes on macOS.
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 1024 if sys.platform != "darwin" else rss / (1024 * 1024)


@router.get("/api/ping")
async def ping() -> dict:
    return {"pong": True, "version": "2.0"}


@router.get("/api/v2/stats")
async def stats(request: Request) -> dict:
    state = request.app.state
    return {
        "uptime": time.time() - state.started,
        "requests": state.requests,
        "memory_mb": round(_mem_mb(), 2),
    }


@router.get("/api/v2/backend")
async def backend(request: Request) -> dict:
    state = request.app.state
    return {
        "node_id": state.settings.node_id,
        "version": "2.0",
        "engines": ["polars", "pyarrow"],
    }


@router.get("/api/v2/backend/summary")
async def backend_summary(request: Request) -> dict:
    return {"node_id": request.app.state.settings.node_id, "status": "ok"}


@router.get("/api/v2/health")
async def health(request: Request) -> dict:
    return {"status": "ok", "uptime": time.time() - request.app.state.started}


@router.get("/api/v2/audit")
async def audit(request: Request, limit: int = 20) -> dict:
    return {"entries": request.app.state.audit.recent(limit)}


@router.get("/api/v2/pyfunc")
async def pyfunc(request: Request) -> dict:
    funcs = await request.app.state.function.list()
    return {"functions": [f.model_dump(mode="json") for f in funcs]}


@router.get("/api/v2/pyenv")
async def pyenv(request: Request) -> dict:
    return {"envs": list(request.app.state.envs)}


# ---------------------------------------------------------------------------
# FX rates — trading-oriented endpoint backed by yggdrasil.fxrate
# ---------------------------------------------------------------------------

@router.get("/api/v2/fxrate")
async def fxrate(pair: list[str] | None = None) -> dict:
    """Fetch latest FX quotes for the requested pairs.

    ``?pair=EUR/USD&pair=GBP/USD`` — defaults to the eight major pairs.
    Uses the live backend chain (Frankfurter → Fawaz → ER-API) with
    automatic fallback and result assembled into a plain list of quote dicts.
    """
    from yggdrasil.fxrate import FxRate
    from yggdrasil.fxrate.backends import Frankfurter, Fawaz, ErApi

    default_pairs = [
        ("EUR", "USD"), ("EUR", "GBP"), ("EUR", "JPY"),
        ("USD", "JPY"), ("GBP", "USD"), ("USD", "CHF"),
        ("AUD", "USD"), ("USD", "CAD"),
    ]
    if pair:
        parsed = []
        for p in pair:
            parts = p.replace("-", "/").split("/")
            if len(parts) == 2:
                parsed.append((parts[0].upper(), parts[1].upper()))
        pairs = parsed or default_pairs
    else:
        pairs = default_pairs

    fx = FxRate(backends=[Frankfurter(), Fawaz(), ErApi()])
    try:
        df = fx.latest(pairs=pairs)
        quotes = df.to_dicts()
        # Convert datetime objects to ISO strings for JSON serialisation
        for q in quotes:
            for k in ("from_timestamp", "to_timestamp"):
                if hasattr(q.get(k), "isoformat"):
                    q[k] = q[k].isoformat()
        return {"quotes": quotes, "pairs": len(quotes), "source": "live"}
    except Exception as exc:
        return {"quotes": [], "pairs": 0, "source": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Analysis — polars lazy scan + OHLC + forecast (trading-focused)
# ---------------------------------------------------------------------------

@router.post("/api/v2/analysis/aggregate")
async def analysis_aggregate(request: Request, body: dict) -> dict:
    from ..api.schemas.analysis import AggregateRequest
    from ..api.services.analysis import AnalysisService
    from ..api.services.fs import FsService

    settings = request.app.state.settings
    svc = AnalysisService(settings, fs=FsService(settings))
    req = AggregateRequest(**body)
    result = await svc.aggregate(req)
    return result.model_dump(mode="json")


@router.post("/api/v2/analysis/ohlc")
async def analysis_ohlc(request: Request, body: dict) -> dict:
    from ..api.schemas.analysis import OhlcRequest
    from ..api.services.analysis import AnalysisService
    from ..api.services.fs import FsService

    settings = request.app.state.settings
    svc = AnalysisService(settings, fs=FsService(settings))
    req = OhlcRequest(**body)
    result = await svc.ohlc(req)
    return result.model_dump(mode="json")


@router.post("/api/v2/analysis/series")
async def analysis_series(request: Request, body: dict) -> dict:
    from ..api.schemas.analysis import SeriesRequest
    from ..api.services.analysis import AnalysisService
    from ..api.services.fs import FsService

    settings = request.app.state.settings
    svc = AnalysisService(settings, fs=FsService(settings))
    req = SeriesRequest(**body)
    result = await svc.series(req)
    return result.model_dump(mode="json")


@router.post("/api/v2/analysis/pivot")
async def analysis_pivot(request: Request, body: dict) -> dict:
    from ..api.schemas.analysis import PivotRequest
    from ..api.services.analysis import AnalysisService
    from ..api.services.fs import FsService

    settings = request.app.state.settings
    svc = AnalysisService(settings, fs=FsService(settings))
    req = PivotRequest(**body)
    result = await svc.pivot(req)
    return result.model_dump(mode="json")


@router.post("/api/v2/analysis/forecast")
async def analysis_forecast(request: Request, body: dict) -> dict:
    from ..api.schemas.analysis import ForecastRequest
    from ..api.services.analysis import AnalysisService
    from ..api.services.fs import FsService

    settings = request.app.state.settings
    svc = AnalysisService(settings, fs=FsService(settings))
    req = ForecastRequest(**body)
    result = await svc.forecast(req)
    return result.model_dump(mode="json")
