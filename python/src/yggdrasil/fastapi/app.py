"""FastAPI app factory.

The app is intentionally thin: a config bag, a :class:`TabularEngine`,
three routers (catalog / sources / data), and a local-only middleware
that 403s remote callers unless ``YGG_API_ALLOW_REMOTE`` is set. Every
heavy lift lives inside :mod:`yggdrasil.io.tabular` — the API just
exposes it.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from yggdrasil.io.tabular import SYSTEM_ENGINE, TabularEngine

from .config import Settings, get_settings
from .exceptions import register_exception_handlers
from .routers import catalog_router, data_router, sources_router


def create_app(
    settings: "Settings | None" = None,
    *,
    engine: "TabularEngine | None" = None,
) -> FastAPI:
    """Build a fresh :class:`FastAPI` instance.

    *settings* defaults to :func:`get_settings` (env-driven). *engine*
    defaults to the process-wide :data:`SYSTEM_ENGINE` so any
    registrations a test or launcher made before instantiating the
    app are immediately visible.
    """
    settings = settings or get_settings()
    engine = engine or SYSTEM_ENGINE

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
    )
    app.state.settings = settings
    app.state.engine = engine

    register_exception_handlers(app)

    @app.middleware("http")
    async def _local_only(request: Request, call_next):
        if settings.allow_remote:
            return await call_next(request)
        host = request.client.host if request.client else None
        if host and host not in settings.local_clients:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "Remote access is disabled. Bind locally or set "
                        "YGG_API_ALLOW_REMOTE=1 to allow non-local clients."
                    )
                },
            )
        return await call_next(request)

    @app.get("/health", tags=["meta"], response_class=PlainTextResponse)
    def _health() -> str:  # noqa: D401 — tiny health probe
        return "ok"

    prefix = settings.api_prefix or ""
    app.include_router(catalog_router, prefix=prefix)
    app.include_router(sources_router, prefix=prefix)
    app.include_router(data_router, prefix=prefix)

    return app


app = create_app()
