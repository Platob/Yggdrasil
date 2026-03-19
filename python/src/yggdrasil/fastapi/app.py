from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .exceptions import register_exception_handlers
from .routers import excel_router, python_router, system_router
from .services import DatabricksExcelService, PythonService, SystemService


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
    )

    app.state.settings = settings
    app.state.system_service = SystemService(settings)
    app.state.python_service = PythonService(settings)
    app.state.databricks_excel_service = DatabricksExcelService(settings)

    @app.middleware("http")
    async def local_only_middleware(request: Request, call_next):
        if settings.allow_remote:
            return await call_next(request)

        client_host = request.client.host if request.client else None
        if client_host and client_host not in settings.local_clients:
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        "Remote access is disabled. Bind locally or set "
                        "YGG_FASTAPI_ALLOW_REMOTE=1 to allow non-local clients."
                    )
                },
            )

        return await call_next(request)

    register_exception_handlers(app)
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

    app.include_router(
        system_router,
        prefix=f"{settings.api_prefix}{settings.system_prefix}",
    )
    app.include_router(
        python_router,
        prefix=f"{settings.api_prefix}{settings.python_prefix}",
    )
    app.include_router(
        excel_router,
        prefix=f"{settings.api_prefix}{settings.python_prefix}{settings.excel_prefix}",
    )

    return app


app = create_app()
