from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .exceptions import register_exception_handlers
from .routers import (
    call_router,
    cmd_router,
    dag_router,
    discovery_router,
    env_router,
    environment_router,
    filesystem_router,
    function_router,
    job_router,
    messenger_router,
    python_router,
    run_router,
)
from .routers.monitor import router as monitor_router
from .services import (
    CallService,
    CmdService,
    DagService,
    DiscoveryService,
    EnvService,
    EnvironmentService,
    FilesystemService,
    FunctionService,
    JobService,
    MessengerService,
    PythonExecService,
    RunService,
)
from .services.monitor import MonitorService


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
    app.state.env_service = EnvService(settings)
    app.state.cmd_service = CmdService(settings)
    app.state.python_service = PythonExecService(settings)
    app.state.job_service = JobService(settings)
    app.state.call_service = CallService(settings)
    app.state.messenger_service = MessengerService(settings)
    app.state.discovery_service = DiscoveryService(settings, messenger_service=app.state.messenger_service)
    app.state.monitor_service = MonitorService(settings)
    app.state.function_service = FunctionService(settings)
    app.state.environment_service = EnvironmentService(settings)
    app.state.run_service = RunService(
        settings,
        function_service=app.state.function_service,
        environment_service=app.state.environment_service,
    )
    app.state.dag_service = DagService(
        settings,
        function_service=app.state.function_service,
        environment_service=app.state.environment_service,
        run_service=app.state.run_service,
    )
    app.state.filesystem_service = FilesystemService(settings)

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
                        "YGG_NODE_ALLOW_REMOTE=1 to allow non-local clients."
                    )
                },
            )

        return await call_next(request)

    register_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

    prefix = settings.api_prefix
    app.include_router(env_router, prefix=f"{prefix}/env")
    app.include_router(cmd_router, prefix=f"{prefix}/cmd")
    app.include_router(python_router, prefix=f"{prefix}/python")
    app.include_router(job_router, prefix=f"{prefix}/job")
    app.include_router(call_router, prefix=f"{prefix}/call")
    app.include_router(messenger_router, prefix=f"{prefix}/messenger")
    app.include_router(discovery_router, prefix=f"{prefix}/hello")
    app.include_router(function_router, prefix=f"{prefix}/function")
    app.include_router(environment_router, prefix=f"{prefix}/environment")
    app.include_router(run_router, prefix=f"{prefix}/run")
    app.include_router(monitor_router, prefix=f"{prefix}/monitor")
    app.include_router(dag_router, prefix=f"{prefix}/dag")
    app.include_router(filesystem_router, prefix=f"{prefix}/fs")

    return app


app = create_app()
