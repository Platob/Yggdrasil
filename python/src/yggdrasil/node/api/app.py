from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from ..config import Settings, get_settings
from ..exceptions import register_exception_handlers
from .routers import (
    backend_router,
    card_router,
    dag_router,
    fs_router,
    messenger_router,
    network_router,
    pyenv_router,
    pyfunc_router,
    pyfuncrun_router,
    replicate_router,
    user_router,
)
from .services.audit import AuditLog
from .services.backend import BackendService
from .services.dag import DAGService
from .services.fs import FsService
from .services.network import NetworkService
from .services.pyenv import PyEnvService
from .services.pyfunc import PyFuncService
from .services.pyfuncrun import PyFuncRunService
from .services.messenger import MessengerService as V2MessengerService
from .services.replicate import ReplicateService
from .services.user import UserService


def create_api(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=f"{settings.app_name}-v2",
        version=settings.app_version,
        docs_url="/v2/docs",
        redoc_url="/v2/redoc",
        openapi_url="/v2/openapi.json",
    )

    app.state.settings = settings

    # -- Services -----------------------------------------------------------
    audit = AuditLog()
    app.state.audit = audit

    fs = FsService(settings)
    app.state.fs_service = fs

    pyenv = PyEnvService(settings, audit=audit)
    pyfunc = PyFuncService(settings, audit=audit)
    pyfuncrun = PyFuncRunService(settings, pyenv, pyfunc)
    dag = DAGService(settings, pyfuncrun)
    backend = BackendService(settings)
    backend.bind_run_counters(
        lambda: pyfuncrun.active_count,
        lambda: pyfuncrun.total_count,
    )
    network = NetworkService(settings, backend)
    replicate = ReplicateService(settings, pyenv, pyfunc, dag)
    user_svc = UserService(settings)
    v2_messenger = V2MessengerService(settings)

    app.state.pyenv_service = pyenv
    app.state.pyfunc_service = pyfunc
    app.state.pyfuncrun_service = pyfuncrun
    app.state.dag_service = dag
    app.state.backend_service = backend
    app.state.network_service = network
    app.state.replicate_service = replicate
    app.state.user_service = user_svc
    app.state.v2_messenger_service = v2_messenger

    # -- Middleware ----------------------------------------------------------

    @app.middleware("http")
    async def local_only_middleware(request: Request, call_next):
        if settings.allow_remote:
            return await call_next(request)
        client_host = request.client.host if request.client else None
        if client_host and client_host not in settings.local_clients:
            return JSONResponse(
                status_code=403,
                content={"detail": "Remote access disabled."},
            )
        return await call_next(request)

    register_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

    # -- Ping (fastest possible, no middleware deps) -------------------------
    _ping_start = time.monotonic()
    _ping_node_id = settings.node_id

    @app.get(f"{settings.api_prefix}/ping")
    async def ping():
        return {"pong": True, "node_id": _ping_node_id, "uptime": round(time.monotonic() - _ping_start, 1)}

    # -- Routers ------------------------------------------------------------
    prefix = f"{settings.api_prefix}/v2"
    app.include_router(card_router, prefix=f"{settings.api_prefix}/card")
    app.include_router(pyenv_router, prefix=f"{prefix}/pyenv")
    app.include_router(pyfunc_router, prefix=f"{prefix}/pyfunc")
    app.include_router(pyfuncrun_router, prefix=f"{prefix}/pyfuncrun")
    app.include_router(dag_router, prefix=f"{prefix}/dag")
    app.include_router(backend_router, prefix=f"{prefix}/backend")
    app.include_router(network_router, prefix=f"{prefix}/network")
    app.include_router(replicate_router, prefix=f"{prefix}/replicate")
    app.include_router(fs_router, prefix=f"{prefix}/fs")
    app.include_router(user_router, prefix=f"{prefix}/user")
    app.include_router(messenger_router, prefix=f"{prefix}/messenger")

    @app.get(f"{prefix}/audit")
    async def get_audit(limit: int = 100):
        return {"entries": audit.recent(limit=limit)}

    return app


api_app = create_api()
