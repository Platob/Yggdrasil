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
    audit = AuditLog(settings)
    app.state.audit = audit

    fs = FsService(settings)
    app.state.fs_service = fs

    pyenv = PyEnvService(settings, audit=audit)
    pyfunc = PyFuncService(settings, audit=audit)
    pyfuncrun = PyFuncRunService(settings, pyenv, pyfunc)
    backend = BackendService(settings)
    backend.bind_run_counters(
        lambda: pyfuncrun.active_count,
        lambda: pyfuncrun.total_count,
    )
    network = NetworkService(settings, backend)
    dag = DAGService(settings, pyfuncrun, network_service=network, backend_service=backend)
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
        allow_origin_regex=".*",
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

    # -- Aggregate cluster stats (one call instead of N) --------------------
    prefix = f"{settings.api_prefix}/v2"

    @app.get(f"{prefix}/stats")
    async def get_stats(request: Request):
        """Aggregate cluster-wide statistics in one call."""
        state = request.app.state
        snap = state.backend_service.snapshot()
        envs = await state.pyenv_service.list()
        funcs = await state.pyfunc_service.list()
        dags = await state.dag_service.list()
        peers = await state.network_service.get_peers()
        mem_pct = (
            round(snap.memory_used_mb / snap.memory_total_mb * 100, 1)
            if snap.memory_total_mb else 0
        )
        return {
            "node_id": settings.node_id,
            "uptime": snap.uptime_seconds,
            "cpu_percent": snap.cpu_percent,
            "memory_percent": mem_pct,
            "active_runs": state.pyfuncrun_service.active_count,
            "total_runs": state.pyfuncrun_service.total_count,
            "env_count": len(envs.envs),
            "func_count": len(funcs.funcs),
            "dag_count": len(dags.dags),
            "scheduled_dags": sum(1 for d in dags.dags if d.schedule_active),
            "peer_count": len(peers.peers),
            "gpu_count": len(snap.gpus),
        }

    @app.get(f"{prefix}/metrics")
    async def get_metrics(request: Request):
        """Detailed metrics: top functions by runs, top by duration, recent activity."""
        state = request.app.state
        funcs = await state.pyfunc_service.list()
        runs = await state.pyfuncrun_service.list()

        top_by_runs = sorted(funcs.funcs, key=lambda f: f.run_count, reverse=True)[:5]
        top_by_duration = sorted(
            [f for f in funcs.funcs if f.avg_duration_ms > 0],
            key=lambda f: f.avg_duration_ms,
            reverse=True,
        )[:5]
        success_rate_funcs = sorted(
            [f for f in funcs.funcs if (f.success_count + f.failure_count) > 0],
            key=lambda f: f.success_count / (f.success_count + f.failure_count),
            reverse=True,
        )[:5]

        recent_runs = sorted(runs.runs, key=lambda r: r.started_at or "", reverse=True)[:10]

        return {
            "node_id": settings.node_id,
            "top_by_runs": [{"id": f.id, "name": f.name, "runs": f.run_count} for f in top_by_runs],
            "top_by_duration": [{"id": f.id, "name": f.name, "avg_ms": f.avg_duration_ms} for f in top_by_duration],
            "success_rate": [
                {"id": f.id, "name": f.name, "rate": round(f.success_count / (f.success_count + f.failure_count) * 100, 1)}
                for f in success_rate_funcs
            ],
            "recent_runs": [
                {"id": r.id, "func_id": r.func_id, "status": r.status, "duration": r.duration, "started_at": r.started_at}
                for r in recent_runs
            ],
        }

    @app.get(f"{prefix}/topology")
    async def get_topology(request: Request):
        """Full cluster view: this node + all peers with their cards."""
        state = request.app.state
        snap = state.backend_service.snapshot()
        peers_resp = await state.network_service.get_peers()
        self_node = {
            "node_id": settings.node_id,
            "host": settings.host,
            "port": settings.port,
            "role": str(state.backend_service.role),
            "cpu_percent": snap.cpu_percent,
            "memory_percent": (round(snap.memory_used_mb / snap.memory_total_mb * 100, 1) if snap.memory_total_mb else 0),
            "active_runs": state.pyfuncrun_service.active_count,
            "gpu_count": len(snap.gpus),
            "self": True,
        }
        peer_nodes = [
            {
                "node_id": p.node_id, "host": p.host, "port": p.port,
                "role": str(p.role), "cpu_percent": p.cpu_percent,
                "memory_percent": p.memory_percent, "active_runs": p.active_runs,
                "gpu_count": p.gpu_count, "self": False,
            }
            for p in peers_resp.peers
        ]
        return {
            "nodes": [self_node] + peer_nodes,
            "total_cpu_percent": (self_node["cpu_percent"] + sum(p["cpu_percent"] for p in peer_nodes)) / (len(peer_nodes) + 1),
            "total_active_runs": self_node["active_runs"] + sum(p["active_runs"] for p in peer_nodes),
            "total_gpus": self_node["gpu_count"] + sum(p["gpu_count"] for p in peer_nodes),
        }

    @app.get(f"{prefix}/health")
    async def health_check(request: Request):
        """Health check with subsystem status."""
        state = request.app.state
        checks: dict = {}

        try:
            snap = state.backend_service.snapshot()
            checks["backend"] = {"status": "ok", "cpu": snap.cpu_percent}
        except Exception as e:
            checks["backend"] = {"status": "error", "error": str(e)}

        try:
            envs = await state.pyenv_service.list()
            checks["pyenv"] = {"status": "ok", "count": len(envs.envs)}
        except Exception as e:
            checks["pyenv"] = {"status": "error", "error": str(e)}

        try:
            funcs = await state.pyfunc_service.list()
            checks["pyfunc"] = {"status": "ok", "count": len(funcs.funcs)}
        except Exception as e:
            checks["pyfunc"] = {"status": "error", "error": str(e)}

        try:
            peers = await state.network_service.get_peers()
            checks["network"] = {"status": "ok", "peers": len(peers.peers)}
        except Exception as e:
            checks["network"] = {"status": "error", "error": str(e)}

        all_ok = all(c["status"] == "ok" for c in checks.values())
        return {
            "status": "healthy" if all_ok else "degraded",
            "node_id": settings.node_id,
            "checks": checks,
        }

    # -- Routers ------------------------------------------------------------
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
