from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .exceptions import register_exception_handlers
from .api.routers import (
    backend_router as v2_backend_router,
    card_router as v2_card_router,
    dag_router as v2_dag_router,
    fs_router as v2_fs_router,
    messenger_router as v2_messenger_router,
    network_router as v2_network_router,
    pyenv_router as v2_pyenv_router,
    pyfunc_router as v2_pyfunc_router,
    pyfuncrun_router as v2_pyfuncrun_router,
    replicate_router as v2_replicate_router,
    user_router as v2_user_router,
)
from .api.services.audit import AuditLog
from .api.services.backend import BackendService
from .api.services.dag import DAGService as V2DagService
from .api.services.fs import FsService
from .api.services.network import NetworkService
from .api.services.pyenv import PyEnvService
from .api.services.pyfunc import PyFuncService
from .api.services.pyfuncrun import PyFuncRunService
from .api.services.messenger import MessengerService as V2MessengerService
from .api.services.replicate import ReplicateService
from .api.services.user import UserService
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
    monitor_router,
    python_router,
    run_router,
)
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
    MonitorService,
    PythonExecService,
    RunService,
)


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

    # -- v2 API services (PyEnv / PyFunc / PyFuncRun / Fs) -------------------
    audit = AuditLog(settings)
    app.state.audit = audit

    v2_fs = FsService(settings)
    pyenv = PyEnvService(settings, audit=audit)
    pyfunc = PyFuncService(settings, audit=audit)
    pyfuncrun = PyFuncRunService(settings, pyenv, pyfunc)
    backend = BackendService(settings)
    backend.bind_run_counters(
        lambda: pyfuncrun.active_count,
        lambda: pyfuncrun.total_count,
    )
    network = NetworkService(settings, backend)
    v2_dag = V2DagService(settings, pyfuncrun, network_service=network, backend_service=backend)
    replicate = ReplicateService(settings, pyenv, pyfunc, v2_dag)
    user_svc = UserService(settings)
    v2_messenger = V2MessengerService(settings)

    app.state.fs_service = v2_fs
    app.state.pyenv_service = pyenv
    app.state.pyfunc_service = pyfunc
    app.state.pyfuncrun_service = pyfuncrun
    app.state.v2_dag_service = v2_dag
    app.state.backend_service = backend
    app.state.network_service = network
    app.state.replicate_service = replicate
    app.state.user_service = user_svc
    app.state.v2_messenger_service = v2_messenger

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
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=5)

    _ping_start = time.monotonic()
    _ping_node_id = settings.node_id

    @app.get(f"{settings.api_prefix}/ping")
    async def ping():
        return {"pong": True, "node_id": _ping_node_id, "uptime": round(time.monotonic() - _ping_start, 1)}

    prefix = settings.api_prefix

    @app.get(f"{prefix}/v2/stats")
    async def get_stats(request: Request):
        """Aggregate cluster-wide statistics in one call."""
        state = request.app.state
        snap = state.backend_service.snapshot()
        envs = await state.pyenv_service.list()
        funcs = await state.pyfunc_service.list()
        dags = await state.v2_dag_service.list()
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

    @app.get(f"{prefix}/v2/metrics")
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

    @app.get(f"{prefix}/v2/topology")
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

    @app.get(f"{prefix}/v2/health")
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

    # -- v2 API routers (PyEnv / PyFunc / PyFuncRun / Backend / Network) ----
    app.include_router(v2_card_router, prefix=f"{prefix}/card")
    app.include_router(v2_pyenv_router, prefix=f"{prefix}/v2/pyenv")
    app.include_router(v2_pyfunc_router, prefix=f"{prefix}/v2/pyfunc")
    app.include_router(v2_pyfuncrun_router, prefix=f"{prefix}/v2/pyfuncrun")
    app.include_router(v2_dag_router, prefix=f"{prefix}/v2/dag")
    app.include_router(v2_backend_router, prefix=f"{prefix}/v2/backend")
    app.include_router(v2_network_router, prefix=f"{prefix}/v2/network")
    app.include_router(v2_replicate_router, prefix=f"{prefix}/v2/replicate")
    app.include_router(v2_fs_router, prefix=f"{prefix}/v2/fs")
    app.include_router(v2_user_router, prefix=f"{prefix}/v2/user")
    app.include_router(v2_messenger_router, prefix=f"{prefix}/v2/messenger")

    @app.get(f"{prefix}/v2/audit")
    async def get_audit(limit: int = 100):
        return {"entries": audit.recent(limit=limit)}

    return app


app = create_app()
