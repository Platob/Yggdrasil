from __future__ import annotations

import hashlib
import platform
import sys

from fastapi import APIRouter, Depends

from ...geo import get_location
from ..deps import (
    get_backend_service,
    get_network_service,
    get_pyenv_service,
    get_pyfunc_service,
    get_pyfuncrun_service,
)
from ..schemas.card import NodeCard
from ..services.backend import BackendService
from ..services.network import NetworkService
from ..services.pyenv import PyEnvService
from ..services.pyfunc import PyFuncService
from ..services.pyfuncrun import PyFuncRunService

router = APIRouter(tags=["card"])


@router.get("", response_model=NodeCard)
async def get_card(
    backend: BackendService = Depends(get_backend_service),
    network: NetworkService = Depends(get_network_service),
    pyenv: PyEnvService = Depends(get_pyenv_service),
    pyfunc: PyFuncService = Depends(get_pyfunc_service),
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> NodeCard:
    settings = backend.settings
    snap = backend.snapshot()
    lat, lon = get_location()
    peers_resp = await network.get_peers()

    envs = await pyenv.list()
    funcs = await pyfunc.list()

    env_count = len(envs.envs)
    func_count = len(funcs.funcs)

    # Fast identity fingerprint for change detection across peers
    content_hash = hashlib.sha256(
        f"{settings.node_id}:{settings.app_version}:{env_count}:{func_count}".encode()
    ).hexdigest()

    return NodeCard(
        node_id=settings.node_id,
        host=settings.host,
        port=settings.port,
        url=f"http://{settings.host}:{settings.port}",
        role=backend.role,
        version=settings.app_version,
        hostname=platform.node(),
        platform=f"{platform.system()} {platform.release()}",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        lat=lat,
        lon=lon,
        cpu_count=snap.cpu_count,
        cpu_percent=snap.cpu_percent,
        memory_used_mb=snap.memory_used_mb,
        memory_total_mb=snap.memory_total_mb,
        gpu_count=len(snap.gpus),
        active_runs=pyfuncrun.active_count,
        total_runs=pyfuncrun.total_count,
        env_count=env_count,
        func_count=func_count,
        uptime_seconds=snap.uptime_seconds,
        node_home=str(settings.node_home),
        peers=[p.node_id for p in peers_resp.peers],
        content_hash=content_hash,
    )
