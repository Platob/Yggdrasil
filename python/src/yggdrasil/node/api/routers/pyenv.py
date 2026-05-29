from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import Field

from ..deps import get_pyenv_service, get_pyfunc_service, get_pyfuncrun_service
from ..schemas.common import StrictModel
from ..schemas.pyenv import (
    PyEnvCreate,
    PyEnvListResponse,
    PyEnvPackagesResponse,
    PyEnvResponse,
    PyEnvUpdate,
)
from ..schemas.pyfunc import PyFuncCreate
from ..schemas.pyfuncrun import PyFuncRunCreate, PyFuncRunResponse
from ..services.pyenv import PyEnvService
from ..services.pyfunc import PyFuncService
from ..services.pyfuncrun import PyFuncRunService

router = APIRouter(tags=["pyenv"])


@router.get("", response_model=PyEnvListResponse)
async def list_envs(
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvListResponse:
    return await service.list()


@router.post("", response_model=PyEnvResponse)
async def create_env(
    req: PyEnvCreate,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.create(req)


@router.get("/{env_id}", response_model=PyEnvResponse)
async def get_env(
    env_id: int,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    entry = await service.get(env_id)
    return PyEnvResponse(env=entry)


@router.get("/by-name/{name}/packages", response_model=PyEnvPackagesResponse)
async def list_env_packages_by_name(
    name: str,
    refresh: bool = False,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvPackagesResponse:
    """Like :func:`list_env_packages`, keyed by the env's (unique) name.

    PyEnvs are upserted by name, so the name is a stable string id — the
    web UI uses this route because the int64 ``env_id`` can't survive a
    JavaScript ``JSON.parse`` losslessly.
    """
    return await service.packages_by_name(name, refresh=refresh)


@router.get("/{env_id}/packages", response_model=PyEnvPackagesResponse)
async def list_env_packages(
    env_id: int,
    refresh: bool = False,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvPackagesResponse:
    """Resolved interpreter version + libraries installed in the env's venv.

    TTL-cached server-side, so polling this for a live view won't flood
    the node with ``pip list`` subprocesses. ``?refresh=true`` forces a
    fresh read.
    """
    return await service.packages(env_id, refresh=refresh)


@router.head("/{env_id}")
async def head_env(
    env_id: int,
    service: PyEnvService = Depends(get_pyenv_service),
) -> dict:
    """Existence check by ID. Returns 200 if found, 404 otherwise."""
    await service.get(env_id)
    return {}


@router.put("/{env_id}", response_model=PyEnvResponse)
async def update_env(
    env_id: int,
    req: PyEnvUpdate,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.update(env_id, req)


@router.patch("/{env_id}", response_model=PyEnvResponse)
async def patch_env(
    env_id: int,
    req: PyEnvUpdate,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    """Partial update on an existing env. 404s if missing."""
    return await service.update(env_id, req)


@router.delete("/{env_id}", response_model=PyEnvResponse)
async def delete_env(
    env_id: int,
    service: PyEnvService = Depends(get_pyenv_service),
) -> PyEnvResponse:
    return await service.delete(env_id)


class _BulkDeleteEnvRequest(StrictModel):
    """Body for bulk-delete: list of env IDs to delete."""
    ids: list[int]


@router.post("/bulk/delete")
async def bulk_delete_envs(
    req: _BulkDeleteEnvRequest,
    service: PyEnvService = Depends(get_pyenv_service),
) -> dict:
    """Delete many envs in one request. Continues on per-ID failures."""
    deleted = 0
    failed: list[dict] = []
    for eid in req.ids:
        try:
            await service.delete(eid)
            deleted += 1
        except Exception as exc:
            failed.append({"id": eid, "error": str(exc)})
    return {"deleted": deleted, "failed": failed}


class _EnvRunRequest(StrictModel):
    """Run code directly in a specific environment."""
    code: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_memory_mb: int | None = None


@router.post("/{env_id}/run", response_model=PyFuncRunResponse)
async def run_in_env(
    env_id: int,
    req: _EnvRunRequest,
    pyenv: PyEnvService = Depends(get_pyenv_service),
    pyfunc: PyFuncService = Depends(get_pyfunc_service),
    pyfuncrun: PyFuncRunService = Depends(get_pyfuncrun_service),
) -> PyFuncRunResponse:
    """Execute code in a specific environment. Creates an ephemeral PyFunc,
    spawns the run, and returns the pending entry immediately — the subprocess
    runs in the background. Follow with /pyfuncrun/{id}/wait or /logs."""
    # Validate env exists
    await pyenv.get(env_id)
    # Create an ephemeral function for this code
    import time
    func_resp = await pyfunc.create(PyFuncCreate(
        name=f"_env_run_{env_id}_{int(time.monotonic() * 1000)}",
        code=req.code,
    ))
    create_req = PyFuncRunCreate(
        func_id=func_resp.func.id,
        env_id=env_id,
        args=list(req.args),
        kwargs=dict(req.kwargs),
        timeout=req.timeout,
        max_memory_mb=req.max_memory_mb,
    )
    return await pyfuncrun.create(create_req)
