from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_dag_service
from ..schemas.dag import (
    DAGCreate,
    DAGListResponse,
    DAGResponse,
    DAGRunListResponse,
    DAGRunResponse,
    DAGScheduleRequest,
)
from ..services.dag import DAGService

router = APIRouter(tags=["dag"])


@router.get("", response_model=DAGListResponse)
async def list_dags(
    service: DAGService = Depends(get_dag_service),
) -> DAGListResponse:
    return await service.list()


@router.post("", response_model=DAGResponse)
async def create_dag(
    req: DAGCreate,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    return await service.create(req)


@router.get("/{dag_id}", response_model=DAGResponse)
async def get_dag(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    entry = await service.get(dag_id)
    return DAGResponse(dag=entry)


@router.head("/{dag_id}")
async def head_dag(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> dict:
    """Existence check by ID. Returns 200 if found, 404 otherwise."""
    await service.get(dag_id)
    return {}


@router.put("/{dag_id}", response_model=DAGResponse)
async def update_dag(
    dag_id: int,
    req: DAGCreate,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    """Update a DAG by upserting with the same name."""
    return await service.create(req)


@router.delete("/{dag_id}", response_model=DAGResponse)
async def delete_dag(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    return await service.delete(dag_id)


@router.post("/{dag_id}/run", response_model=DAGRunResponse)
async def execute_dag(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGRunResponse:
    return await service.execute(dag_id)


@router.get("/{dag_id}/run", response_model=DAGRunListResponse)
async def list_dag_runs(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGRunListResponse:
    return await service.list_runs(dag_id)


@router.get("/{dag_id}/run/{run_id}", response_model=DAGRunResponse)
async def get_dag_run(
    dag_id: int,
    run_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGRunResponse:
    entry = await service.get_run(dag_id, run_id)
    return DAGRunResponse(run=entry)


@router.post("/{dag_id}/schedule", response_model=DAGResponse)
async def schedule_dag(
    dag_id: int,
    req: DAGScheduleRequest,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    """Schedule a DAG to run at intervals."""
    return await service.schedule(dag_id, req.interval_seconds, req.max_runs)


@router.delete("/{dag_id}/schedule", response_model=DAGResponse)
async def unschedule_dag(
    dag_id: int,
    service: DAGService = Depends(get_dag_service),
) -> DAGResponse:
    """Stop a scheduled DAG."""
    return await service.unschedule(dag_id)
