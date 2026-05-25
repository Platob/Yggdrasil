from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_dag_service
from ..schemas.dag import (
    DagCreate,
    DagListResponse,
    DagResponse,
    DagRunListResponse,
    DagRunResponse,
)
from ..services.dag import DagService

router = APIRouter(tags=["dag"])


@router.get("", response_model=DagListResponse)
async def list_dags(
    service: DagService = Depends(get_dag_service),
) -> DagListResponse:
    return await service.list()


@router.post("", response_model=DagResponse)
async def create_dag(
    req: DagCreate,
    service: DagService = Depends(get_dag_service),
) -> DagResponse:
    return await service.create(req)


@router.get("/{dag_id}", response_model=DagResponse)
async def get_dag(
    dag_id: int,
    service: DagService = Depends(get_dag_service),
) -> DagResponse:
    entry = await service.get(dag_id)
    return DagResponse(dag=entry)


@router.delete("/{dag_id}", response_model=DagResponse)
async def delete_dag(
    dag_id: int,
    service: DagService = Depends(get_dag_service),
) -> DagResponse:
    return await service.delete(dag_id)


@router.post("/{dag_id}/run", response_model=DagRunResponse)
async def trigger_dag_run(
    dag_id: int,
    service: DagService = Depends(get_dag_service),
) -> DagRunResponse:
    return await service.execute(dag_id)


@router.get("/{dag_id}/run", response_model=DagRunListResponse)
async def list_dag_runs(
    dag_id: int,
    service: DagService = Depends(get_dag_service),
) -> DagRunListResponse:
    return await service.list_runs(dag_id)


@router.get("/{dag_id}/run/{run_id}", response_model=DagRunResponse)
async def get_dag_run(
    dag_id: int,
    run_id: int,
    service: DagService = Depends(get_dag_service),
) -> DagRunResponse:
    entry = await service.get_run(dag_id, run_id)
    return DagRunResponse(run=entry)
