from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_job_service
from ..schemas.job import (
    JobCreateRequest,
    JobListResponse,
    JobResponse,
    RunListResponse,
    RunResponse,
)
from ..services.job import JobService

router = APIRouter(tags=["job"])


# -- job CRUD --------------------------------------------------------------

@router.get("", response_model=JobListResponse)
async def list_jobs(
    service: JobService = Depends(get_job_service),
) -> JobListResponse:
    return await service.list_jobs()


@router.post("", response_model=JobResponse)
async def create_job(
    req: JobCreateRequest,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    return await service.create_job(req)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    return await service.get_job(job_id)


@router.put("/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: str,
    req: JobCreateRequest,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    return await service.update_job(job_id, req)


@router.delete("/{job_id}", response_model=JobResponse)
async def delete_job(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> JobResponse:
    return await service.delete_job(job_id)


# -- run CRUD --------------------------------------------------------------

@router.get("/{job_id}/run", response_model=RunListResponse)
async def list_runs(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> RunListResponse:
    return await service.list_runs(job_id)


@router.post("/{job_id}/run", response_model=RunResponse)
async def trigger_run(
    job_id: str,
    service: JobService = Depends(get_job_service),
) -> RunResponse:
    return await service.trigger_run(job_id)


@router.get("/{job_id}/run/{run_id}", response_model=RunResponse)
async def get_run(
    job_id: str,
    run_id: str,
    service: JobService = Depends(get_job_service),
) -> RunResponse:
    return await service.get_run(job_id, run_id)


@router.put("/{job_id}/run/{run_id}", response_model=RunResponse)
async def update_run(
    job_id: str,
    run_id: str,
    service: JobService = Depends(get_job_service),
) -> RunResponse:
    return await service.update_run(job_id, run_id)


@router.delete("/{job_id}/run/{run_id}", response_model=RunResponse)
async def delete_run(
    job_id: str,
    run_id: str,
    service: JobService = Depends(get_job_service),
) -> RunResponse:
    return await service.delete_run(job_id, run_id)
