from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response as FastAPIResponse

from ..deps import get_python_service
from ..schemas.python import PythonListResponse, PythonRequest, PythonResponse
from ..services.python import PythonExecService

router = APIRouter(tags=["python"])


@router.get("", response_model=PythonListResponse)
async def list_executions(
    service: PythonExecService = Depends(get_python_service),
) -> PythonListResponse:
    return await service.list_history()


@router.post("")
async def execute_python(
    req: PythonRequest,
    service: PythonExecService = Depends(get_python_service),
):
    response = await service.execute(req)
    if req.result_format == "arrow_ipc" and response.result is not None:
        ipc_bytes = PythonExecService.result_to_arrow_ipc(response.result)
        return FastAPIResponse(
            content=ipc_bytes,
            media_type="application/vnd.apache.arrow.file",
            headers={"X-Bot-Exec-Id": response.id},
        )
    return response


@router.get("/{exec_id}", response_model=PythonResponse)
async def get_execution(
    exec_id: str,
    service: PythonExecService = Depends(get_python_service),
) -> PythonResponse:
    return await service.get(exec_id)


@router.delete("/{exec_id}", response_model=PythonResponse)
async def delete_execution(
    exec_id: str,
    service: PythonExecService = Depends(get_python_service),
) -> PythonResponse:
    return await service.delete(exec_id)
