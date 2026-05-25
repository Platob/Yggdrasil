from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import Response as FastAPIResponse, StreamingResponse

from ..deps import get_python_service
from ..schemas.python import PythonListResponse, PythonRequest, PythonResponse
from ..services.python import PythonExecService
from ..transport import (
    CONTENT_TYPE_ARROW_STREAM,
    is_tabular,
    to_arrow_table,
    write_arrow_stream_chunked,
)

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
        if is_tabular(response.result):
            table = to_arrow_table(response.result)
            return StreamingResponse(
                write_arrow_stream_chunked(table),
                media_type=CONTENT_TYPE_ARROW_STREAM,
                headers={
                    "X-Bot-Exec-Id": response.id,
                    "X-Arrow-Num-Rows": str(table.num_rows),
                    "X-Arrow-Num-Columns": str(table.num_columns),
                },
            )

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
