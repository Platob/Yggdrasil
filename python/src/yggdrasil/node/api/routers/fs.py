from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..deps import get_fs_service
from ..schemas.fs import (
    FsEntry,
    FsListResponse,
    FsMoveRequest,
    FsReadResponse,
    FsWriteRequest,
)
from ..services.fs import FsService

router = APIRouter(tags=["fs"])


@router.get("/ls", response_model=FsListResponse)
async def list_directory(
    path: str = "",
    service: FsService = Depends(get_fs_service),
) -> FsListResponse:
    return await service.ls(path)


@router.get("/stat", response_model=FsEntry)
async def stat_path(
    path: str,
    service: FsService = Depends(get_fs_service),
) -> FsEntry:
    return await service.stat(path)


@router.get("/read", response_model=FsReadResponse)
async def read_file(
    path: str,
    service: FsService = Depends(get_fs_service),
) -> FsReadResponse:
    return await service.read(path)


@router.post("/write", response_model=FsEntry)
async def write_file(
    req: FsWriteRequest,
    service: FsService = Depends(get_fs_service),
) -> FsEntry:
    return await service.write(req)


@router.delete("/delete", response_model=None, status_code=204)
async def delete_path(
    path: str,
    service: FsService = Depends(get_fs_service),
) -> None:
    await service.delete(path)


@router.post("/move", response_model=FsEntry)
async def move_path(
    req: FsMoveRequest,
    service: FsService = Depends(get_fs_service),
) -> FsEntry:
    return await service.move(req)


@router.post("/mkdir", response_model=FsEntry)
async def make_directory(
    path: str,
    service: FsService = Depends(get_fs_service),
) -> FsEntry:
    return await service.mkdir(path)


@router.get("/stream")
async def stream_download(
    path: str,
    service: FsService = Depends(get_fs_service),
) -> StreamingResponse:
    info = await service.stat(path)
    return StreamingResponse(
        service.stream_read(path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{info.name}"',
            "Content-Length": str(info.size),
        },
    )


@router.post("/upload", response_model=FsEntry)
async def stream_upload(
    path: str,
    request: Request,
    service: FsService = Depends(get_fs_service),
) -> FsEntry:
    async def _body_chunks():
        async for chunk in request.stream():
            yield chunk

    return await service.stream_write(path, _body_chunks())
