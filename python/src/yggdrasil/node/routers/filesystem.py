from __future__ import annotations

from fastapi import APIRouter, Depends, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from ..deps import get_filesystem_service
from ..schemas.filesystem import (
    DirectoryListing,
    FileContent,
    FileInfo,
    FileMoveRequest,
    FileWriteRequest,
)
from ..services.filesystem import FilesystemService

router = APIRouter(tags=["filesystem"])


@router.get("/ls", response_model=DirectoryListing)
async def list_directory(
    path: str = "",
    service: FilesystemService = Depends(get_filesystem_service),
) -> DirectoryListing:
    return await service.list_dir(path)


@router.get("/read", response_model=FileContent)
async def read_file(
    path: str,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileContent:
    return await service.read_file(path)


@router.post("/write", response_model=FileInfo)
async def write_file(
    req: FileWriteRequest,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileInfo:
    return await service.write_file(req)


@router.delete("/delete", response_model=None, status_code=204)
async def delete_path(
    path: str,
    service: FilesystemService = Depends(get_filesystem_service),
) -> None:
    await service.delete(path)


@router.post("/move", response_model=FileInfo)
async def move_path(
    req: FileMoveRequest,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileInfo:
    return await service.move(req)


@router.post("/mkdir", response_model=FileInfo)
async def make_directory(
    path: str,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileInfo:
    return await service.mkdir(path)


@router.get("/stat", response_model=FileInfo)
async def stat_path(
    path: str,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileInfo:
    return await service.stat(path)


@router.get("/stream")
async def stream_download(
    path: str,
    service: FilesystemService = Depends(get_filesystem_service),
) -> StreamingResponse:
    # Get file info for content-type/size headers
    info = await service.stat(path)
    return StreamingResponse(
        service.stream_read(path),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{info.name}"',
            "Content-Length": str(info.size),
        },
    )


@router.post("/upload", response_model=FileInfo)
async def stream_upload(
    path: str,
    request: Request,
    service: FilesystemService = Depends(get_filesystem_service),
) -> FileInfo:
    async def _body_chunks():
        async for chunk in request.stream():
            yield chunk

    return await service.stream_write(path, _body_chunks())
