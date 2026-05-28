from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..deps import get_fs_service
from ..schemas.common import StrictModel
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


class _SearchRequest(StrictModel):
    path: str = ""
    pattern: str  # glob pattern like "*.py" or substring
    max_results: int = 100


@router.post("/search")
async def search_files(req: _SearchRequest, service: FsService = Depends(get_fs_service)) -> dict:
    """Search for files matching a pattern (glob or substring)."""
    import fnmatch
    from ...exceptions import NotFoundError
    root = service._root / req.path.lstrip("/") if req.path else service._root
    if not root.exists() or not root.is_dir():
        raise NotFoundError(f"Directory {req.path!r} not found")

    matches = []
    for p in root.rglob("*"):
        if len(matches) >= req.max_results:
            break
        rel = str(p.relative_to(service._root))
        name = p.name
        if fnmatch.fnmatch(name, req.pattern) or req.pattern.lower() in name.lower():
            try:
                stat = p.stat()
                matches.append({
                    "path": rel,
                    "name": name,
                    "is_dir": p.is_dir(),
                    "size": stat.st_size if not p.is_dir() else 0,
                })
            except OSError:
                continue
    return {"path": req.path, "pattern": req.pattern, "count": len(matches), "matches": matches}


class _CopyRequest(StrictModel):
    source: str
    destination: str


@router.post("/copy", response_model=FsEntry)
async def copy_path(req: _CopyRequest, service: FsService = Depends(get_fs_service)) -> FsEntry:
    """Copy a file or directory."""
    import shutil
    import datetime as dt
    from ...exceptions import ForbiddenError, NotFoundError
    src = service._root / req.source.lstrip("/")
    dst = service._root / req.destination.lstrip("/")
    if not src.exists():
        raise NotFoundError(f"Source {req.source!r} not found")
    # Prevent traversal
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    root_resolved = service._root.resolve()
    if not str(src_resolved).startswith(str(root_resolved)) or not str(dst_resolved).startswith(str(root_resolved)):
        raise ForbiddenError("Path traversal not allowed")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    else:
        shutil.copy2(str(src), str(dst))
    s = dst.stat()
    return FsEntry(
        path=req.destination,
        name=dst.name,
        is_dir=dst.is_dir(),
        size=s.st_size if not dst.is_dir() else 0,
        modified_at=dt.datetime.fromtimestamp(s.st_mtime, tz=dt.timezone.utc).isoformat(),
    )


@router.get("/tree")
async def tree_listing(path: str = "", depth: int = 3, service: FsService = Depends(get_fs_service)) -> dict:
    """Return a recursive tree of the directory up to N levels deep."""
    from ...exceptions import NotFoundError
    root = service._root / path.lstrip("/") if path else service._root
    if not root.exists() or not root.is_dir():
        raise NotFoundError(f"Directory {path!r} not found")

    def walk(p, level):
        if level > depth:
            return None
        try:
            children = []
            for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if c.is_dir():
                    sub = walk(c, level + 1)
                    children.append({"name": c.name, "is_dir": True, "children": sub})
                else:
                    try:
                        st = c.stat()
                        children.append({"name": c.name, "is_dir": False, "size": st.st_size})
                    except OSError:
                        continue
            return children
        except (PermissionError, OSError):
            return []

    return {"path": path, "depth": depth, "tree": walk(root, 0)}


@router.get("/head")
async def head_file(path: str, n: int = 100, service: FsService = Depends(get_fs_service)) -> dict:
    """First N lines of a file. Default 100."""
    return {"path": path, "lines": service.head_lines(path, n=n)}


@router.get("/tail")
async def tail_file(path: str, n: int = 100, service: FsService = Depends(get_fs_service)) -> dict:
    """Last N lines of a file. Default 100. Uses byte-back scan, no full load."""
    return {"path": path, "lines": service.tail_lines(path, n=n)}


@router.get("/watch")
async def watch_file(path: str, service: FsService = Depends(get_fs_service)) -> StreamingResponse:
    """SSE tail -f. Streams each new line as ``data: {line}\\n\\n``."""
    import orjson
    async def stream():
        async for line in service.watch_tail(path):
            yield b"data: " + orjson.dumps({"line": line}) + b"\n\n"
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


class _GrepRequest(StrictModel):
    path: str = ""
    pattern: str
    max_matches: int = 200
    case_sensitive: bool = False
    regex: bool = False


@router.post("/grep")
async def grep_files(req: _GrepRequest, service: FsService = Depends(get_fs_service)) -> dict:
    """Recursive grep over text files. Returns matches with line numbers."""
    matches = service.grep(
        req.path, req.pattern,
        max_matches=req.max_matches,
        case_sensitive=req.case_sensitive,
        regex=req.regex,
    )
    return {"path": req.path, "pattern": req.pattern, "count": len(matches), "matches": matches}


@router.get("/du")
async def disk_usage(path: str = "", service: FsService = Depends(get_fs_service)) -> dict:
    """Total disk usage of a directory (recursive)."""
    from ...exceptions import NotFoundError
    root = service._root / path.lstrip("/") if path else service._root
    if not root.exists():
        raise NotFoundError(f"Path {path!r} not found")

    total_size = 0
    file_count = 0
    dir_count = 0
    if root.is_file():
        total_size = root.stat().st_size
        file_count = 1
    else:
        for p in root.rglob("*"):
            try:
                if p.is_file():
                    total_size += p.stat().st_size
                    file_count += 1
                elif p.is_dir():
                    dir_count += 1
            except OSError:
                continue
    return {
        "path": path,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / 1024 / 1024, 2),
        "file_count": file_count,
        "dir_count": dir_count,
    }
