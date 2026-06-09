"""Filesystem router — /api/v2/fs/

GET  /api/v2/fs/ls?path=&offset=&limit=    → LsResult
GET  /api/v2/fs/read?path=                 → ReadResult
POST /api/v2/fs/write                      → {"ok": true}
DELETE /api/v2/fs/delete?path=             → {"ok": true}
GET  /api/v2/fs/nodes                      → FsNodeRoot (with mounts key)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    pass

router = APIRouter(prefix="/api/v2/fs", tags=["fs"])


def _fs(request: Request):
    return request.app.state.fs


def _saga(request: Request):
    return request.app.state.saga


@router.get("/ls")
async def ls(path: str = "", offset: int = 0, limit: int | None = None,
             fs=Depends(_fs)) -> dict:
    try:
        result = await fs.ls(path, offset=offset, limit=limit)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/read")
async def read(path: str, fs=Depends(_fs)) -> dict:
    try:
        result = await fs.read(path)
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/write")
async def write(body: dict, fs=Depends(_fs)) -> dict:
    try:
        path = body.get("path", "")
        content = body.get("content", "")
        target = fs._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return {"ok": True, "path": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/delete")
async def delete(path: str, fs=Depends(_fs)) -> dict:
    try:
        target = fs._resolve(path)
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        return {"ok": True, "path": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/nodes")
async def nodes(request: Request, fs=Depends(_fs)) -> dict:
    """Return the node home root plus all registered mounts."""
    try:
        result = await fs.ls("")
    except Exception:
        result = type("_R", (), {"entries": [], "total": 0})()

    # Fetch mounts from saga service if available.
    mounts: list[dict] = []
    try:
        saga = request.app.state.saga
        mount_list = await saga.list_mounts()
        mounts = [m.model_dump() for m in mount_list]
    except Exception:
        pass

    settings = request.app.state.settings
    return {
        "node_id": settings.node_id,
        "home": str(settings.node_home),
        "entries": [e.model_dump() for e in result.entries],
        "mounts": mounts,
    }
