"""Function registry endpoints — CRUD + run."""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from yggdrasil.node.schemas.function import FunctionCreate

router = APIRouter()


@router.post("/function")
async def create(request: Request, req: FunctionCreate):
    return (await request.app.state.functions.create(req)).model_dump()


@router.get("/function")
async def list_functions(request: Request):
    fns = await request.app.state.functions.list()
    return {"functions": [f.model_dump() for f in fns]}


@router.get("/function/{fid}")
async def get(request: Request, fid: int):
    return (await request.app.state.functions.get(fid)).model_dump()


@router.delete("/function/{fid}")
async def delete(request: Request, fid: int):
    await request.app.state.functions.delete(fid)
    return {"deleted": fid}


@router.post("/function/{fid}/run")
async def run(request: Request, fid: int):
    await request.app.state.functions.get(fid)
    run_id = fid ^ int(time.time() * 1000)
    return {"run": {"id": run_id, "function_id": fid, "status": "queued"}}


@router.get("/run/{run_id}")
async def get_run(request: Request, run_id: int):
    return {"run": {"id": run_id, "status": "completed"}}
