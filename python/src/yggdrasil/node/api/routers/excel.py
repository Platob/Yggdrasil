"""HTTP surface for the Excel service.

Tabular endpoints return a binary body (Parquet by default, Arrow, or
JSON records) chosen via ``?format=``; metadata endpoints return JSON.
Designed for the Power Query connector (`Parquet.Document`) and the
Office.js add-in (apache-arrow / JSON) alike.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ..deps import get_excel_service
from ..schemas.excel import (
    ExcelInfo,
    ExcelQueryRequest,
    ExcelTreeResponse,
    ExcelWriteResponse,
)
from ..services.excel import ExcelService

router = APIRouter(tags=["excel"])


@router.get("/info", response_model=ExcelInfo)
async def excel_info(service: ExcelService = Depends(get_excel_service)) -> ExcelInfo:
    """Identity + capability card — the connector/add-in reads this on connect."""
    return service.info()


@router.post("/python")
async def run_python(
    req: ExcelQueryRequest,
    format: str = "parquet",
    service: ExcelService = Depends(get_excel_service),
) -> Response:
    """Run a Python snippet and return the named dataframe as a table."""
    table = await service.run_python(req)
    body, content_type = service.serialize_table(table, format)
    return Response(content=body, media_type=content_type)


@router.get("/fs/tree", response_model=ExcelTreeResponse)
async def fs_tree(
    path: str = "",
    depth: int = 3,
    service: ExcelService = Depends(get_excel_service),
) -> ExcelTreeResponse:
    """Walk the node filesystem for the connector's navigation table."""
    return await service.tree(path, depth)


@router.get("/fs/read")
async def fs_read(
    path: str,
    format: str = "parquet",
    source_format: str | None = None,
    service: ExcelService = Depends(get_excel_service),
) -> Response:
    """Read a file (parquet/csv/json/arrow) and return it as a typed table.

    ``source_format`` overrides how the file is parsed (default: by
    extension); ``format`` picks the wire encoding handed back.
    """
    table = await service.read_table(path, source_format)
    body, content_type = service.serialize_table(table, format)
    return Response(content=body, media_type=content_type)


@router.post("/fs/write", response_model=ExcelWriteResponse)
async def fs_write(
    request: Request,
    path: str,
    service: ExcelService = Depends(get_excel_service),
) -> ExcelWriteResponse:
    """Write an uploaded table (parquet/arrow/csv body) to a file."""
    data = await request.body()
    content_type = request.headers.get("content-type", "")
    return await service.write_table(path, data, content_type)
