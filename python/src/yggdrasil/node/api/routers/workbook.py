from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ...transport import CONTENT_TYPE_ARROW_STREAM
from ..deps import get_network_service, get_tabular_service
from ..schemas.tabular import (
    WorkbookEditRequest,
    WorkbookEditResponse,
    WorkbookSheets,
)
from ..services.network import NetworkService
from ..services.tabular import TabularService

router = APIRouter(tags=["workbook"])


@router.get("/sheets", response_model=WorkbookSheets)
async def sheets(
    path: str,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> WorkbookSheets:
    """List the workbook's sheets with their dims (rows/cols/visible)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "GET", "/api/v2/workbook/sheets", params={"path": path})
    infos = await service.workbook_sheets(path)
    return WorkbookSheets(node_id=service.settings.node_id, path=path, sheets=infos)


@router.get("/read")
async def read_sheet(
    path: str,
    sheet: str | None = None,
    header: bool = True,
    skip_rows: int = 0,
    n_rows: int | None = None,
    columns: str | None = None,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> Response:
    """Read a (windowed) sheet as an Arrow IPC stream — the workbook grid's
    viewport fetch. ``columns`` is a comma-separated subset."""
    if node and node != service.settings.node_id:
        params = {"path": path, "header": header, "skip_rows": skip_rows}
        if sheet is not None:
            params["sheet"] = sheet
        if n_rows is not None:
            params["n_rows"] = n_rows
        if columns:
            params["columns"] = columns
        return StreamingResponse(
            network.proxy_stream(node, "/api/v2/workbook/read", params),
            media_type=CONTENT_TYPE_ARROW_STREAM,
        )
    col_list = [c for c in columns.split(",") if c] if columns else None
    data = await service.read_sheet_arrow(
        path, sheet, header=header, skip_rows=skip_rows, n_rows=n_rows, columns=col_list,
    )
    return Response(content=data, media_type=CONTENT_TYPE_ARROW_STREAM)


@router.post("/edit", response_model=WorkbookEditResponse)
async def edit(
    req: WorkbookEditRequest,
    node: str | None = None,
    service: TabularService = Depends(get_tabular_service),
    network: NetworkService = Depends(get_network_service),
) -> WorkbookEditResponse:
    """Apply a batch of cell edits or a rectangular range write, surgically
    (preserves formulas/formatting/other sheets)."""
    if node and node != service.settings.node_id:
        return await network.proxy_json(node, "POST", "/api/v2/workbook/edit", json_body=req.model_dump())
    n = await service.edit_workbook(req)
    return WorkbookEditResponse(path=req.path, sheet=req.sheet, cells_written=n)
