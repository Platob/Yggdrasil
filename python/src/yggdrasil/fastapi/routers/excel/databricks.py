"""Databricks SQL routes for the Excel router.

Execute SQL queries on Databricks and return results as DataFrame payloads
consumable by Excel / Power Query.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...deps import get_databricks_excel_service
from ...schemas.databricks import DatabricksSQLRequest, DatabricksSQLResponse
from ...services.databricks import DatabricksExcelService

router = APIRouter(prefix="/databricks", tags=["excel", "databricks"])


@router.post("/sql", response_model=DatabricksSQLResponse)
async def execute_sql(
    req: DatabricksSQLRequest,
    service: DatabricksExcelService = Depends(get_databricks_excel_service),
) -> DatabricksSQLResponse:
    return await service.execute_sql(req)

