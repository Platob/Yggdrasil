"""Excel router package.

Aggregates sub-routers for Python/Excel execution and Databricks SQL queries.
All sub-routers share GzipRoute for request-body decompression.
"""
from __future__ import annotations

from fastapi import APIRouter

from ...middleware import GzipRoute
from .pyxl import router as pyxl_router
from .databricks import router as databricks_router

router = APIRouter(route_class=GzipRoute)

router.include_router(pyxl_router)
router.include_router(databricks_router)

__all__ = ["router"]
