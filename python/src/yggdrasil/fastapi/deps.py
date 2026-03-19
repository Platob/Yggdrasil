from __future__ import annotations

from fastapi import Request

from .services.databricks import DatabricksExcelService
from .services.python import PythonService
from .services.system import SystemService


def get_system_service(request: Request) -> SystemService:
    return request.app.state.system_service


def get_python_service(request: Request) -> PythonService:
    return request.app.state.python_service


def get_databricks_excel_service(request: Request) -> DatabricksExcelService:
    return request.app.state.databricks_excel_service

