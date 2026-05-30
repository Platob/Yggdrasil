from __future__ import annotations

from fastapi import Request

from .services.backend import BackendService
from .services.dag import DAGService
from .services.network import NetworkService
from .services.pyenv import PyEnvService
from .services.pyfunc import PyFuncService
from .services.pyfuncrun import PyFuncRunService
from .services.fs import FsService
from .services.tabular import TabularService
from .services.analysis import AnalysisService
from .services.replicate import ReplicateService
from .services.user import UserService
from .services.messenger import MessengerService as V2MessengerService
from .services.excel import ExcelService
from .services.market import MarketService
from .services.ai_insight import AIInsightService


def get_fs_service(request: Request) -> FsService:
    return request.app.state.fs_service


def get_tabular_service(request: Request) -> TabularService:
    return request.app.state.tabular_service


def get_analysis_service(request: Request) -> AnalysisService:
    return request.app.state.analysis_service


def get_pyenv_service(request: Request) -> PyEnvService:
    return request.app.state.pyenv_service


def get_pyfunc_service(request: Request) -> PyFuncService:
    return request.app.state.pyfunc_service


def get_pyfuncrun_service(request: Request) -> PyFuncRunService:
    return request.app.state.pyfuncrun_service


def get_dag_service(request: Request) -> DAGService:
    return request.app.state.dag_service


def get_backend_service(request: Request) -> BackendService:
    return request.app.state.backend_service


def get_network_service(request: Request) -> NetworkService:
    return request.app.state.network_service


def get_replicate_service(request: Request) -> ReplicateService:
    return request.app.state.replicate_service


def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service


def get_messenger_service(request: Request) -> V2MessengerService:
    return request.app.state.v2_messenger_service


def get_excel_service(request: Request) -> ExcelService:
    return request.app.state.excel_service


def get_market_service(request: Request) -> MarketService:
    return request.app.state.market_service


def get_ai_insight_service(request: Request) -> AIInsightService:
    return request.app.state.ai_insight_service
