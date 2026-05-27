from __future__ import annotations

from fastapi import Request

from .services.ai import AIService
from .services.call import CallService
from .services.cmd import CmdService
from .services.dag import DagService
from .services.discovery import DiscoveryService
from .services.env import EnvService
from .services.environment import EnvironmentService
from .services.filesystem import FilesystemService
from .services.function import FunctionService
from .services.job import JobService
from .services.market import MarketService
from .services.messenger import MessengerService
from .services.monitor import MonitorService
from .services.python import PythonExecService
from .services.run import RunService


def get_env_service(request: Request) -> EnvService:
    return request.app.state.env_service


def get_cmd_service(request: Request) -> CmdService:
    return request.app.state.cmd_service


def get_python_service(request: Request) -> PythonExecService:
    return request.app.state.python_service


def get_job_service(request: Request) -> JobService:
    return request.app.state.job_service


def get_call_service(request: Request) -> CallService:
    return request.app.state.call_service


def get_messenger_service(request: Request) -> MessengerService:
    return request.app.state.messenger_service


def get_discovery_service(request: Request) -> DiscoveryService:
    return request.app.state.discovery_service


def get_function_service(request: Request) -> FunctionService:
    return request.app.state.function_service


def get_environment_service(request: Request) -> EnvironmentService:
    return request.app.state.environment_service


def get_run_service(request: Request) -> RunService:
    return request.app.state.run_service


def get_dag_service(request: Request) -> DagService:
    return request.app.state.dag_service


def get_filesystem_service(request: Request) -> FilesystemService:
    return request.app.state.filesystem_service


def get_monitor_service(request: Request) -> MonitorService:
    return request.app.state.monitor_service


def get_market_service(request: Request) -> MarketService:
    return request.app.state.market_service


def get_ai_service(request: Request) -> AIService:
    return request.app.state.ai_service
