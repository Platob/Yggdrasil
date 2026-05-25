from __future__ import annotations

from fastapi import Request

from .services.call import CallService
from .services.cmd import CmdService
from .services.env import EnvService
from .services.job import JobService
from .services.python import PythonExecService


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
