from __future__ import annotations

from fastapi import Request

from .services.backend import BackendService
from .services.dag import DAGService
from .services.network import NetworkService
from .services.pyenv import PyEnvService
from .services.pyfunc import PyFuncService
from .services.pyfuncrun import PyFuncRunService
from .services.fs import FsService
from .services.replicate import ReplicateService


def get_fs_service(request: Request) -> FsService:
    return request.app.state.fs_service


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
