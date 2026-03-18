from __future__ import annotations

from .common import StrictModel


class SystemInfoResponse(StrictModel):
    name: str
    version: str
    docs: str
    openapi: str
    routes: dict[str, str]


class HealthResponse(StrictModel):
    ok: bool
    env_home: str
    python_executable: str
