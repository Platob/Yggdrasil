from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class EnvironmentCreate(StrictModel):
    name: str
    python_version: str = "3.11"
    dependencies: list[str] = Field(default_factory=list)


class EnvironmentUpdate(StrictModel):
    name: str | None = None
    dependencies: list[str] | None = None


class EnvironmentEntry(StrictModel):
    id: int
    name: str
    python_version: str
    dependencies: list[str] = Field(default_factory=list)
    path: str
    status: str = "pending"
    created_at: str
    updated_at: str
    error: str | None = None


class EnvironmentResponse(StrictModel):
    environment: EnvironmentEntry


class EnvironmentListResponse(StrictModel):
    node_id: str
    environments: list[EnvironmentEntry]


class InstallRequest(StrictModel):
    packages: list[str]
