from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class PyEnvCreate(StrictModel):
    name: str
    python_version: str = "3.11"
    dependencies: list[str] = Field(default_factory=list)


class PyEnvUpdate(StrictModel):
    name: str | None = None
    dependencies: list[str] | None = None


class PyEnvEntry(StrictModel):
    id: int
    name: str
    python_version: str
    dependencies: list[str] = Field(default_factory=list)
    path: str
    status: str = "pending"
    created_at: str
    updated_at: str
    error: str | None = None
    last_used_at: str | None = None
    content_hash: str = ""
    replicated_at: str | None = None
    replicated_from: str | None = None


class PyEnvResponse(StrictModel):
    env: PyEnvEntry


class PyEnvListResponse(StrictModel):
    node_id: str
    envs: list[PyEnvEntry]
