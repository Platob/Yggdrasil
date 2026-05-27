from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class PyFuncCreate(StrictModel):
    name: str
    code: str
    description: str = ""
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    env_id: int | None = None


class PyFuncUpdate(StrictModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    python_version: str | None = None
    dependencies: list[str] | None = None
    env_id: int | None = None


class PyFuncEntry(StrictModel):
    id: int
    name: str
    code: str
    description: str = ""
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    env_id: int | None = None
    run_count: int = 0
    created_at: str
    updated_at: str
    last_run_at: str | None = None


class PyFuncResponse(StrictModel):
    func: PyFuncEntry


class PyFuncListResponse(StrictModel):
    node_id: str
    funcs: list[PyFuncEntry]
