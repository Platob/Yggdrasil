from __future__ import annotations

from pydantic import Field

from .common import StrictModel


class FunctionCreate(StrictModel):
    name: str
    language: str = "python"
    code: str
    description: str = ""
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    environment_id: int | None = None


class FunctionUpdate(StrictModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    python_version: str | None = None
    dependencies: list[str] | None = None
    environment_id: int | None = None


class FunctionEntry(StrictModel):
    id: int
    name: str
    language: str
    code: str
    description: str
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    environment_id: int | None = None
    creator: str = "system"
    created_at: str
    updated_at: str
    run_count: int = 0
    deleted_at: str | None = None
    last_used_at: str | None = None
    state: str = "ready"  # ready, running, disabled


class FunctionCloneRequest(StrictModel):
    name: str | None = None


class FunctionResponse(StrictModel):
    function: FunctionEntry


class FunctionListResponse(StrictModel):
    node_id: str
    functions: list[FunctionEntry]
