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
    environment_id: str | None = None


class FunctionUpdate(StrictModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    python_version: str | None = None
    dependencies: list[str] | None = None
    environment_id: str | None = None


class FunctionEntry(StrictModel):
    id: str
    name: str
    language: str
    code: str
    description: str
    python_version: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    environment_id: str | None = None
    creator: str = "system"
    created_at: str
    updated_at: str
    run_count: int = 0


class FunctionResponse(StrictModel):
    function: FunctionEntry


class FunctionListResponse(StrictModel):
    node_id: str
    functions: list[FunctionEntry]
