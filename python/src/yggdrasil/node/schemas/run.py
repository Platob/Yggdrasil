from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class RunCreate(StrictModel):
    function_id: int
    environment_id: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class RunEntry(StrictModel):
    id: int
    function_id: int
    environment_id: int | None = None
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    result: Any = None
    node_id: str = ""


class RunResponse(StrictModel):
    run: RunEntry


class RunListResponse(StrictModel):
    node_id: str
    runs: list[RunEntry]
