from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class RunCreate(StrictModel):
    function_id: str
    environment_id: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class RunEntry(StrictModel):
    id: str
    function_id: str
    environment_id: str | None = None
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
