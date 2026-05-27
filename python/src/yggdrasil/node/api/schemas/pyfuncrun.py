from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class PyFuncRunCreate(StrictModel):
    func_id: int
    env_id: int | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_memory_mb: int | None = None


class PyFuncRunSubmit(StrictModel):
    """Submit a run by function name instead of ID."""
    func_name: str
    env_id: int | None = None
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None
    max_memory_mb: int | None = None


class PyFuncRunEntry(StrictModel):
    id: int
    func_id: int
    env_id: int | None = None
    status: str = "pending"
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    result: Any = None
    result_type: str | None = None
    error: str | None = None
    node_id: str = ""
    progress: float = 0.0
    log_lines: int = 0


class PyFuncRunResponse(StrictModel):
    run: PyFuncRunEntry


class PyFuncRunListResponse(StrictModel):
    node_id: str
    runs: list[PyFuncRunEntry]
