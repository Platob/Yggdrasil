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
    # Extra environment variables for this run, layered over the node
    # environment and the env's own stored vars (these win).
    env_vars: dict[str, str] = Field(default_factory=dict)


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
    pid: int | None = None
    heartbeat_at: str | None = None
    cancellation_requested: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class PyFuncRunResponse(StrictModel):
    run: PyFuncRunEntry


class PyFuncRunListResponse(StrictModel):
    node_id: str
    runs: list[PyFuncRunEntry]
