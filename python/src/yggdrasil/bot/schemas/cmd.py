from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class CmdRequest(StrictModel):
    command: list[str] = Field(
        ...,
        min_length=1,
        description="Command and arguments, e.g. ['ls', '-la'].",
    )
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout: float | None = Field(
        default=None,
        description="Timeout in seconds. Capped by server max_cmd_timeout.",
    )
    stdin: str | None = None


class CmdResponse(StrictModel):
    id: str
    node_id: str
    command: list[str]
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration: float | None = None
    status: str = "completed"


class CmdEntry(StrictModel):
    id: str
    command: list[str]
    status: str
    returncode: int | None = None
    duration: float | None = None
    created_at: str | None = None


class CmdListResponse(StrictModel):
    node_id: str
    items: list[CmdEntry]
