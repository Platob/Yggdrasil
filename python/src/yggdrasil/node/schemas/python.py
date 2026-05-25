from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class PythonRequest(StrictModel):
    code: str = Field(
        ...,
        description="Python source code to execute.",
    )
    env: dict[str, str] = Field(default_factory=dict)
    timeout: float | None = Field(
        default=None,
        description="Timeout in seconds. Capped by server max_python_timeout.",
    )
    result_format: str = Field(
        default="json",
        description="Response format: 'json' or 'arrow_ipc'.",
    )


class PythonResponse(StrictModel):
    id: str
    node_id: str
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    result: Any = None
    duration: float | None = None
    status: str = "completed"


class PythonEntry(StrictModel):
    id: str
    status: str
    returncode: int | None = None
    duration: float | None = None
    created_at: str | None = None


class PythonListResponse(StrictModel):
    node_id: str
    items: list[PythonEntry]
