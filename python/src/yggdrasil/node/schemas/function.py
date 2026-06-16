"""Python function schemas — create payloads, stored data, run results."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FunctionCreate(BaseModel):
    """Inbound payload to create/update a python function (upsert by name)."""

    name: str
    code: str
    language: str = "python"


class FunctionData(BaseModel):
    """A stored function. ``id`` is immutable once assigned."""

    id: str
    name: str
    code: str
    language: str = "python"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class FunctionResponse(BaseModel):
    """Envelope returned by create/get."""

    function: FunctionData


class RunData(BaseModel):
    """A single function execution."""

    id: str
    function_id: str
    status: str = "completed"
    stdout: str = ""
    stderr: str = ""
    created_at: datetime = Field(default_factory=_now)


class RunResponse(BaseModel):
    """Envelope returned by run."""

    run: RunData
