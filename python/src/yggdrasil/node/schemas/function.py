"""Function management schemas."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class FunctionCreate(BaseModel):
    name: str
    code: str
    language: str = "python"


class Function(BaseModel):
    id: int
    name: str
    code: str
    language: str
    created_at: float


class FunctionResponse(BaseModel):
    function: Function


class RunResponse(BaseModel):
    id: int
    function_id: int
    status: str
    stdout: str = ""
    stderr: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
