"""Function contracts: create/upsert a stored function, run results."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FunctionCreate(BaseModel):
    name: str
    code: str
    language: str = "python"


class Function(BaseModel):
    id: int
    name: str
    code: str
    language: str = "python"


class FunctionResponse(BaseModel):
    function: Function


class RunResult(BaseModel):
    id: int
    function_id: int
    status: str  # "running" | "succeeded" | "failed"
    stdout: str = ""
    stderr: str = ""
    result: Any = None
