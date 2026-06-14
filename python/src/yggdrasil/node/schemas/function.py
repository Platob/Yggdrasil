"""Function registry schemas."""
from __future__ import annotations

from pydantic import BaseModel


class FunctionCreate(BaseModel):
    name: str
    code: str
    language: str = "python"


class FunctionInfo(BaseModel):
    id: int
    name: str
    language: str
    created_at: float
    updated_at: float


class Function(FunctionInfo):
    code: str


class FunctionResponse(BaseModel):
    function: FunctionInfo


class RunResponse(BaseModel):
    run: dict
