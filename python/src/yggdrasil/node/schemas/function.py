"""Wire models for the function store + runner service."""
from __future__ import annotations

from pydantic import BaseModel


class FunctionCreate(BaseModel):
    name: str
    code: str
    language: str = "python"


class FunctionRecord(BaseModel):
    id: str
    name: str
    code: str
    language: str
    created_at: str  # ISO 8601, UTC


class FunctionResponse(BaseModel):
    function: FunctionRecord


class RunRecord(BaseModel):
    id: str
    function_id: str
    status: str  # "success" | "error"
    output: str | None = None
    error: str | None = None
    started_at: str
    finished_at: str | None = None


class RunResponse(BaseModel):
    run: RunRecord
