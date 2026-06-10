from __future__ import annotations

from pydantic import BaseModel


class FunctionCreate(BaseModel):
    name: str
    code: str
    language: str = "python"
    description: str = ""


class FunctionRecord(BaseModel):
    id: int
    name: str
    code: str
    language: str
    description: str
    created_at: str


class FunctionResponse(BaseModel):
    function: FunctionRecord


class FunctionListResponse(BaseModel):
    functions: list[FunctionRecord]
