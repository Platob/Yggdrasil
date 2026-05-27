from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class DAGNodeRef(StrictModel):
    node_url: str | None = None
    func_id: int
    env_id: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class DAGEdge(StrictModel):
    from_step: str
    to_step: str
    output_key: str = "result"
    input_key: str = "input"


class DAGStep(StrictModel):
    id: str
    ref: DAGNodeRef
    depends_on: list[str] = Field(default_factory=list)


class DAGCreate(StrictModel):
    name: str
    description: str = ""
    steps: list[DAGStep]
    edges: list[DAGEdge] = Field(default_factory=list)


class DAGEntry(StrictModel):
    id: int
    name: str
    description: str = ""
    steps: list[DAGStep]
    edges: list[DAGEdge] = Field(default_factory=list)
    created_at: str
    updated_at: str
    run_count: int = 0
    content_hash: str = ""
    replicated_at: str | None = None
    replicated_from: str | None = None


class DAGResponse(StrictModel):
    dag: DAGEntry


class DAGListResponse(StrictModel):
    node_id: str
    dags: list[DAGEntry]


class DAGRunEntry(StrictModel):
    id: int
    dag_id: int
    status: str = "pending"
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)
    node_id: str = ""


class DAGRunResponse(StrictModel):
    run: DAGRunEntry


class DAGRunListResponse(StrictModel):
    node_id: str
    runs: list[DAGRunEntry]
