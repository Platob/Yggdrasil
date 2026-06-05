from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class DagNodeRef(StrictModel):
    """Reference to a function on a specific node."""
    node_url: str | None = None  # None = local node
    function_id: int
    environment_id: int | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class DagEdge(StrictModel):
    """Connection between two DAG nodes. output_key maps source output to dest input."""
    from_step: str  # step ID
    to_step: str
    output_key: str = "result"  # which output to pass
    input_key: str = "input"   # which input arg to set


class DagStep(StrictModel):
    """A step in a DAG."""
    id: str  # human-readable step name like "extract", "transform", "load"
    ref: DagNodeRef
    depends_on: list[str] = Field(default_factory=list)


class DagCreate(StrictModel):
    name: str
    description: str = ""
    steps: list[DagStep]
    edges: list[DagEdge] = Field(default_factory=list)


class DagEntry(StrictModel):
    id: int
    name: str
    description: str
    steps: list[DagStep]
    edges: list[DagEdge] = Field(default_factory=list)
    created_at: str
    updated_at: str
    run_count: int = 0
    deleted_at: str | None = None
    last_used_at: str | None = None


class DagResponse(StrictModel):
    dag: DagEntry


class DagListResponse(StrictModel):
    node_id: str
    dags: list[DagEntry]


class DagRunEntry(StrictModel):
    id: int
    dag_id: int
    status: str = "pending"  # pending, running, completed, failed
    started_at: str | None = None
    completed_at: str | None = None
    duration: float | None = None
    step_results: dict[str, Any] = Field(default_factory=dict)  # step_id -> result


class DagRunResponse(StrictModel):
    run: DagRunEntry


class DagRunListResponse(StrictModel):
    node_id: str
    runs: list[DagRunEntry]
