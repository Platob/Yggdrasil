from __future__ import annotations

from typing import Any

from pydantic import Field

from .common import StrictModel


class TaskDefinition(StrictModel):
    type: str = Field(
        ...,
        description="Task type: 'cmd' or 'python'.",
    )
    command: list[str] | None = None
    code: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout: float | None = None
    depends_on: list[str] = Field(default_factory=list)


class JobCreateRequest(StrictModel):
    name: str | None = None
    tasks: dict[str, TaskDefinition] = Field(
        ...,
        min_length=1,
        description="Mapping of task_key -> task definition.",
    )
    schedule: str | None = Field(
        default=None,
        description="Optional cron expression for recurring execution.",
    )


class RunEntry(StrictModel):
    run_id: str
    job_id: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration: float | None = None
    task_results: dict[str, Any] = Field(default_factory=dict)


class JobEntry(StrictModel):
    job_id: str
    name: str | None = None
    task_keys: list[str] = Field(default_factory=list)
    schedule: str | None = None
    created_at: str | None = None
    run_count: int = 0


class JobResponse(StrictModel):
    node_id: str
    job: JobEntry


class JobListResponse(StrictModel):
    node_id: str
    items: list[JobEntry]


class RunResponse(StrictModel):
    node_id: str
    run: RunEntry


class RunListResponse(StrictModel):
    node_id: str
    items: list[RunEntry]
