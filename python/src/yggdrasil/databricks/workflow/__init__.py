"""Prefect-style declarative workflow API for Databricks.

Stack:

* :func:`task` decorates a Python callable as a single workflow step.
  Inside a flow trace the call records a :class:`TaskNode` future;
  outside a trace it runs unchanged so unit tests need no workspace.
* :func:`flow` decorates the orchestrating function. Calling
  ``flow.deploy()`` traces the body, stages every captured task as a
  Databricks :class:`Task`, and upserts the bundle as one
  :class:`Job` via :meth:`Jobs.create_or_update`. ``flow.run()`` then
  triggers it.
* :func:`secret` builds a :class:`SecretRef` placeholder that
  resolves at runtime via :func:`runtime.secret` — cleartext never
  lives on disk in the staged source.
* :mod:`runtime` holds the helpers the staged tasks call:
  :func:`runtime.secret`, :func:`runtime.task_value`,
  :func:`runtime.publish_return`. Imported on every staged script
  via ``from yggdrasil.databricks.workflow import runtime as
  _ygg_runtime``.

Worked example::

    from yggdrasil.databricks.workflow import flow, task, secret

    @task
    def extract(date: str) -> str:
        return f"/Volumes/raw/{date}"

    @task(retries=2)
    def load(path: str, api_key: str = secret("vendor", "api-key")) -> str:
        import requests
        return requests.post(URL, json={"path": path}, auth=api_key).text

    @flow(name="daily-etl", schedule="0 2 * * *")
    def daily_etl(date: str = "2025-01-01"):
        p = extract(date)
        load(p)

    daily_etl.deploy()                          # upsert the Databricks Job
    run = daily_etl.run(date="2025-01-15", wait=True)
"""
from __future__ import annotations

from . import runtime
from .context import TraceContext, current_trace
from .flow import Flow, flow
from .nodes import FlowParam, TaskNode
from .resources import SecretRef, secret
from .task import WorkflowTask, task

__all__ = [
    "Flow",
    "FlowParam",
    "SecretRef",
    "TaskNode",
    "TraceContext",
    "WorkflowTask",
    "current_trace",
    "flow",
    "runtime",
    "secret",
    "task",
]
