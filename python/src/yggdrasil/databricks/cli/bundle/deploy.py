"""Bundle deploy — translate a resolved bundle config into API calls."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from yggdrasil.databricks.client import DatabricksClient

from .config import load_bundle, resolve_target
from .sync import sync_files

LOGGER = logging.getLogger(__name__)


def deploy(
    bundle_path: Path,
    target_name: str | None = None,
    *,
    client: DatabricksClient | None = None,
) -> int:
    """Deploy a Databricks Asset Bundle.

    Parses the bundle YAML, resolves the target, syncs workspace files,
    and upserts every job defined in ``resources.jobs``.

    Returns an exit code (0 on success).
    """
    raw = load_bundle(bundle_path)
    bundle_name = (raw.get("bundle") or {}).get("name", bundle_path.parent.name)
    target_cfg, resolved = resolve_target(raw, target_name)

    workspace_cfg = target_cfg.get("workspace") or {}
    host = workspace_cfg.get("host")

    if client is None:
        kwargs: dict[str, Any] = {}
        if host:
            kwargs["host"] = host
        client = DatabricksClient(**kwargs)

    sys.stderr.write(
        f"Deploying bundle {bundle_name!r}"
        f" to {client.base_url.to_string()}"
        f" (target={target_name or 'default'})\n"
    )

    bundle_root = bundle_path.parent

    uploaded = sync_files(client, bundle_root, target_cfg, resolved)
    if uploaded:
        sys.stderr.write(f"  Synced {len(uploaded)} file(s) to workspace\n")

    resources = resolved.get("resources") or {}
    jobs_cfg = resources.get("jobs") or {}

    for job_key, job_cfg in jobs_cfg.items():
        _deploy_job(client, job_key, job_cfg)

    sys.stderr.write("Deploy complete.\n")
    return 0


def _deploy_job(
    client: DatabricksClient,
    job_key: str,
    job_cfg: dict[str, Any],
) -> None:
    """Translate a single job config dict into a Jobs API upsert."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import (
        CronSchedule,
        ForEachTask,
        JobEnvironment,
        JobParameterDefinition,
        NotebookTask,
        PauseStatus,
        Task,
        TaskDependency,
    )

    job_name = job_cfg.get("name", job_key)
    sys.stderr.write(f"  Deploying job {job_name!r} …\n")

    tasks = [
        _build_task(task_cfg)
        for task_cfg in (job_cfg.get("tasks") or [])
    ]

    settings: dict[str, Any] = {}

    schedule_cfg = job_cfg.get("schedule")
    if schedule_cfg:
        pause = schedule_cfg.get("pause_status")
        settings["schedule"] = CronSchedule(
            quartz_cron_expression=schedule_cfg["quartz_cron_expression"],
            timezone_id=schedule_cfg.get("timezone_id", "UTC"),
            pause_status=PauseStatus(pause) if pause else None,
        )

    params_cfg = job_cfg.get("parameters")
    if params_cfg:
        settings["parameters"] = [
            JobParameterDefinition(
                name=p["name"],
                default=str(p.get("default", "")),
            )
            for p in params_cfg
        ]

    envs_cfg = job_cfg.get("environments")
    if envs_cfg:
        settings["environments"] = [
            JobEnvironment(
                environment_key=e["environment_key"],
                spec=Environment(
                    client=e["spec"].get("client"),
                    dependencies=e["spec"].get("dependencies"),
                ) if e.get("spec") else None,
            )
            for e in envs_cfg
        ]

    tags = job_cfg.get("tags")
    if tags:
        settings["tags"] = dict(tags)

    max_concurrent = job_cfg.get("max_concurrent_runs")
    if max_concurrent is not None:
        settings["max_concurrent_runs"] = int(max_concurrent)

    description = job_cfg.get("description")
    if description:
        settings["description"] = description

    job = client.jobs.create_or_update(
        name=job_name,
        tasks=tasks,
        **settings,
    )
    sys.stderr.write(f"    → {job.explore_url}\n")


def _build_task(task_cfg: dict[str, Any]) -> "Task":
    """Translate a single task YAML dict into a Databricks SDK Task."""
    from databricks.sdk.service.jobs import (
        ForEachTask,
        NotebookTask,
        Task,
        TaskDependency,
    )

    task_key = task_cfg["task_key"]

    kwargs: dict[str, Any] = {"task_key": task_key}

    deps = task_cfg.get("depends_on")
    if deps:
        kwargs["depends_on"] = [
            TaskDependency(task_key=d["task_key"])
            for d in deps
        ]

    cluster_id = task_cfg.get("existing_cluster_id")
    if cluster_id:
        kwargs["existing_cluster_id"] = cluster_id

    env_key = task_cfg.get("environment_key")
    if env_key:
        kwargs["environment_key"] = env_key

    timeout = task_cfg.get("timeout_seconds")
    if timeout is not None:
        kwargs["timeout_seconds"] = int(timeout)

    nb_cfg = task_cfg.get("notebook_task")
    if nb_cfg:
        kwargs["notebook_task"] = NotebookTask(
            notebook_path=nb_cfg["notebook_path"],
            base_parameters=nb_cfg.get("base_parameters"),
            source=None,
        )

    for_each_cfg = task_cfg.get("for_each_task")
    if for_each_cfg:
        inner_task = _build_task(for_each_cfg["task"])
        concurrency = for_each_cfg.get("concurrency")
        kwargs["for_each_task"] = ForEachTask(
            inputs=str(for_each_cfg["inputs"]),
            task=inner_task,
            concurrency=int(concurrency) if concurrency is not None else None,
        )

    max_retries = task_cfg.get("max_retries")
    if max_retries is not None:
        kwargs["max_retries"] = int(max_retries)

    retry_interval = task_cfg.get("min_retry_interval_millis")
    if retry_interval is not None:
        kwargs["min_retry_interval_millis"] = int(retry_interval)

    description = task_cfg.get("description")
    if description:
        kwargs["description"] = description

    return Task(**kwargs)
