"""Resource deployers — translate bundle YAML dicts into yggdrasil service calls.

Each ``deploy_<resource_type>(client, key, cfg)`` function accepts a
:class:`DatabricksClient`, the YAML key, and the resolved config dict
for that resource. It returns the created/updated resource handle.

Bundle YAML ``resources:`` maps directly:

.. code-block:: yaml

    resources:
      jobs:
        my_job: { ... }      # → deploy_job(client, "my_job", {...})
      clusters:
        my_cluster: { ... }  # → deploy_cluster(client, "my_cluster", {...})
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


# ======================================================================
# Jobs
# ======================================================================

def deploy_job(
    client: "DatabricksClient",
    job_key: str,
    cfg: dict[str, Any],
) -> "Job":
    """Translate a job config dict into a Jobs API upsert.

    Handles: name, tags, schedule (CronSchedule), parameters
    (JobParameterDefinition), environments (JobEnvironment with
    serverless spec), max_concurrent_runs, description, tasks
    (notebook_task, for_each_task, spark_python_task, depends_on,
    existing_cluster_id, environment_key, timeout, retries).
    """
    from yggdrasil.databricks.client import DatabricksClient

    job_name = cfg.get("name", job_key)
    sys.stderr.write(f"  Deploying job {job_name!r} …\n")

    tasks = [build_task(t) for t in (cfg.get("tasks") or [])]
    settings = build_job_settings(cfg)

    job = client.jobs.create_or_update(
        name=job_name,
        tasks=tasks,
        **settings,
    )
    sys.stderr.write(f"    → {job.explore_url}\n")
    return job


def build_job_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract job-level settings from a YAML config dict."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import (
        CronSchedule,
        JobEnvironment,
        JobParameterDefinition,
        PauseStatus,
    )

    settings: dict[str, Any] = {}

    # Schedule
    schedule_cfg = cfg.get("schedule")
    if schedule_cfg:
        pause = schedule_cfg.get("pause_status")
        settings["schedule"] = CronSchedule(
            quartz_cron_expression=schedule_cfg["quartz_cron_expression"],
            timezone_id=schedule_cfg.get("timezone_id", "UTC"),
            pause_status=PauseStatus(pause) if pause else None,
        )

    # Parameters
    params_cfg = cfg.get("parameters")
    if params_cfg:
        settings["parameters"] = [
            JobParameterDefinition(
                name=p["name"],
                default=str(p.get("default", "")),
            )
            for p in params_cfg
        ]

    # Environments
    envs_cfg = cfg.get("environments")
    if envs_cfg:
        settings["environments"] = [
            build_job_environment(e) for e in envs_cfg
        ]

    # Tags
    tags = cfg.get("tags")
    if tags:
        settings["tags"] = {str(k): str(v) for k, v in tags.items()}

    # Max concurrent runs
    max_concurrent = cfg.get("max_concurrent_runs")
    if max_concurrent is not None:
        settings["max_concurrent_runs"] = int(max_concurrent)

    # Description
    description = cfg.get("description")
    if description:
        settings["description"] = description

    # Email notifications
    notifications = cfg.get("email_notifications")
    if notifications:
        settings["email_notifications"] = notifications

    # Webhook notifications
    webhook = cfg.get("webhook_notifications")
    if webhook:
        settings["webhook_notifications"] = webhook

    # Timeout
    timeout = cfg.get("timeout_seconds")
    if timeout is not None:
        settings["timeout_seconds"] = int(timeout)

    return settings


def build_job_environment(cfg: dict[str, Any]) -> "JobEnvironment":
    """Build a JobEnvironment SDK object from a YAML dict."""
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import JobEnvironment

    spec = cfg.get("spec")
    return JobEnvironment(
        environment_key=cfg["environment_key"],
        spec=Environment(
            client=spec.get("client"),
            dependencies=spec.get("dependencies"),
        ) if spec else None,
    )


def build_task(cfg: dict[str, Any]) -> "Task":
    """Translate a single task YAML dict into a Databricks SDK Task.

    Supports: notebook_task, spark_python_task, spark_jar_task,
    sql_task, python_wheel_task, for_each_task (recursive),
    depends_on, existing_cluster_id, job_cluster_key,
    environment_key, timeout_seconds, max_retries, description,
    run_if, libraries.
    """
    from databricks.sdk.service.jobs import (
        ForEachTask,
        NotebookTask,
        RunIf,
        SparkJarTask,
        SparkPythonTask,
        SqlTask,
        Task,
        TaskDependency,
    )

    task_key = cfg["task_key"]
    kwargs: dict[str, Any] = {"task_key": task_key}

    # Dependencies
    deps = cfg.get("depends_on")
    if deps:
        kwargs["depends_on"] = [
            TaskDependency(
                task_key=d["task_key"],
                outcome=d.get("outcome"),
            )
            for d in deps
        ]

    # Compute
    for field in (
        "existing_cluster_id", "job_cluster_key", "environment_key",
    ):
        val = cfg.get(field)
        if val:
            kwargs[field] = val

    # Timeout & retries
    for int_field in (
        "timeout_seconds", "max_retries", "min_retry_interval_millis",
    ):
        val = cfg.get(int_field)
        if val is not None:
            kwargs[int_field] = int(val)

    # Description
    description = cfg.get("description")
    if description:
        kwargs["description"] = description

    # run_if
    run_if = cfg.get("run_if")
    if run_if:
        kwargs["run_if"] = RunIf(run_if)

    # Notebook task
    nb_cfg = cfg.get("notebook_task")
    if nb_cfg:
        kwargs["notebook_task"] = NotebookTask(
            notebook_path=nb_cfg["notebook_path"],
            base_parameters=nb_cfg.get("base_parameters"),
            source=None,
            warehouse_id=nb_cfg.get("warehouse_id"),
        )

    # Spark Python task
    sp_cfg = cfg.get("spark_python_task")
    if sp_cfg:
        kwargs["spark_python_task"] = SparkPythonTask(
            python_file=sp_cfg["python_file"],
            parameters=sp_cfg.get("parameters"),
            source=None,
        )

    # Spark JAR task
    jar_cfg = cfg.get("spark_jar_task")
    if jar_cfg:
        kwargs["spark_jar_task"] = SparkJarTask(
            main_class_name=jar_cfg.get("main_class_name"),
            parameters=jar_cfg.get("parameters"),
            jar_uri=jar_cfg.get("jar_uri"),
        )

    # SQL task
    sql_cfg = cfg.get("sql_task")
    if sql_cfg:
        kwargs["sql_task"] = _build_sql_task(sql_cfg)

    # For-each task (recursive)
    for_each_cfg = cfg.get("for_each_task")
    if for_each_cfg:
        inner = build_task(for_each_cfg["task"])
        concurrency = for_each_cfg.get("concurrency")
        kwargs["for_each_task"] = ForEachTask(
            inputs=str(for_each_cfg["inputs"]),
            task=inner,
            concurrency=int(concurrency) if concurrency is not None else None,
        )

    # Libraries
    libs = cfg.get("libraries")
    if libs:
        kwargs["libraries"] = [_build_library(lib) for lib in libs]

    return Task(**kwargs)


def _build_sql_task(cfg: dict[str, Any]) -> "SqlTask":
    """Build a SqlTask from YAML config."""
    from databricks.sdk.service.jobs import (
        SqlTask,
        SqlTaskAlert,
        SqlTaskDashboard,
        SqlTaskFile,
        SqlTaskQuery,
    )

    kwargs: dict[str, Any] = {}
    if "warehouse_id" in cfg:
        kwargs["warehouse_id"] = cfg["warehouse_id"]
    if "query" in cfg:
        kwargs["query"] = SqlTaskQuery(query_id=cfg["query"]["query_id"])
    if "dashboard" in cfg:
        kwargs["dashboard"] = SqlTaskDashboard(dashboard_id=cfg["dashboard"]["dashboard_id"])
    if "alert" in cfg:
        kwargs["alert"] = SqlTaskAlert(alert_id=cfg["alert"]["alert_id"])
    if "file" in cfg:
        kwargs["file"] = SqlTaskFile(path=cfg["file"]["path"])

    return SqlTask(**kwargs)


def _build_library(cfg: dict[str, Any]) -> "Library":
    """Build a Library from YAML config."""
    from databricks.sdk.service.compute import (
        Library,
        PythonPyPiLibrary,
        MavenLibrary,
        RCranLibrary,
    )

    if isinstance(cfg, str):
        return Library(pypi=PythonPyPiLibrary(package=cfg))
    if "pypi" in cfg:
        pypi = cfg["pypi"]
        return Library(pypi=PythonPyPiLibrary(
            package=pypi.get("package", pypi) if isinstance(pypi, str) else pypi.get("package"),
            repo=pypi.get("repo") if isinstance(pypi, dict) else None,
        ))
    if "jar" in cfg:
        return Library(jar=cfg["jar"])
    if "whl" in cfg:
        return Library(whl=cfg["whl"])
    if "maven" in cfg:
        m = cfg["maven"]
        return Library(maven=MavenLibrary(
            coordinates=m.get("coordinates"),
            repo=m.get("repo"),
            exclusions=m.get("exclusions"),
        ))
    if "cran" in cfg:
        c = cfg["cran"]
        return Library(cran=RCranLibrary(
            package=c.get("package"),
            repo=c.get("repo"),
        ))
    return Library()


# ======================================================================
# Clusters
# ======================================================================

def deploy_cluster(
    client: "DatabricksClient",
    cluster_key: str,
    cfg: dict[str, Any],
) -> "Cluster":
    """Deploy a cluster definition from a bundle YAML.

    Uses ``Clusters.find_cluster`` → update or ``Clusters.create``
    depending on whether the cluster already exists (by name).
    """
    cluster_name = cfg.get("cluster_name", cluster_key)
    sys.stderr.write(f"  Deploying cluster {cluster_name!r} …\n")

    existing = client.compute.clusters.find_cluster(
        cluster_name=cluster_name, raise_error=False,
    )

    spec = _build_cluster_spec(cfg)

    if existing is not None:
        existing.update(**spec)
        sys.stderr.write(f"    → Updated cluster {cluster_name!r}\n")
        return existing

    cluster = client.compute.clusters.create(
        cluster_name=cluster_name, **spec,
    )
    sys.stderr.write(f"    → Created cluster {cluster_name!r}\n")
    return cluster


def _build_cluster_spec(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract cluster spec kwargs from YAML config."""
    spec: dict[str, Any] = {}

    for field in (
        "spark_version", "node_type_id", "driver_node_type_id",
        "num_workers", "autotermination_minutes",
        "spark_conf", "spark_env_vars", "custom_tags",
        "instance_pool_id", "driver_instance_pool_id",
        "enable_elastic_disk", "data_security_mode",
        "single_user_name", "runtime_engine",
    ):
        val = cfg.get(field)
        if val is not None:
            spec[field] = val

    # Autoscale
    autoscale = cfg.get("autoscale")
    if autoscale:
        from databricks.sdk.service.compute import AutoScale
        spec["autoscale"] = AutoScale(
            min_workers=autoscale.get("min_workers", 1),
            max_workers=autoscale.get("max_workers", 2),
        )

    # Libraries (post-creation)
    libs = cfg.get("libraries")
    if libs:
        spec["libraries"] = [
            lib if isinstance(lib, str) else lib.get("pypi", {}).get("package", "")
            for lib in libs
        ]

    return spec


# ======================================================================
# Pipelines (DLT) — stub for future expansion
# ======================================================================

def deploy_pipeline(
    client: "DatabricksClient",
    pipeline_key: str,
    cfg: dict[str, Any],
) -> None:
    """Deploy a DLT pipeline (not yet implemented)."""
    sys.stderr.write(
        f"  Skipping pipeline {pipeline_key!r} — DLT pipeline deploy not yet supported.\n"
    )


# ======================================================================
# Dispatcher
# ======================================================================

#: Maps ``resources.<type>`` keys to their deployer function.
RESOURCE_DEPLOYERS: dict[str, Any] = {
    "jobs": deploy_job,
    "clusters": deploy_cluster,
    "pipelines": deploy_pipeline,
}


def deploy_all_resources(
    client: "DatabricksClient",
    resources: dict[str, Any],
) -> dict[str, list]:
    """Deploy every resource type found in the ``resources`` dict.

    Returns a dict mapping resource type to list of deployed handles.
    """
    deployed: dict[str, list] = {}

    for resource_type, deployer in RESOURCE_DEPLOYERS.items():
        entries = resources.get(resource_type) or {}
        if not entries:
            continue

        deployed[resource_type] = []
        for key, cfg in entries.items():
            result = deployer(client, key, cfg)
            if result is not None:
                deployed[resource_type].append(result)

    return deployed
