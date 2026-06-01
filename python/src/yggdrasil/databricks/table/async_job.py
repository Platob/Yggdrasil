"""Async, file-arrival-triggered table loader.

A table's ``insert(..., wait=False)`` turns the synchronous warehouse insert
into a *drop-and-aggregate* pipeline:

- :meth:`Table.async_insert` writes the staged Parquet to the table's default
  tmp staging path and drops a small JSON *operation log* at ``.sql/async/logs/``
  that **records where the data was written** (its uniform URL, so the data can
  live anywhere) — no warehouse statement runs at call time.
- A **file-arrival trigger** on the ``logs/`` directory wakes a deployed
  serverless job whose Python entry point (``ygg-job table-async-load
  <full_name>``) drives :meth:`Tables.async_insert`: read every pending log,
  group by ``(target table, mode)``, run one aggregated ``INSERT`` per group,
  then clear the consumed logs + data.

Only ``OVERWRITE`` and ``APPEND`` (no ``match_by``) are supported for now.

:func:`ensure_async_job` get-or-creates the file-arrival job for a table; reach
it via :meth:`yggdrasil.databricks.table.table.Table.async_job`.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from yggdrasil.enums.mode import Mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.table.table import Table

__all__ = [
    "ASYNC_ROOT",
    "LOGS_SUBDIR",
    "ASYNC_MODES",
    "BUFFER_SECONDS",
    "job_name",
    "logs_path",
    "ensure_async_job",
]

logger = logging.getLogger(__name__)

#: Root (under a table's staging volume) for the async drop pipeline. Only the
#: ``logs/`` directory is fixed — the file-arrival trigger watches it. The
#: staged Parquet lives wherever the producer wrote it (the table's default tmp
#: staging path); each operation log records that location.
ASYNC_ROOT = ".sql/async"
LOGS_SUBDIR = f"{ASYNC_ROOT}/logs"

#: Modes the async path accepts — keyed merges have no aggregation story here.
ASYNC_MODES = (Mode.OVERWRITE, Mode.APPEND)

#: File-arrival buffering window (seconds): wait this long after the last
#: dropped log before firing, and fire at most this often — so a burst of
#: ``async_insert`` drops batches into one aggregated load. Default 2 min.
BUFFER_SECONDS = 120

#: Job-name prefix so the file-arrival loader jobs are easy to spot.
_NAME_PREFIX = "[YGG][ASYNC]"


def job_name(table: "Table") -> str:
    """The deployed loader job's name for *table*."""
    return f"{_NAME_PREFIX} {table.catalog_name}.{table.schema_name}.{table.table_name}"


def logs_path(table: "Table") -> "VolumePath":
    """``<staging_volume>/.sql/async/logs`` — the trigger's watch dir and where
    :meth:`Table.async_insert` drops its operation logs."""
    return table.staging_volume.path(LOGS_SUBDIR)


def _trigger_url(table: "Table") -> str:
    # Databricks requires the file-arrival URL to end with '/'.
    url = logs_path(table).full_path()
    return url if url.endswith("/") else url + "/"


def ensure_async_job(table: "Table", *, client: Any = None) -> Any:
    """Get-or-create the file-arrival loader job for *table*, return the Job.

    Creates the watched ``logs/`` dir, builds + uploads the full ygg wheel
    (:func:`~yggdrasil.databricks.job.wheel.ensure_ygg_wheel` — ygg +
    databricks-sdk), and upserts a serverless job whose single python-wheel
    task runs ``ygg-job table-async-load <full_name>`` when a log lands. Any
    stale job watching the same logs dir is pruned so a single job owns the
    trigger.
    """
    from databricks.sdk.service.compute import Environment
    from databricks.sdk.service.jobs import (
        FileArrivalTriggerConfiguration,
        JobEnvironment,
        PythonWheelTask,
        Task as DBTask,
        TriggerSettings,
    )

    from yggdrasil.databricks.job.skeleton import ensure_console_logging
    from yggdrasil.databricks.job.wheel import ensure_ygg_wheel

    ensure_console_logging()  # surface the deploy CRUD interactively
    client = client or table.client

    # The trigger watches the logs dir — create it first so Databricks accepts
    # the URL (and the first drop lands cleanly).
    logs = logs_path(table)
    logger.info("async job: ensuring logs dir %s", logs.full_path())
    logs.mkdir(parents=True, exist_ok=True)

    wheels = ensure_ygg_wheel(client)

    name = job_name(table)
    logger.info("create-or-update async job %r", name)
    job = client.jobs.create_or_update(
        name=name,
        tasks=[
            DBTask(
                task_key="async-load",
                environment_key="default",
                python_wheel_task=PythonWheelTask(
                    package_name="ygg",
                    entry_point="ygg-job",
                    parameters=["table-async-load", table.full_name()],
                ),
            )
        ],
        environments=[
            JobEnvironment(
                environment_key="default",
                spec=Environment(environment_version="5", dependencies=wheels),
            )
        ],
        trigger=TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(
                url=_trigger_url(table),
                wait_after_last_change_seconds=BUFFER_SECONDS,
                min_time_between_triggers_seconds=BUFFER_SECONDS,
            ),
        ),
    )
    logger.info("deployed async job %r (id=%s)", name, getattr(job, "job_id", None))
    _prune_duplicates(client, _trigger_url(table), keep=getattr(job, "job_id", None))
    return job


def _prune_duplicates(client: Any, url: str, *, keep: Any) -> None:
    """Delete any *other* job whose file-arrival trigger watches the same logs
    dir — orphans left by an earlier naming scheme keep firing on the shared
    trigger (and fail), so the deploy collapses to a single job."""
    try:
        for other in client.jobs.list():
            if other.job_id == keep:
                continue
            trigger = getattr(other.settings, "trigger", None)
            file_arrival = getattr(trigger, "file_arrival", None)
            if file_arrival is not None and file_arrival.url == url:
                try:
                    other.delete()
                    logger.info("removed stale async job %s (%s)", other.job_id, url)
                except Exception:
                    logger.warning("could not delete stale async job %s", other.job_id)
    except Exception:
        logger.debug("stale-async-job prune skipped", exc_info=True)
