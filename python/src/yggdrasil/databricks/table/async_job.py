"""Async, file-arrival-triggered table loader.

A table's ``insert(..., wait=False)`` turns the synchronous warehouse insert
into a *drop-and-aggregate* pipeline:

- :meth:`Table.async_insert` writes the staged Parquet to the table's default
  tmp staging path and drops a small JSON *operation log* at ``.sql/async/logs/``
  that **records where the data was written** (its uniform URL, so the data can
  live anywhere) — no warehouse statement runs at call time.
- A **file-arrival trigger** on the ``logs/`` directory wakes a deployed
  serverless job whose Python entry point (``ygg databricks table
  execute_async_insert --logs <dir>``) drives :meth:`Tables.async_insert`:
  read every pending log, group by ``(target table, mode)``, run one
  aggregated ``INSERT`` per group, then clear the consumed logs + data.

Only ``OVERWRITE`` and ``APPEND`` (no ``match_by``) are supported for now.

:func:`ensure_async_job` get-or-creates the file-arrival job for a table; reach
it via :meth:`yggdrasil.databricks.table.table.Table.async_job`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
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
    "AsyncInsert",
    "stage_async_insert",
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


def _new_op_id() -> str:
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


@dataclass
class AsyncInsert:
    """One async-insert operation — the typed content of an op-log.

    Carries everything the loader needs: the ``target`` table (full name), the
    ``mode``, and the staged ``data``'s **uniform URL**. Owns the op-log
    schema — :meth:`from_log` parses a dropped log and :meth:`to_json`
    serializes one — so the producer and loader share one safe type instead of
    ad-hoc dicts / positional tuples.
    """

    target: str
    mode: str
    data: str
    op_id: str = field(default_factory=_new_op_id)
    ts: float = field(default_factory=time.time)
    #: the on-disk op-log (set when read back) — cleaned up after a load.
    log_file: Any = None

    def __post_init__(self) -> None:
        if Mode.from_(self.mode, default=Mode.APPEND) not in ASYNC_MODES:
            raise ValueError(
                f"async insert supports only OVERWRITE / APPEND, got {self.mode!r}"
            )

    @classmethod
    def from_log(cls, log_file: Any) -> "AsyncInsert":
        """Parse an op-log file into an :class:`AsyncInsert` (keeps *log_file*)."""
        r = json.loads(bytes(log_file.read_bytes()))
        return cls(
            target=r["target"],
            mode=r["mode"],
            data=r["data"],
            op_id=r.get("op_id") or _new_op_id(),
            ts=r.get("ts") or time.time(),
            log_file=log_file,
        )

    def to_json(self) -> bytes:
        """Serialize to the JSON op-log payload."""
        return json.dumps({
            "op_id": self.op_id,
            "target": self.target,
            "mode": self.mode,
            "data": self.data,
            "ts": self.ts,
        }).encode()

    @property
    def group_key(self) -> "tuple[str, str]":
        """``(target, mode)`` — the loader aggregates one ``INSERT`` per group."""
        return (self.target, self.mode)

    def data_path(self, client: Any) -> Any:
        """Reconstruct the staged-data :class:`Path` from its uniform URL."""
        from yggdrasil.databricks.path import DatabricksPath

        return DatabricksPath.from_(self.data, client=client)


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


def stage_async_insert(
    table: "Table",
    data: Any,
    *,
    mode: Any = None,
    match_by: "list[str] | None" = None,
    cast_options: Any = None,
) -> "VolumePath":
    """Stage *data* as Parquet + drop an :class:`AsyncInsert` op-log — no warehouse.

    The producer behind :meth:`Table.async_insert`: write the rows to the
    table's default tmp staging path and drop a JSON op-log under
    :func:`logs_path` recording the staged data's uniform URL (so it can live
    anywhere). A path/URL *string* source is read into Arrow first. Returns the
    op-log path; only ``OVERWRITE`` / ``APPEND`` with no ``match_by``.
    """
    mode_enum = Mode.from_(mode, default=Mode.APPEND)
    if mode_enum not in ASYNC_MODES:
        raise ValueError(
            f"async insert (wait=False) supports only OVERWRITE / APPEND, "
            f"got {mode_enum.name}"
        )
    if match_by:
        raise ValueError("async insert (wait=False) does not support match_by")

    if isinstance(data, str):
        from yggdrasil.io.holder import IO
        data = IO.from_(data).read_arrow_table()

    # Data goes to the default tmp staging path (kept until consumed); the log
    # records its uniform URL so the loader reads it wherever it landed.
    data_file = table.insert_volume_path(table, temporary=False)
    data_file.write_table(data, cast_options, mode=Mode.OVERWRITE)

    op = AsyncInsert(
        target=table.full_name(),
        mode=mode_enum.name.lower(),
        data=data_file.to_url().to_string(),
    )
    log_file = logs_path(table) / f"{op.op_id}.json"
    log_file.write_bytes(op.to_json())
    return log_file


def ensure_async_job(table: "Table", *, client: Any = None) -> Any:
    """Get-or-create the file-arrival loader job for *table*, return the Job.

    Creates the watched ``logs/`` dir, builds + uploads the full ygg wheel
    (:func:`~yggdrasil.databricks.job.wheel.ensure_ygg_wheel` — ygg +
    databricks-sdk), and upserts a serverless job whose single python-wheel
    task runs ``ygg databricks table execute_async_insert --logs <dir>`` when a
    log lands. Any stale job watching the same logs dir is pruned so a single
    job owns the trigger.
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
                    # Run the ygg CLI on the cluster:
                    #   ygg databricks table execute_async_insert --logs <dir>
                    package_name="ygg",
                    entry_point="ygg",
                    parameters=[
                        "databricks", "table", "execute_async_insert",
                        "--logs", logs.full_path(),
                    ],
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
