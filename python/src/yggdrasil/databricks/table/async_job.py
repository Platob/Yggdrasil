"""Async, file-arrival-triggered table loader.

:class:`TableJob` turns a table's synchronous warehouse insert into a
*drop-and-aggregate* pipeline:

- ``table.insert(..., wait=False)`` writes the staged Parquet under the table's
  staging volume at ``.sql/async/data/`` and drops a small JSON *operation log*
  next to it at ``.sql/async/logs/`` — no warehouse statement runs at call time.
- A **file-arrival trigger** on the ``logs/`` directory wakes this job, which
  reads every pending log, groups the operations by ``(target table, mode)``,
  builds one aggregated ``INSERT`` per group that reads all the staged Parquet
  at once, runs it through ygg (:meth:`Table.insert`), then clears the consumed
  logs + data.

Only ``OVERWRITE`` and ``APPEND`` (no ``match_by``) are supported for now.

Reach it lazily via :attr:`yggdrasil.databricks.table.table.Table.async_job`,
which get-or-creates the job from :meth:`TableJob.definition`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.databricks.job.job import Job
from yggdrasil.databricks.resource import DatabricksResource
from yggdrasil.enums.mode import Mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.table.table import Table

__all__ = ["TableJob", "ASYNC_ROOT", "LOGS_SUBDIR", "DATA_SUBDIR"]

logger = logging.getLogger(__name__)

#: Root (under a table's staging volume) for the async drop pipeline, plus the
#: two sibling directories it splits into — ``logs/`` (the JSON operation logs
#: the file-arrival trigger watches) and ``data/`` (the staged Parquet).
ASYNC_ROOT = ".sql/async"
LOGS_SUBDIR = f"{ASYNC_ROOT}/logs"
DATA_SUBDIR = f"{ASYNC_ROOT}/data"

#: Modes the async path accepts — keyed merges have no aggregation story here.
ASYNC_MODES = (Mode.OVERWRITE, Mode.APPEND)


class TableJob(Job):
    """A file-arrival-triggered Databricks job that aggregates async inserts.

    Bound to a single :class:`Table` (its ``.sql/async`` area). Construct via
    ``TableJob(table)`` and :meth:`ensure` it into existence, or reach the
    cached handle through :attr:`Table.async_job`.
    """

    _NAME_PREFIX: ClassVar[str] = "ygg-async-insert"

    # -- paths ----------------------------------------------------------
    @staticmethod
    def job_name(table: "Table") -> str:
        return f"{TableJob._NAME_PREFIX}-{table.catalog_name}.{table.schema_name}.{table.table_name}"

    @staticmethod
    def logs_path(table: "Table") -> "VolumePath":
        """``<staging_volume>/.sql/async/logs`` — the trigger's watch dir."""
        return table.staging_volume.path(LOGS_SUBDIR)

    @staticmethod
    def data_path(table: "Table") -> "VolumePath":
        """``<staging_volume>/.sql/async/data`` — the staged Parquet."""
        return table.staging_volume.path(DATA_SUBDIR)

    # -- singleton identity / construction ------------------------------
    @classmethod
    def _singleton_key(
        cls,
        table: "Table | None" = None,
        *,
        service: Any = None,
        job_id: "int | str | None" = None,
        name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        if name is None and table is not None:
            name = cls.job_name(table)
        svc = service if service is not None else (table.client.jobs if table is not None else None)
        return (cls, svc, job_id if job_id is not None else name)

    def __init__(
        self,
        table: "Table | None" = None,
        *,
        service: Any = None,
        job_id: "int | str | None" = None,
        name: str | None = None,
        details: Any = None,
        singleton_ttl: Any = ...,
    ) -> None:
        del singleton_ttl
        if getattr(self, "_initialized", False):
            return
        if name is None and table is not None:
            name = self.job_name(table)
        if service is None and table is not None:
            service = table.client.jobs
        # Bypass ``Job.__init__``'s auto-lookup-by-name: the async job may not
        # exist yet (it's get-or-created lazily in :meth:`ensure`).
        DatabricksResource.__init__(self, service=service)
        self.service = service
        self.job_id = int(job_id) if isinstance(job_id, (int, str)) and str(job_id).isdigit() else None
        self.name = name
        self._details = details
        self._table = table
        self._initialized = True

    # -- definition + get-or-create -------------------------------------
    def definition(self) -> dict:
        """JobSettings kwargs for :meth:`Jobs.create_or_update`.

        A single ``async-load`` task (the ygg loader, parameterised with the
        table's full name) plus a **file-arrival trigger** on the ``logs/``
        directory, so a dropped operation log wakes the job.
        """
        from databricks.sdk.service.jobs import (
            FileArrivalTriggerConfiguration,
            PythonWheelTask,
            Task,
            TriggerSettings,
        )

        table = self._table
        task = Task(
            task_key="async-load",
            python_wheel_task=PythonWheelTask(
                package_name="yggdrasil",
                entry_point="table-async-load",
                parameters=[table.full_name()],
            ),
        )
        return dict(
            name=self.name,
            tasks=[task],
            trigger=TriggerSettings(
                file_arrival=FileArrivalTriggerConfiguration(
                    url=self.logs_path(table).full_path(),
                ),
            ),
        )

    def ensure(self) -> "TableJob":
        """Get-or-create the underlying Databricks job from :meth:`definition`."""
        if self.job_id is not None:
            return self
        spec = self.definition()
        job = self.service.create_or_update(
            name=spec.pop("name"),
            **spec,
        )
        self.job_id = job.job_id
        self._details = job._details
        return self

    # -- the loader the trigger runs ------------------------------------
    def process(self, *, wait: Any = True, limit: Optional[int] = None) -> int:
        """Consume pending operation logs and load them into their targets.

        Reads every JSON log under :meth:`logs_path`, groups by
        ``(target, mode)``, builds one aggregated ``INSERT`` per group that
        reads all the group's staged Parquet at once (``parquet.`…` UNION
        ALL …``), runs it through ygg, then deletes the consumed logs + data.
        Returns the number of operations processed.
        """
        table = self._table
        logs_dir = self.logs_path(table)
        data_dir = self.data_path(table)
        if not logs_dir.exists():
            return 0

        # Parse pending logs into (target, mode, data-leaf, log-path).
        ops: list[tuple[str, str, str, Any]] = []
        for log_file in logs_dir.iterdir():
            if not str(log_file.name).endswith(".json"):
                continue
            try:
                record = json.loads(bytes(log_file.read_bytes()))
            except Exception:
                logger.warning("skipping unreadable async log %s", log_file)
                continue
            ops.append((record["target"], record["mode"], record["data"], log_file))
            if limit is not None and len(ops) >= limit:
                break
        if not ops:
            return 0

        groups: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        for target, mode, data, log_file in ops:
            groups.setdefault((target, mode), []).append((data, log_file))

        processed = 0
        for (target_name, mode), items in groups.items():
            target = self._resolve_target(target_name)
            union = " UNION ALL ".join(
                f"SELECT * FROM parquet.`{(data_dir / leaf).full_path()}`"
                for leaf, _ in items
            )
            target.insert(union, mode=mode, wait=wait)
            # Clear consumed logs + data only after a successful load.
            for leaf, log_file in items:
                _best_effort_unlink(log_file)
                _best_effort_unlink(data_dir / leaf)
            processed += len(items)
        return processed

    def _resolve_target(self, full_name: str) -> "Table":
        table = self._table
        if table is not None and table.full_name() == full_name:
            return table
        return table.service[full_name]


def _best_effort_unlink(path: Any) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("async cleanup: failed to remove %s", path, exc_info=True)
