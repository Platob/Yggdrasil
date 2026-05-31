"""Async, file-arrival-triggered table loader.

:class:`TableJob` is a :class:`~yggdrasil.databricks.job.JobSkeleton` (a
declarative Python-job definition) that turns a table's synchronous warehouse
insert into a *drop-and-aggregate* pipeline:

- ``table.insert(..., wait=False)`` writes the staged Parquet under the table's
  staging volume at ``.sql/async/data/`` and drops a small JSON *operation log*
  next to it at ``.sql/async/logs/`` — no warehouse statement runs at call time.
- A **file-arrival trigger** on the ``logs/`` directory wakes the deployed job,
  whose Python entry point calls :meth:`TableJob.run`: read every pending log,
  group the operations by ``(target table, mode)``, build one aggregated
  ``INSERT`` per group that reads all the staged Parquet at once, run it through
  ygg (:meth:`Table.insert`), then clear the consumed logs + data.

Only ``OVERWRITE`` and ``APPEND`` (no ``match_by``) are supported for now.

Reach it lazily via :attr:`yggdrasil.databricks.table.table.Table.async_job`,
which get-or-creates the live :class:`~yggdrasil.databricks.job.Job` from this
skeleton's :meth:`definition`.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from yggdrasil.databricks.job.skeleton import JobSkeleton
from yggdrasil.enums.mode import Mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.table.table import Table

__all__ = ["TableJob", "ASYNC_ROOT", "LOGS_SUBDIR", "DATA_SUBDIR", "ASYNC_MODES"]

logger = logging.getLogger(__name__)

#: Root (under a table's staging volume) for the async drop pipeline, plus the
#: two sibling directories it splits into — ``logs/`` (the JSON operation logs
#: the file-arrival trigger watches) and ``data/`` (the staged Parquet).
ASYNC_ROOT = ".sql/async"
LOGS_SUBDIR = f"{ASYNC_ROOT}/logs"
DATA_SUBDIR = f"{ASYNC_ROOT}/data"

#: Modes the async path accepts — keyed merges have no aggregation story here.
ASYNC_MODES = (Mode.OVERWRITE, Mode.APPEND)


class TableJob(JobSkeleton):
    """File-arrival job skeleton that aggregates a table's async inserts.

    Bound to a single :class:`Table` (its ``.sql/async`` area). Build via
    ``TableJob(table)``; :meth:`ensure` / :attr:`job` get-or-create the live
    Databricks job from :meth:`definition`, and :meth:`run` is the loader the
    deployed task executes. Usually reached through :attr:`Table.async_job`.
    """

    entry_point: ClassVar[str] = "ygg-table-async-load"
    task_key: ClassVar[str] = "async-load"
    _NAME_PREFIX: ClassVar[str] = "ygg-async-insert"

    def __init__(self, table: "Table") -> None:
        self._table = table
        self._job: "Job | None" = None

    # -- identity / paths -----------------------------------------------
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

    # -- JobSkeleton definition surface ---------------------------------
    @property
    def name(self) -> str:
        return self.job_name(self._table)

    def parameters(self) -> list[str]:
        return [self._table.full_name()]

    def trigger(self) -> Any:
        from databricks.sdk.service.jobs import (
            FileArrivalTriggerConfiguration,
            TriggerSettings,
        )

        return TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(
                url=self.logs_path(self._table).full_path(),
            ),
        )

    # -- get-or-create the live Job -------------------------------------
    def ensure(self) -> "TableJob":
        """Get-or-create the underlying Databricks job from :meth:`definition`."""
        if self._job is None:
            self._job = self.deploy(self._table.client.jobs)
        return self

    @property
    def job(self) -> "Job":
        """The live :class:`Job` (deployed on first access)."""
        return self.ensure()._job

    # -- the loader the trigger runs ------------------------------------
    def run(self, *, wait: Any = True, limit: Optional[int] = None) -> int:
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

    #: Descriptive alias for :meth:`run` (the JobSkeleton entry point).
    process = run

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
