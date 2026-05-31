"""Async, file-arrival-triggered table loader.

:class:`TableJob` is a :class:`~yggdrasil.databricks.job.JobSkeleton` (a
declarative Python-job definition) that turns a table's synchronous warehouse
insert into a *drop-and-aggregate* pipeline:

- ``table.insert(..., wait=False)`` writes the staged Parquet to the table's
  default tmp staging path and drops a small JSON *operation log* at
  ``.sql/async/logs/`` that **records where the data was written** (so the data
  can live anywhere) — no warehouse statement runs at call time.
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

from yggdrasil.databricks.job.skeleton import Flow
from yggdrasil.enums.mode import Mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.fs.volume_path import VolumePath
    from yggdrasil.databricks.job.job import Job
    from yggdrasil.databricks.table.table import Table

__all__ = ["TableJob", "ASYNC_ROOT", "LOGS_SUBDIR", "ASYNC_MODES"]

logger = logging.getLogger(__name__)

#: Root (under a table's staging volume) for the async drop pipeline. Only the
#: ``logs/`` directory is fixed — the file-arrival trigger watches it. The
#: staged Parquet lives wherever the producer wrote it (the table's default tmp
#: staging path); each operation log records that location.
ASYNC_ROOT = ".sql/async"
LOGS_SUBDIR = f"{ASYNC_ROOT}/logs"

#: Modes the async path accepts — keyed merges have no aggregation story here.
ASYNC_MODES = (Mode.OVERWRITE, Mode.APPEND)


class TableJob(Flow):
    """File-arrival :class:`~yggdrasil.databricks.job.Flow` that aggregates a
    table's async inserts.

    A single-task serverless flow bound to one :class:`Table` (its ``.sql/async``
    area). Build via ``TableJob(table)``; :meth:`ensure` / :attr:`job`
    get-or-create the live Databricks job from :meth:`definition`, and
    :meth:`run` (the flow body, callable via ``TableJob(table)()``) is the
    loader the deployed task executes. Usually reached through
    :attr:`Table.async_job`.
    """

    entry_point: ClassVar[str] = "ygg-table-async-load"
    task_key: ClassVar[str] = "async-load"
    _NAME_PREFIX: ClassVar[str] = "ygg-async-insert"

    def __init__(self, table: "Table") -> None:
        super().__init__(name=self.job_name(table))
        self.table = table
        self._job: "Job | None" = None

    # -- identity / paths -----------------------------------------------
    @staticmethod
    def job_name(table: "Table") -> str:
        return f"{TableJob._NAME_PREFIX}-{table.catalog_name}.{table.schema_name}.{table.table_name}"

    @staticmethod
    def logs_path(table: "Table") -> "VolumePath":
        """``<staging_volume>/.sql/async/logs`` — the trigger's watch dir."""
        return table.staging_volume.path(LOGS_SUBDIR)

    # -- Flow deploy surface (name set in __init__) ---------------------
    def parameters(self) -> list[str]:
        # The wheel param is the table's full name, not the live handle.
        return [self.table.full_name()]

    def trigger(self) -> Any:
        from databricks.sdk.service.jobs import (
            FileArrivalTriggerConfiguration,
            TriggerSettings,
        )

        return TriggerSettings(
            file_arrival=FileArrivalTriggerConfiguration(
                url=self.logs_path(self.table).full_path(),
            ),
        )

    # -- get-or-create the live Job -------------------------------------
    def ensure(self) -> "TableJob":
        """Get-or-create the underlying Databricks job from :meth:`definition`."""
        if self._job is None:
            self._job = self.deploy(self.table.client.jobs)
        return self

    @property
    def job(self) -> "Job":
        """The live :class:`Job` (deployed on first access)."""
        return self.ensure()._job

    # -- the loader the trigger runs (the job body) ---------------------
    def run(self, *, wait: Any = True, limit: Optional[int] = None) -> int:
        """Consume pending operation logs and load them into their targets.

        Reads every JSON log under :meth:`logs_path`, groups by
        ``(target, mode)``, builds one aggregated ``INSERT`` per group that
        reads all the group's staged Parquet at once (``parquet.`…` UNION
        ALL …``), runs it through ygg, then deletes the consumed logs + data.
        Returns the number of operations processed.
        """
        table = self.table
        logs_dir = self.logs_path(table)
        if not logs_dir.exists():
            return 0

        # Parse pending logs into (target, mode, data-path, log-path).
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
                f"SELECT * FROM parquet.`{data}`" for data, _ in items
            )
            target.insert(union, mode=mode, wait=wait)
            # Clear consumed logs + data only after a successful load.
            for data, log_file in items:
                _best_effort_unlink(log_file)
                _best_effort_unlink(self._data_file(data))
            processed += len(items)
        return processed

    #: Descriptive alias for :meth:`run` — calling the skeleton runs it too.
    def process(self, *, wait: Any = True, limit: Optional[int] = None) -> int:
        return self.run(wait=wait, limit=limit)

    def _resolve_target(self, full_name: str) -> "Table":
        table = self.table
        if table is not None and table.full_name() == full_name:
            return table
        return table.service[full_name]

    def _data_file(self, path: str) -> Any:
        """Reconstruct the staged-Parquet :class:`Path` from its logged path."""
        from yggdrasil.databricks.path import DatabricksPath

        return DatabricksPath.from_(path, client=self.table.client)


def _best_effort_unlink(path: Any) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("async cleanup: failed to remove %s", path, exc_info=True)
