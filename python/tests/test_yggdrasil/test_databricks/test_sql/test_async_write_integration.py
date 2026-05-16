"""Live-integration tests for :class:`AsyncInsert` + the applier-job flow.

Skipped unless ``DATABRICKS_HOST`` (and the matching credentials)
are exported via the standard SDK env vars — see
:class:`DatabricksIntegrationCase`.

Scope
-----
The fixture is pinned to ``trading.unittest`` (override via
``DATABRICKS_INTEGRATION_CATALOG`` / ``DATABRICKS_INTEGRATION_SCHEMA``).
Each test creates and tears down its own table and applier job so a
partial failure leaves at most one orphan in the workspace.

These tests exercise:

- :meth:`Table.async_insert` against a real Volume staging path,
- :meth:`AsyncInsert.merge` reading the staged JSON metadata back,
- :meth:`AsyncInsert.to_sql` + :meth:`AsyncInsert.execute` against
  the live SQL engine (data lands in the target table, staged files
  are cleaned up on success),
- :meth:`AsyncInsert.job` + scheduling against the live
  Jobs API, plus :meth:`Job.run` for the trigger path.
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Any, ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.jobs.job import Job
from yggdrasil.databricks.jobs.run import JobRun
from yggdrasil.databricks.sql.engine import SQLEngine
from yggdrasil.databricks.table.async_write import AsyncInsert
from yggdrasil.databricks.table.table import Table
from yggdrasil.io.url import URL

from .. import DatabricksIntegrationCase


__all__ = [
    "TestAsyncWriteIntegration",
    "TestAsyncWriteJobIntegration",
]


class _AsyncWriteIntegrationBase(DatabricksIntegrationCase):
    """Shared fixture for the async-write integration suites."""

    catalog_name: ClassVar[str]
    schema_name: ClassVar[str]
    engine: ClassVar[SQLEngine]
    created_tables: ClassVar[list[str]]
    created_jobs: ClassVar[list[int]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.catalog_name = (
            os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading").strip()
            or "trading"
        )
        cls.schema_name = (
            os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "unittest").strip()
            or "unittest"
        )

        cls.engine = cls.client.sql(
            catalog_name=cls.catalog_name, schema_name=cls.schema_name,
        )
        cls.created_tables = []
        cls.created_jobs = []

        try:
            catalog = cls.engine.catalogs.catalog(cls.catalog_name)
            catalog.ensure_created(comment="yggdrasil async-write integration catalog")
            schema = cls.engine.schemas.schema(
                f"{cls.catalog_name}.{cls.schema_name}",
            )
            schema.ensure_created(comment="yggdrasil async-write integration schema")
        except DatabricksError as exc:
            import unittest as _ut
            raise _ut.SkipTest(
                f"Cannot create or access {cls.catalog_name}.{cls.schema_name}: "
                f"{exc}. Set DATABRICKS_INTEGRATION_CATALOG / "
                "DATABRICKS_INTEGRATION_SCHEMA to a location the test identity "
                "can write to."
            ) from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            for full_name in cls.created_tables:
                try:
                    cls.engine.table(full_name).delete(missing_ok=True)
                except DatabricksError:
                    pass
            for job_id in cls.created_jobs:
                try:
                    cls.client.jobs.find(job_id=job_id).delete()
                except Exception:
                    pass
        finally:
            super().tearDownClass()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _unique_table(self, prefix: str = "async") -> Table:
        name = f"yg_{prefix}_{secrets.token_hex(4)}"
        full_name = f"{self.catalog_name}.{self.schema_name}.{name}"
        type(self).created_tables.append(full_name)
        return self.engine.table(full_name)

    @staticmethod
    def _sample_schema() -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.int64(), nullable=False),
            pa.field("label", pa.string()),
            pa.field("amount", pa.float64()),
        ])

    @staticmethod
    def _batch(ids: list[int], label: str = "x", amount: float = 1.0) -> pa.Table:
        return pa.table({
            "id": pa.array(ids, type=pa.int64()),
            "label": pa.array([label] * len(ids), type=pa.string()),
            "amount": pa.array([amount] * len(ids), type=pa.float64()),
        })

    @staticmethod
    def _record_for(table: Table) -> AsyncInsert:
        """Build a minimal :class:`AsyncInsert` carrying *table*'s schema identity.

        The applier-job test cases don't need any staged payload — they
        only need a record whose ``target_catalog_name`` /
        ``target_schema_name`` match the live table so
        :meth:`AsyncInsert.job` resolves the right schema.
        """
        return AsyncInsert(
            target_full_name=table.full_name(safe=False),
            target_catalog_name=table.catalog_name,
            target_schema_name=table.schema_name,
            target_table_name=table.table_name,
        )

    def _volume_path(self, entry: Any) -> VolumePath:
        """Return a :class:`VolumePath` for an :class:`AsyncInsert`
        ``parquet_paths`` / ``metadata_paths`` entry.

        :func:`stage_async_insert` now carries the live
        :class:`VolumePath` on the record — pass it through. Records
        loaded from disk metadata still hand back strings (URL form
        ``/Volumes/cat/sch/...``); coerce those into a path bound to
        the test client.
        """
        if isinstance(entry, VolumePath):
            return entry
        inner = entry.removeprefix("/Volumes") or "/"
        return VolumePath(
            url=URL(scheme=VolumePath.scheme, path=inner),
            client=self.client,
        )


class TestAsyncWriteIntegration(_AsyncWriteIntegrationBase):
    """End-to-end stage → merge → execute flow against a real workspace."""

    def test_async_insert_round_trip(self):
        table = self._unique_table("async_rt")
        table.ensure_created(self._sample_schema())

        # Stage two independent async inserts on the same target.
        first = table.async_insert(self._batch([1, 2], label="a"))
        second = table.async_insert(self._batch([3, 4], label="b"))
        first_parquet = self._volume_path(first.parquet_paths[0])
        second_parquet = self._volume_path(second.parquet_paths[0])
        assert first_parquet.exists()
        assert second_parquet.exists()

        # Both metadata records list the same target.
        folder = table.staging_folder(temporary=False, async_write=True)
        records = AsyncInsert.merge(folder)
        assert len(records) == 1
        merged = records[0]
        assert merged.target_full_name.replace("`", "") == table.full_name(safe=False).replace("`", "")
        assert len(merged.parquet_paths) == 2

        # Apply the merged record.
        result = merged.execute(self.engine, wait=True, raise_error=True)
        assert result is not None

        # Target table now carries every staged row.
        rows = self.engine.execute(
            f"SELECT count(*) AS n FROM {table.full_name(safe=True)}"
        ).to_pylist()
        assert rows[0]["n"] == 4

        # Cleanup happened — staged Parquet + metadata files are gone.
        # ``WarehouseStatementBatch.clear_temporary_resources`` fires
        # unlinks via ``Job.make(...).fire_and_forget()`` and returns
        # before the ThreadJobs complete, so ``execute(wait=True)`` can
        # land before cleanup finishes — poll instead of asserting.
        first_parquet.wait_until_gone({"timeout": 30.0, "interval": 0.2})
        second_parquet.wait_until_gone({"timeout": 30.0, "interval": 0.2})

    def test_overwrite_drops_earlier_appends(self):
        table = self._unique_table("async_ovw")
        table.ensure_created(self._sample_schema())

        # Seed the target with one append.
        self.engine.execute(
            f"INSERT INTO {table.full_name(safe=True)} VALUES (1, 'seed', 1.0)"
        )

        # Stage an append, then an overwrite — the overwrite drops the
        # append's data from the SQL projection.
        table.async_insert(self._batch([10, 11], label="appended"), mode="append")
        # Tiny sleep so the created_at ordering is unambiguous when the
        # second op is staged in the same millisecond bucket.
        time.sleep(0.05)
        table.async_insert(
            self._batch([100, 101], label="overwritten"), mode="overwrite",
        )

        records = AsyncInsert.merge(
            table.staging_folder(temporary=False, async_write=True),
        )
        assert len(records) == 1
        merged = records[0]
        assert merged.is_overwrite

        merged.execute(self.engine, wait=True, raise_error=True)

        # Only the overwrite payload survives — the seed and the
        # earlier appended rows are gone.
        rows = self.engine.execute(
            f"SELECT id FROM {table.full_name(safe=True)} ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in rows] == [100, 101]


class TestAsyncWriteJobIntegration(_AsyncWriteIntegrationBase):
    """Schema-level applier-job find-or-create + scheduling + trigger paths.

    The applier job is keyed off ``(catalog, schema)`` — every table
    in the same schema is drained by the same job.
    """

    def test_ensure_job_is_keyed_by_schema_not_by_table(self):
        """Two tables in the same schema → one shared applier job."""
        table_a = self._unique_table("async_share_a")
        table_a.ensure_created(self._sample_schema())
        table_b = self._unique_table("async_share_b")
        table_b.ensure_created(self._sample_schema())

        first = self._record_for(table_a).job()
        assert isinstance(first, Job)
        assert first.job_id is not None
        type(self).created_jobs.append(first.job_id)

        second = self._record_for(table_b).job()
        assert second.job_id == first.job_id
        assert (
            first.job_name
            == f"ygg-async-insert-{self.catalog_name}-{self.schema_name}"
        )

    def test_ensure_job_with_cron_schedule_is_visible_on_the_job(self):
        from databricks.sdk.service.jobs import PauseStatus

        table = self._unique_table("async_sched")
        table.ensure_created(self._sample_schema())

        job = self._record_for(table).job(
            schedule="0 0 */6 * * ?",          # every 6 hours
            schedule_timezone="UTC",
            schedule_pause_status="paused",    # keep it idle in the test
        )
        type(self).created_jobs.append(job.job_id)

        job.refresh()
        settings = job.settings
        assert settings is not None
        schedule = settings.schedule
        assert schedule is not None
        assert schedule.quartz_cron_expression == "0 0 */6 * * ?"
        assert schedule.timezone_id == "UTC"
        assert schedule.pause_status == PauseStatus.PAUSED

    def test_run_now_returns_job_run(self):
        """Trigger the job via :meth:`Job.run`. We attach a no-op
        ``condition_task`` (``1 == 1``) so the run terminates fast —
        Databricks rejects ``run_now`` on an empty-task job with
        ``InvalidParameterValue``, so the default ``record.job()`` shape
        is not directly triggerable."""
        from databricks.sdk.service.jobs import (
            ConditionTask,
            ConditionTaskOp,
            Task,
        )

        table = self._unique_table("async_trig")
        table.ensure_created(self._sample_schema())

        job = self._record_for(table).job(
            task=Task(
                task_key="noop",
                condition_task=ConditionTask(
                    op=ConditionTaskOp.EQUAL_TO,
                    left="1",
                    right="1",
                ),
            ),
        )
        if job.job_id not in type(self).created_jobs:
            type(self).created_jobs.append(job.job_id)

        run = job.run()
        assert isinstance(run, JobRun)
        assert run.run_id is not None

        # Let it land in a terminal state. Empty-task jobs typically
        # finish in seconds, but we cap the wait so a slow workspace
        # doesn't hang the suite.
        run.wait_for_status(
            wait={"timeout": 120.0, "interval": 2.0},
            raise_error=False,
        )
        assert run.is_terminal

    def test_apply_schema_drains_multiple_tables(self):
        """Two tables in the same schema, each with a staged insert →
        :meth:`AsyncInsert.apply_schema` drains both in one call."""
        table_a = self._unique_table("apply_a")
        table_a.ensure_created(self._sample_schema())
        table_b = self._unique_table("apply_b")
        table_b.ensure_created(self._sample_schema())

        table_a.async_insert(self._batch([1, 2], label="a"))
        table_b.async_insert(self._batch([10, 20], label="b"))

        AsyncInsert.apply_schema(
            self.engine,
            self.catalog_name,
            self.schema_name,
            client=self.client,
            wait=True,
            raise_error=True,
        )

        rows_a = self.engine.execute(
            f"SELECT id FROM {table_a.full_name(safe=True)} ORDER BY id"
        ).to_pylist()
        rows_b = self.engine.execute(
            f"SELECT id FROM {table_b.full_name(safe=True)} ORDER BY id"
        ).to_pylist()
        assert [r["id"] for r in rows_a] == [1, 2]
        assert [r["id"] for r in rows_b] == [10, 20]
