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
- :class:`AsyncInsertJob` create-or-update + scheduling + the
  file-arrival trigger against the live Jobs API, plus
  :meth:`Job.run` for the manual trigger path.
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
    "TestAsyncApplierTaskIntegration",
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
    """Per-table :class:`AsyncInsertJob` create-or-update + scheduling +
    file-arrival trigger paths.

    The applier job is keyed off ``(catalog, schema, table)`` — one
    Databricks Job per target table, fired by file arrival in the
    table's async staging ``data/`` folder.
    """

    def test_job_is_keyed_by_table_and_carries_file_arrival_trigger(self):
        from databricks.sdk.service.jobs import (
            FileArrivalTriggerConfiguration,
        )

        table = self._unique_table("async_trig_url")
        table.ensure_created(self._sample_schema())

        job = table.async_job()
        assert isinstance(job, Job)
        assert job.job_id is not None
        type(self).created_jobs.append(job.job_id)
        assert job.name == (
            f"ygg-async-insert-{self.catalog_name}-{self.schema_name}-"
            f"{table.table_name}"
        )

        job.refresh()
        trigger = job.settings.trigger if job.settings else None
        assert trigger is not None
        assert isinstance(trigger.file_arrival, FileArrivalTriggerConfiguration)
        # Trigger points at the table's own async staging data folder.
        assert (
            f"/Volumes/{self.catalog_name}/{self.schema_name}/stg_"
            in trigger.file_arrival.url
        )
        assert trigger.file_arrival.url.endswith("/.sql/async/insert/logs/")

    def test_async_job_with_cron_schedule_is_visible_on_the_job(self):
        from databricks.sdk.service.jobs import PauseStatus

        table = self._unique_table("async_sched")
        table.ensure_created(self._sample_schema())

        job = table.async_job(
            schedule="0 0 */6 * * ?",          # every 6 hours
            schedule_timezone="UTC",
            schedule_pause_status="paused",    # keep it idle in the test
            file_arrival_trigger=False,
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
        ``InvalidParameterValue``, so the default no-task shape is
        not directly triggerable."""
        from databricks.sdk.service.jobs import (
            ConditionTask,
            ConditionTaskOp,
            Task,
        )

        table = self._unique_table("async_trig")
        table.ensure_created(self._sample_schema())

        job = table.async_job(
            task=Task(
                task_key="noop",
                condition_task=ConditionTask(
                    op=ConditionTaskOp.EQUAL_TO,
                    left="1",
                    right="1",
                ),
            ),
            file_arrival_trigger=False,
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


class TestAsyncApplierTaskIntegration(_AsyncWriteIntegrationBase):
    """Stage the applier as a notebook vs. a Spark Python task — live workspace.

    Drives :meth:`Table.async_job` against a real workspace, downloads
    the staged file from the workspace, prints the generated source
    to stdout (visible under ``pytest -s``), and asserts the
    task-spec shape end-to-end:

    * ``task_type="notebook"`` (default) — workspace stores a
      Databricks-format ``.py`` notebook; ``NotebookTask.notebook_path``
      drops the ``.py`` extension; widget reads in the invocation
      cell pull job parameters at run time.
    * ``task_type="spark"`` — workspace stores a flat Python script;
      ``SparkPythonTask.parameters`` carries
      ``{{job.parameters.*}}`` placeholders that Databricks
      substitutes into the script's ``sys.argv`` reads.

    Both shapes also kick a single ``Job.run`` and wait for it to
    reach a terminal state so a regression in the renderer surfaces
    as a real failed run, not just a settings-shape assertion.
    """

    @staticmethod
    def _format_section(title: str, body: str) -> str:
        rule = "=" * 78
        return (
            f"\n{rule}\n"
            f"{title}\n"
            f"{rule}\n"
            f"{body.rstrip()}\n"
            f"{rule}\n"
        )

    def _workspace_read(self, path: str) -> bytes:
        ws = self.client.workspace_client().workspace
        from databricks.sdk.service.workspace import ExportFormat
        # ``ExportFormat.SOURCE`` returns the raw Databricks ``.py``
        # source notebook bytes (with the ``# Databricks notebook
        # source`` magic header + ``# COMMAND ----------`` cell
        # separators) — same shape the staging path uploaded.
        try:
            export = ws.export(path, format=ExportFormat.SOURCE)
        except DatabricksError:
            # Path resolution on notebook objects sometimes needs the
            # extension dropped; retry once on the bare name.
            if path.endswith(".py"):
                export = ws.export(path[:-3], format=ExportFormat.SOURCE)
            else:
                raise
        # Newer SDKs return base64-encoded content under ``.content``;
        # older ones expose a ``.read()`` stream.
        content = getattr(export, "content", None)
        if isinstance(content, str):
            import base64
            return base64.b64decode(content)
        if isinstance(content, (bytes, bytearray)):
            return bytes(content)
        if hasattr(export, "read"):
            return export.read()
        raise AssertionError(
            f"Unexpected workspace.export return shape: {export!r}"
        )

    def test_notebook_applier_is_staged_and_visible_in_workspace(self):
        from databricks.sdk.service.jobs import NotebookTask

        table = self._unique_table("async_nb")
        table.ensure_created(self._sample_schema())

        # ``force=True`` re-stages even if a prior test left a stale
        # job under the same name; lets this test run idempotently.
        job = table.async_job(force=True)  # task_type="notebook" default
        type(self).created_jobs.append(job.job_id)
        job.refresh()

        tasks = (job.settings.tasks if job.settings else None) or []
        assert len(tasks) == 1, f"expected one staged task, got {tasks!r}"
        task = tasks[0]
        assert isinstance(task.notebook_task, NotebookTask), (
            f"task_type=notebook should wire a NotebookTask, got {task!r}"
        )
        assert task.spark_python_task is None
        notebook_path = task.notebook_task.notebook_path
        # Databricks references notebooks without the ``.py`` source
        # extension — the staging path strips it when wiring the task.
        assert notebook_path
        assert not notebook_path.endswith(".py"), notebook_path

        # Download the staged source straight from the workspace and
        # print it so the developer running the suite can eyeball what
        # actually landed (visible under ``pytest -s``).
        body = self._workspace_read(notebook_path).decode()
        print(self._format_section(
            f"Staged NOTEBOOK applier — workspace path: {notebook_path}", body,
        ))

        # Magic header + cell separators must round-trip back through
        # ``workspace.export`` — proves the upload landed as a real
        # notebook, not a workspace file.
        assert body.startswith("# Databricks notebook source\n"), body[:80]
        assert "\n# COMMAND ----------\n" in body
        # Invocation cell reads job parameters via the widget helper.
        assert "def _yggdrasil_widget(name):" in body
        assert "catalog_name=_yggdrasil_widget('catalog_name')" in body
        # Compile sanity — same call shape the Databricks notebook host
        # uses (``exec(compile(f.read(), filename, 'exec'))``).
        compile(body, f"<staged:{notebook_path}>", "exec")

    def test_spark_applier_is_staged_with_job_parameter_substitution(self):
        from databricks.sdk.service.jobs import SparkPythonTask

        table = self._unique_table("async_sp")
        table.ensure_created(self._sample_schema())

        job = table.async_job(force=True, task_type="spark")
        type(self).created_jobs.append(job.job_id)
        job.refresh()

        tasks = (job.settings.tasks if job.settings else None) or []
        assert len(tasks) == 1, f"expected one staged task, got {tasks!r}"
        task = tasks[0]
        assert isinstance(task.spark_python_task, SparkPythonTask), (
            f"task_type=spark should wire a SparkPythonTask, got {task!r}"
        )
        assert task.notebook_task is None
        python_file = task.spark_python_task.python_file
        assert python_file.endswith(".py"), python_file

        # ``parameters`` carries one ``{{job.parameters.<name>}}``
        # placeholder per unbound applier param so Databricks
        # substitutes the Job's ``catalog_name`` / ``schema_name`` /
        # ``table_name`` values into the script's ``sys.argv`` reads.
        assert task.spark_python_task.parameters == [
            "{{job.parameters.catalog_name}}",
            "{{job.parameters.schema_name}}",
            "{{job.parameters.table_name}}",
        ]

        body = self._workspace_read(python_file).decode()
        print(self._format_section(
            f"Staged SPARK applier — workspace path: {python_file}", body,
        ))

        # Script shape: future import, checkargs wrap, sys.argv reader,
        # positional invocation against the parameter list above.
        assert body.startswith("from __future__ import annotations\n")
        assert "from yggdrasil.dataclasses.safe_function import checkargs" in body
        assert "def _yggdrasil_argv(idx):" in body
        assert "catalog_name=_yggdrasil_argv(1)" in body
        assert "schema_name=_yggdrasil_argv(2)" in body
        assert "table_name=_yggdrasil_argv(3)" in body
        compile(body, f"<staged:{python_file}>", "exec")

    def test_applier_task_run_completes_in_terminal_state(self):
        """End-to-end: trigger the notebook applier and wait for terminal.

        The first run on an empty staging folder is a no-op
        (``AsyncInsertJob.load`` returns ``[]`` and the body returns
        early) so the job exits in seconds — long enough to flush any
        renderer-level breakage (NameError at class-body time,
        SyntaxError in the staged cells, missing ``ygg`` dependency
        in the auto-resolved environment) without burning workspace
        budget on a long-running insert.
        """
        table = self._unique_table("async_run")
        table.ensure_created(self._sample_schema())

        job = table.async_job(force=True)
        type(self).created_jobs.append(job.job_id)

        run = job.run()
        assert isinstance(run, JobRun)
        run.wait_for_status(
            wait={"timeout": 600.0, "interval": 5.0},
            raise_error=False,
        )
        assert run.is_terminal, (
            f"Applier run did not reach terminal state in time — "
            f"last status: {run!r}"
        )

        # Print the final run state + any task-level error message so
        # ``pytest -s`` surfaces the diagnosis when this guard catches
        # a renderer regression.
        details = getattr(run, "_details", None) or getattr(run, "details", None)
        print(self._format_section(
            f"Applier run terminal state — run_id={run.run_id}",
            f"state: {getattr(run, 'state', None)!r}\n"
            f"is_success: {getattr(run, 'is_success', None)}\n"
            f"details: {details!r}",
        ))

