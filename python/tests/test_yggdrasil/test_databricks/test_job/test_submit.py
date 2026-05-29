"""Tests for one-time run submission via :meth:`Jobs.submit`.

Exercises the production wiring end-to-end against the mocked SDK boundary:
task coercion (dict → SubmitTask), default-cluster backfill, and that the
returned handle is an awaitable :class:`JobRun` with a ``run_id`` but no
``job_id`` (one-time runs are not backed by a persisted job).
"""
from __future__ import annotations

from databricks.sdk.service.jobs import NotebookTask, SubmitTask

from yggdrasil.databricks.job.run import JobRun
from yggdrasil.databricks.tests import DatabricksTestCase


class TestJobsSubmit(DatabricksTestCase):

    def setUp(self) -> None:
        super().setUp()
        JobRun._INSTANCES.clear()
        self.workspace_client.jobs.submit.return_value.run_id = 555

    def tearDown(self) -> None:
        JobRun._INSTANCES.clear()
        super().tearDown()

    def test_submit_calls_sdk_and_returns_run(self):
        run = self.client.jobs.submit(
            run_name="one-shot",
            tasks=[
                SubmitTask(
                    task_key="ingest",
                    existing_cluster_id="cluster-1",
                    notebook_task=NotebookTask(notebook_path="/Repos/x/ingest"),
                )
            ],
        )
        self.workspace_client.jobs.submit.assert_called_once()
        self.assertIsInstance(run, JobRun)
        self.assertEqual(run.run_id, 555)
        self.assertIsNone(run.job_id)

    def test_submit_coerces_dict_tasks(self):
        self.client.jobs.submit(
            run_name="from-dict",
            tasks=[{"task_key": "t", "existing_cluster_id": "c"}],
        )
        _, kwargs = self.workspace_client.jobs.submit.call_args
        tasks = kwargs["tasks"]
        self.assertEqual(len(tasks), 1)
        self.assertIsInstance(tasks[0], SubmitTask)
        self.assertEqual(tasks[0].task_key, "t")

    def test_submit_backfills_default_cluster(self):
        self.client.jobs.submit(
            run_name="defaulted",
            tasks=[SubmitTask(task_key="t")],
            cluster="cluster-default",
        )
        _, kwargs = self.workspace_client.jobs.submit.call_args
        self.assertEqual(kwargs["tasks"][0].existing_cluster_id, "cluster-default")

    def test_submit_does_not_override_pinned_cluster(self):
        self.client.jobs.submit(
            tasks=[SubmitTask(task_key="t", existing_cluster_id="pinned")],
            cluster="cluster-default",
        )
        _, kwargs = self.workspace_client.jobs.submit.call_args
        self.assertEqual(kwargs["tasks"][0].existing_cluster_id, "pinned")

    def test_submit_rejects_bad_task_type(self):
        with self.assertRaises(TypeError):
            self.client.jobs.submit(tasks=[object()])
