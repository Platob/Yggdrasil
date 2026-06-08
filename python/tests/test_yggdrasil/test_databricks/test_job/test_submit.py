"""Tests for one-time run submission via :meth:`Jobs.submit`.

Exercises the production wiring end-to-end against the mocked SDK boundary:
task coercion (dict → SubmitTask), default-cluster backfill, and that the
returned handle is an awaitable :class:`JobRun` carrying the run's
``run_id`` and ``run_page_url`` — fetched right after submission so logs
deep-link to the run page rather than the vanity-host jobs list.  One-time
runs are not backed by a persisted job, so the handle has no ``job_id``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import NotebookTask, SubmitTask

from yggdrasil.databricks.job.run import JobRun
from yggdrasil.databricks.tests import DatabricksTestCase


class TestJobsSubmit(DatabricksTestCase):

    def setUp(self) -> None:
        super().setUp()
        # Force a mocked workspace client regardless of the environment: when
        # DATABRICKS_HOST is set, the base case hands back a *real* client, but
        # this is a pure unit test of the submit wiring against the SDK boundary.
        self.workspace_client = MagicMock(spec=WorkspaceClient)
        object.__setattr__(self.client, "_workspace_client", self.workspace_client)
        JobRun._INSTANCES.clear()
        self.workspace_client.jobs.submit.return_value.run_id = 555
        # submit() fetches the run right after submission so the handle
        # carries its canonical run_page_url for repr/logs.
        got = self.workspace_client.jobs.get_run.return_value
        got.run_id = 555
        got.job_id = None
        got.run_page_url = "https://dbc-test.cloud.databricks.com/jobs/900/runs/555"

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
        self.workspace_client.jobs.get_run.assert_called_once_with(run_id=555)
        self.assertIsInstance(run, JobRun)
        self.assertEqual(run.run_id, 555)
        self.assertIsNone(run.job_id)
        self.assertEqual(
            run.explore_url.to_string(),
            "https://dbc-test.cloud.databricks.com/jobs/900/runs/555",
        )

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

    def test_submit_no_environment_by_default(self):
        # Backward compatible: without an explicit ``environment``, submit
        # attaches nothing (no auto-resolution on the generic path).
        self.client.jobs.submit(tasks=[SubmitTask(task_key="t")])
        _, kwargs = self.workspace_client.jobs.submit.call_args
        self.assertIsNone(kwargs["environments"])
        self.assertIsNone(kwargs["tasks"][0].environment_key)

    def test_submit_attaches_explicit_job_environment(self):
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        env = JobEnvironment(
            environment_key="meteologica",
            spec=Environment(environment_version="5", dependencies=["ygg"]),
        )
        self.client.jobs.submit(
            tasks=[SubmitTask(task_key="t", notebook_task=NotebookTask(notebook_path="/x"))],
            environment=env,
        )
        _, kwargs = self.workspace_client.jobs.submit.call_args
        self.assertEqual(kwargs["environments"], [env])
        # Key backfilled onto the cluster-less, key-less task.
        self.assertEqual(kwargs["tasks"][0].environment_key, "meteologica")

    def test_submit_pinned_cluster_skips_environment_backfill(self):
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        env = JobEnvironment(
            environment_key="e", spec=Environment(environment_version="5"),
        )
        self.client.jobs.submit(
            tasks=[SubmitTask(task_key="t", existing_cluster_id="pinned")],
            environment=env,
        )
        _, kwargs = self.workspace_client.jobs.submit.call_args
        # The env is still attached, but a cluster-pinned task isn't rekeyed.
        self.assertEqual(kwargs["environments"], [env])
        self.assertIsNone(kwargs["tasks"][0].environment_key)
