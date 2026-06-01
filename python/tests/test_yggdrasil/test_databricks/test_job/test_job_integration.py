"""Live-integration smoke test for the Databricks jobs orchestration surface.

Validates the whole chain against a real workspace, end to end:

- :meth:`Jobs.submit` runs a one-time **multi-task DAG** (two notebook tasks,
  ``b`` depends on ``a``) on serverless;
- the returned :class:`JobRun` is awaitable — ``run.wait()`` blocks to terminal
  success;
- :meth:`JobRun.dag` reflects the task graph + live per-task state, and
  :meth:`JobRun.task` hands back an awaitable :class:`JobTask` whose ``wait()``
  resolves to a terminal state;
- a persisted multi-task :class:`Job` exposes the same static DAG via
  :meth:`Job.dag` and runs via :meth:`Job.run` (awaitable).

Skipped wholesale unless ``DATABRICKS_HOST`` is set. Each test provisions a
throw-away notebook + (for the persisted case) a throw-away job, both cleaned up
in teardown. Permission / serverless-availability failures degrade to
``unittest.SkipTest`` rather than failing the suite.
"""
from __future__ import annotations

import base64
import secrets
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied
from databricks.sdk.service.jobs import NotebookTask, SubmitTask, Task, TaskDependency
from databricks.sdk.service.workspace import ImportFormat, Language

from yggdrasil.databricks.job.dag import JobDag
from yggdrasil.databricks.job.run import JobRun, JobTask
from yggdrasil.enums.state import State

from .. import DatabricksIntegrationCase

_SMOKE_DIR = "/Workspace/Shared/.ygg/_smoke"
# Generous because serverless cold-start dominates a trivial notebook run.
_WAIT_SECONDS = 360


class TestJobsOrchestrationIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.notebook = f"{_SMOKE_DIR}/nb_{secrets.token_hex(4)}"
        cls.failing_notebook = f"{_SMOKE_DIR}/nb_fail_{secrets.token_hex(4)}"
        cls.param_notebook = f"{_SMOKE_DIR}/nb_param_{secrets.token_hex(4)}"
        # Reads a run parameter via a widget and returns it as the task output —
        # the standard "parameters in, result out" notebook contract.
        param_src = (
            'dbutils.widgets.text("msg", "default")\n'
            'dbutils.notebook.exit(dbutils.widgets.get("msg"))\n'
        )
        w = cls.client.workspace_client()
        try:
            w.workspace.mkdirs(path=_SMOKE_DIR)
            w.workspace.import_(
                path=cls.notebook,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                content=base64.b64encode(b'print("ygg jobs smoke ok")\n').decode(),
                overwrite=True,
            )
            w.workspace.import_(
                path=cls.failing_notebook,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                content=base64.b64encode(b'raise Exception("boom-xyz")\n').decode(),
                overwrite=True,
            )
            w.workspace.import_(
                path=cls.param_notebook,
                format=ImportFormat.SOURCE,
                language=Language.PYTHON,
                content=base64.b64encode(param_src.encode()).decode(),
                overwrite=True,
            )
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot provision smoke notebook: {exc}") from exc
        cls._jobs_to_delete: list[int] = []

    @classmethod
    def tearDownClass(cls) -> None:
        w = cls.client.workspace_client()
        for job_id in getattr(cls, "_jobs_to_delete", []):
            try:
                w.jobs.delete(job_id=job_id)
            except Exception:
                pass
        try:
            w.workspace.delete(path=_SMOKE_DIR, recursive=True)
        except Exception:
            pass
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        JobRun._INSTANCES.clear()

    def _nb_task(self, key: str, deps: list[str] | None = None) -> SubmitTask:
        return SubmitTask(
            task_key=key,
            notebook_task=NotebookTask(notebook_path=self.notebook),
            depends_on=[TaskDependency(task_key=d) for d in (deps or [])] or None,
        )

    def test_submit_multitask_dag_is_awaitable_end_to_end(self):
        try:
            run = self.client.jobs.submit(
                run_name=f"ygg-smoke-{secrets.token_hex(3)}",
                tasks=[self._nb_task("a"), self._nb_task("b", ["a"])],
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"submit needs job/serverless access: {exc}")

        # Awaitable JobRun — block to terminal.
        self.assertIsInstance(run, JobRun)
        run.wait(wait=_WAIT_SECONDS)
        self.assertTrue(run.is_done, f"run not terminal: {run.state}")
        self.assertTrue(
            run.is_succeeded,
            f"run failed: {run.state} — {run.state_message}",
        )
        # The awaitable handle carries the run id (one-time runs may still get
        # an ephemeral job id from the platform — don't assume it's absent).
        self.assertIsNotNone(run.run_id)

        # DAG reflects the two tasks + the a→b edge, with live success states.
        dag = run.dag()
        self.assertIsInstance(dag, JobDag)
        self.assertEqual(sorted(dag.keys), ["a", "b"])
        self.assertIn(("a", "b"), dag.edges())
        self.assertEqual(dag.roots(), ["a"])
        self.assertEqual(dag.leaves(), ["b"])
        for node in dag:
            self.assertEqual(node.state, State.SUCCEEDED)

        # Per-task awaitable handles resolve to terminal success.
        self.assertEqual(len(run.tasks), 2)
        task_a = run.task("a")
        self.assertIsInstance(task_a, JobTask)
        task_a.wait(wait=_WAIT_SECONDS)
        self.assertTrue(task_a.is_succeeded)
        self.assertTrue(run.task("b").wait(wait=_WAIT_SECONDS).is_succeeded)
        self.assertIsNone(run.task("missing"))

    def test_persisted_job_exposes_static_dag_and_runs(self):
        name = f"[YGG][SMOKE] {secrets.token_hex(4)}"
        try:
            job = self.client.jobs.create_or_update(
                name=name,
                tasks=[
                    Task(
                        task_key="a",
                        notebook_task=NotebookTask(notebook_path=self.notebook),
                    ),
                    Task(
                        task_key="b",
                        notebook_task=NotebookTask(notebook_path=self.notebook),
                        depends_on=[TaskDependency(task_key="a")],
                    ),
                ],
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"job create needs access: {exc}")
        self._jobs_to_delete.append(job.job_id)

        # Static DAG from the persisted settings — no run needed.
        dag = job.dag()
        self.assertEqual(sorted(dag.keys), ["a", "b"])
        self.assertIn(("a", "b"), dag.edges())
        # Static definition carries no run state.
        self.assertIsNone(dag.node("a").state)

        # Run it — Job.run returns an awaitable JobRun.
        run = job.run(wait=_WAIT_SECONDS)
        self.assertTrue(run.is_succeeded, f"run failed: {run.state_message}")
        self.assertEqual(run.job_id, job.job_id)
        self.assertEqual(sorted(run.dag().keys), ["a", "b"])

    def test_run_passes_parameters_and_task_receives_them(self):
        # How to pass parameters when starting a run and have the job receive
        # them: Job.run(notebook_params={...}) overrides the notebook's widgets;
        # the task reads them with dbutils.widgets.get and returns via
        # dbutils.notebook.exit, which surfaces as the task's notebook output.
        name = f"[YGG][SMOKE] params {secrets.token_hex(4)}"
        try:
            job = self.client.jobs.create_or_update(
                name=name,
                tasks=[
                    Task(
                        task_key="echo",
                        notebook_task=NotebookTask(notebook_path=self.param_notebook),
                    )
                ],
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"job create needs access: {exc}")
        self._jobs_to_delete.append(job.job_id)

        run = job.run(notebook_params={"msg": "hello-xyz"}, wait=_WAIT_SECONDS)
        self.assertTrue(run.is_succeeded, f"run failed:\n{run.debug()}")

        # The parameter made it into the task and came back as its output.
        out = run.task_output("echo")
        self.assertIsNotNone(out)
        self.assertEqual(out.notebook_output.result, "hello-xyz")

    def test_failed_task_exposes_stderr_for_debugging(self):
        # A failing task surfaces its error through the debug accessors so a
        # caller can diagnose a run without reaching into the SDK.
        try:
            run = self.client.jobs.submit(
                run_name=f"ygg-fail-{secrets.token_hex(3)}",
                tasks=[
                    SubmitTask(
                        task_key="boom",
                        notebook_task=NotebookTask(notebook_path=self.failing_notebook),
                    )
                ],
            )
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"submit needs job/serverless access: {exc}")

        # wait without raising so we can inspect the terminal failed state
        run.wait(wait=_WAIT_SECONDS, raise_error=False)
        self.assertTrue(run.is_failed, f"expected failure, got {run.state}")

        task = run.task("boom")
        self.assertTrue(task.is_failed)
        self.assertIn("boom-xyz", task.error_message or "")
        self.assertIn("boom-xyz", task.stderr or "")
        # Run-level aggregates + the debug dump carry the error too.
        self.assertIn("boom-xyz", run.stderr)
        dump = run.debug()
        self.assertIn("boom", dump)            # task key
        self.assertIn("boom-xyz", dump)        # error detail
