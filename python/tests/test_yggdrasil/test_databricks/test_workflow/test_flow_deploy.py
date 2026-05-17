"""End-to-end tests for :meth:`Flow.deploy`.

Stages a small DAG against a mocked ``Jobs`` service and asserts:

* every captured :class:`TaskNode` lands as a :class:`Task` in the
  upserted :class:`JobSettings`,
* ``depends_on`` edges flow from the trace into the staged task,
* ``SecretRef`` defaults render as ``ygg.secret(...)``
  invocations in the staged ``.py``,
* ``TaskNode`` outputs render as ``ygg.task_value(...)``
  reads in the downstream invocation,
* the staged script wraps the function call in
  ``ygg.publish_return`` so downstream tasks can read the
  result.

Source-bearing tasks live at module scope so ``inspect.getsource``
can fetch their bodies — the workflow staging pipeline relies on
that.
"""
from __future__ import annotations

import logging
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.fs.workspace_path import WorkspacePath
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.databricks.workflow import flow, secret, task


# ----------------------------------------------------------------- #
# Source-bearing fixtures
# ----------------------------------------------------------------- #


@task
def extract(date: str) -> str:
    return f"/Volumes/raw/{date}"


@task(retries=2, environment_key="ygg-default")
def load_to_warehouse(
    path: str,
    api_key: str = secret("vendor", "api-key"),
) -> str:
    return f"loaded {path}"


@task
def notify():
    logging.getLogger("test").info("done")


@flow(name="daily-etl", schedule="0 2 * * *", timezone="UTC", prefix=False)
def daily_etl(date: str = "2025-01-01"):
    p = extract(date)
    load_to_warehouse(p)


@flow(name="ordering-flow", prefix=False)
def ordering_flow():
    e = extract("now")
    after_wrapped = notify.after(e)(notify)
    after_wrapped()


@flow(name="prefixed-flow")
def prefixed_flow():
    extract("now")


@flow(name="literal-prefix", prefix="[CUSTOM] ")
def literal_prefix_flow():
    extract("now")


# ----------------------------------------------------------------- #
# Test fixtures
# ----------------------------------------------------------------- #


class _CapturingWorkspacePathMixin:
    """Replace ``WorkspacePath.write_bytes`` to capture staged sources."""

    captured: dict

    def _install_capture(self):
        self.captured = {}
        orig_write = WorkspacePath.write_bytes

        def _write(this: WorkspacePath, data: bytes):
            # ``full_path`` already resolves any ``<me>`` placeholder and
            # builds the canonical ``/Workspace/...`` form — keep using it
            # so the captured map keys match what ``SparkPythonTask.python_file``
            # / ``NotebookTask.notebook_path`` end up referencing on the job spec.
            self.captured[this.url.path or this.full_path()] = data.decode()

        WorkspacePath.write_bytes = _write
        self.addCleanup(setattr, WorkspacePath, "write_bytes", orig_write)


class TestFlowDeploy(_CapturingWorkspacePathMixin, DatabricksTestCase):
    """Trace + stage + upsert path."""

    def setUp(self) -> None:
        super().setUp()
        self._install_capture()

    def test_deploy_calls_create_or_update_with_tasks(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])

        with patch.object(self.client.jobs, "create_or_update") as create_or_update:
            create_or_update.return_value = MagicMock(
                job_id=42, job_name="daily-etl",
            )
            daily_etl.deploy(service=self.client.jobs)

        create_or_update.assert_called_once()
        call = create_or_update.call_args
        self.assertEqual(call.kwargs["name"], "daily-etl")
        tasks = call.kwargs["tasks"]
        self.assertEqual({t.task_key for t in tasks}, {"extract", "load_to_warehouse"})
        # Schedule + parameters both land on the spec.
        self.assertIsNotNone(call.kwargs.get("schedule"))
        params = call.kwargs.get("parameters") or []
        self.assertIn("date", {p.name for p in params})

    def test_depends_on_edges_match_trace(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        tasks = create_or_update.call_args.kwargs["tasks"]
        load = next(t for t in tasks if t.task_key == "load_to_warehouse")
        self.assertEqual([d.task_key for d in (load.depends_on or [])], ["extract"])
        # Extract has no upstream.
        extract_task = next(t for t in tasks if t.task_key == "extract")
        self.assertFalse(extract_task.depends_on)

    def test_retries_and_environment_key_propagate(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        tasks = create_or_update.call_args.kwargs["tasks"]
        load = next(t for t in tasks if t.task_key == "load_to_warehouse")
        self.assertEqual(load.max_retries, 2)
        self.assertEqual(load.environment_key, "ygg-default")

    def test_staged_script_renders_secret_at_invocation(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        # Every captured body should mention the runtime import.
        for body in self.captured.values():
            self.assertIn(
                "from yggdrasil.databricks.workflow import ygg",
                body,
            )
        # ``load_to_warehouse`` invocation must materialise the secret
        # via ygg.secret(...).
        load_bodies = [
            b for b in self.captured.values()
            if "load_to_warehouse" in b and "publish_return" in b
        ]
        self.assertTrue(load_bodies, "no staged body found for load_to_warehouse")
        body = load_bodies[0]
        self.assertIn("ygg.secret('vendor', 'api-key')", body)
        self.assertIn("ygg.task_value('extract')", body)
        self.assertIn("ygg.publish_return(", body)

    def test_explicit_after_edge_lands_on_task(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as create_or_update:
            create_or_update.return_value = MagicMock(job_id=1, job_name="ordering-flow")
            ordering_flow.deploy(service=self.client.jobs)
        tasks = create_or_update.call_args.kwargs["tasks"]
        notify_task = next(t for t in tasks if t.task_key == "notify")
        self.assertEqual(
            [d.task_key for d in (notify_task.depends_on or [])], ["extract"],
        )


class TestPrefixPolicy(_CapturingWorkspacePathMixin, DatabricksTestCase):
    """``@flow(prefix=...)`` controls the deployed job-name prefix."""

    def setUp(self) -> None:
        super().setUp()
        self._install_capture()

    def test_prefix_true_auto_derives_ygg_prefix(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="prefixed-flow")
            prefixed_flow.deploy(service=self.client.jobs)
        deployed_name = cou.call_args.kwargs["name"]
        self.assertTrue(
            deployed_name.startswith("[YGG]["),
            f"expected prefix [YGG][…], got {deployed_name!r}",
        )
        self.assertTrue(deployed_name.endswith(" prefixed-flow"))

    def test_prefix_false_keeps_raw_name(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        self.assertEqual(cou.call_args.kwargs["name"], "daily-etl")

    def test_prefix_literal_string(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="literal-prefix")
            literal_prefix_flow.deploy(service=self.client.jobs)
        self.assertEqual(
            cou.call_args.kwargs["name"],
            "[CUSTOM] literal-prefix",
        )


class TestMetadataAttribution(_CapturingWorkspacePathMixin, DatabricksTestCase):
    """Source metadata lands on both job tags and task descriptions."""

    def setUp(self) -> None:
        super().setUp()
        self._install_capture()

    def test_job_tags_include_source_metadata(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        tags = cou.call_args.kwargs["tags"] or {}
        # Auto-derived attribution: module / qualname / yggdrasil version.
        self.assertIn("ygg.flow", tags)
        self.assertEqual(tags["ygg.flow"], "daily-etl")
        self.assertIn("ygg.module", tags)
        self.assertIn("ygg.version", tags)

    def test_task_description_carries_source_footer(self) -> None:
        self.workspace_client.jobs.list.return_value = iter([])
        with patch.object(self.client.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="daily-etl")
            daily_etl.deploy(service=self.client.jobs)
        tasks = cou.call_args.kwargs["tasks"]
        extract_task = next(t for t in tasks if t.task_key == "extract")
        self.assertIn("Task source:", extract_task.description or "")
        self.assertIn("extract", extract_task.description or "")


class TestClientPinning(_CapturingWorkspacePathMixin, DatabricksTestCase):
    """``@flow(client=…)`` and ``@task(client=…)`` pin a target workspace."""

    def setUp(self) -> None:
        super().setUp()
        self._install_capture()

    def test_flow_client_overrides_current(self) -> None:
        # Build a sibling client targeting a different host; deploy under it.
        from yggdrasil.databricks.client import DatabricksClient

        other = DatabricksClient(
            host="https://other-workspace.cloud.databricks.com",
            token="other-token",
        )

        @flow(name="pinned-flow", prefix=False, client=other)
        def pinned_flow():
            extract("x")

        with patch.object(other.jobs, "create_or_update") as cou:
            cou.return_value = MagicMock(job_id=1, job_name="pinned-flow")
            pinned_flow.deploy()
        cou.assert_called_once()

    def test_task_client_overrides_staging_workspace(self) -> None:
        # The per-task ``client`` is read at stage time. Use a sentinel
        # task that pins ``self.client`` to a different ``DatabricksClient``
        # — assert ``stage`` routes the staging call through it.
        from unittest.mock import sentinel as _s


        captured_client: list = []

        def _spy_stage_python_callable(client, func, *args, **kwargs):
            captured_client.append(client)
            from databricks.sdk.service.jobs import SparkPythonTask, Task

            return (
                Task(
                    task_key=kwargs.get("task_key", "spy"),
                    spark_python_task=SparkPythonTask(python_file="/Workspace/x.py"),
                ),
                [],
                [],
            )

        @task(client=_s.pinned_client)
        def pinned_task(date: str):
            return date

        with patch(
            "yggdrasil.databricks.jobs.task.stage_python_callable",
            _spy_stage_python_callable,
        ):
            from yggdrasil.databricks.workflow.nodes import TaskNode

            node = TaskNode(spec=pinned_task, task_key="pinned_task", args=("now",))
            # ``stage`` should prefer the pinned client over its arg.
            pinned_task.stage(_s.flow_client, node)

        self.assertEqual(captured_client, [_s.pinned_client])


class TestSecretClientPinning(unittest.TestCase):
    """``secret(..., client=...)`` carries a workspace hint through to runtime."""

    def test_secret_repr_includes_host_when_pinned(self) -> None:
        from yggdrasil.databricks.workflow import secret

        ref = secret("vendor", "key", client="https://prod-eu.cloud.databricks.com")
        self.assertEqual(
            repr(ref),
            "ygg.secret('vendor', 'key', host='https://prod-eu.cloud.databricks.com')",
        )

    def test_secret_repr_no_host_when_unpinned(self) -> None:
        from yggdrasil.databricks.workflow import secret

        self.assertEqual(
            repr(secret("vendor", "key")),
            "ygg.secret('vendor', 'key')",
        )

    def test_secret_accepts_live_client(self) -> None:
        from yggdrasil.databricks.client import DatabricksClient
        from yggdrasil.databricks.workflow import secret

        client = DatabricksClient(host="https://x.example", token="t")
        ref = secret("vendor", "key", client=client)
        self.assertTrue(ref.host)
        self.assertIn("x.example", ref.host or "")


class TestLocalExecution(DatabricksTestCase):
    """The flow body must run locally without touching the workspace."""

    def test_local_call_runs_func_bodies_inline(self) -> None:
        # Tasks return real Python values; downstream tasks receive
        # them directly (no taskValues hop).
        with patch(
            "yggdrasil.databricks.client.DatabricksClient.current",
        ) as current:
            current.return_value = self.client
            self.client.__dict__["_secrets"] = MagicMock()
            self.client.__dict__["_secrets"].__getitem__.return_value.svalue.return_value = "FAKE"
            # ``daily_etl()`` runs both tasks in process — load_to_warehouse
            # receives the SecretRef materialised via runtime.secret.
            daily_etl(date="2025-01-15")

    def test_local_call_passes_explicit_secret_through_unchanged(self) -> None:
        # Caller-supplied secret wins over SecretRef default.
        self.assertEqual(
            load_to_warehouse(path="/p", api_key="explicit-key"),
            "loaded /p",
        )


if __name__ == "__main__":  # pragma: no cover
    import unittest

    unittest.main()
