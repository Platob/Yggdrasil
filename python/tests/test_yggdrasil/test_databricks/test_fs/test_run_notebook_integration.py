"""Live integration for :meth:`WorkspacePath.create_notebook` /
:meth:`WorkspacePath.run_notebook` against a real workspace.

Where :mod:`test_job_integration` drives the low-level ``jobs.submit`` /
``NotebookTask`` SDK surface, this exercises the high-level ergonomic path a
caller actually reaches for:

* :meth:`WorkspacePath.create_notebook` imports a real notebook (round-trips
  through ``get_status`` as an ``object_type`` of ``NOTEBOOK``);
* :meth:`WorkspacePath.run_notebook` wraps it in a one-time **serverless** run,
  threads ``base_parameters`` onto the notebook's widgets, blocks to terminal,
  and surfaces the ``dbutils.notebook.exit`` value as the task output;
* a failing notebook reports its error through the run/task debug accessors.

Provisions a per-run scratch directory under
:envvar:`DATABRICKS_INTEGRATION_WORKSPACE_DIR` (default
``/Workspace/Users/<current-user>/yggdrasil-integration``) and removes it on
teardown. Skipped wholesale unless ``DATABRICKS_HOST`` is set; permission /
serverless-availability failures degrade to ``unittest.SkipTest``.
"""
from __future__ import annotations

import os
import secrets
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.databricks.fs import WorkspacePath
from yggdrasil.databricks.job.run import JobRun

from .. import DatabricksIntegrationCase


__all__ = ["TestRunNotebookIntegration"]

# Serverless cold-start dominates a trivial notebook run — be generous.
_WAIT_SECONDS = 600

# A param-echo notebook: read a widget, hand it straight back as the run output.
_ECHO_SRC = (
    'dbutils.widgets.text("msg", "default")\n'
    'dbutils.notebook.exit(dbutils.widgets.get("msg"))\n'
)


class TestRunNotebookIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        base = os.environ.get("DATABRICKS_INTEGRATION_WORKSPACE_DIR", "").strip()
        if not base:
            user = cls.workspace.current_user.me().user_name
            base = f"/Workspace/Users/{user}/yggdrasil-integration"
        cls.root = WorkspacePath(
            f"{base.rstrip('/')}/nb-run-{secrets.token_hex(4)}", client=cls.client,
        )
        try:
            cls.root.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {base}: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            root = getattr(cls, "root", None)
            if root is not None:
                root.remove(recursive=True, missing_ok=True)
        finally:
            super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        JobRun._INSTANCES.clear()

    def _notebook(self, content: str) -> WorkspacePath:
        """Create a throw-away Python notebook under the scratch root."""
        nb = self.root / f"nb_{secrets.token_hex(4)}"
        nb.invalidate_singleton()
        nb.create_notebook("python", content=content, overwrite=True)
        return nb

    # ------------------------------------------------------------------ #
    def test_create_notebook_round_trips_as_real_notebook(self) -> None:
        nb = self._notebook('print("ygg create_notebook ok")\n')
        nb.invalidate_singleton()
        self.assertTrue(nb.exists())
        # Imported as a real notebook (not a plain workspace file).
        info = self.workspace.workspace.get_status(nb.api_path)
        self.assertEqual(
            str(getattr(info.object_type, "name", info.object_type)).upper(),
            "NOTEBOOK",
        )

    def test_run_notebook_passes_parameters_and_returns_output(self) -> None:
        nb = self._notebook(_ECHO_SRC)
        try:
            run = nb.run_notebook({"msg": "hello-ygg"}, wait=_WAIT_SECONDS)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"run_notebook needs job/serverless access: {exc}")

        self.assertIsInstance(run, JobRun)
        self.assertTrue(run.is_succeeded, f"run failed:\n{run.debug()}")
        self.assertIsNotNone(run.run_id)
        # The widget value threaded in came back as the notebook's exit result.
        out = run.task_output(nb.name)
        self.assertIsNotNone(out)
        self.assertEqual(out.notebook_output.result, "hello-ygg")

    def test_run_notebook_default_serverless_no_params(self) -> None:
        nb = self._notebook('print("ygg serverless default ok")\n')
        try:
            run = nb.run_notebook(wait=_WAIT_SECONDS)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"run_notebook needs job/serverless access: {exc}")
        self.assertTrue(run.is_succeeded, f"run failed:\n{run.debug()}")

    def test_run_notebook_default_env_uses_seeded_ygg(self) -> None:
        # The default (environment=None) serverless path auto-resolves to the
        # seeded ygg base environment under WORKSPACE_ENV_DIR. Skip when it
        # isn't deployed (seeding builds a wheel closure — too heavy for a
        # test); otherwise prove it's genuinely the active image by importing
        # yggdrasil *inside* the run, which only succeeds against the seeded
        # zero-PyPI ygg environment.
        from yggdrasil.databricks.environments import service as W
        from yggdrasil.databricks.job.service import _resolve_submit_environment

        env = self.client.environments.get('ygg')
        if env is None or not env.serverless:
            self.skipTest(
                f"ygg base environment not seeded under "
                f"{W.WORKSPACE_ENV_DIR} (run `ygg databricks environment`)."
            )
        # Auto-resolution points the run at the seeded .yml.
        resolved = _resolve_submit_environment(self.client, None)
        self.assertEqual(resolved.spec.base_environment, env.serverless)

        nb = self._notebook(
            "import yggdrasil\n"
            'dbutils.notebook.exit("ygg=" + yggdrasil.__version__)\n'
        )
        try:
            run = nb.run_notebook(wait=_WAIT_SECONDS)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"run_notebook needs job/serverless access: {exc}")

        self.assertTrue(run.is_succeeded, f"run failed:\n{run.debug()}")
        out = run.task_output(nb.name)
        self.assertIsNotNone(out)
        # yggdrasil imported from the seeded env and reported its version.
        self.assertTrue(
            out.notebook_output.result.startswith("ygg="),
            f"unexpected notebook output: {out.notebook_output.result!r}",
        )

    def test_run_notebook_failure_surfaces_error(self) -> None:
        nb = self._notebook('raise Exception("boom-ygg")\n')
        try:
            run = nb.run_notebook(wait=_WAIT_SECONDS, raise_error=False)
        except (DatabricksError, PermissionDenied) as exc:
            self.skipTest(f"run_notebook needs job/serverless access: {exc}")

        self.assertTrue(run.is_failed, f"expected failure, got {run.state}")
        self.assertIn("boom-ygg", run.stderr)
        task = run.task(nb.name)
        self.assertTrue(task.is_failed)
        self.assertIn("boom-ygg", task.error_message or "")
