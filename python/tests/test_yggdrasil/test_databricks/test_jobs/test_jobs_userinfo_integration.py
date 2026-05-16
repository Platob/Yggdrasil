"""Live-integration tests for UserInfo-driven defaults + auto-dependency sniffing.

Skipped unless ``DATABRICKS_HOST`` is exported (see
:class:`DatabricksIntegrationCase`). Exercises:

- :meth:`Jobs.userinfo_defaults` — settings dict built from the
  running process's :class:`UserInfo`; the values land on the job
  spec when the dict is splatted into :meth:`create`.
- :meth:`Jobs.create_for_user` — the same path packaged as one call.
- :meth:`JobTask.from_callable(auto_dependencies=True)` — imports
  sniffed from the staged script's source surface as
  :attr:`JobTask.extra_dependencies` and merge into the matching
  serverless :class:`JobEnvironment` on :meth:`JobTask.create`.
- :meth:`JobTask.from_callable(workspace_pypi=True)` — non-PyPI
  imports are uploaded as wheels to the workspace-side
  :class:`WorkspacePyPI` and the pinned ``project @ workspace://…``
  spec lands in the env.
- :class:`WorkspacePyPI` — publish a real local package and re-import
  it from the workspace simple index.
"""
from __future__ import annotations

import tempfile
import textwrap
import secrets
from pathlib import Path as _LocalPath
from typing import ClassVar, List

from databricks.sdk.errors import DatabricksError
from databricks.sdk.service.jobs import (
    ConditionTask,
    ConditionTaskOp,
    Task,
)

from yggdrasil.databricks.jobs import (
    Job,
    JobTask,
    WorkspacePyPI,
)
from yggdrasil.databricks.jobs.task import DEFAULT_ENVIRONMENT_KEY
from yggdrasil.databricks.fs.workspace_path import WorkspacePath

from .. import DatabricksIntegrationCase


__all__ = [
    "TestJobsUserInfoIntegration",
    "TestJobTaskAutoDepsIntegration",
    "TestWorkspacePyPIIntegration",
]


def _noop_condition_task(task_key: str = "noop") -> Task:
    return Task(
        task_key=task_key,
        condition_task=ConditionTask(
            op=ConditionTaskOp.EQUAL_TO, left="1", right="1",
        ),
    )


class _UserInfoIntegrationBase(DatabricksIntegrationCase):
    """Shared cleanup fixture (mirrors ``_JobsIntegrationBase``)."""

    created_jobs: ClassVar[List[int]]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.created_jobs = []

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            for job_id in cls.created_jobs:
                try:
                    cls.client.jobs.find(job_id=job_id).delete()
                except (DatabricksError, Exception):
                    pass
        finally:
            super().tearDownClass()

    @staticmethod
    def _unique_job_name(prefix: str) -> str:
        return f"yg_int_{prefix}_{secrets.token_hex(4)}"

    @classmethod
    def _track(cls, job: Job) -> Job:
        if job.job_id is not None and job.job_id not in cls.created_jobs:
            cls.created_jobs.append(job.job_id)
        return job


class TestJobsUserInfoIntegration(_UserInfoIntegrationBase):
    """``client.jobs.userinfo_defaults`` + ``create_for_user`` round-trip."""

    def test_userinfo_defaults_returns_real_dict(self):
        # Just exercises that the helper doesn't blow up against a
        # live workspace; the content depends on whether the runner
        # has a resolvable git remote / email.
        defaults = self.client.jobs.userinfo_defaults()
        self.assertIsInstance(defaults, dict)
        for key in defaults:
            self.assertIn(
                key, {"git_source", "email_notifications", "tags"},
                f"unexpected key {key!r} in userinfo_defaults",
            )

    def test_create_for_user_persists_userinfo_settings(self):
        name = self._unique_job_name("for_user")
        job = self.client.jobs.create_for_user(
            name=name,
            tasks=[_noop_condition_task()],
            tags={"IntegrationTag": "true"},
        )
        self._track(job)

        # Caller tag survives the merge; auto-derived tags are layered
        # under (or absent when UserInfo can't resolve them).
        job.refresh()
        tags = (job.settings.tags or {}) if job.settings else {}
        self.assertEqual(tags.get("IntegrationTag"), "true")


class TestJobTaskAutoDepsIntegration(_UserInfoIntegrationBase):
    """``JobTask.from_callable(auto_dependencies=True)`` against a live job."""

    def _fresh_job(self, prefix: str) -> Job:
        name = self._unique_job_name(prefix)
        job = self.client.jobs.create(
            name=name, tasks=[_noop_condition_task("seed")],
        )
        return self._track(job)

    def test_auto_dependencies_lands_on_serverless_environment(self):
        """Sniffed pip requirements ride into the matching JobEnvironment."""
        job = self._fresh_job("autodeps")

        def step_with_imports():
            """Function that explicitly imports a third-party package."""
            import pyarrow  # noqa: F401  - sniffed
            print("ok")

        jt = JobTask.from_callable(
            job, step_with_imports, auto_dependencies=True,
        )
        # The sniffer should at minimum surface pyarrow.
        self.assertTrue(
            any(spec.startswith("pyarrow") for spec in jt.extra_dependencies),
            f"expected pyarrow in extra_dependencies, got "
            f"{jt.extra_dependencies!r}",
        )

        jt.create()
        job.refresh()
        envs = job.settings.environments or []
        default_env = next(
            (e for e in envs if e.environment_key == DEFAULT_ENVIRONMENT_KEY),
            None,
        )
        self.assertIsNotNone(default_env)
        deps = default_env.spec.dependencies or []
        self.assertTrue(
            any(d.startswith("pyarrow") for d in deps),
            f"expected pyarrow in env deps, got {deps!r}",
        )

    def test_sniffed_env_vars_surface_on_jobtask(self):
        job = self._fresh_job("autoenv")

        def step_with_env():
            import os
            os.getenv("DATABRICKS_INTEGRATION_FLAG")

        jt = JobTask.from_callable(job, step_with_env)
        self.assertIn("DATABRICKS_INTEGRATION_FLAG", jt.sniffed_env_vars)


class TestWorkspacePyPIIntegration(_UserInfoIntegrationBase):
    """End-to-end: publish a local wheel to the workspace and re-import it."""

    @staticmethod
    def _make_pkg(tmp: _LocalPath) -> _LocalPath:
        """Create a tiny installable Python package on disk."""
        pkg_root = tmp / "src" / "ygg_int_pkg"
        pkg_root.mkdir(parents=True)
        (pkg_root.parent / "pyproject.toml").write_text(textwrap.dedent("""
            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "ygg_int_pkg"
            version = "0.0.1"

            [tool.setuptools.packages.find]
            where = ["."]
        """))
        (pkg_root / "__init__.py").write_text(
            "INTEGRATION_VALUE = 'live-from-workspace'\n",
        )
        return pkg_root.parent

    def test_publish_then_import_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="ygg-wpypi-int-") as tmp:
            tmp_path = _LocalPath(tmp)
            pkg_root = self._make_pkg(tmp_path)

            # Anchor the workspace index under the user's home so this
            # test is hermetic (the shared default at
            # /Workspace/Shared/.ygg/pypi/simple is fine but we want a
            # cleaner cleanup contract on integration runs).
            root_path = (
                f"/Workspace/Users/<me>/.ygg/integration-tests/pypi-"
                f"{secrets.token_hex(4)}/simple"
            )
            pypi = WorkspacePyPI(self.client, root=root_path)
            try:
                published = pypi.publish(
                    "ygg_int_pkg", source_path=str(pkg_root),
                )
                self.assertTrue(
                    published.full_path().endswith(".whl"),
                    f"expected a .whl path, got {published!r}",
                )
                # The per-project index page was written too.
                index = pypi.root / "ygg-int-pkg" / "index.html"
                self.assertTrue(index.exists())
                body = index.read_bytes().decode()
                self.assertIn("ygg_int_pkg", body)

                # Round-trip: re-import the wheel through the index.
                import importlib
                import sys

                sys.modules.pop("ygg_int_pkg", None)
                try:
                    mod = pypi.import_module("ygg_int_pkg", install=True)
                    self.assertEqual(
                        getattr(mod, "INTEGRATION_VALUE", None),
                        "live-from-workspace",
                    )
                finally:
                    sys.modules.pop("ygg_int_pkg", None)
                    importlib.invalidate_caches()
            finally:
                # Best-effort cleanup of the per-test index folder.
                try:
                    WorkspacePath(root_path, client=self.client).rmdir()
                except Exception:
                    pass
