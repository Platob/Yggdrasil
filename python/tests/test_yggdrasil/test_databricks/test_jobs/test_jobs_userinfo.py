"""Tests for UserInfo-driven defaults on the Jobs service and JobTask."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks.sdk.service.jobs import (
    GitProvider,
    GitSource,
    JobEmailNotifications,
    SparkPythonTask,
    Task,
)

from yggdrasil.databricks.jobs import Job, JobTask
from yggdrasil.databricks.jobs.task import DEFAULT_ENVIRONMENT_KEY
from yggdrasil.databricks.tests import DatabricksTestCase
from yggdrasil.io.url import URL


def _job_info_for(job_id, name="t"):
    from databricks.sdk.service.jobs import Job as JobInfo, JobSettings
    return JobInfo(job_id=job_id, settings=JobSettings(name=name))


def _userinfo_stub():
    info = MagicMock()
    info.email = "alice@example.com"
    info.cwd = "/repo"
    info.git_url = URL.from_str("https://github.com/acme/widgets#abc1234567")
    info.url = URL.from_str("https://workspace.example.com/?o=42")
    return info


def _git_info_stub(_cwd):
    return {"git_branch": "main", "git_remote": "https://github.com/acme/widgets"}


class TestJobsUserInfoDefaults(DatabricksTestCase):
    def test_userinfo_defaults_returns_settings_dict(self):
        with patch(
            "yggdrasil.databricks.jobs.userinfo._git_info", _git_info_stub,
        ), patch(
            "yggdrasil.environ.UserInfo.current", return_value=_userinfo_stub(),
        ):
            defaults = self.jobs.userinfo_defaults()

        assert isinstance(defaults["git_source"], GitSource)
        assert defaults["git_source"].git_provider == GitProvider.GIT_HUB
        assert isinstance(defaults["email_notifications"], JobEmailNotifications)
        assert defaults["email_notifications"].on_failure == ["alice@example.com"]
        assert "GitUrl" in defaults["tags"]

    def test_userinfo_defaults_empty_on_resolution_failure(self):
        with patch(
            "yggdrasil.environ.UserInfo.current", side_effect=RuntimeError("nope"),
        ):
            assert self.jobs.userinfo_defaults() == {}

    def test_create_for_user_merges_caller_overrides(self):
        self.jobs_api.create.return_value = MagicMock(job_id=42)
        self.jobs_api.get.return_value = _job_info_for(42, "etl")

        with patch(
            "yggdrasil.databricks.jobs.userinfo._git_info", _git_info_stub,
        ), patch(
            "yggdrasil.environ.UserInfo.current", return_value=_userinfo_stub(),
        ):
            self.jobs.create_for_user(
                name="etl", tasks=[], tags={"Env": "prod"},
            )

        _, kwargs = self.jobs_api.create.call_args
        # Auto-derived git source + notifications are present.
        assert isinstance(kwargs.get("git_source"), GitSource)
        assert isinstance(
            kwargs.get("email_notifications"), JobEmailNotifications,
        )
        # Auto tags + caller tags merged, caller wins on key collision.
        merged = kwargs["tags"]
        assert merged["Env"] == "prod"          # caller override
        assert merged["GitUrl"] == "https://github.com/acme/widgets"

    def test_create_for_user_can_opt_out_of_userinfo(self):
        self.jobs_api.create.return_value = MagicMock(job_id=42)
        self.jobs_api.get.return_value = _job_info_for(42, "etl")

        with patch(
            "yggdrasil.environ.UserInfo.current", return_value=_userinfo_stub(),
        ):
            self.jobs.create_for_user(
                name="etl", tasks=[], userinfo_defaults=False,
            )

        _, kwargs = self.jobs_api.create.call_args
        assert "git_source" not in kwargs
        assert "email_notifications" not in kwargs


class TestJobTaskAutoDependencies(DatabricksTestCase):
    """from_callable sniffs imports and threads them into the env spec."""

    def _job(self, *, job_id: int = 42) -> Job:
        return Job(
            service=self.jobs,
            job_id=job_id,
            job_name="t",
            details=_job_info_for(job_id, "t"),
        )

    def test_from_callable_populates_extra_dependencies(self):
        job = self._job()

        def _staged_with_imports():
            import polars  # noqa: F401
            import os
            os.getenv("PROD_API_TOKEN")

        with patch(
            "yggdrasil.databricks.jobs.task.WorkspacePath",
            create=True,
        ) as mock_path:
            mock_instance = MagicMock()
            mock_instance.full_path.return_value = "/Workspace/foo.py"
            mock_path.return_value = mock_instance
            with patch(
                "yggdrasil.databricks.jobs.introspect.resolve_module_dependency"
            ) as mock_resolve:
                from yggdrasil.databricks.jobs.introspect import ModuleDependency
                mock_resolve.side_effect = lambda m: ModuleDependency(
                    module=m,
                    project=m,
                    version="1.0.0" if m == "polars" else None,
                    kind="stdlib" if m == "os" else "pypi",
                )
                task = JobTask.from_callable(job, _staged_with_imports)

        assert "polars==1.0.0" in task.extra_dependencies
        assert "PROD_API_TOKEN" in task.sniffed_env_vars

    def test_create_merges_extra_deps_into_default_environment(self):
        job = self._job()
        job.settings.tasks = []

        jt = JobTask(
            job=job,
            task_key="step",
            details=Task(
                task_key="step",
                spark_python_task=SparkPythonTask(python_file="/x.py"),
                environment_key=DEFAULT_ENVIRONMENT_KEY,
            ),
            extra_dependencies=["polars==1.0.0", "httpx==0.27.0"],
        )
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        envs = kwargs["new_settings"].environments
        assert envs is not None
        deps = envs[0].spec.dependencies
        assert "polars==1.0.0" in deps
        assert "httpx==0.27.0" in deps
        # Default ygg dependency still present.
        assert any(d.startswith("ygg") for d in deps)

    def test_create_extends_existing_environment_dependencies(self):
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        job = self._job()
        job.settings.tasks = []
        job.settings.environments = [
            JobEnvironment(
                environment_key=DEFAULT_ENVIRONMENT_KEY,
                spec=Environment(client="1", dependencies=["ygg[data]"]),
            ),
        ]

        jt = JobTask(
            job=job,
            task_key="step",
            details=Task(
                task_key="step",
                spark_python_task=SparkPythonTask(python_file="/x.py"),
                environment_key=DEFAULT_ENVIRONMENT_KEY,
            ),
            extra_dependencies=["polars==1.0.0"],
        )
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        envs = kwargs["new_settings"].environments
        assert envs is not None
        deps = envs[0].spec.dependencies
        assert deps == ["ygg[data]", "polars==1.0.0"]

    def test_create_skips_environments_when_extras_already_declared(self):
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        job = self._job()
        job.settings.tasks = []
        job.settings.environments = [
            JobEnvironment(
                environment_key=DEFAULT_ENVIRONMENT_KEY,
                spec=Environment(
                    client="1",
                    dependencies=["ygg[data]", "polars==1.0.0"],
                ),
            ),
        ]

        jt = JobTask(
            job=job,
            task_key="step",
            details=Task(
                task_key="step",
                spark_python_task=SparkPythonTask(python_file="/x.py"),
                environment_key=DEFAULT_ENVIRONMENT_KEY,
            ),
            extra_dependencies=["polars==1.0.0"],
        )
        jt.create()

        _, kwargs = self.jobs_api.update.call_args
        # No environments update — extras already present.
        assert kwargs["new_settings"].environments is None
