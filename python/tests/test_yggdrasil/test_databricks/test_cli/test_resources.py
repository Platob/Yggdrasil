"""Tests for ``yggdrasil.databricks.cli.bundle.resources``."""
from __future__ import annotations

import unittest

from databricks.sdk.service.compute import Environment
from databricks.sdk.service.jobs import (
    CronSchedule,
    ForEachTask,
    JobEnvironment,
    JobParameterDefinition,
    NotebookTask,
    PauseStatus,
    RunIf,
    SparkJarTask,
    SparkPythonTask,
    Task,
    TaskDependency,
)

from yggdrasil.databricks.cli.bundle.resources import (
    RESOURCE_DEPLOYERS,
    build_job_environment,
    build_job_settings,
    build_task,
    deploy_all_resources,
)


class TestBuildTask(unittest.TestCase):

    def test_notebook_task(self):
        cfg = {
            "task_key": "ingest",
            "existing_cluster_id": "c-123",
            "timeout_seconds": 600,
            "notebook_task": {
                "notebook_path": "/Shared/ingest",
                "base_parameters": {"date": "2025-01-01"},
            },
        }
        t = build_task(cfg)
        self.assertIsInstance(t, Task)
        self.assertEqual(t.task_key, "ingest")
        self.assertEqual(t.existing_cluster_id, "c-123")
        self.assertEqual(t.timeout_seconds, 600)
        self.assertIsInstance(t.notebook_task, NotebookTask)
        self.assertEqual(t.notebook_task.notebook_path, "/Shared/ingest")
        self.assertEqual(t.notebook_task.base_parameters, {"date": "2025-01-01"})

    def test_spark_python_task(self):
        cfg = {
            "task_key": "etl",
            "spark_python_task": {
                "python_file": "/Workspace/etl.py",
                "parameters": ["--date", "2025-01-01"],
            },
        }
        t = build_task(cfg)
        self.assertIsInstance(t.spark_python_task, SparkPythonTask)
        self.assertEqual(t.spark_python_task.python_file, "/Workspace/etl.py")
        self.assertEqual(t.spark_python_task.parameters, ["--date", "2025-01-01"])

    def test_spark_jar_task(self):
        cfg = {
            "task_key": "jar_step",
            "spark_jar_task": {
                "main_class_name": "com.example.Main",
                "parameters": ["arg1"],
            },
        }
        t = build_task(cfg)
        self.assertIsInstance(t.spark_jar_task, SparkJarTask)
        self.assertEqual(t.spark_jar_task.main_class_name, "com.example.Main")

    def test_depends_on(self):
        cfg = {
            "task_key": "step2",
            "depends_on": [
                {"task_key": "step1"},
                {"task_key": "step1b", "outcome": "true"},
            ],
            "notebook_task": {"notebook_path": "/nb"},
        }
        t = build_task(cfg)
        self.assertEqual(len(t.depends_on), 2)
        self.assertIsInstance(t.depends_on[0], TaskDependency)
        self.assertEqual(t.depends_on[0].task_key, "step1")
        self.assertIsNone(t.depends_on[0].outcome)
        self.assertEqual(t.depends_on[1].outcome, "true")

    def test_for_each_task(self):
        cfg = {
            "task_key": "fan_out",
            "depends_on": [{"task_key": "producer"}],
            "for_each_task": {
                "inputs": "{{tasks.producer.values.items}}",
                "concurrency": 8,
                "task": {
                    "task_key": "inner",
                    "existing_cluster_id": "c-abc",
                    "notebook_task": {
                        "notebook_path": "/Shared/process",
                        "base_parameters": {"item": "{{input}}"},
                    },
                },
            },
        }
        t = build_task(cfg)
        self.assertEqual(t.task_key, "fan_out")
        self.assertIsInstance(t.for_each_task, ForEachTask)
        self.assertEqual(t.for_each_task.concurrency, 8)
        self.assertEqual(t.for_each_task.inputs, "{{tasks.producer.values.items}}")
        inner = t.for_each_task.task
        self.assertEqual(inner.task_key, "inner")
        self.assertEqual(inner.existing_cluster_id, "c-abc")
        self.assertEqual(
            inner.notebook_task.base_parameters["item"], "{{input}}",
        )

    def test_retries(self):
        cfg = {
            "task_key": "retry_me",
            "max_retries": 3,
            "min_retry_interval_millis": 5000,
            "notebook_task": {"notebook_path": "/nb"},
        }
        t = build_task(cfg)
        self.assertEqual(t.max_retries, 3)
        self.assertEqual(t.min_retry_interval_millis, 5000)

    def test_environment_key(self):
        cfg = {
            "task_key": "serverless",
            "environment_key": "ygg-default",
            "notebook_task": {"notebook_path": "/nb"},
        }
        t = build_task(cfg)
        self.assertEqual(t.environment_key, "ygg-default")
        self.assertIsNone(t.existing_cluster_id)

    def test_run_if(self):
        cfg = {
            "task_key": "conditional",
            "run_if": "ALL_SUCCESS",
            "notebook_task": {"notebook_path": "/nb"},
        }
        t = build_task(cfg)
        self.assertEqual(t.run_if, RunIf.ALL_SUCCESS)

    def test_notebook_warehouse_id(self):
        cfg = {
            "task_key": "sql_nb",
            "notebook_task": {
                "notebook_path": "/nb",
                "warehouse_id": "wh-123",
            },
        }
        t = build_task(cfg)
        self.assertEqual(t.notebook_task.warehouse_id, "wh-123")

    def test_string_timeout_coerced_to_int(self):
        cfg = {
            "task_key": "t",
            "timeout_seconds": "300",
            "notebook_task": {"notebook_path": "/nb"},
        }
        t = build_task(cfg)
        self.assertEqual(t.timeout_seconds, 300)
        self.assertIsInstance(t.timeout_seconds, int)


class TestBuildJobSettings(unittest.TestCase):

    def test_schedule(self):
        cfg = {
            "schedule": {
                "quartz_cron_expression": "0 0 * * * ?",
                "timezone_id": "America/New_York",
                "pause_status": "PAUSED",
            },
        }
        s = build_job_settings(cfg)
        sched = s["schedule"]
        self.assertIsInstance(sched, CronSchedule)
        self.assertEqual(sched.quartz_cron_expression, "0 0 * * * ?")
        self.assertEqual(sched.timezone_id, "America/New_York")
        self.assertEqual(sched.pause_status, PauseStatus.PAUSED)

    def test_schedule_defaults_utc(self):
        cfg = {"schedule": {"quartz_cron_expression": "0 0 * * * ?"}}
        s = build_job_settings(cfg)
        self.assertEqual(s["schedule"].timezone_id, "UTC")

    def test_parameters(self):
        cfg = {
            "parameters": [
                {"name": "date", "default": "2025-01-01"},
                {"name": "mode", "default": "live"},
            ],
        }
        s = build_job_settings(cfg)
        params = s["parameters"]
        self.assertEqual(len(params), 2)
        self.assertIsInstance(params[0], JobParameterDefinition)
        self.assertEqual(params[0].name, "date")
        self.assertEqual(params[0].default, "2025-01-01")

    def test_environments(self):
        cfg = {
            "environments": [
                {
                    "environment_key": "v5",
                    "spec": {"client": "5", "dependencies": ["ygg"]},
                },
            ],
        }
        s = build_job_settings(cfg)
        envs = s["environments"]
        self.assertEqual(len(envs), 1)
        self.assertIsInstance(envs[0], JobEnvironment)
        self.assertEqual(envs[0].environment_key, "v5")
        self.assertEqual(envs[0].spec.client, "5")
        self.assertEqual(envs[0].spec.dependencies, ["ygg"])

    def test_tags(self):
        cfg = {"tags": {"owner": "team-a", "env": "prd"}}
        s = build_job_settings(cfg)
        self.assertEqual(s["tags"], {"owner": "team-a", "env": "prd"})

    def test_max_concurrent_runs(self):
        cfg = {"max_concurrent_runs": "4"}
        s = build_job_settings(cfg)
        self.assertEqual(s["max_concurrent_runs"], 4)
        self.assertIsInstance(s["max_concurrent_runs"], int)

    def test_description(self):
        cfg = {"description": "My important job."}
        s = build_job_settings(cfg)
        self.assertEqual(s["description"], "My important job.")

    def test_empty_config_returns_empty_dict(self):
        self.assertEqual(build_job_settings({}), {})


class TestBuildJobEnvironment(unittest.TestCase):

    def test_full_spec(self):
        cfg = {
            "environment_key": "ygg-v5",
            "spec": {
                "client": "5",
                "dependencies": ["ygg[databricks]>=0.8"],
            },
        }
        env = build_job_environment(cfg)
        self.assertIsInstance(env, JobEnvironment)
        self.assertEqual(env.environment_key, "ygg-v5")
        self.assertIsInstance(env.spec, Environment)
        self.assertEqual(env.spec.client, "5")
        self.assertEqual(env.spec.dependencies, ["ygg[databricks]>=0.8"])

    def test_no_spec(self):
        cfg = {"environment_key": "bare"}
        env = build_job_environment(cfg)
        self.assertEqual(env.environment_key, "bare")
        self.assertIsNone(env.spec)


class TestBuildLibrary(unittest.TestCase):

    def test_string_becomes_pypi(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_library
        lib = _build_library("requests==2.31")
        self.assertIsNotNone(lib.pypi)
        self.assertEqual(lib.pypi.package, "requests==2.31")

    def test_pypi_dict(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_library
        lib = _build_library({"pypi": {"package": "numpy", "repo": "https://pypi.org/simple"}})
        self.assertEqual(lib.pypi.package, "numpy")
        self.assertEqual(lib.pypi.repo, "https://pypi.org/simple")

    def test_jar(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_library
        lib = _build_library({"jar": "dbfs:/jars/my.jar"})
        self.assertEqual(lib.jar, "dbfs:/jars/my.jar")

    def test_whl(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_library
        lib = _build_library({"whl": "dbfs:/wheels/my.whl"})
        self.assertEqual(lib.whl, "dbfs:/wheels/my.whl")

    def test_maven(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_library
        lib = _build_library({
            "maven": {
                "coordinates": "com.example:lib:1.0",
                "repo": "https://maven.example.com",
                "exclusions": ["org.slf4j:*"],
            },
        })
        self.assertIsNotNone(lib.maven)
        self.assertEqual(lib.maven.coordinates, "com.example:lib:1.0")
        self.assertEqual(lib.maven.exclusions, ["org.slf4j:*"])


class TestBuildSqlTask(unittest.TestCase):

    def test_query(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_sql_task
        t = _build_sql_task({
            "warehouse_id": "wh-1",
            "query": {"query_id": "q-123"},
        })
        self.assertEqual(t.warehouse_id, "wh-1")
        self.assertEqual(t.query.query_id, "q-123")

    def test_file(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_sql_task
        t = _build_sql_task({
            "warehouse_id": "wh-2",
            "file": {"path": "/sql/report.sql"},
        })
        self.assertEqual(t.file.path, "/sql/report.sql")
        self.assertEqual(t.warehouse_id, "wh-2")


class TestBuildClusterSpec(unittest.TestCase):

    def test_basic_fields(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_cluster_spec
        spec = _build_cluster_spec({
            "spark_version": "15.4.x-scala2.12",
            "node_type_id": "i3.xlarge",
            "num_workers": 4,
            "autotermination_minutes": 60,
            "custom_tags": {"team": "data"},
        })
        self.assertEqual(spec["spark_version"], "15.4.x-scala2.12")
        self.assertEqual(spec["node_type_id"], "i3.xlarge")
        self.assertEqual(spec["num_workers"], 4)
        self.assertEqual(spec["autotermination_minutes"], 60)
        self.assertEqual(spec["custom_tags"], {"team": "data"})

    def test_autoscale(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_cluster_spec
        spec = _build_cluster_spec({
            "autoscale": {"min_workers": 2, "max_workers": 10},
        })
        self.assertIn("autoscale", spec)
        self.assertEqual(spec["autoscale"].min_workers, 2)
        self.assertEqual(spec["autoscale"].max_workers, 10)

    def test_libraries_extracted(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_cluster_spec
        spec = _build_cluster_spec({
            "libraries": [
                "requests",
                {"pypi": {"package": "numpy"}},
            ],
        })
        self.assertEqual(spec["libraries"], ["requests", "numpy"])

    def test_empty_config(self):
        from yggdrasil.databricks.cli.bundle.resources import _build_cluster_spec
        self.assertEqual(_build_cluster_spec({}), {})


class TestResourceDeployers(unittest.TestCase):

    def test_deployer_registry_has_expected_keys(self):
        self.assertIn("jobs", RESOURCE_DEPLOYERS)
        self.assertIn("clusters", RESOURCE_DEPLOYERS)
        self.assertIn("pipelines", RESOURCE_DEPLOYERS)

    def test_deploy_all_empty_resources(self):
        result = deploy_all_resources(None, {})
        self.assertEqual(result, {})
