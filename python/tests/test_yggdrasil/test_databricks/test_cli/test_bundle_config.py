"""Tests for ``yggdrasil.databricks.cli.bundle.config``."""
from __future__ import annotations

import os
import textwrap
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from yggdrasil.databricks.cli.bundle.config import (
    load_bundle,
    resolve_target,
)


SAMPLE_BUNDLE = textwrap.dedent("""\
    bundle:
      name: test-bundle

    variables:
      cluster_id:
        default: "abc-123"
      notebook_root:
        default: "/Shared/test"
      concurrency:
        default: 8

    resources:
      jobs:
        test_job:
          name: test-job
          tags:
            owner: test
          schedule:
            quartz_cron_expression: "0 0/15 * * * ?"
            timezone_id: UTC
            pause_status: UNPAUSED
          max_concurrent_runs: 2
          parameters:
            - name: mode
              default: "live"
            - name: date
              default: ""
          environments:
            - environment_key: env1
              spec:
                client: "5"
                dependencies: ["ygg[databricks]"]
          tasks:
            - task_key: step_one
              existing_cluster_id: ${var.cluster_id}
              timeout_seconds: 300
              notebook_task:
                notebook_path: ${var.notebook_root}/step_one
                base_parameters:
                  mode: "{{job.parameters.mode}}"
            - task_key: step_two
              depends_on:
                - task_key: step_one
              for_each_task:
                inputs: "{{tasks.step_one.values.items}}"
                concurrency: ${var.concurrency}
                task:
                  task_key: step_two_inner
                  existing_cluster_id: ${var.cluster_id}
                  timeout_seconds: 600
                  notebook_task:
                    notebook_path: ${var.notebook_root}/step_two
                    base_parameters:
                      item: "{{input}}"

    targets:
      dev:
        workspace:
          host: https://dev.cloud.databricks.com
        variables:
          cluster_id:
            default: "dev-cluster"
      prd:
        default: true
        workspace:
          host: https://prd.cloud.databricks.com
        sync:
          paths:
            - .
          include:
            - "step_one.py"
            - "step_two.py"
""")


class TestLoadBundle(unittest.TestCase):

    def test_load_parses_yaml(self):
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(SAMPLE_BUNDLE)
            f.flush()
            path = Path(f.name)

        try:
            raw = load_bundle(path)
            self.assertEqual(raw["bundle"]["name"], "test-bundle")
            self.assertIn("variables", raw)
            self.assertIn("resources", raw)
            self.assertIn("targets", raw)
        finally:
            path.unlink()


class TestResolveTarget(unittest.TestCase):

    def _load(self) -> dict:
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(SAMPLE_BUNDLE)
            f.flush()
            self._path = Path(f.name)
        return load_bundle(self._path)

    def tearDown(self):
        if hasattr(self, "_path"):
            self._path.unlink(missing_ok=True)

    def test_default_target_selected(self):
        raw = self._load()
        target_cfg, resolved = resolve_target(raw, None)
        self.assertEqual(
            target_cfg["workspace"]["host"],
            "https://prd.cloud.databricks.com",
        )

    def test_explicit_target(self):
        raw = self._load()
        target_cfg, _ = resolve_target(raw, "dev")
        self.assertEqual(
            target_cfg["workspace"]["host"],
            "https://dev.cloud.databricks.com",
        )

    def test_variable_resolution_default(self):
        raw = self._load()
        _, resolved = resolve_target(raw, "prd")
        job = resolved["resources"]["jobs"]["test_job"]
        t0 = job["tasks"][0]
        self.assertEqual(t0["existing_cluster_id"], "abc-123")
        self.assertEqual(
            t0["notebook_task"]["notebook_path"],
            "/Shared/test/step_one",
        )

    def test_variable_override_per_target(self):
        raw = self._load()
        _, resolved = resolve_target(raw, "dev")
        job = resolved["resources"]["jobs"]["test_job"]
        t0 = job["tasks"][0]
        self.assertEqual(t0["existing_cluster_id"], "dev-cluster")

    def test_runtime_expressions_preserved(self):
        raw = self._load()
        _, resolved = resolve_target(raw, "prd")
        job = resolved["resources"]["jobs"]["test_job"]
        t0 = job["tasks"][0]
        self.assertEqual(
            t0["notebook_task"]["base_parameters"]["mode"],
            "{{job.parameters.mode}}",
        )
        t1 = job["tasks"][1]
        inner = t1["for_each_task"]["task"]
        self.assertEqual(
            inner["notebook_task"]["base_parameters"]["item"],
            "{{input}}",
        )

    def test_for_each_concurrency_resolved(self):
        raw = self._load()
        _, resolved = resolve_target(raw, "prd")
        job = resolved["resources"]["jobs"]["test_job"]
        t1 = job["tasks"][1]
        self.assertEqual(t1["for_each_task"]["concurrency"], "8")

    def test_env_var_override(self):
        raw = self._load()
        old = os.environ.get("BUNDLE_VAR_cluster_id")
        try:
            os.environ["BUNDLE_VAR_cluster_id"] = "env-cluster"
            _, resolved = resolve_target(raw, "prd")
            job = resolved["resources"]["jobs"]["test_job"]
            t0 = job["tasks"][0]
            self.assertEqual(t0["existing_cluster_id"], "env-cluster")
        finally:
            if old is None:
                os.environ.pop("BUNDLE_VAR_cluster_id", None)
            else:
                os.environ["BUNDLE_VAR_cluster_id"] = old


class TestBuildTask(unittest.TestCase):

    def _resolved_tasks(self):
        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(SAMPLE_BUNDLE)
            f.flush()
            path = Path(f.name)
        try:
            raw = load_bundle(path)
            _, resolved = resolve_target(raw, "prd")
            return resolved["resources"]["jobs"]["test_job"]["tasks"]
        finally:
            path.unlink()

    def test_simple_notebook_task(self):
        from yggdrasil.databricks.cli.bundle.deploy import _build_task

        tasks_cfg = self._resolved_tasks()
        t0 = _build_task(tasks_cfg[0])
        self.assertEqual(t0.task_key, "step_one")
        self.assertEqual(t0.existing_cluster_id, "abc-123")
        self.assertEqual(t0.timeout_seconds, 300)
        self.assertEqual(
            t0.notebook_task.notebook_path, "/Shared/test/step_one",
        )
        self.assertIsNone(t0.depends_on)
        self.assertIsNone(t0.for_each_task)

    def test_for_each_task(self):
        from yggdrasil.databricks.cli.bundle.deploy import _build_task

        tasks_cfg = self._resolved_tasks()
        t1 = _build_task(tasks_cfg[1])
        self.assertEqual(t1.task_key, "step_two")
        self.assertEqual(len(t1.depends_on), 1)
        self.assertEqual(t1.depends_on[0].task_key, "step_one")

        fe = t1.for_each_task
        self.assertIsNotNone(fe)
        self.assertEqual(fe.inputs, "{{tasks.step_one.values.items}}")
        self.assertEqual(fe.concurrency, 8)
        self.assertEqual(fe.task.task_key, "step_two_inner")
        self.assertEqual(fe.task.existing_cluster_id, "abc-123")
        self.assertEqual(
            fe.task.notebook_task.notebook_path, "/Shared/test/step_two",
        )


class TestCLIHelp(unittest.TestCase):

    def test_main_help_exits_zero(self):
        from yggdrasil.databricks.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_bundle_deploy_help_exits_zero(self):
        from yggdrasil.databricks.cli import main

        with self.assertRaises(SystemExit) as ctx:
            main(["bundle", "deploy", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_command_returns_one(self):
        from yggdrasil.databricks.cli import main

        self.assertEqual(main([]), 1)

    def test_validate_with_bundle(self):
        from yggdrasil.databricks.cli import main

        with NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(SAMPLE_BUNDLE)
            f.flush()
            path = f.name

        try:
            result = main(["bundle", "validate", "-f", path, "-t", "prd"])
            self.assertEqual(result, 0)
        finally:
            Path(path).unlink()
