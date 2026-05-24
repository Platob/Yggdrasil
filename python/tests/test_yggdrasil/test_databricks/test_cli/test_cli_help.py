"""Tests for CLI help output and basic dispatching."""
from __future__ import annotations

import textwrap
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from yggdrasil.databricks.cli import main


SAMPLE_BUNDLE = textwrap.dedent("""\
    bundle:
      name: test-bundle
    variables:
      cluster_id:
        default: "abc-123"
      notebook_root:
        default: "/Shared/test"
    resources:
      jobs:
        test_job:
          name: test-job
          tasks:
            - task_key: step_one
              existing_cluster_id: ${var.cluster_id}
              notebook_task:
                notebook_path: ${var.notebook_root}/step_one
      clusters:
        my_cluster:
          cluster_name: test-cluster
          num_workers: 2
    targets:
      prd:
        default: true
        workspace:
          host: https://prd.cloud.databricks.com
""")


class TestMainHelp(unittest.TestCase):

    def test_main_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_no_command_returns_one(self):
        self.assertEqual(main([]), 1)

    def test_bundle_deploy_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["bundle", "deploy", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_bundle_run_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["bundle", "run", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_bundle_validate_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["bundle", "validate", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestJobsHelp(unittest.TestCase):

    def test_jobs_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_jobs_list_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_jobs_get_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "get", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_jobs_create_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "create", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_jobs_delete_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "delete", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_jobs_run_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["jobs", "run", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestClustersHelp(unittest.TestCase):

    def test_clusters_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_list_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_create_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "create", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_clusters_start_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["clusters", "start", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestWarehousesHelp(unittest.TestCase):

    def test_warehouses_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_warehouses_list_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_warehouses_create_help(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["warehouses", "create", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestValidateWithResources(unittest.TestCase):

    def test_validate_shows_all_resource_types(self):
        f = NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        f.write(SAMPLE_BUNDLE)
        f.flush()
        f.close()

        try:
            result = main(["bundle", "validate", "-f", f.name, "-t", "prd"])
            self.assertEqual(result, 0)
        finally:
            Path(f.name).unlink()

    def test_validate_no_bundle_file_returns_error(self):
        result = main(["bundle", "validate", "-f", "/nonexistent/path.yml"])
        self.assertEqual(result, 1)
