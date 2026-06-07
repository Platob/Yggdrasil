"""Dispatch tests for ``ygg databricks environment`` (mocked wheel machinery)."""
from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from yggdrasil.databricks.cli import main


def _env(python=None):
    key = "py" + (python or "3X").replace(".", "")
    name = f"ygg-1.0-{key}"
    env_dir = f"/Workspace/Shared/environments/{name}"
    return {
        "python": python, "key": key, "env_name": name, "env_dir": env_dir,
        "n_wheels": 7,
        "serverless": f"{env_dir}/{name}.yml",
        "cluster": f"{env_dir}/{name}.requirements.txt",
    }


class TestEnvironmentHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["environment", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestEnvironmentEnsure(unittest.TestCase):
    def test_bare_command_builds_all_supported_pythons(self):
        from yggdrasil.databricks.job.wheel import SUPPORTED_PYTHONS

        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(v) for v in SUPPORTED_PYTHONS]) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["environment", "--rebuild"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        # default → every supported Python
        self.assertEqual(ensure.call_args.kwargs["versions"], list(SUPPORTED_PYTHONS))
        self.assertTrue(ensure.call_args.kwargs["rebuild"])

    def test_current_get_or_installs_local_python_env(self):
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]) as ensure, \
             contextlib.redirect_stdout(buf):
            rc = main(["environment", "--current"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()
        # ``--current`` → local Python only, no forced rebuild
        self.assertEqual(ensure.call_args.kwargs["versions"], [None])
        self.assertFalse(ensure.call_args.kwargs["rebuild"])
        out = buf.getvalue()
        self.assertIn("ygg-1.0-py3X.yml", out)
        self.assertIn("ygg-1.0-py3X.requirements.txt", out)

    def test_alias_env_works(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.ensure_environments",
                   return_value=[_env(None)]) as ensure, \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["env"])
        self.assertEqual(rc, 0)
        ensure.assert_called_once()


class TestEnvironmentList(unittest.TestCase):
    def test_list_prints_deployed_files(self):
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments",
                   return_value=["/Workspace/Shared/environments/ygg-1.0-py312/ygg-1.0-py312.yml"]), \
             contextlib.redirect_stdout(buf):
            rc = main(["environment", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("ygg-1.0-py312.yml", buf.getvalue())

    def test_list_empty_returns_one(self):
        with patch("yggdrasil.databricks.client.DatabricksClient"), \
             patch("yggdrasil.cli.style.print_logo"), \
             patch("yggdrasil.databricks.job.wheel.deployed_environments", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["environment", "list"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
