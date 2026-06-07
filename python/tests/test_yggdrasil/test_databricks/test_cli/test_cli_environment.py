"""Dispatch tests for ``ygg databricks environment`` (mocked ``dbc.environments``)."""
from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main


def _env(python=None):
    """An ``Environment``-shaped handle (the CLI reads ``.name`` / ``.serverless`` / …)."""
    key = "py" + (python or "3X").replace(".", "")
    name = f"ygg-1.0-{key}"
    env_dir = f"/Workspace/Shared/environments/{name}"
    return types.SimpleNamespace(
        name=name, project="ygg", env_dir=env_dir, python=python,
        serverless=f"{env_dir}/{name}.yml",
        cluster=f"{env_dir}/{name}.requirements.txt",
        dependencies=["a", "b", "c", "d", "e", "f", "g"],
    )


class TestEnvironmentHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["environment", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestEnvironmentEnsure(unittest.TestCase):
    def test_bare_command_builds_all_supported_pythons(self):
        from yggdrasil.databricks.wheels.service import SUPPORTED_PYTHONS

        client = MagicMock()
        client.environments.deploy_ygg.return_value = [_env(v) for v in SUPPORTED_PYTHONS]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["environment", "--rebuild"])
        self.assertEqual(rc, 0)
        client.environments.deploy_ygg.assert_called_once()
        # default → every supported Python
        self.assertEqual(client.environments.deploy_ygg.call_args.kwargs["versions"], list(SUPPORTED_PYTHONS))
        self.assertTrue(client.environments.deploy_ygg.call_args.kwargs["rebuild"])

    def test_current_get_or_installs_local_python_env(self):
        client = MagicMock()
        client.environments.deploy_ygg.return_value = [_env(None)]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["environment", "--current"])
        self.assertEqual(rc, 0)
        client.environments.deploy_ygg.assert_called_once()
        # ``--current`` → local Python only, no forced rebuild
        self.assertEqual(client.environments.deploy_ygg.call_args.kwargs["versions"], [None])
        self.assertFalse(client.environments.deploy_ygg.call_args.kwargs["rebuild"])
        out = buf.getvalue()
        self.assertIn("ygg-1.0-py3X.yml", out)
        self.assertIn("ygg-1.0-py3X.requirements.txt", out)

    def test_alias_env_works(self):
        client = MagicMock()
        client.environments.deploy_ygg.return_value = [_env(None)]
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["env"])
        self.assertEqual(rc, 0)
        client.environments.deploy_ygg.assert_called_once()


class TestEnvironmentList(unittest.TestCase):
    def test_list_prints_deployed_files(self):
        client = MagicMock()
        client.environments.list.return_value = [_env("3.12")]
        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(buf):
            rc = main(["environment", "list"])
        self.assertEqual(rc, 0)
        self.assertIn("ygg-1.0-py312.yml", buf.getvalue())

    def test_list_empty_returns_one(self):
        client = MagicMock()
        client.environments.list.return_value = []
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()):
            rc = main(["environment", "list"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
