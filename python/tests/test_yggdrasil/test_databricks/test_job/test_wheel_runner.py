"""Unit tests for the ygg-job runner + wheel build/upload (no live cluster)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import runner, wheel


# --------------------------------------------------------------------------- #
# runner — ygg-job CLI
# --------------------------------------------------------------------------- #
class TestRunner:
    def test_table_async_load_runs_loader(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.async_job.TableJob") as TJ:
            TJ.return_value.run.return_value = 3
            n = runner.table_async_load("c.s.t")
        client.tables.__getitem__.assert_called_once_with("c.s.t")
        TJ.assert_called_once_with(table)
        TJ.return_value.run.assert_called_once_with(wait=True)
        assert n == 3

    def test_main_dispatches_subcommand(self):
        with patch.object(runner, "table_async_load", return_value=2) as f:
            rc = runner.main(["table-async-load", "c.s.t"])
        f.assert_called_once_with("c.s.t")
        assert rc == 0

    def test_unknown_command_exits(self):
        with pytest.raises(SystemExit):
            runner.main(["nope"])

    def test_missing_command_exits(self):
        with pytest.raises(SystemExit):
            runner.main([])

    def test_missing_table_arg_exits(self):
        with pytest.raises(SystemExit):
            runner.main(["table-async-load"])


# --------------------------------------------------------------------------- #
# wheel — build + upload + ensure
# --------------------------------------------------------------------------- #
class TestWheel:
    def test_find_project_root_walks_up(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        nested = tmp_path / "pkg" / "sub"
        nested.mkdir(parents=True)
        assert wheel.find_project_root(nested / "mod.py") == tmp_path.resolve()
        assert wheel.find_project_root(nested) == tmp_path.resolve()

    def test_find_project_root_raises_when_absent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            wheel.find_project_root(tmp_path / "nope" / "mod.py")

    def test_upload_writes_to_workspace_dir(self, tmp_path):
        client = MagicMock()
        wf = tmp_path / "ygg-1.2.3-py3-none-any.whl"
        wf.write_bytes(b"WHEELBYTES")
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = path
            dest = wheel.upload_wheel(client, wf)
        assert dest == "/Workspace/Shared/.ygg/whl/ygg-1.2.3-py3-none-any.whl"
        DP.from_.assert_called_once_with(dest, client=client)
        path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        path.write_bytes.assert_called_once_with(b"WHEELBYTES")

    def test_build_wheel_runs_pip_wheel_with_extras_and_reqs(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        out = tmp_path / "out"
        out.mkdir()
        (out / "x-1.0-py3-none-any.whl").write_bytes(b"")
        (out / "dep-2.0-py3-none-any.whl").write_bytes(b"")
        with patch("yggdrasil.databricks.job.wheel.subprocess.run") as run, \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value=str(out)):
            wheels = wheel.build_wheel(tmp_path / "mod.py", extras=["databricks"], requirements=["databricks-sdk"])
        cmd = run.call_args.args[0]
        assert "wheel" in cmd and f"{tmp_path.resolve()}[databricks]" in cmd
        assert "databricks-sdk" in cmd                  # extra requirement bundled too
        assert sorted(w.name for w in wheels) == ["dep-2.0-py3-none-any.whl", "x-1.0-py3-none-any.whl"]

    def test_ensure_builds_with_deps_and_uploads_all(self):
        client = MagicMock()
        built = [Path("/tmp/x-1.0-py3-none-any.whl"), Path("/tmp/dep-2.0-py3-none-any.whl")]
        with patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            dests = wheel.ensure_wheel(client, "/proj/mod.py", workspace_dir="/ws/job", extras=["databricks"])
        bw.assert_called_once_with("/proj/mod.py", extras=["databricks"], requirements=())
        assert dests == ["/ws/job/x-1.0-py3-none-any.whl", "/ws/job/dep-2.0-py3-none-any.whl"]
