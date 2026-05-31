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
    def test_project_dependencies_flattens_requested_extras(self):
        reqs = [
            "pyarrow>=10",                              # base
            'pandas>=2; extra == "data"',              # requested extra → keep
            'pyspark>=3; extra == "bigdata"',          # other extra → drop
            'tomli; python_version < "3.11"',          # non-extra marker → keep
        ]
        with patch("yggdrasil.databricks.job.wheel.ilmd.requires", return_value=reqs):
            deps = wheel._project_dependencies("ygg", {"data"})
        assert "pyarrow>=10" in deps
        assert "pandas>=2" in deps
        assert all("pyspark" not in d for d in deps)
        assert any("tomli" in d for d in deps)

    def test_render_pyproject_has_scripts_and_deps(self):
        text = wheel._render_pyproject(
            "ygg", "1.2.3", "yggdrasil",
            ["pyarrow>=10"],
            {"ygg-job": "yggdrasil.databricks.job.runner:main"},
        )
        assert 'name = "ygg"' in text and 'version = "1.2.3"' in text
        assert '"pyarrow>=10",' in text
        assert 'ygg-job = "yggdrasil.databricks.job.runner:main"' in text
        assert 'include = ["yggdrasil*"]' in text

    def test_synthesize_project_copies_package_and_writes_pyproject(self, tmp_path):
        # point at a fake on-disk package; real ygg metadata fills the pyproject
        module = MagicMock()
        module.__file__ = str(tmp_path / "yggdrasil" / "__init__.py")
        (tmp_path / "yggdrasil").mkdir()
        (tmp_path / "yggdrasil" / "__init__.py").write_text("# pkg\n")
        out = tmp_path / "synth"
        with patch("yggdrasil.databricks.job.wheel.importlib.import_module", return_value=module), \
             patch("yggdrasil.databricks.job.wheel.distribution_for", return_value="ygg"):
            project = wheel.synthesize_project("yggdrasil", dest_dir=out)
        assert (project / "yggdrasil" / "__init__.py").exists()    # live files copied
        py = (project / "pyproject.toml").read_text()
        assert 'name = "ygg"' in py and "[project.scripts]" in py
        assert 'include = ["yggdrasil*"]' in py

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

    def test_build_wheel_synthesizes_then_pip_wheels(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "ygg-1.0-py3-none-any.whl").write_bytes(b"")
        (out / "pyarrow-1-py3-none-any.whl").write_bytes(b"")
        with patch("yggdrasil.databricks.job.wheel.synthesize_project", return_value=Path("/synth")) as sp, \
             patch("yggdrasil.databricks.job.wheel.subprocess.run") as run, \
             patch("yggdrasil.databricks.job.wheel.tempfile.mkdtemp", return_value=str(out)):
            wheels = wheel.build_wheel("yggdrasil", extras=["databricks"], requirements=["databricks-sdk"])
        sp.assert_called_once_with("yggdrasil", extras=["databricks"])
        cmd = run.call_args.args[0]
        assert "wheel" in cmd and "/synth" in cmd and "databricks-sdk" in cmd
        assert sorted(w.name for w in wheels) == ["pyarrow-1-py3-none-any.whl", "ygg-1.0-py3-none-any.whl"]

    def test_ensure_builds_with_deps_and_uploads_all(self):
        client = MagicMock()
        built = [Path("/tmp/ygg-1.0-py3-none-any.whl"), Path("/tmp/pyarrow-1-py3-none-any.whl")]
        with patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", side_effect=lambda c, w, *, workspace_dir: f"{workspace_dir}/{w.name}"):
            dests = wheel.ensure_wheel(client, "yggdrasil", workspace_dir="/ws/job", extras=["databricks"])
        bw.assert_called_once_with("yggdrasil", extras=["databricks"], requirements=())
        assert dests == ["/ws/job/ygg-1.0-py3-none-any.whl", "/ws/job/pyarrow-1-py3-none-any.whl"]
