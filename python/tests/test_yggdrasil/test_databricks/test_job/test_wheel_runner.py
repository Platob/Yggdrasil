"""Unit tests for the ygg-job runner + wheel build/upload (no live cluster)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.job import runner, wheel


# --------------------------------------------------------------------------- #
# runner — ygg-job <kind> <args>
# --------------------------------------------------------------------------- #
class TestRunner:
    def test_dispatches_table_async_load(self):
        client = MagicMock()
        table = MagicMock()
        client.tables.__getitem__.return_value = table
        table.async_job  # noqa: B018
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("yggdrasil.databricks.table.async_job.TableJob") as TJ:
            TJ.return_value.run.return_value = 3
            n = runner.run("table-async-load", ["c.s.t"])
        client.tables.__getitem__.assert_called_once_with("c.s.t")
        TJ.assert_called_once_with(table)
        TJ.return_value.run.assert_called_once_with(wait=True)
        assert n == 3

    def test_unknown_kind_raises(self):
        with pytest.raises(SystemExit):
            runner.run("nope", [])

    def test_main_parses_argv(self):
        with patch.object(runner, "run", return_value=0) as r:
            runner.main(["table-async-load", "c.s.t"])
        r.assert_called_once_with("table-async-load", ["c.s.t"])

    def test_main_requires_kind(self):
        with pytest.raises(SystemExit):
            runner.main([])


# --------------------------------------------------------------------------- #
# wheel — build + upload + ensure
# --------------------------------------------------------------------------- #
class TestWheel:
    def test_wheel_name_uses_version(self):
        with patch("yggdrasil.version.__version__", "1.2.3"):
            assert wheel.wheel_name() == "ygg-1.2.3-py3-none-any.whl"

    def test_upload_writes_to_workspace_dir(self, tmp_path):
        client = MagicMock()
        wf = tmp_path / "ygg-1.2.3-py3-none-any.whl"
        wf.write_bytes(b"WHEELBYTES")
        path = MagicMock()
        with patch("yggdrasil.databricks.path.DatabricksPath") as DP:
            DP.from_.return_value = path
            dest = wheel.upload_wheel(client, wf)
        assert dest == "/Workspace/Shared/.ygg/jobs/ygg-1.2.3-py3-none-any.whl"
        DP.from_.assert_called_once_with(dest, client=client)
        path.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        path.write_bytes.assert_called_once_with(b"WHEELBYTES")

    def test_ensure_reuses_existing_wheel(self):
        client = MagicMock()
        existing = MagicMock()
        existing.exists.return_value = True
        with patch("yggdrasil.databricks.job.wheel.wheel_name", return_value="ygg-1.0-py3-none-any.whl"), \
             patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.job.wheel.build_wheel") as bw:
            DP.from_.return_value = existing
            dest = wheel.ensure_wheel(client)
        assert dest == "/Workspace/Shared/.ygg/jobs/ygg-1.0-py3-none-any.whl"
        bw.assert_not_called()                         # reused, no rebuild

    def test_ensure_builds_and_uploads_when_absent(self):
        client = MagicMock()
        missing = MagicMock()
        missing.exists.return_value = False
        built = Path("/tmp/ygg-1.0-py3-none-any.whl")
        with patch("yggdrasil.databricks.job.wheel.wheel_name", return_value="ygg-1.0-py3-none-any.whl"), \
             patch("yggdrasil.databricks.path.DatabricksPath") as DP, \
             patch("yggdrasil.databricks.job.wheel.build_wheel", return_value=built) as bw, \
             patch("yggdrasil.databricks.job.wheel.upload_wheel", return_value="/Workspace/Shared/.ygg/jobs/ygg-1.0-py3-none-any.whl") as up:
            DP.from_.return_value = missing
            dest = wheel.ensure_wheel(client)
        bw.assert_called_once()
        up.assert_called_once()
        assert dest.endswith("ygg-1.0-py3-none-any.whl")
