"""Unit tests for the async, file-arrival table loader (no live Databricks).

Covers:
* ``Table.insert(wait=False)`` routing → ``async_insert`` (OVERWRITE/APPEND,
  no match_by), and the sync path otherwise;
* ``async_insert`` staging a Parquet + dropping a JSON operation log;
* ``TableJob.ensure`` get-or-create with a file-arrival trigger;
* ``TableJob.run`` aggregating logs into one INSERT per (target, mode),
  then cleaning up consumed logs + data.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.table import async_job as aj
from yggdrasil.databricks.table.async_job import LOGS_SUBDIR, TableJob
from yggdrasil.databricks.table.table import Table


def _table_mock(full_name="c.s.t"):
    t = MagicMock()
    t.catalog_name, t.schema_name, t.table_name = full_name.split(".")
    t.full_name.return_value = full_name
    return t


# --------------------------------------------------------------------------- #
# Table.insert(wait=False) routing
# --------------------------------------------------------------------------- #
class TestInsertRouting:
    def test_wait_false_routes_to_async(self):
        t = MagicMock()
        out = Table.insert(t, {"a": [1]}, mode="append", wait=False)
        t.async_insert.assert_called_once()
        assert t.async_insert.call_args.kwargs["mode"] == "append"
        assert out is t.async_insert.return_value
        t.insert_into.assert_not_called()

    def test_wait_true_uses_sync_path(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append")  # wait defaults True
        t.insert_into.assert_called_once()
        t.async_insert.assert_not_called()

    def test_match_by_stays_sync(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append", wait=False, match_by=["id"])
        t.insert_into.assert_called_once()
        t.async_insert.assert_not_called()

    def test_merge_mode_stays_sync(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="merge", wait=False)
        t.insert_into.assert_called_once()
        t.async_insert.assert_not_called()


# --------------------------------------------------------------------------- #
# Table.async_insert
# --------------------------------------------------------------------------- #
class TestAsyncInsert:
    def test_rejects_non_overwrite_append(self):
        t = _table_mock()
        with pytest.raises(ValueError, match="OVERWRITE / APPEND"):
            Table.async_insert(t, object(), mode="merge")

    def test_rejects_match_by(self):
        t = _table_mock()
        with pytest.raises(ValueError, match="match_by"):
            Table.async_insert(t, object(), mode="append", match_by=["id"])

    def test_writes_parquet_to_staging_and_logs_its_path(self):
        t = _table_mock()
        # data goes to the default tmp staging path
        data_file = MagicMock()
        data_file.full_path.return_value = "/Volumes/c/s/t/.sql/tmp/tmp-1-ab.parquet"
        t.insert_volume_path.return_value = data_file
        # log dir
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file

        with patch.object(TableJob, "logs_path", staticmethod(lambda tbl: logs_dir)):
            result = Table.async_insert(t, {"a": [1]}, mode="append")

        assert result is log_file
        data_file.write_table.assert_called_once()           # staged Parquet
        log_file.write_bytes.assert_called_once()            # operation log
        payload = json.loads(log_file.write_bytes.call_args[0][0])
        assert payload["target"] == "c.s.t"
        assert payload["mode"] == "append"
        assert payload["data"] == "/Volumes/c/s/t/.sql/tmp/tmp-1-ab.parquet"
        # touched async_job so the file-arrival trigger exists
        assert t.async_job is not None


# --------------------------------------------------------------------------- #
# TableJob.ensure (get-or-create with a file-arrival trigger)
# --------------------------------------------------------------------------- #
class TestEnsure:
    def test_creates_job_with_file_arrival_trigger(self):
        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        created = MagicMock()
        created.job_id = 42
        created._details = None
        jobs.create_or_update.return_value = created
        t.staging_volume.path.return_value.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"

        wheel = "/Workspace/Shared/.ygg/jobs/ygg-1.2.3-py3-none-any.whl"
        with patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=wheel) as ew:
            tj = TableJob(t)
            assert tj.ensure() is tj
        assert tj.job is created
        ew.assert_called_once_with(t.client)               # built + uploaded the wheel
        # the watched logs dir is created so the trigger URL is valid
        t.staging_volume.path.return_value.mkdir.assert_called_with(
            parents=True, exist_ok=True
        )

        kwargs = jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "[YGG][ASYNC] c.s.t"
        assert kwargs["trigger"].file_arrival.url == "/Volumes/c/s/t/.sql/async/logs/"
        task = kwargs["tasks"][0]
        # ygg-job table-async-load <full_name> on the cluster
        assert task.python_wheel_task.package_name == "ygg"
        assert task.python_wheel_task.entry_point == "ygg-job"
        assert task.python_wheel_task.parameters == ["table-async-load", "c.s.t"]
        # serverless v5; deps = the uploaded wheel (not a direct package) +
        # databricks-sdk (latest)
        env = kwargs["environments"][0]
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == [wheel, "databricks-sdk"]
        assert task.environment_key == env.environment_key

    def test_ensure_is_noop_when_already_deployed(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        tj = TableJob(t)
        tj._job = MagicMock()                # already deployed
        assert tj.ensure() is tj
        t.client.jobs.create_or_update.assert_not_called()

    def test_deploy_prunes_stale_jobs_on_same_trigger(self):
        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        created = MagicMock()
        created.job_id = 99
        jobs.create_or_update.return_value = created
        t.staging_volume.path.return_value.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"
        url = "/Volumes/c/s/t/.sql/async/logs/"

        def _job(job_id, trigger_url):
            j = MagicMock()
            j.job_id = job_id
            j.settings.trigger.file_arrival.url = trigger_url
            return j

        keep = _job(99, url)            # the one we just deployed
        stale = _job(7, url)            # an orphan watching the same logs dir
        other = _job(8, "/Volumes/other/.sql/async/logs/")  # unrelated job
        jobs.list.return_value = [keep, stale, other]

        with patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value="w.whl"):
            TableJob(t).deploy(t.client)

        stale.delete.assert_called_once()      # orphan on the shared trigger removed
        keep.delete.assert_not_called()
        other.delete.assert_not_called()


# --------------------------------------------------------------------------- #
# TableJob.run (aggregate logs → INSERT per (target, mode))
# --------------------------------------------------------------------------- #
class TestProcess:
    def _wire_logs(self, table):
        logs_dir = MagicMock()
        table.staging_volume.path.side_effect = (
            lambda sub, *a, **k: logs_dir if sub == LOGS_SUBDIR else MagicMock()
        )
        return logs_dir

    @staticmethod
    def _log(op, *, target="c.s.t", mode="append"):
        f = MagicMock()
        f.name = f"{op}.json"
        # the log records the data's full path (it can live anywhere)
        f.read_bytes.return_value = json.dumps(
            {"target": target, "mode": mode, "data": f"/Volumes/c/s/t/.sql/tmp/{op}.parquet"}
        ).encode()
        return f

    def test_no_logs_returns_zero(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir = self._wire_logs(t)
        logs_dir.exists.return_value = False
        assert TableJob(t).run() == 0
        t.insert.assert_not_called()

    def test_aggregates_same_group_into_one_insert(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir = self._wire_logs(t)
        logs_dir.exists.return_value = True
        log_a, log_b = self._log("a"), self._log("b")
        logs_dir.iterdir.return_value = [log_a, log_b]

        data_files: dict[str, MagicMock] = {}
        with patch.object(
            TableJob, "_data_file",
            lambda self, p: data_files.setdefault(p, MagicMock()),
        ):
            processed = TableJob(t).run(wait=False)

        assert processed == 2
        t.insert.assert_called_once()
        union = t.insert.call_args.args[0]
        assert "UNION ALL" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/b.parquet`" in union
        assert t.insert.call_args.kwargs["mode"] == "append"
        # consumed logs + data cleaned up (data resolved from the logged path)
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()
        data_files["/Volumes/c/s/t/.sql/tmp/a.parquet"].unlink.assert_called_once()
        data_files["/Volumes/c/s/t/.sql/tmp/b.parquet"].unlink.assert_called_once()

    def test_callable_runs_the_loader(self):
        # TableJob is callable like a function — calling it runs the loader.
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir = self._wire_logs(t)
        logs_dir.exists.return_value = True
        logs_dir.iterdir.return_value = [self._log("a")]

        with patch.object(TableJob, "_data_file", lambda self, p: MagicMock()):
            processed = TableJob(t)(wait=False)       # __call__ → run

        assert processed == 1
        t.insert.assert_called_once()

    def test_splits_by_mode(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir = self._wire_logs(t)
        logs_dir.exists.return_value = True
        logs_dir.iterdir.return_value = [
            self._log("a", mode="append"),
            self._log("b", mode="overwrite"),
        ]

        with patch.object(TableJob, "_data_file", lambda self, p: MagicMock()):
            processed = TableJob(t).run()

        assert processed == 2
        modes = {c.kwargs["mode"] for c in t.insert.call_args_list}
        assert modes == {"append", "overwrite"}      # one INSERT per mode
