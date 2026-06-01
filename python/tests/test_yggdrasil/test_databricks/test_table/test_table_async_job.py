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

    def test_writes_parquet_to_staging_and_logs_its_uniform_url(self):
        t = _table_mock()
        # data goes to the default tmp staging path
        data_file = MagicMock()
        # the log records the project's uniform URL (round-trippable)
        data_file.to_url.return_value.to_string.return_value = (
            "dbfs+volume:/c/s/t/.sql/tmp/tmp-1-ab.parquet"
        )
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
        assert payload["data"] == "dbfs+volume:/c/s/t/.sql/tmp/tmp-1-ab.parquet"

    def test_string_source_is_read_then_staged(self):
        t = _table_mock()
        data_file = MagicMock()
        data_file.to_url.return_value.to_string.return_value = (
            "dbfs+volume:/c/s/t/.sql/tmp/x.parquet"
        )
        t.insert_volume_path.return_value = data_file
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file
        src = MagicMock()
        src.read_arrow_table.return_value = {"a": [1]}
        with patch.object(TableJob, "logs_path", staticmethod(lambda tbl: logs_dir)), \
             patch("yggdrasil.io.holder.IO.from_", return_value=src) as io_from:
            Table.async_insert(t, "s3://b/data.parquet", mode="append")
        io_from.assert_called_once_with("s3://b/data.parquet")
        src.read_arrow_table.assert_called_once_with()
        data_file.write_table.assert_called_once()

    def test_execute_async_insert_loads_synchronously(self):
        t = MagicMock()
        out = Table.execute_async_insert(t, "SELECT 1", mode="append")
        t.insert_into.assert_called_once()
        assert t.insert_into.call_args.kwargs["mode"] == "append"
        t.insert_volume_path.assert_not_called()      # no staging
        assert out is t.insert_into.return_value


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

        tj = TableJob(t)
        wheels = ["/Workspace/Shared/.ygg/whl/YGG_ASYNC_c.s.t/ygg-9.9-py3-none-any.whl",
                  "/Workspace/Shared/.ygg/whl/YGG_ASYNC_c.s.t/databricks_sdk-1.2-py3-none-any.whl"]
        with patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=wheels):
            assert tj.ensure() is tj
        assert tj.job is created
        # the watched logs dir is created so the trigger URL is valid
        t.staging_volume.path.return_value.mkdir.assert_called_with(
            parents=True, exist_ok=True
        )

        kwargs = jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "[YGG][ASYNC] c.s.t"
        fa = kwargs["trigger"].file_arrival
        assert fa.url == "/Volumes/c/s/t/.sql/async/logs/"
        assert fa.wait_after_last_change_seconds == 120        # 2-min buffering
        assert fa.min_time_between_triggers_seconds == 120
        task = kwargs["tasks"][0]
        # ygg-job table-async-load <full_name> on the cluster
        assert task.python_wheel_task.package_name == "ygg"
        assert task.python_wheel_task.entry_point == "ygg-job"
        assert task.python_wheel_task.parameters == ["table-async-load", "c.s.t"]
        # serverless v5; by default ygg + databricks-sdk are bundled as wheels
        env = kwargs["environments"][0]
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == wheels
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

        with patch("yggdrasil.databricks.job.wheel.ensure_wheel", return_value=["w.whl"]):
            TableJob(t).deploy(t.client)

        stale.delete.assert_called_once()      # orphan on the shared trigger removed
        keep.delete.assert_not_called()
        other.delete.assert_not_called()


# --------------------------------------------------------------------------- #
# Tables.async_insert — the loader (driven by a log path; groups by table)
# --------------------------------------------------------------------------- #
def _service():
    from yggdrasil.databricks.table.tables import Tables
    return Tables(client=MagicMock())


def _log(op, *, target="c.s.t", mode="append"):
    f = MagicMock()
    f.name = f"{op}.json"
    # the log records the project's uniform URL for the staged data
    f.read_bytes.return_value = json.dumps(
        {"target": target, "mode": mode,
         "data": f"dbfs+volume:/c/s/t/.sql/tmp/{op}.parquet"}
    ).encode()
    return f


def _logs_dir(*entries):
    d = MagicMock()
    d.exists.return_value = True
    d.is_dir.return_value = True
    d.iterdir.return_value = list(entries)
    return d


def _fake_databricks_from():
    """``DatabricksPath.from_`` stand-in: maps a uniform URL to a mock path
    whose ``full_path()`` is the ``/Volumes/...`` display form. Returns the
    side_effect plus the per-URL cache so cleanup can be asserted."""
    cache: dict = {}

    def _from(url, **_kwargs):
        if url not in cache:
            m = MagicMock()
            # dbfs+volume:/c/s/t/x.parquet → /Volumes/c/s/t/x.parquet
            m.full_path.return_value = "/Volumes" + url.split(":", 1)[1]
            cache[url] = m
        return cache[url]

    return _from, cache


class TestTablesAsyncInsertLoader:
    def test_no_logs_returns_zero(self):
        logs = MagicMock()
        logs.exists.return_value = False
        assert _service().async_insert(logs) == 0

    def test_aggregates_same_group_into_one_insert(self):
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        log_a, log_b = _log("a"), _log("b")
        logs = _logs_dir(log_a, log_b)
        target = MagicMock()
        from_fn, data_paths = _fake_databricks_from()
        with patch.object(Tables, "__getitem__", return_value=target), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = svc.async_insert(logs, wait=False)

        assert processed == 2
        # one execute_async_insert load per (target, mode) group, with the union
        target.execute_async_insert.assert_called_once()
        union = target.execute_async_insert.call_args.args[0]
        assert "UNION ALL" in union
        # the uniform URL is resolved to the warehouse-facing path for the query
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/b.parquet`" in union
        assert target.execute_async_insert.call_args.kwargs["mode"] == "append"
        # consumed logs + data (reconstructed from the uniform URL) cleaned up
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/a.parquet"].unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/b.parquet"].unlink.assert_called_once()

    def test_splits_by_mode(self):
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        logs = _logs_dir(_log("a", mode="append"), _log("b", mode="overwrite"))
        target = MagicMock()
        with patch.object(Tables, "__getitem__", return_value=target), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_"):
            processed = svc.async_insert(logs)
        assert processed == 2
        modes = {c.kwargs["mode"] for c in target.execute_async_insert.call_args_list}
        assert modes == {"append", "overwrite"}      # one INSERT per mode

    def test_groups_by_target_table_from_logs(self):
        # logs name their own target — the loader resolves each, no table arg.
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        logs = _logs_dir(
            _log("a", target="c.s.t1"), _log("b", target="c.s.t2"),
        )
        tables: dict[str, MagicMock] = {}

        def getitem(self_, name):
            return tables.setdefault(name, MagicMock())

        with patch.object(Tables, "__getitem__", getitem), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_"):
            processed = svc.async_insert(logs)

        assert processed == 2
        assert set(tables) == {"c.s.t1", "c.s.t2"}
        tables["c.s.t1"].execute_async_insert.assert_called_once()
        tables["c.s.t2"].execute_async_insert.assert_called_once()

    def test_single_log_file_path_string(self):
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        log = _log("a")
        log.exists.return_value = True
        log.is_dir.return_value = False
        target = MagicMock()
        with patch.object(Tables, "__getitem__", return_value=target), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_",
                   side_effect=lambda p, **k: log if p == "/logs/a.json" else MagicMock()):
            processed = svc.async_insert("/logs/a.json")
        assert processed == 1
        target.execute_async_insert.assert_called_once()

    def test_log_files_arg_skips_the_directory_scan(self):
        # Pre-gathered log files are consumed directly — no scan of a dir.
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        log_a, log_b = _log("a"), _log("b")
        target = MagicMock()
        from_fn, _ = _fake_databricks_from()
        with patch.object(Tables, "__getitem__", return_value=target), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = svc.async_insert(log_files=[log_a, log_b], wait=False)
        assert processed == 2
        target.execute_async_insert.assert_called_once()   # one (target, mode) group
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()

    def test_dispatch_async_groups_preparsed_ops(self):
        # dispatch_async takes already-parsed ops directly.
        from yggdrasil.databricks.table.tables import Tables
        svc = _service()
        log_a, log_b = MagicMock(), MagicMock()
        ops = [
            ("c.s.t", "append", "dbfs+volume:/c/s/t/.sql/tmp/a.parquet", log_a),
            ("c.s.t", "append", "dbfs+volume:/c/s/t/.sql/tmp/b.parquet", log_b),
        ]
        target = MagicMock()
        from_fn, _ = _fake_databricks_from()
        with patch.object(Tables, "__getitem__", return_value=target), \
             patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = svc.dispatch_async(ops)
        assert processed == 2
        target.execute_async_insert.assert_called_once()
        union = target.execute_async_insert.call_args.args[0]
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union


class TestTableJobRunDelegates:
    def test_run_delegates_to_service_loader(self):
        t = _table_mock()
        logs = MagicMock()
        # logs_path(table) → staging_volume.path(LOGS_SUBDIR)
        t.staging_volume.path.return_value = logs
        t.service.async_insert.return_value = 7

        out = TableJob(t).run(wait=False, limit=3)

        t.service.async_insert.assert_called_once_with(logs, wait=False, limit=3)
        assert out == 7
