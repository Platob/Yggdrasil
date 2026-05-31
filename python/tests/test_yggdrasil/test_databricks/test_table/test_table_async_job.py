"""Unit tests for the async, file-arrival table loader (no live Databricks).

Covers:
* ``Table.insert(wait=False)`` routing → ``_async_insert`` (OVERWRITE/APPEND,
  no match_by), and the sync path otherwise;
* ``_async_insert`` staging a Parquet + dropping a JSON operation log;
* ``TableJob.ensure`` get-or-create with a file-arrival trigger;
* ``TableJob.process`` aggregating logs into one INSERT per (target, mode),
  then cleaning up consumed logs + data.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.table import async_job as aj
from yggdrasil.databricks.table.async_job import (
    DATA_SUBDIR,
    LOGS_SUBDIR,
    TableJob,
)
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
        t._async_insert.assert_called_once()
        assert t._async_insert.call_args.kwargs["mode"] == "append"
        assert out is t._async_insert.return_value
        t.insert_into.assert_not_called()

    def test_wait_true_uses_sync_path(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append")  # wait defaults True
        t.insert_into.assert_called_once()
        t._async_insert.assert_not_called()

    def test_match_by_stays_sync(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append", wait=False, match_by=["id"])
        t.insert_into.assert_called_once()
        t._async_insert.assert_not_called()

    def test_merge_mode_stays_sync(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="merge", wait=False)
        t.insert_into.assert_called_once()
        t._async_insert.assert_not_called()


# --------------------------------------------------------------------------- #
# Table._async_insert
# --------------------------------------------------------------------------- #
class TestAsyncInsert:
    def test_rejects_non_overwrite_append(self):
        t = _table_mock()
        with pytest.raises(ValueError, match="OVERWRITE / APPEND"):
            Table._async_insert(t, object(), mode="merge")

    def test_rejects_match_by(self):
        t = _table_mock()
        with pytest.raises(ValueError, match="match_by"):
            Table._async_insert(t, object(), mode="append", match_by=["id"])

    def test_writes_parquet_and_log(self):
        t = _table_mock()
        data_dir, logs_dir = MagicMock(), MagicMock()
        data_file, log_file = MagicMock(), MagicMock()
        data_dir.__truediv__.return_value = data_file
        logs_dir.__truediv__.return_value = log_file

        with patch.object(TableJob, "data_path", staticmethod(lambda tbl: data_dir)), \
             patch.object(TableJob, "logs_path", staticmethod(lambda tbl: logs_dir)):
            result = Table._async_insert(t, {"a": [1]}, mode="append")

        assert result is log_file
        data_file.write_table.assert_called_once()           # staged Parquet
        log_file.write_bytes.assert_called_once()            # operation log
        payload = json.loads(log_file.write_bytes.call_args[0][0])
        assert payload["target"] == "c.s.t"
        assert payload["mode"] == "append"
        assert payload["data"].endswith(".parquet")
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

        tj = TableJob(t, service=jobs)
        assert tj.ensure() is tj
        assert tj.job_id == 42

        kwargs = jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "ygg-async-insert-c.s.t"
        assert kwargs["trigger"].file_arrival.url == "/Volumes/c/s/t/.sql/async/logs"
        task = kwargs["tasks"][0]
        assert task.python_wheel_task.parameters == ["c.s.t"]

    def test_ensure_is_noop_when_already_resolved(self):
        t = _table_mock()
        jobs = MagicMock()
        tj = TableJob(t, service=jobs, job_id=7)
        assert tj.ensure() is tj
        jobs.create_or_update.assert_not_called()


# --------------------------------------------------------------------------- #
# TableJob.process (aggregate logs → INSERT per (target, mode))
# --------------------------------------------------------------------------- #
class TestProcess:
    def _wire_dirs(self, table):
        logs_dir, data_dir = MagicMock(), MagicMock()
        table.staging_volume.path.side_effect = (
            lambda sub, *a, **k: logs_dir if sub == LOGS_SUBDIR else data_dir
        )
        data_files: dict[str, MagicMock] = {}

        def _div(leaf):
            m = MagicMock()
            m.full_path.return_value = f"/Volumes/c/s/t/{DATA_SUBDIR}/{leaf}"
            data_files[leaf] = m
            return m

        data_dir.__truediv__.side_effect = _div
        return logs_dir, data_dir, data_files

    @staticmethod
    def _log(op, *, target="c.s.t", mode="append"):
        f = MagicMock()
        f.name = f"{op}.json"
        f.read_bytes.return_value = json.dumps(
            {"target": target, "mode": mode, "data": f"{op}.parquet"}
        ).encode()
        return f

    def test_no_logs_returns_zero(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir, _, _ = self._wire_dirs(t)
        logs_dir.exists.return_value = False
        assert TableJob(t).process() == 0
        t.insert.assert_not_called()

    def test_aggregates_same_group_into_one_insert(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir, data_dir, data_files = self._wire_dirs(t)
        logs_dir.exists.return_value = True
        log_a, log_b = self._log("a"), self._log("b")
        logs_dir.iterdir.return_value = [log_a, log_b]

        processed = TableJob(t).process(wait=False)

        assert processed == 2
        t.insert.assert_called_once()
        union = t.insert.call_args.args[0]
        assert "UNION ALL" in union
        assert f"parquet.`/Volumes/c/s/t/{DATA_SUBDIR}/a.parquet`" in union
        assert f"parquet.`/Volumes/c/s/t/{DATA_SUBDIR}/b.parquet`" in union
        assert t.insert.call_args.kwargs["mode"] == "append"
        # consumed logs + data cleaned up
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()
        data_files["a.parquet"].unlink.assert_called_once()
        data_files["b.parquet"].unlink.assert_called_once()

    def test_splits_by_mode(self):
        t = _table_mock()
        t.client.jobs = MagicMock()
        logs_dir, data_dir, _ = self._wire_dirs(t)
        logs_dir.exists.return_value = True
        logs_dir.iterdir.return_value = [
            self._log("a", mode="append"),
            self._log("b", mode="overwrite"),
        ]

        processed = TableJob(t).process()

        assert processed == 2
        modes = {c.kwargs["mode"] for c in t.insert.call_args_list}
        assert modes == {"append", "overwrite"}      # one INSERT per mode
