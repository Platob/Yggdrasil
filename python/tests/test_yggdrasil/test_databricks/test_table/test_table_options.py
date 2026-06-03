"""``TableOptions`` + the ``use_databricks`` → native-DeltaFolder dispatch.

``use_databricks=True`` keeps reads/writes on Databricks (the SQL warehouse);
``False`` prefers the native DeltaFolder when the table is Delta-backed
(reads: managed or external; writes: external only). ``None`` (default)
guesses: active Spark → Databricks; else a small Delta table (< 128 MiB) →
native, larger → Databricks. Any UC-credential failure on the native path
falls back to Databricks.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from databricks.sdk.service.catalog import (
    DataSourceFormat,
    TableInfo,
    TableOperation,
    TableType,
)

from yggdrasil.data.options import CastOptions
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.databricks.table.table import Table, _NATIVE_DELTA_MAX_BYTES


class TestTableOptions:
    def test_options_class_is_table_options(self):
        assert Table.options_class() is TableOptions

    def test_use_databricks_defaults_none(self):
        assert TableOptions().use_databricks is None

    def test_rehome_from_cast_options_keeps_fields(self):
        homed = TableOptions.check(CastOptions(row_limit=7))
        assert isinstance(homed, TableOptions)
        assert homed.row_limit == 7
        assert homed.use_databricks is None

    def test_override(self):
        o = TableOptions.check(None, use_databricks=False, row_limit=3)
        assert o.use_databricks is False and o.row_limit == 3


def _table(table_type, fmt, storage="s3://b/x"):
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(
        table_type=table_type, data_source_format=fmt, storage_location=storage,
    )
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


class TestExplicitDispatch:
    def test_use_databricks_true_never_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t._native_delta(TableOptions(use_databricks=True), write=False) is False
        assert t._native_delta(TableOptions(use_databricks=True), write=True) is False

    def test_use_databricks_false_external_delta_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t.is_delta is True
        assert t._native_delta(TableOptions(use_databricks=False), write=False) is True
        assert t._native_delta(TableOptions(use_databricks=False), write=True) is True

    def test_use_databricks_false_managed_reads_native_writes_sql(self):
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        assert t._native_delta(TableOptions(use_databricks=False), write=False) is True
        assert t._native_delta(TableOptions(use_databricks=False), write=True) is False

    def test_non_delta_never_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.PARQUET)
        assert t._native_delta(TableOptions(use_databricks=False), write=False) is False

    def test_view_never_native(self):
        t = _table(TableType.VIEW, DataSourceFormat.DELTA, storage=None)
        assert t._native_delta(TableOptions(use_databricks=False), write=False) is False


class TestGuessDispatch:
    """``use_databricks=None`` — active Spark vs table-size heuristic."""

    def _ext(self):
        return _table(TableType.EXTERNAL, DataSourceFormat.DELTA)

    def test_active_spark_uses_databricks(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: True))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: 1)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_small_table_uses_native(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES - 1)
        assert t._native_delta(TableOptions(), write=False) is True
        assert t._native_delta(TableOptions(), write=True) is True

    def test_large_table_uses_databricks(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_unknown_size_falls_back_to_databricks(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: None)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_spark_session_on_options_counts_as_active(self):
        assert Table._has_active_spark(TableOptions(spark_session=object())) is True


class TestCredentialFallback:
    """A UC-credential / storage failure makes the native probe return None,
    so the caller falls back to Databricks. The probe vends the credential
    scope that matches the operation (read vs read-write)."""

    def test_probe_returns_none_on_credential_error(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        t.delta = MagicMock(side_effect=RuntimeError("could not generate credentials"))
        assert t._native_delta_folder(write=True) is None

    def test_probe_returns_folder_when_snapshot_ok(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        folder = MagicMock()
        t.delta = MagicMock(return_value=folder)
        assert t._native_delta_folder(write=False) is folder
        folder.snapshot.assert_called_once()

    def test_probe_returns_none_when_snapshot_raises(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        folder = MagicMock()
        folder.snapshot.side_effect = RuntimeError("403 vend failed")
        t.delta = MagicMock(return_value=folder)
        assert t._native_delta_folder(write=True) is None

    def test_read_probe_vends_read_scope_write_probe_vends_write(self):
        # The probe must ask delta() for the matching credential scope, so a
        # read works even when the principal can't write.
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        t.delta = MagicMock(return_value=MagicMock())
        t._native_delta_folder(write=False)
        t._native_delta_folder(write=True)
        assert t.delta.call_args_list[0].kwargs == {"write": False}
        assert t.delta.call_args_list[1].kwargs == {"write": True}

    def test_write_denied_but_read_ok_falls_back_on_write_only(self):
        # READ_WRITE vend denied, READ vend fine — read native, write falls back.
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)

        def delta(*, write):
            f = MagicMock()
            if write:
                f.snapshot.side_effect = RuntimeError("403: no write creds")
            return f

        t.delta = MagicMock(side_effect=delta)
        assert t._native_delta_folder(write=False) is not None
        assert t._native_delta_folder(write=True) is None


class TestStoragePathCredentialScope:
    """``storage_path(write=...)`` asks ``aws`` for the matching UC scope."""

    def _t(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        t.storage_location = MagicMock(return_value="s3://b/x")
        t.aws = MagicMock()
        return t

    def test_write_false_requests_read(self):
        t = self._t()
        t.storage_path(write=False)
        t.aws.assert_called_once_with(TableOperation.READ)

    def test_write_true_requests_read_write(self):
        t = self._t()
        t.storage_path(write=True)
        t.aws.assert_called_once_with(TableOperation.READ_WRITE)

    def test_write_none_uses_table_type_default(self):
        t = self._t()
        t.storage_path(write=None)
        t.aws.assert_called_once_with()
