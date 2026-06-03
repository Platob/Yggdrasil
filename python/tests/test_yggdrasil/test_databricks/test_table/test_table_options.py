"""``TableOptions.engine`` (EngineName) → compute-engine dispatch.

``engine`` picks YGGDRASIL (native DeltaFolder) / DATABRICKS_SQL_WAREHOUSE /
SPARK explicitly; ``None`` guesses best (active Spark → SPARK; small Delta
< 128 MiB → YGGDRASIL; else → DATABRICKS_SQL_WAREHOUSE). A YGGDRASIL pick on
a table that can't take the native path, or a UC-credential failure, degrades
to the warehouse.
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
from yggdrasil.enums import EngineName
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.databricks.table.table import Table, _NATIVE_DELTA_MAX_BYTES

WH = EngineName.DATABRICKS_SQL_WAREHOUSE
YG = EngineName.YGGDRASIL
SP = EngineName.SPARK


class TestEngineName:
    def test_members(self):
        assert (YG.value, WH.value, SP.value) == (0, 1, 2)

    def test_from_aliases(self):
        assert EngineName.from_("api") is WH
        assert EngineName.from_("warehouse") is WH
        assert EngineName.from_("ygg") is YG
        assert EngineName.from_("native") is YG
        assert EngineName.from_("spark") is SP
        assert EngineName.from_(2) is SP
        assert EngineName.from_(None) is None


class TestTableOptions:
    def test_options_class_is_table_options(self):
        assert Table.options_class() is TableOptions

    def test_engine_defaults_none(self):
        assert TableOptions().engine is None

    def test_engine_string_coerced(self):
        assert TableOptions(engine="warehouse").engine is WH
        assert TableOptions(engine="spark").engine is SP

    def test_rehome_from_cast_options_keeps_fields(self):
        homed = TableOptions.check(CastOptions(row_limit=7))
        assert isinstance(homed, TableOptions)
        assert homed.row_limit == 7 and homed.engine is None


def _table(table_type, fmt, storage="s3://b/x"):
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(
        table_type=table_type, data_source_format=fmt, storage_location=storage,
    )
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


class TestResolveEngineExplicit:
    def test_explicit_engines_pass_through(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t._resolve_engine(TableOptions(engine=WH), write=False) is WH
        assert t._resolve_engine(TableOptions(engine=SP), write=True) is SP
        assert t._resolve_engine(TableOptions(engine=YG), write=False) is YG
        assert t._resolve_engine(TableOptions(engine=YG), write=True) is YG

    def test_yggdrasil_degrades_when_not_delta(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.PARQUET)
        assert t._resolve_engine(TableOptions(engine=YG), write=False) is WH

    def test_yggdrasil_write_on_managed_degrades(self):
        # Managed Delta can't take a native write (read-only creds) → warehouse.
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        assert t._resolve_engine(TableOptions(engine=YG), write=True) is WH
        # but a managed read can be native
        assert t._resolve_engine(TableOptions(engine=YG), write=False) is YG


class TestResolveEngineGuess:
    """``engine=None`` — active Spark vs Delta-size heuristic."""

    def _ext(self):
        return _table(TableType.EXTERNAL, DataSourceFormat.DELTA)

    def test_active_spark_guesses_spark(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: True))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: 1)
        assert t._resolve_engine(TableOptions(), write=False) is SP

    def test_small_delta_guesses_yggdrasil(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES - 1)
        assert t._resolve_engine(TableOptions(), write=False) is YG
        assert t._resolve_engine(TableOptions(), write=True) is YG

    def test_large_delta_guesses_warehouse(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES)
        assert t._resolve_engine(TableOptions(), write=False) is WH

    def test_unknown_size_guesses_warehouse(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: None)
        assert t._resolve_engine(TableOptions(), write=False) is WH

    def test_managed_write_guess_warehouse_even_when_small(self, monkeypatch):
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: 1)
        # managed write isn't delta-capable → warehouse; managed read is native
        assert t._resolve_engine(TableOptions(), write=True) is WH
        assert t._resolve_engine(TableOptions(), write=False) is YG


class TestCredentialFallback:
    """A UC-credential failure makes the native probe return None (caller →
    warehouse). The probe vends the scope that matches the operation."""

    def test_probe_none_on_delta_error(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        t.delta = MagicMock(side_effect=RuntimeError("could not generate credentials"))
        assert t._native_delta_folder(write=True) is None

    def test_probe_folder_when_ok(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        folder = MagicMock()
        t.delta = MagicMock(return_value=folder)
        assert t._native_delta_folder(write=False) is folder
        folder.snapshot.assert_called_once()

    def test_read_probe_reads_write_probe_writes(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        t.delta = MagicMock(return_value=MagicMock())
        t._native_delta_folder(write=False)
        t._native_delta_folder(write=True)
        assert t.delta.call_args_list[0].kwargs == {"write": False}
        assert t.delta.call_args_list[1].kwargs == {"write": True}

    def test_write_denied_read_ok(self):
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

    def test_write_none_uses_default(self):
        t = self._t()
        t.storage_path(write=None)
        t.aws.assert_called_once_with()
