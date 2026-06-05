"""``TableOptions.engine`` (EngineType) → compute-engine dispatch.

``engine`` picks YGGDRASIL (native DeltaFolder) / DATABRICKS_SQL_WAREHOUSE /
SPARK explicitly; ``None`` guesses best (active Spark → SPARK; else →
DATABRICKS_SQL_WAREHOUSE). The native DeltaFolder path is **never** auto-routed
— for a read or a write — because reading/writing straight off the table's
storage ``_delta_log`` bypasses the warehouse (and its governance / staging);
it is taken only when ``engine=YGGDRASIL`` is set explicitly. A YGGDRASIL pick
on a table that can't take the native path, or a UC-credential failure,
degrades to the warehouse.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from databricks.sdk.service.catalog import (
    DataSourceFormat,
    TableInfo,
    TableOperation,
    TableType,
)

from yggdrasil.data.options import CastOptions
from yggdrasil.enums import EngineType
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.databricks.table.table import Table

WH = EngineType.DATABRICKS_SQL_WAREHOUSE
YG = EngineType.YGGDRASIL
SP = EngineType.SPARK


class TestEngineType:
    def test_members(self):
        assert (YG.value, WH.value, SP.value) == (0, 1, 2)

    def test_from_str_aliases(self):
        assert EngineType.from_str("warehouse") is WH
        assert EngineType.from_str("sql") is WH
        assert EngineType.from_str("ygg") is YG
        assert EngineType.from_str("native") is YG
        assert EngineType.from_str("spark") is SP
        assert EngineType.from_str("connect") is SP

    def test_api_alias_removed(self):
        # ``api`` is too broad — no longer maps to an engine.
        with pytest.raises(ValueError):
            EngineType.from_str("api")
        assert EngineType.from_str("api", default=WH) is WH

    def test_from_numeric(self):
        assert EngineType.from_numeric(0) is YG
        assert EngineType.from_numeric(2) is SP
        with pytest.raises(ValueError):
            EngineType.from_numeric(9)
        assert EngineType.from_numeric(9, default=None) is None
        with pytest.raises(ValueError):
            EngineType.from_numeric(True)  # bool rejected

    def test_from_dispatch(self):
        assert EngineType.from_(SP) is SP
        assert EngineType.from_("warehouse") is WH
        assert EngineType.from_(2) is SP
        assert EngineType.from_(None) is None      # unset → None
        assert EngineType.from_(...) is None
        with pytest.raises(ValueError):
            EngineType.from_("nope")
        assert EngineType.from_("nope", default=WH) is WH


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


class TestDeltaCapable:
    """``_delta_capable`` — when the storage path is natively read/writable."""

    def test_plain_external_is_read_and_write_capable(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA, storage="s3://b/ext/t")
        assert t._delta_capable(write=False) is True
        assert t._delta_capable(write=True) is True

    def test_uc_managed_external_is_not_writable(self):
        # A ``__unitycatalog`` layout is UC-governed: direct PutObject is denied,
        # so the storage path is not writable even though the table is external.
        # Reads may still go direct.
        t = _table(
            TableType.EXTERNAL, DataSourceFormat.DELTA,
            storage="s3://b/metastore/__unitycatalog/catalogs/c/tables/t",
        )
        assert t._delta_capable(write=False) is True
        assert t._delta_capable(write=True) is False

    def test_managed_is_not_writable(self):
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        assert t._delta_capable(write=True) is False


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
    """``engine=None`` — active Spark → SPARK, else warehouse. The native
    DeltaFolder path is never auto-selected (read or write)."""

    def _ext(self):
        return _table(TableType.EXTERNAL, DataSourceFormat.DELTA)

    def test_active_spark_guesses_spark(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: True))
        assert t._resolve_engine(TableOptions(), write=False) is SP
        assert t._resolve_engine(TableOptions(), write=True) is SP

    def test_external_delta_read_guesses_warehouse(self, monkeypatch):
        # Even a small external Delta table no longer auto-reads native — the
        # native path must be requested explicitly with ``engine=YGGDRASIL``.
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        assert t._resolve_engine(TableOptions(), write=False) is WH
        assert t._resolve_engine(TableOptions(), write=True) is WH

    def test_managed_delta_read_guesses_warehouse(self, monkeypatch):
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        assert t._resolve_engine(TableOptions(), write=False) is WH
        assert t._resolve_engine(TableOptions(), write=True) is WH


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
