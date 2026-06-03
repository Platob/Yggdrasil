"""``TableOptions`` + the ``use_warehouse`` → native-DeltaFolder dispatch.

``use_warehouse=True`` keeps reads/writes on the SQL warehouse; ``False``
prefers the native DeltaFolder when the table is Delta-backed (reads:
managed or external; writes: external only). ``None`` (default) guesses:
active Spark → Databricks; else a small Delta table (< 128 MiB) → native,
larger → Databricks.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from databricks.sdk.service.catalog import DataSourceFormat, TableInfo, TableType

from yggdrasil.data.options import CastOptions
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.databricks.table.table import Table, _NATIVE_DELTA_MAX_BYTES


class TestTableOptions:
    def test_options_class_is_table_options(self):
        assert Table.options_class() is TableOptions

    def test_use_warehouse_defaults_none(self):
        assert TableOptions().use_warehouse is None

    def test_rehome_from_cast_options_keeps_fields(self):
        homed = TableOptions.check(CastOptions(row_limit=7))
        assert isinstance(homed, TableOptions)
        assert homed.row_limit == 7
        assert homed.use_warehouse is None

    def test_override(self):
        o = TableOptions.check(None, use_warehouse=False, row_limit=3)
        assert o.use_warehouse is False and o.row_limit == 3


def _table(table_type, fmt, storage="s3://b/x"):
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(
        table_type=table_type, data_source_format=fmt, storage_location=storage,
    )
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


class TestExplicitDispatch:
    def test_use_warehouse_true_never_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t._native_delta(TableOptions(use_warehouse=True), write=False) is False
        assert t._native_delta(TableOptions(use_warehouse=True), write=True) is False

    def test_use_warehouse_false_external_delta_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t.is_delta is True
        assert t._native_delta(TableOptions(use_warehouse=False), write=False) is True
        assert t._native_delta(TableOptions(use_warehouse=False), write=True) is True

    def test_use_warehouse_false_managed_reads_native_writes_sql(self):
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        assert t._native_delta(TableOptions(use_warehouse=False), write=False) is True
        assert t._native_delta(TableOptions(use_warehouse=False), write=True) is False

    def test_non_delta_never_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.PARQUET)
        assert t._native_delta(TableOptions(use_warehouse=False), write=False) is False

    def test_view_never_native(self):
        t = _table(TableType.VIEW, DataSourceFormat.DELTA, storage=None)
        assert t._native_delta(TableOptions(use_warehouse=False), write=False) is False


class TestGuessDispatch:
    """``use_warehouse=None`` — active Spark vs table-size heuristic."""

    def _ext(self):
        return _table(TableType.EXTERNAL, DataSourceFormat.DELTA)

    def test_active_spark_uses_warehouse(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: True))
        # size would say native, but Spark wins → warehouse
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: 1)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_small_table_uses_native(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES - 1)
        assert t._native_delta(TableOptions(), write=False) is True
        assert t._native_delta(TableOptions(), write=True) is True

    def test_large_table_uses_warehouse(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: _NATIVE_DELTA_MAX_BYTES)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_unknown_size_falls_back_to_warehouse(self, monkeypatch):
        t = self._ext()
        monkeypatch.setattr(Table, "_has_active_spark", staticmethod(lambda o: False))
        monkeypatch.setattr(Table, "_delta_total_bytes", lambda self: None)
        assert t._native_delta(TableOptions(), write=False) is False

    def test_spark_session_on_options_counts_as_active(self):
        # _has_active_spark short-circuits on a bound session — no pyspark import.
        assert Table._has_active_spark(TableOptions(spark_session=object())) is True
