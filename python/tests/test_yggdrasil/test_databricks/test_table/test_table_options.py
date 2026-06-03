"""``TableOptions`` + the ``prefer_sql`` → native-DeltaFolder dispatch.

``prefer_sql=True`` (default) keeps reads/writes on the SQL warehouse;
``prefer_sql=False`` prefers the native DeltaFolder path when the table is
Delta-backed (reads: managed or external; writes: external only, since UC
vends READ_WRITE creds for external tables only).
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

from databricks.sdk.service.catalog import DataSourceFormat, TableInfo, TableType

from yggdrasil.data.options import CastOptions
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.databricks.table.table import Table


class TestTableOptions:
    def test_options_class_is_table_options(self):
        assert Table.options_class() is TableOptions

    def test_prefer_sql_defaults_true(self):
        assert TableOptions().prefer_sql is True

    def test_rehome_from_cast_options_keeps_fields(self):
        homed = TableOptions.check(CastOptions(row_limit=7))
        assert isinstance(homed, TableOptions)
        assert homed.row_limit == 7
        assert homed.prefer_sql is True

    def test_override(self):
        o = TableOptions.check(None, prefer_sql=False, row_limit=3)
        assert o.prefer_sql is False and o.row_limit == 3


def _table(table_type, fmt, storage="s3://b/x"):
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(
        table_type=table_type, data_source_format=fmt, storage_location=storage,
    )
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


class TestDeltaDispatch:
    def test_prefer_sql_true_never_uses_delta(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t._prefer_delta_read(TableOptions(prefer_sql=True)) is False
        assert t._prefer_delta_write(TableOptions(prefer_sql=True)) is False

    def test_external_delta_reads_and_writes_native(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA)
        assert t.is_delta is True
        assert t._prefer_delta_read(TableOptions(prefer_sql=False)) is True
        assert t._prefer_delta_write(TableOptions(prefer_sql=False)) is True

    def test_managed_delta_reads_native_but_writes_sql(self):
        # Managed Delta: UC vends read-only creds → native read, SQL write.
        t = _table(TableType.MANAGED, DataSourceFormat.DELTA)
        assert t._prefer_delta_read(TableOptions(prefer_sql=False)) is True
        assert t._prefer_delta_write(TableOptions(prefer_sql=False)) is False

    def test_non_delta_never_uses_delta(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.PARQUET)
        assert t.is_delta is False
        assert t._prefer_delta_read(TableOptions(prefer_sql=False)) is False
        assert t._prefer_delta_write(TableOptions(prefer_sql=False)) is False

    def test_view_never_uses_delta(self):
        t = _table(TableType.VIEW, DataSourceFormat.DELTA, storage=None)
        assert t.is_delta is False
        assert t._prefer_delta_read(TableOptions(prefer_sql=False)) is False

    def test_no_storage_location_falls_back(self):
        t = _table(TableType.EXTERNAL, DataSourceFormat.DELTA, storage=None)
        assert t._prefer_delta_read(TableOptions(prefer_sql=False)) is False
        assert t._prefer_delta_write(TableOptions(prefer_sql=False)) is False
