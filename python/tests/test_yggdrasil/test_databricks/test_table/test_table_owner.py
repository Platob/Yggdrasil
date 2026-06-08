"""``Table.owner`` getter + setter (Unity Catalog ownership)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks.table.table import Table


def _table(table_type=TableType.MANAGED, owner="alice@x.com") -> Table:
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(table_type=table_type, owner=owner)
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


def _capture(t: Table) -> list[str]:
    out: list[str] = []
    t.sql.execute = lambda stmt, **kw: out.append(stmt)
    t.invalidate_singleton = lambda **kw: None  # type: ignore[method-assign]
    return out


class TestOwner:
    def test_get_owner(self):
        assert _table(owner="data_eng").owner == "data_eng"

    def test_set_owner_table_emits_alter_table(self):
        t = _table(TableType.MANAGED)
        sql = _capture(t)
        t.owner = "data_eng"
        assert sql == ["ALTER TABLE `c`.`s`.`t` OWNER TO `data_eng`"]

    def test_set_owner_view_emits_alter_view(self):
        t = _table(TableType.VIEW)
        sql = _capture(t)
        t.owner = "analysts"
        assert sql == ["ALTER VIEW `c`.`s`.`t` OWNER TO `analysts`"]

    def test_set_owner_quotes_principal(self):
        t = _table(TableType.MANAGED)
        sql = _capture(t)
        t.owner = "group with space"
        assert sql == ["ALTER TABLE `c`.`s`.`t` OWNER TO `group with space`"]

    def test_set_owner_invalidates_cache(self):
        t = _table(TableType.MANAGED)
        t.sql.execute = lambda stmt, **kw: None
        seen = {}
        t.invalidate_singleton = lambda **kw: seen.update(kw)  # type: ignore[method-assign]
        t.owner = "x"
        assert seen.get("remove_global") is True

    def test_empty_owner_raises(self):
        t = _table()
        with pytest.raises(ValueError):
            t.owner = ""
