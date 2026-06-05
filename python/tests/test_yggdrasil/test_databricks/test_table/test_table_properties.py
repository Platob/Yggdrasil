"""``Table.properties`` — transparent ``TBLPROPERTIES`` MutableMapping.

Asserts the value-diff contract: reads come off cached ``infos`` (no
network), and a mutation only hits the warehouse when it actually changes
the catalog.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks.table.table import Table, TableProperties


def _table(table_type=TableType.MANAGED, properties=None) -> Table:
    t = Table(service=MagicMock(name="Tables"), catalog_name="c", schema_name="s", table_name="t")
    t._infos = TableInfo(table_type=table_type, properties=dict(properties or {}))
    t._infos_fetched_at = time.time()  # keep the cache fresh (no network)
    return t


def _capture(t: Table) -> list[str]:
    out: list[str] = []
    t.sql.execute = lambda stmt, **kw: out.append(stmt)
    t.invalidate_singleton = lambda **kw: None  # type: ignore[method-assign]
    return out


class TestRead:
    def test_is_mutable_mapping(self):
        assert isinstance(_table().properties, TableProperties)

    def test_get_iter_len_contains(self):
        p = _table(properties={"a": "1", "b": "2"}).properties
        assert p["a"] == "1"
        assert len(p) == 2
        assert set(p) == {"a", "b"}
        assert "a" in p and "z" not in p
        assert dict(p) == {"a": "1", "b": "2"}

    def test_get_missing_raises_keyerror_no_network(self):
        t = _table(properties={"a": "1"})
        sql = _capture(t)
        with pytest.raises(KeyError):
            _ = t.properties["nope"]
        assert sql == []


class TestSet:
    def test_set_changed_emits_one_alter(self):
        t = _table(properties={"a": "1"})
        sql = _capture(t)
        t.properties["a"] = "2"
        assert sql == ["ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('a' = '2')"]

    def test_set_new_key_emits_alter(self):
        t = _table(properties={})
        sql = _capture(t)
        t.properties["delta.appendOnly"] = "true"
        assert sql == [
            "ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('delta.appendOnly' = 'true')"
        ]

    def test_set_unchanged_skips_network(self):
        t = _table(properties={"a": "1"})
        sql = _capture(t)
        t.properties["a"] = "1"
        assert sql == []

    def test_set_coerces_value_to_str(self):
        t = _table(properties={})
        sql = _capture(t)
        t.properties["n"] = 5
        assert sql == ["ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('n' = '5')"]

    def test_set_unchanged_after_coercion_skips_network(self):
        t = _table(properties={"n": "5"})
        sql = _capture(t)
        t.properties["n"] = 5  # str(5) == "5" already
        assert sql == []

    def test_set_escapes_quotes(self):
        t = _table(properties={})
        sql = _capture(t)
        t.properties["k"] = "a'b"
        assert sql == ["ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('k' = 'a''b')"]

    def test_set_on_view_emits_alter_view(self):
        t = _table(TableType.VIEW, properties={})
        sql = _capture(t)
        t.properties["x"] = "y"
        assert sql == ["ALTER VIEW `c`.`s`.`t` SET TBLPROPERTIES ('x' = 'y')"]

    def test_set_invalidates_cache(self):
        t = _table(properties={})
        t.sql.execute = lambda stmt, **kw: None
        seen = {}
        t.invalidate_singleton = lambda **kw: seen.update(kw)  # type: ignore[method-assign]
        t.properties["a"] = "1"
        assert seen.get("remove_global") is True


class TestDelete:
    def test_delete_present_emits_unset(self):
        t = _table(properties={"a": "1"})
        sql = _capture(t)
        del t.properties["a"]
        assert sql == [
            "ALTER TABLE `c`.`s`.`t` UNSET TBLPROPERTIES IF EXISTS ('a')"
        ]

    def test_delete_absent_raises_no_network(self):
        t = _table(properties={"a": "1"})
        sql = _capture(t)
        with pytest.raises(KeyError):
            del t.properties["nope"]
        assert sql == []


class TestUpdate:
    def test_update_batches_only_changed_keys(self):
        t = _table(properties={"a": "1", "b": "2"})
        sql = _capture(t)
        t.properties.update({"a": "1", "b": "9", "c": "3"})  # a unchanged
        assert sql == [
            "ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('b' = '9', 'c' = '3')"
        ]

    def test_update_all_noop_skips_network(self):
        t = _table(properties={"a": "1", "b": "2"})
        sql = _capture(t)
        t.properties.update({"a": "1", "b": "2"})
        assert sql == []

    def test_update_kwargs(self):
        t = _table(properties={})
        sql = _capture(t)
        t.properties.update(x="1")
        assert sql == ["ALTER TABLE `c`.`s`.`t` SET TBLPROPERTIES ('x' = '1')"]
