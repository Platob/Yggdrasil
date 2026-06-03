"""``requalify_table_refs`` + view-clone requalification.

When a view is cloned to a different schema, its stored ``view_definition``
still points at the *source* catalog/schema; ``Table.clone`` must re-point
the inner query at the target. These check the textual requalifier and that
``clone`` applies it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from databricks.sdk.service.catalog import TableInfo, TableType

from yggdrasil.databricks.sql.sql_utils import requalify_table_refs
from yggdrasil.databricks.table.table import Table


SRC = ("c", "s")
TGT = ("c2", "s2")


class TestRequalifyTableRefs:
    def test_three_part_backticked(self):
        assert requalify_table_refs(
            "SELECT * FROM `c`.`s`.`t`", source=SRC, target=TGT
        ) == "SELECT * FROM `c2`.`s2`.`t`"

    def test_three_part_bare(self):
        assert requalify_table_refs(
            "SELECT * FROM c.s.t", source=SRC, target=TGT
        ) == "SELECT * FROM `c2`.`s2`.t"

    def test_bare_schema_qualified(self):
        assert requalify_table_refs(
            "select a from s.t join s.u on a=b", source=SRC, target=TGT
        ) == "select a from `s2`.t join `s2`.u on a=b"

    def test_other_catalog_same_schema_untouched(self):
        # other_c.s.u must keep its schema — only c.s.* is rewritten.
        out = requalify_table_refs(
            "SELECT * FROM `c`.`s`.`t` JOIN other_c.s.u ON 1=1",
            source=SRC, target=TGT,
        )
        assert out == "SELECT * FROM `c2`.`s2`.`t` JOIN other_c.s.u ON 1=1"

    def test_substring_schema_not_matched(self):
        assert requalify_table_refs(
            "SELECT * FROM s_other.t", source=SRC, target=TGT
        ) == "SELECT * FROM s_other.t"
        assert requalify_table_refs(
            "SELECT * FROM `c`.`s_x`.`t`", source=SRC, target=TGT
        ) == "SELECT * FROM `c`.`s_x`.`t`"

    def test_noop_when_source_equals_target(self):
        assert requalify_table_refs(
            "SELECT * FROM c.s.t", source=SRC, target=SRC
        ) == "SELECT * FROM c.s.t"

    def test_same_catalog_different_schema(self):
        assert requalify_table_refs(
            "SELECT * FROM c.s.t", source=("c", "s"), target=("c", "s2")
        ) == "SELECT * FROM `c`.`s2`.t"

    def test_missing_schema_returns_unchanged(self):
        assert requalify_table_refs(
            "SELECT 1", source=("c", None), target=TGT
        ) == "SELECT 1"


class TestCloneViewRequalifies:
    def _view(self, definition: str) -> tuple[Table, list[str]]:
        svc = MagicMock(name="Tables")
        src = Table(service=svc, catalog_name="c", schema_name="s", table_name="v")
        src._infos = TableInfo(table_type=TableType.VIEW, view_definition=definition)
        captured: list[str] = []
        src.sql.execute = lambda stmt, **kw: captured.append(stmt)
        return src, captured

    def test_clone_view_rewrites_inner_schema(self):
        src, captured = self._view("SELECT id FROM `c`.`s`.`t` WHERE id > 1")
        target = Table(service=src.sql.tables, catalog_name="c2", schema_name="s2", table_name="v2")
        src.clone(target)
        ddl = captured[0]
        assert "`c2`.`s2`.`v2`" in ddl          # target view name
        assert "`c2`.`s2`.`t`" in ddl           # inner ref requalified
        assert "`c`.`s`.`t`" not in ddl         # source ref gone
        assert "WHERE id > 1" in ddl            # rest of the query preserved

    def test_clone_view_same_schema_keeps_definition(self):
        src, captured = self._view("SELECT id FROM `c`.`s`.`t`")
        # Different table name, same schema → inner query unchanged.
        target = Table(service=src.sql.tables, catalog_name="c", schema_name="s", table_name="v2")
        src.clone(target)
        assert "`c`.`s`.`t`" in captured[0]
