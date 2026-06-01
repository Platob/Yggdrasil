"""Unit tests for Table._delete dispatch (no live Databricks needed).

* a predicate → server-side ``DELETE FROM … WHERE …`` via the SQL engine;
* no predicate → the UC tables API drop, inline (NOT via ``self.delete``,
  which dispatches back into ``_delete`` and would recurse forever), so no
  SQL warehouse is started just to empty the table.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from yggdrasil.databricks.table.table import Table


def test_delete_with_predicate_runs_sql_delete():
    t = MagicMock()
    t.full_name.return_value = "`c`.`s`.`tbl`"

    result = Table._delete(t, "id = 1", wait=True)

    assert result == 0
    t.sql.execute.assert_called_once()
    query = t.sql.execute.call_args[0][0]
    assert query == "DELETE FROM `c`.`s`.`tbl` WHERE id = 1"
    # No UC asset drop when only filtering rows.
    t.client.workspace_client.return_value.tables.delete.assert_not_called()


def test_delete_without_predicate_drops_asset_no_warehouse():
    t = MagicMock()
    t.full_name.return_value = "c.s.tbl"
    uc = t.client.workspace_client.return_value.tables

    result = Table._delete(t, None, wait=True, missing_ok=True, delete_staging=True)

    assert result == 0
    # Dropped through the UC tables API — no SQL warehouse, and crucially
    # not via self.delete() (that dispatches back into _delete → recursion).
    uc.delete.assert_called_once_with(full_name="c.s.tbl")
    t.sql.execute.assert_not_called()
    t._staging_volume.delete.assert_called_once_with(wait=False)
    t.invalidate_singleton.assert_called_once_with(remove_global=True)


def test_delete_without_predicate_keeps_staging_when_asked():
    t = MagicMock()
    t.full_name.return_value = "c.s.tbl"

    Table._delete(t, None, wait=True, delete_staging=False)

    t.client.workspace_client.return_value.tables.delete.assert_called_once()
    t._staging_volume.delete.assert_not_called()   # staging kept for re-create
