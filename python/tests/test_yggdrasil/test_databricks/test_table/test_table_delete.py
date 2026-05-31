"""Unit tests for Table._delete dispatch (no live Databricks needed).

* a predicate → server-side ``DELETE FROM … WHERE …`` via the SQL engine;
* no predicate → the UC tables API (``Table.delete``), so no SQL warehouse
  is started just to empty the table.
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
    t.delete.assert_not_called()          # no API drop when filtering rows


def test_delete_without_predicate_uses_api_no_warehouse():
    t = MagicMock()

    result = Table._delete(t, None, wait=False, missing_ok=True, delete_staging=False)

    assert result == 0
    t.delete.assert_called_once_with(wait=False, missing_ok=True, delete_staging=False)
    t.sql.execute.assert_not_called()      # no SQL warehouse spun up
