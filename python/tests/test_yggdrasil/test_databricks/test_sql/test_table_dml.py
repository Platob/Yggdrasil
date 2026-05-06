"""Tests for the Databricks :class:`Table` insert SQL generation.

The legacy code used :sql:`MERGE INTO` for keyed inserts and a
DELETE+INSERT fallback wrapped in a multi-stage retry. The
rewritten path is simpler:

* **APPEND / AUTO with keys** → single
  ``INSERT INTO ... SELECT ... WHERE NOT EXISTS (...)`` against
  the target. Existing rows are filtered out by the ``EXISTS``
  subquery; the engine never has to touch (let alone rewrite)
  rows that already match.
* **UPSERT / MERGE with keys** → ``DELETE`` matching keys, then
  plain ``INSERT``. Incoming wins on overlap. Same shape as the
  legacy fallback, now the only path.
* **OVERWRITE** → caller pre-deletes; we just ``INSERT``.
* **TRUNCATE** with keys → keyed DELETE + INSERT.
* **TRUNCATE** without keys → ``TRUNCATE TABLE`` + INSERT.
* **No keys, plain INSERT** → unchanged.

These tests pin the SQL output shape so a future change can't
silently regress the keyed-dedup behavior.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums import Mode
from yggdrasil.databricks.sql.table import (
    _build_anti_join_insert,
    _build_delete_insert_statements,
    _build_dml_statements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_ws(sql: str) -> str:
    """Collapse whitespace runs so order/indentation isn't load-bearing."""
    return " ".join(sql.split())


def _texts(stmts: list[str]) -> list[str]:
    return [_normalize_ws(s) for s in stmts]


# ---------------------------------------------------------------------------
# _build_anti_join_insert
# ---------------------------------------------------------------------------


class TestAntiJoinInsert:

    def test_basic_shape(self) -> None:
        out = _build_anti_join_insert(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            match_by=["id"],
            prune_predicates=[],
        )
        assert len(out) == 1
        sql = _normalize_ws(out[0])
        # Single INSERT statement, no MERGE.
        assert "MERGE INTO" not in sql
        assert sql.startswith("INSERT INTO cat.sch.t (`id`, `v`)")
        # Filter via NOT EXISTS over the target — keyed-existence check.
        assert "NOT EXISTS" in sql
        assert "FROM cat.sch.t AS T" in sql
        assert "T.`id` <=> S.`id`" in sql

    def test_composite_key(self) -> None:
        out = _build_anti_join_insert(
            target_location="cat.sch.t",
            source_sql="SELECT a, b, v FROM staging",
            columns=["a", "b", "v"],
            match_by=["a", "b"],
            prune_predicates=[],
        )
        sql = _normalize_ws(out[0])
        # Both key columns AND-joined under the EXISTS subquery.
        assert "T.`a` <=> S.`a`" in sql
        assert "T.`b` <=> S.`b`" in sql
        assert " AND " in sql

    def test_prune_predicates_lift_into_exists(self) -> None:
        out = _build_anti_join_insert(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            match_by=["id"],
            prune_predicates=["T.`region` = 'us'"],
        )
        sql = _normalize_ws(out[0])
        # Prune predicate AND-joined to the key match inside the
        # ``NOT EXISTS`` subquery — narrows the target scan without
        # leaking into the outer SELECT.
        assert "NOT EXISTS" in sql
        not_exists_block = sql[sql.index("NOT EXISTS"):]
        assert "T.`region` = 'us'" in not_exists_block
        assert "T.`id` <=> S.`id`" in not_exists_block
        assert "AND" in not_exists_block

    def test_columns_quoted(self) -> None:
        out = _build_anti_join_insert(
            target_location="cat.sch.t",
            source_sql="SELECT * FROM staging",
            columns=["id with space", "value"],
            match_by=["id with space"],
            prune_predicates=[],
        )
        sql = _normalize_ws(out[0])
        # Backtick-quoted to handle the space.
        assert "`id with space`" in sql


# ---------------------------------------------------------------------------
# _build_dml_statements — mode dispatch
# ---------------------------------------------------------------------------


class TestDMLDispatch:

    def test_append_with_keys_uses_anti_join_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.APPEND,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        sql = _normalize_ws(stmts[0])
        # Single INSERT with NOT EXISTS — no MERGE, no DELETE.
        assert sql.startswith("INSERT INTO cat.sch.t")
        assert "NOT EXISTS" in sql
        assert "MERGE" not in sql
        assert all("DELETE" not in _normalize_ws(s) for s in stmts)

    def test_auto_with_keys_uses_anti_join_insert(self) -> None:
        # AUTO defaults to "append only new keys" when match_by is set.
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.AUTO,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert "NOT EXISTS" in _normalize_ws(stmts[0])

    def test_upsert_with_keys_uses_delete_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        # Two statements: DELETE then INSERT.
        assert len(stmts) >= 2
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")
        # No MERGE anywhere.
        assert all("MERGE" not in _normalize_ws(s) for s in stmts)

    def test_merge_mode_aliases_to_delete_insert(self) -> None:
        # ``Mode.MERGE`` no longer triggers a real MERGE statement.
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.MERGE,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")

    def test_overwrite_emits_plain_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.OVERWRITE,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        assert len(stmts) == 1
        assert _normalize_ws(stmts[0]).startswith("INSERT INTO cat.sch.t")
        assert "DELETE" not in _normalize_ws(stmts[0])
        assert "NOT EXISTS" not in _normalize_ws(stmts[0])

    def test_truncate_no_keys_emits_truncate_then_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.TRUNCATE,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        assert _normalize_ws(stmts[0]).startswith("TRUNCATE TABLE cat.sch.t")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO cat.sch.t")

    def test_truncate_with_keys_emits_delete_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.TRUNCATE,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")

    def test_no_keys_plain_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.APPEND,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("INSERT INTO cat.sch.t")
        assert "NOT EXISTS" not in sql
        assert "DELETE" not in sql

    def test_update_column_names_is_ignored_now(self) -> None:
        """Legacy ``update_column_names`` is dropped — incoming wins on UPSERT."""
        stmts_default = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        stmts_explicit = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=["v"],
            prune_predicates=[],
        )
        # Both produce the same DELETE+INSERT — no UPDATE branch.
        assert _texts(stmts_default) == _texts(stmts_explicit)


# ---------------------------------------------------------------------------
# Maintenance tail
# ---------------------------------------------------------------------------


class TestMaintenance:

    def test_zorder_appended_after_dml(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.APPEND,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
            zorder_by=["id"],
        )
        # ZORDER as a final OPTIMIZE statement.
        assert any("ZORDER BY" in _normalize_ws(s) for s in stmts)

    def test_optimize_only_when_keyed(self) -> None:
        no_keys = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.APPEND,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
            optimize_after_merge=True,
        )
        # No OPTIMIZE without keys.
        assert all("OPTIMIZE" not in _normalize_ws(s) for s in no_keys)

        keyed = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.APPEND,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            optimize_after_merge=True,
        )
        assert any("OPTIMIZE cat.sch.t" in _normalize_ws(s) for s in keyed)

    def test_vacuum_appended(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.APPEND,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
            vacuum_hours=168,
        )
        assert any("VACUUM cat.sch.t RETAIN 168 HOURS" in _normalize_ws(s) for s in stmts)


# ---------------------------------------------------------------------------
# Legacy-shape DELETE+INSERT (used internally by UPSERT path)
# ---------------------------------------------------------------------------


class TestDeleteInsertShape:

    def test_keyed_delete_uses_exists_subquery(self) -> None:
        stmts = _build_delete_insert_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            match_by=["id"],
            prune_predicates=[],
        )
        delete_sql = _normalize_ws(stmts[0])
        insert_sql = _normalize_ws(stmts[1])
        assert delete_sql.startswith("DELETE FROM cat.sch.t AS T")
        assert "EXISTS" in delete_sql
        assert "T.`id` <=> S.`id`" in delete_sql
        assert insert_sql.startswith("INSERT INTO cat.sch.t (`id`, `v`)")

    def test_prune_predicates_narrow_target(self) -> None:
        stmts = _build_delete_insert_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            match_by=["id"],
            prune_predicates=["T.`region` = 'us'"],
        )
        delete_sql = _normalize_ws(stmts[0])
        # Prune predicate AND-joined before the EXISTS subquery
        # (so the target scan is partition-pruned first).
        assert "T.`region` = 'us'" in delete_sql
        assert delete_sql.index("T.`region` = 'us'") < delete_sql.index("EXISTS")


# ---------------------------------------------------------------------------
# Removed legacy machinery — make sure callers can't reach back to it
# ---------------------------------------------------------------------------


class TestNoMergeFallbackMachinery:
    """The legacy retry / fallback funnel is gone; assert no shadows."""

    def test_no_drain_helper(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        assert not hasattr(_t, "_drain_batch_with_retry")

    def test_no_fallback_funnel(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        assert not hasattr(_t, "_execute_with_merge_fallback")
        assert not hasattr(_t, "_build_merge_fallback_statements")
        assert not hasattr(_t, "_build_merge_statement")

    def test_execute_dml_helper_present(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        assert hasattr(_t, "_execute_dml")
