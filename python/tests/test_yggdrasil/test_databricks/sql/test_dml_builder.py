"""Unit tests for the SQL builder in :mod:`yggdrasil.databricks.sql.table`.

These exercise :func:`_build_dml_statements` and the merge-fallback
factory directly — no Databricks needed. They lock in the post-cleanup
behaviour:

- ``Mode.AUTO`` / ``Mode.APPEND`` with ``match_by`` emit a single
  insert-only ``MERGE … WHEN NOT MATCHED THEN INSERT`` (no update
  branch — the engine only probes target by match keys, matched rows
  are never rewritten).
- ``Mode.UPSERT`` / ``Mode.MERGE`` emit a full update-and-insert
  ``MERGE INTO``; the keyed ``DELETE`` + ``INSERT`` pair stays as the
  fallback for backends without a native MERGE.
- ``Mode.TRUNCATE`` with ``match_by`` is a keyed ``DELETE`` + ``INSERT``;
  without ``match_by`` it's a real ``TRUNCATE TABLE`` + ``INSERT``.
- Prune predicates are AND-stitched onto MERGE ``ON`` clauses (and the
  outer ``WHERE`` of the keyed-DELETE form) but never onto a plain
  ``INSERT``.
- Maintenance statements (``OPTIMIZE`` / ``VACUUM``) tail the DML in the
  expected order.
"""

from __future__ import annotations

import pytest

from yggdrasil.databricks.sql.table import (
    _build_delete_insert_statements,
    _build_dml_statements,
    _build_merge_fallback_statements,
)
from yggdrasil.io.enums import Mode


TARGET = "`cat`.`sch`.`tbl`"
COLUMNS = ["id", "name", "value"]
SOURCE_SQL = "SELECT `id`, `name`, `value` FROM source"


# ---------------------------------------------------------------------------
# match_by + insert-only modes (AUTO / APPEND)
# ---------------------------------------------------------------------------


class TestInsertOnlyMerge:

    @pytest.mark.parametrize("mode", [Mode.AUTO, Mode.APPEND])
    def test_match_by_emits_single_merge_with_no_update_branch(self, mode):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=mode,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert len(sqls) == 1
        merge = sqls[0]

        assert merge.startswith(f"MERGE INTO {TARGET} AS T")
        assert "WHEN NOT MATCHED THEN INSERT" in merge
        # Insert-only is the whole point — never emit an UPDATE branch.
        assert "WHEN MATCHED THEN UPDATE" not in merge
        # Null-safe match condition.
        assert "T.`id` <=> S.`id`" in merge

    def test_auto_no_match_by_is_plain_insert(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.AUTO,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        assert sqls == [
            f"INSERT INTO {TARGET} (`id`, `name`, `value`)\n{SOURCE_SQL}"
        ]

    def test_append_match_by_threads_prune_predicates_into_merge_on(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.APPEND,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=["T.`region` = 'eu'"],
        )
        merge = sqls[0]
        # Prune predicate lands on the ON clause, not as a separate WHERE.
        assert "ON T.`id` <=> S.`id` AND T.`region` = 'eu'" in merge
        # Plain INSERT is never produced for the match_by path.
        assert "INSERT INTO" not in merge.split("WHEN NOT MATCHED")[0]


# ---------------------------------------------------------------------------
# match_by + full-merge modes (UPSERT / MERGE)
# ---------------------------------------------------------------------------


class TestUpsertAndMerge:

    @pytest.mark.parametrize("mode", [Mode.UPSERT, Mode.MERGE])
    def test_match_by_emits_single_merge_with_update_branch(self, mode):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=mode,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert len(sqls) == 1
        merge = sqls[0]
        assert merge.startswith(f"MERGE INTO {TARGET} AS T")
        # Both branches present — UPSERT no longer dispatches DELETE+INSERT
        # by default; that's now the merge-fallback code path.
        assert "WHEN MATCHED THEN UPDATE SET" in merge
        assert "WHEN NOT MATCHED THEN INSERT" in merge
        # Default update set excludes match-by columns.
        assert "T.`id` = S.`id`" not in merge.split("WHEN MATCHED")[1].split(
            "WHEN NOT MATCHED"
        )[0]
        assert "T.`name` = S.`name`" in merge
        assert "T.`value` = S.`value`" in merge

    def test_upsert_does_not_emit_a_separate_delete(self):
        # The historical UPSERT path produced ``DELETE … EXISTS`` +
        # ``INSERT INTO …``; the cleanup folds it into MERGE INTO.
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        assert all(not sql.startswith("DELETE FROM") for sql in sqls)
        assert all(not sql.startswith("INSERT INTO") for sql in sqls)

    def test_update_column_names_narrows_update_set(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.MERGE,
            match_by=["id"],
            update_column_names=["value"],
            prune_predicates=[],
        )
        merge = sqls[0]
        update_clause = merge.split("WHEN MATCHED")[1].split("WHEN NOT MATCHED")[0]
        assert "T.`value` = S.`value`" in update_clause
        # ``name`` was excluded by ``update_column_names``.
        assert "T.`name` = S.`name`" not in update_clause


# ---------------------------------------------------------------------------
# OVERWRITE / TRUNCATE
# ---------------------------------------------------------------------------


class TestOverwriteAndTruncate:

    def test_overwrite_emits_plain_insert(self):
        # OVERWRITE without match_by lets the call site issue DELETE
        # before the INSERT; the SQL builder only emits the insert.
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.OVERWRITE,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        assert sqls == [
            f"INSERT INTO {TARGET} (`id`, `name`, `value`)\n{SOURCE_SQL}"
        ]

    def test_truncate_no_match_by_emits_truncate_then_insert(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.TRUNCATE,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
        )
        assert sqls == [
            f"TRUNCATE TABLE {TARGET}",
            f"INSERT INTO {TARGET} (`id`, `name`, `value`)\n{SOURCE_SQL}",
        ]

    def test_truncate_with_match_by_emits_keyed_delete_insert(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.TRUNCATE,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        # Two statements: DELETE then INSERT, both targeting `cat`.`sch`.`tbl`.
        assert len(sqls) == 2
        assert sqls[0].startswith(f"DELETE FROM {TARGET} AS T")
        assert "EXISTS" in sqls[0]
        assert sqls[1] == f"INSERT INTO {TARGET} (`id`, `name`, `value`)\n{SOURCE_SQL}"


# ---------------------------------------------------------------------------
# Prune predicates on the keyed-delete form
# ---------------------------------------------------------------------------


class TestPrunePredicates:

    def test_keyed_delete_insert_lifts_prune_predicates_to_outer_where(self):
        sqls = _build_delete_insert_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            match_by=["id"],
            prune_predicates=["T.`region` = 'eu'", "T.`tier` IN ('gold', 'plat')"],
        )
        delete_sql, insert_sql = sqls
        # Both predicates land on the outer DELETE WHERE alongside the
        # EXISTS clause — they bound the target scan before the EXISTS
        # subquery runs.
        assert "WHERE T.`region` = 'eu'" in delete_sql
        assert "AND T.`tier` IN ('gold', 'plat')" in delete_sql
        assert "EXISTS" in delete_sql
        # The INSERT half doesn't re-apply the predicates — source rows
        # are filtered by whatever ``source_sql`` already encodes.
        assert "WHERE" not in insert_sql

    def test_plain_insert_path_ignores_prune_predicates(self):
        # Mode.AUTO without match_by produces a plain INSERT INTO …; the
        # prune predicates have no place to land and must NOT be silently
        # appended to the INSERT.
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.AUTO,
            match_by=None,
            update_column_names=None,
            prune_predicates=["T.`region` = 'eu'"],
        )
        assert "WHERE" not in sqls[0]
        assert "region" not in sqls[0]


# ---------------------------------------------------------------------------
# Maintenance suffix
# ---------------------------------------------------------------------------


class TestMaintenanceSuffix:

    def test_optimize_after_merge_only_fires_when_keyed(self):
        # No match_by → no OPTIMIZE even with optimize_after_merge=True.
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.AUTO,
            match_by=None,
            update_column_names=None,
            prune_predicates=[],
            optimize_after_merge=True,
        )
        assert all("OPTIMIZE" not in sql for sql in sqls)

    def test_optimize_after_merge_fires_when_keyed(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.AUTO,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            optimize_after_merge=True,
        )
        assert sqls[-1] == f"OPTIMIZE {TARGET}"

    def test_zorder_and_vacuum_tail_in_order(self):
        sqls = _build_dml_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            mode=Mode.AUTO,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            zorder_by=["id"],
            optimize_after_merge=True,
            vacuum_hours=24,
        )
        assert sqls[-3] == f"OPTIMIZE {TARGET} ZORDER BY (`id`)"
        assert sqls[-2] == f"OPTIMIZE {TARGET}"
        assert sqls[-1] == f"VACUUM {TARGET} RETAIN 24 HOURS"


# ---------------------------------------------------------------------------
# Merge-fallback factory used by Mode.MERGE / Mode.UPSERT
# ---------------------------------------------------------------------------


class TestMergeFallbackStatements:

    def test_fallback_builds_keyed_delete_then_insert(self):
        sqls = _build_merge_fallback_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            match_by=["id"],
            prune_predicates=[],
        )
        assert len(sqls) == 2
        assert sqls[0].startswith(f"DELETE FROM {TARGET} AS T")
        assert "EXISTS" in sqls[0]
        assert sqls[1] == f"INSERT INTO {TARGET} (`id`, `name`, `value`)\n{SOURCE_SQL}"

    def test_fallback_appends_maintenance_when_requested(self):
        sqls = _build_merge_fallback_statements(
            target_location=TARGET,
            source_sql=SOURCE_SQL,
            columns=COLUMNS,
            match_by=["id"],
            prune_predicates=[],
            optimize_after_merge=True,
            vacuum_hours=12,
        )
        # DELETE, INSERT, OPTIMIZE, VACUUM — in that order.
        assert sqls[0].startswith("DELETE FROM")
        assert sqls[1].startswith("INSERT INTO")
        assert sqls[2] == f"OPTIMIZE {TARGET}"
        assert sqls[3] == f"VACUUM {TARGET} RETAIN 12 HOURS"
