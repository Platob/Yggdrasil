"""Tests for the Databricks :class:`Table` insert SQL generation.

Two strategies live side-by-side, picked by the ``safe_merge``
flag (``CastOptions.safe_merge`` / ``insert(..., safe_merge=...)``):

* **safe_merge=False (default)** — single ``MERGE INTO`` statement.
  Databricks / Delta plans the keyed dedup once;
  :attr:`Mode.UPSERT` / :attr:`Mode.MERGE` get the full
  update-and-insert MERGE, :attr:`Mode.APPEND` /
  :attr:`Mode.AUTO` get the insert-only variant.
* **safe_merge=True** — sidesteps MERGE entirely. UPSERT becomes
  keyed ``DELETE`` + plain ``INSERT``; APPEND becomes
  ``INSERT ... WHERE NOT EXISTS (...)``. Useful for backends
  without native MERGE, for callers that want explicit dedup
  semantics, or for the Spark fast path where the DataFrame
  anti-join one layer up turns the SQL submission into a plain
  ``INSERT``.

Mode dispatch (independent of ``safe_merge``):

* **OVERWRITE** → caller pre-deletes; plain ``INSERT``.
* **TRUNCATE** with keys → keyed DELETE + INSERT.
* **TRUNCATE** without keys → ``TRUNCATE TABLE`` + INSERT.
* **No keys, plain INSERT** → unchanged.

These tests pin the SQL output for both branches so a future
change can't silently regress the keyed-dedup behavior on either
side.
"""
from __future__ import annotations

import pytest

import pyarrow as pa

from yggdrasil.data.enums import Mode
from yggdrasil.data.schema import Schema
from yggdrasil.databricks.sql.table import (
    _build_anti_join_insert,
    _build_cast_projection,
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


class TestNativeMerge:
    """``safe_merge=False`` (the default) → engine MERGE INTO."""

    def test_append_with_keys_uses_insert_only_merge(self) -> None:
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
        assert sql.startswith("MERGE INTO cat.sch.t AS T")
        # Insert-only — no UPDATE branch when mode is APPEND/AUTO.
        assert "WHEN NOT MATCHED THEN INSERT" in sql
        assert "WHEN MATCHED" not in sql

    def test_upsert_with_keys_uses_full_merge(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
        )
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("MERGE INTO cat.sch.t AS T")
        # Full merge — both branches present.
        assert "WHEN MATCHED THEN UPDATE SET" in sql
        assert "WHEN NOT MATCHED THEN INSERT" in sql

    def test_update_column_names_narrows_update_set(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v, ts FROM staging",
            columns=["id", "v", "ts"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=["v"],  # only update v, not ts
            prune_predicates=[],
        )
        sql = _normalize_ws(stmts[0])
        assert "UPDATE SET T.`v` = S.`v`" in sql
        assert "T.`ts` = S.`ts`" not in sql


class TestSafeMerge:
    """``safe_merge=True`` → DELETE+INSERT or NOT EXISTS INSERT, no MERGE."""

    def test_append_with_keys_uses_anti_join_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.APPEND,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            safe_merge=True,
        )
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("INSERT INTO cat.sch.t")
        assert "NOT EXISTS" in sql
        assert "MERGE" not in sql
        assert all("DELETE" not in _normalize_ws(s) for s in stmts)

    def test_auto_with_keys_uses_anti_join_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.AUTO,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            safe_merge=True,
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
            safe_merge=True,
        )
        assert len(stmts) >= 2
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")
        assert all("MERGE" not in _normalize_ws(s) for s in stmts)

    def test_merge_mode_uses_delete_insert(self) -> None:
        stmts = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id FROM staging",
            columns=["id"],
            mode=Mode.MERGE,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            safe_merge=True,
        )
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")


class TestDMLDispatch:
    """Mode dispatch behavior that's independent of ``safe_merge``."""

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

    def test_safe_merge_ignores_update_column_names(self) -> None:
        """``safe_merge=True`` discards ``update_column_names`` —
        the DELETE+INSERT pair always lets incoming win on overlap."""
        stmts_default = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=None,
            prune_predicates=[],
            safe_merge=True,
        )
        stmts_explicit = _build_dml_statements(
            target_location="cat.sch.t",
            source_sql="SELECT id, v FROM staging",
            columns=["id", "v"],
            mode=Mode.UPSERT,
            match_by=["id"],
            update_column_names=["v"],
            prune_predicates=[],
            safe_merge=True,
        )
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
    """The legacy retry / fallback funnel is gone; the simple
    helpers (MERGE template, anti-join, _execute_dml) remain."""

    def test_no_drain_helper(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        assert not hasattr(_t, "_drain_batch_with_retry")

    def test_no_fallback_funnel(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        assert not hasattr(_t, "_execute_with_merge_fallback")
        assert not hasattr(_t, "_build_merge_fallback_statements")

    def test_helpers_present(self) -> None:
        from yggdrasil.databricks.sql import table as _t
        # MERGE template back, anti-join INSERT alongside, and the
        # simplified executor instead of the fallback funnel.
        assert hasattr(_t, "_build_merge_statement")
        assert hasattr(_t, "_build_anti_join_insert")
        assert hasattr(_t, "_execute_dml")
        assert hasattr(_t, "_spark_filter_existing_keys")


# ---------------------------------------------------------------------------
# _build_cast_projection — schema-on-INSERT projection list
# ---------------------------------------------------------------------------


def _schema(*pairs: tuple[str, "pa.DataType"]) -> Schema:
    return Schema.from_arrow(pa.schema([pa.field(n, t) for n, t in pairs]))


class TestBuildCastProjection:
    """The CAST projection is what carries the target schema onto the
    INSERT statement. ``arrow_insert`` / ``spark_insert`` /
    ``sql_insert`` all funnel through here, so pin the shape."""

    def test_unaliased_emits_bare_quoted_column_name(self) -> None:
        sql = _build_cast_projection(
            _schema(("id", pa.int64()), ("v", pa.float64())).fields,
        )
        assert sql == "CAST(`id` AS BIGINT) AS `id`, CAST(`v` AS DOUBLE) AS `v`"

    def test_source_alias_qualifies_left_side_only(self) -> None:
        sql = _build_cast_projection(
            _schema(("id", pa.int64()), ("v", pa.float64())).fields,
            source_alias="raw_src",
        )
        # Source side gets the alias; the AS target stays unqualified.
        assert sql == (
            "CAST(raw_src.`id` AS BIGINT) AS `id`, "
            "CAST(raw_src.`v` AS DOUBLE) AS `v`"
        )

    def test_empty_fields_returns_empty_string(self) -> None:
        assert _build_cast_projection(_schema().fields) == ""

    def test_special_characters_in_name_are_quoted(self) -> None:
        sql = _build_cast_projection(_schema(("a b", pa.string())).fields)
        # Backtick quoting protects spaces / reserved words; DDL stays
        # name-free so the cast never re-emits the identifier inline.
        assert sql == "CAST(`a b` AS STRING) AS `a b`"

    def test_ddl_omits_nullable_and_comment(self) -> None:
        # Nullable / comment in the source schema must NOT leak into
        # the CAST DDL — Delta MERGE refuses ``CAST(... AS x NOT NULL)``
        # inside a struct (``DATATYPE_MISMATCH.CAST_WITHOUT_SUGGESTION``).
        schema = Schema.from_arrow(pa.schema([
            pa.field("id", pa.int64(), nullable=False),
        ]))
        sql = _build_cast_projection(schema.fields)
        assert "NOT NULL" not in sql
        assert "COMMENT" not in sql

    def test_nested_struct_renders_inner_types(self) -> None:
        inner = pa.struct([
            pa.field("k", pa.int32()),
            pa.field("name", pa.string()),
        ])
        sql = _build_cast_projection(_schema(("payload", inner)).fields)
        # Struct inner types come through to_spark_name; nullable on
        # the children stays off so the cast is permissive.
        assert sql.startswith("CAST(`payload` AS STRUCT<")
        assert "`k` INT" in sql
        assert "`name` STRING" in sql
        assert sql.endswith(" AS `payload`")
        assert "NOT NULL" not in sql


class TestSparkInsertSQLProjection:
    """Pin the shape of the source projection used by spark_insert /
    arrow_insert. Validates the helper through the call sites — if
    the helper changes contract here, the SQL generators need to
    change too."""

    def test_view_projection_uses_cast_per_column(self) -> None:
        # Reconstruct what spark_insert emits: ``SELECT <cast_proj>
        # FROM <view>``. The full method needs a Spark session; this
        # verifies the projection helper plugs into a plain SELECT.
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        proj = _build_cast_projection(schema.fields)
        sql = f"SELECT {proj} FROM `_yg_src`"
        assert "CAST(`id` AS BIGINT) AS `id`" in sql
        assert "CAST(`v` AS DOUBLE) AS `v`" in sql

    def test_staging_projection_is_unqualified(self) -> None:
        # arrow_insert reads the staged Parquet through an external-data
        # placeholder; the projection therefore must NOT prefix columns.
        schema = _schema(("id", pa.int64()))
        proj = _build_cast_projection(schema.fields)
        assert "raw_src." not in proj

    def test_sql_insert_projection_is_qualified(self) -> None:
        # sql_insert wraps the user's source query as ``... AS raw_src``;
        # the projection must qualify the source side so the column
        # reference resolves against the wrapper alias.
        schema = _schema(("id", pa.int64()))
        proj = _build_cast_projection(schema.fields, source_alias="raw_src")
        assert proj.startswith("CAST(raw_src.`id` AS BIGINT)")


# ---------------------------------------------------------------------------
# CastOptions.safe_merge
# ---------------------------------------------------------------------------


class TestCastOptionsSafeMerge:

    def test_default_is_false(self) -> None:
        from yggdrasil.data.options import CastOptions
        assert CastOptions().safe_merge is False

    def test_set_via_constructor(self) -> None:
        from yggdrasil.data.options import CastOptions
        assert CastOptions(safe_merge=True).safe_merge is True
