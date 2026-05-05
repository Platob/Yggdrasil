"""Unit tests for the SQL builder in :mod:`yggdrasil.databricks.sql.table`.

These exercise :func:`_build_dml_statements`, the merge-fallback
factory, and the Databricks-side prune helpers directly — no
Databricks needed. They lock in the post-cleanup behaviour:

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
- The Databricks-side prune-value collectors (Polars-on-Parquet for
  the warehouse staging path, distinct collect for the Spark path)
  feed the predicate builder with the right shape.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from unittest.mock import MagicMock

from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.sql import Table, Tables
from yggdrasil.databricks.sql.table import (
    _build_delete_insert_statements,
    _build_dml_statements,
    _build_merge_fallback_statements,
    _collect_prune_values_polars,
    _render_source_predicate,
    _resolve_dispatch_targets,
    _resolve_prune_by,
)
from yggdrasil.io.buffer.primitive import ParquetIO
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


# ---------------------------------------------------------------------------
# Databricks-side prune helpers
#
# These adapters bridge the engine-agnostic SQL builder above with the
# two write paths: ``arrow_insert`` stages a Parquet buffer and reads
# distinct prune values via Polars; ``spark_insert`` keeps the source
# DataFrame and uses ``distinct().collect()``.
# ---------------------------------------------------------------------------


class TestResolvePruneBy:

    def test_explicit_list_passes_through(self):
        partitions = [pa.field("year", pa.int32())]  # ignored when not "auto"
        assert _resolve_prune_by(["region"], partitions) == ["region"]

    def test_auto_pulls_from_partition_fields(self):
        partitions = [
            pa.field("year", pa.int32()),
            pa.field("month", pa.int32()),
        ]
        assert _resolve_prune_by("auto", partitions) == ["year", "month"]

    def test_auto_with_no_partitions_is_none(self):
        # No partition fields → AUTO gracefully degrades to "no prune".
        # Empty list, not "auto" string, lands here.
        assert _resolve_prune_by("auto", []) is None

    def test_none_returns_none(self):
        assert _resolve_prune_by(None, [pa.field("x", pa.int32())]) is None

    def test_empty_list_returns_none(self):
        assert _resolve_prune_by([], [pa.field("x", pa.int32())]) is None


class TestCollectPruneValuesPolars:
    """Verifies the warehouse-staging path's prune-value capture.

    ``arrow_insert`` writes the source to a ``ParquetIO`` buffer, then
    runs this helper to harvest the distinct values per prune column
    *before* shipping to Volume staging. The values flow back into
    :func:`_build_prune_predicates` to bound the target rewrite.
    """

    @pytest.fixture
    def buffer(self):
        polars = pytest.importorskip("polars")  # noqa: F841
        table = pa.table({
            "region": ["eu", "us", "eu", "ap"],
            "tier":   ["gold", "gold", "plat", "gold"],
            "id":     [1, 2, 3, 4],
        })
        buf = ParquetIO()
        buf.write_arrow_table(table)
        buf.seek(0)
        try:
            yield buf
        finally:
            buf.clear()

    def test_single_column_returns_distinct_values(self, buffer):
        out = _collect_prune_values_polars(buffer, ["region"])
        # Order isn't promised by ``.unique()``; assert as a set.
        assert set(out) == {"region"}
        assert set(out["region"]) == {"eu", "us", "ap"}

    def test_multiple_columns_each_distinct_independently(self, buffer):
        out = _collect_prune_values_polars(buffer, ["region", "tier"])
        # Each key gets its own distinct list — *not* a Cartesian product
        # of (region, tier) tuples. The downstream predicate builder
        # ANDs per-column ``IN`` lists, so independent distincts are
        # what it expects.
        assert set(out) == {"region", "tier"}
        assert set(out["region"]) == {"eu", "us", "ap"}
        assert set(out["tier"]) == {"gold", "plat"}

    def test_nullable_values_survive_distinct(self):
        polars = pytest.importorskip("polars")  # noqa: F841
        table = pa.table({
            "region": pa.array(["eu", None, "eu", None, "us"]),
        })
        buf = ParquetIO()
        try:
            buf.write_arrow_table(table)
            buf.seek(0)
            out = _collect_prune_values_polars(buf, ["region"])
            # NULL match keys ride through distinct as Python ``None`` —
            # the predicate builder uses ``is_in`` with ``includes_null``,
            # so the None is meaningful.
            assert set(out["region"]) == {"eu", "us", None}
        finally:
            buf.clear()


# ---------------------------------------------------------------------------
# table_dispatch helpers — multi-target insert fan-out
# ---------------------------------------------------------------------------


def _make_table(catalog: str, schema: str, name: str) -> Table:
    """Build a :class:`Table` bound to a stub :class:`Tables` service.

    The mock client/service combo lets us exercise the dispatch
    resolver and predicate renderer without any Databricks round trip.
    """
    client = MagicMock(spec=DatabricksClient)
    service = MagicMock(spec=Tables)
    service.client = client
    service.catalog_name = catalog
    service.schema_name = schema

    def _from_(obj, *, service, **_kw):
        loc = str(obj)
        parts = loc.split(".")
        if len(parts) == 3:
            cat, sch, tbl = parts
        else:
            cat, sch, tbl = catalog, schema, parts[-1]
        return Table(
            service=service,
            catalog_name=cat,
            schema_name=sch,
            table_name=tbl,
        )

    # ``Table.from_`` is bound at the class level — the resolver hits it
    # for str keys, so we route it through the same stubbed service.
    service.parse_check_location_params.side_effect = (
        lambda location=None, catalog_name=None, schema_name=None, table_name=None, **kw: (
            location or f"{catalog_name}.{schema_name}.{table_name}",
            catalog_name, schema_name, table_name,
        )
    )

    return Table(
        service=service,
        catalog_name=catalog,
        schema_name=schema,
        table_name=name,
    )


class TestRenderSourcePredicate:
    def test_none_returns_empty(self):
        assert _render_source_predicate(None) == ""

    def test_simple_predicate_renders_unaliased(self):
        from yggdrasil.data.expr import col
        sql = _render_source_predicate(col("region") == "eu")
        # Source-side fragment must NOT carry an ``T.`` / ``S.`` alias —
        # callers compose it onto whatever projection they built.
        assert "T." not in sql and "S." not in sql
        assert "`region`" in sql
        assert "'eu'" in sql

    def test_compound_predicate_is_parenthesized(self):
        from yggdrasil.data.expr import col
        pred = (col("region") == "eu") & (col("tier") == "gold")
        sql = _render_source_predicate(pred)
        # Compound forms must be parenthesized so AND/OR nesting stays
        # explicit when the fragment is concatenated onto a WHERE clause.
        assert sql.startswith("(") and sql.endswith(")")
        assert " AND " in sql


class TestResolveDispatchTargets:
    def test_none_returns_empty_list(self):
        primary = _make_table("cat", "sch", "primary")
        assert _resolve_dispatch_targets(None, primary=primary) == []

    def test_empty_mapping_returns_empty_list(self):
        primary = _make_table("cat", "sch", "primary")
        assert _resolve_dispatch_targets({}, primary=primary) == []

    def test_table_keys_pass_through(self):
        from yggdrasil.data.expr import col
        primary = _make_table("cat", "sch", "primary")
        extra = _make_table("cat", "sch", "extra")
        pred = col("region") == "eu"

        out = _resolve_dispatch_targets({extra: pred}, primary=primary)

        assert len(out) == 1
        assert out[0][0] is extra
        assert out[0][1] is pred

    def test_self_dispatch_raises_with_actionable_message(self):
        from yggdrasil.data.expr import col
        primary = _make_table("cat", "sch", "primary")

        with pytest.raises(ValueError) as exc:
            _resolve_dispatch_targets(
                {primary: col("x") == 1}, primary=primary,
            )
        msg = str(exc.value)
        # Helpful-error contract from AGENTS.md: name what failed and
        # what to do next.
        assert "primary" in msg
        assert "Drop the entry" in msg or "different table" in msg

    def test_unsupported_key_type_raises_typeerror(self):
        from yggdrasil.data.expr import col
        primary = _make_table("cat", "sch", "primary")

        with pytest.raises(TypeError) as exc:
            _resolve_dispatch_targets(
                {123: col("x") == 1}, primary=primary,
            )
        assert "Table or str" in str(exc.value)

    def test_string_predicate_is_lifted_via_predicate_from_(self):
        from yggdrasil.data.expr.nodes import Predicate
        primary = _make_table("cat", "sch", "primary")
        extra = _make_table("cat", "sch", "extra")

        # Raw SQL on the value side — Predicate.from_ routes str to
        # from_sql, so callers don't have to build expr trees by hand
        # for the simple "filter rows where ..." case.
        out = _resolve_dispatch_targets(
            {extra: "region = 'eu' AND tier = 'gold'"},
            primary=primary,
        )

        assert len(out) == 1
        target, predicate = out[0]
        assert target is extra
        # Coercion lands a real Predicate AST — downstream rendering
        # (``_render_source_predicate``) doesn't need to special-case
        # strings.
        assert isinstance(predicate, Predicate)

        rendered = _render_source_predicate(predicate)
        assert "`region`" in rendered
        assert "`tier`" in rendered
        assert "AND" in rendered

    def test_existing_predicate_passes_through_unchanged(self):
        from yggdrasil.data.expr import col
        primary = _make_table("cat", "sch", "primary")
        extra = _make_table("cat", "sch", "extra")
        pred = col("region") == "eu"

        out = _resolve_dispatch_targets({extra: pred}, primary=primary)

        # Already-a-Predicate skips the lift entirely — same object
        # comes back so callers can rely on identity for caching keys.
        assert out[0][1] is pred
