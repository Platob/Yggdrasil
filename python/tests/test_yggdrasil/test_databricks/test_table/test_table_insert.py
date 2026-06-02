"""Unit tests for the centralized table insert module (no live Databricks).

Covers:
* ``Table.insert(wait=False)`` routing → ``stage_async_insert`` (OVERWRITE /
  APPEND with no keys, or MERGE / UPSERT with match_by), and the sync path
  otherwise;
* :class:`DatabricksTableInsert` parsing / serialization / validation and the
  per-op + per-batch SQL generators (``make_sql_select`` / ``make_sql_insert``,
  including the keyed-batch MERGE dedup);
* ``stage_async_insert`` staging a Parquet + dropping a JSON operation log;
* ``ensure_async_job`` get-or-create with a file-arrival trigger;
* ``load_async`` / ``dispatch_async`` aggregating logs into one load per
  ``(target, mode)``, then cleaning up consumed logs + data.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

from yggdrasil.data.schema import Schema
from yggdrasil.databricks.table.table import Table


def _table_mock(full_name="c.s.t"):
    t = MagicMock()
    t.catalog_name, t.schema_name, t.table_name = full_name.split(".")
    t.full_name.return_value = full_name
    return t


def _schema(*pairs):
    return Schema.from_arrow(pa.schema([pa.field(n, t) for n, t in pairs]))


def _normalize_ws(sql: str) -> str:
    return " ".join(sql.split())


# --------------------------------------------------------------------------- #
# Table.insert(wait=False) routing
# --------------------------------------------------------------------------- #
class TestInsertRouting:
    def test_wait_false_routes_to_async(self):
        t = MagicMock()
        with patch(
            "yggdrasil.databricks.table.insert.stage_async_insert",
        ) as stage:
            out = Table.insert(t, {"a": [1]}, mode="append", wait=False)
        stage.assert_called_once()
        assert stage.call_args.args[0] is t
        assert stage.call_args.kwargs["mode"] == "append"
        assert out is stage.return_value
        t.insert_into.assert_not_called()

    def test_wait_true_uses_sync_path(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append")  # wait defaults True
        t.insert_into.assert_called_once()

    def test_match_by_stays_sync(self):
        t = MagicMock()
        with patch("yggdrasil.databricks.table.insert.stage_async_insert") as stage:
            Table.insert(t, {"a": [1]}, mode="append", wait=False, match_by=["id"])
        t.insert_into.assert_called_once()
        stage.assert_not_called()

    def test_merge_mode_without_keys_stays_sync(self):
        # MERGE needs keys to qualify for the async drop; without them it falls
        # through to the synchronous path.
        t = MagicMock()
        with patch("yggdrasil.databricks.table.insert.stage_async_insert") as stage:
            Table.insert(t, {"a": [1]}, mode="merge", wait=False)
        t.insert_into.assert_called_once()
        stage.assert_not_called()

    def test_merge_with_keys_routes_to_async(self):
        # MERGE/UPSERT + match_by qualifies for the async drop, forwarding keys.
        t = MagicMock()
        with patch("yggdrasil.databricks.table.insert.stage_async_insert") as stage:
            out = Table.insert(
                t, {"a": [1]}, mode="merge", wait=False, match_by=["id"],
            )
        stage.assert_called_once()
        assert stage.call_args.kwargs["mode"] == "merge"
        assert stage.call_args.kwargs["match_by"] == ["id"]
        assert out is stage.return_value
        t.insert_into.assert_not_called()


# --------------------------------------------------------------------------- #
# DatabricksTableInsert — parsing / serialization / validation / SQL
# --------------------------------------------------------------------------- #
class TestDatabricksTableInsert:
    def test_round_trips_through_the_log(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        from yggdrasil.enums.mode import Mode
        op = DatabricksTableInsert(
            target="c.s.t", mode="append", data="dbfs+volume:/x.parquet",
        )
        log = MagicMock()
        log.read_bytes.return_value = op.to_json()
        parsed = DatabricksTableInsert.from_log(log)
        # mode is the typed enum; target / data keep their serialized form
        assert parsed.target == "c.s.t"
        assert parsed.mode is Mode.APPEND
        assert parsed.data == "dbfs+volume:/x.parquet"
        assert parsed.op_id == op.op_id and parsed.ts == op.ts
        assert parsed.log_file is log            # keeps the file for cleanup
        assert parsed.group_key == "c.s.t"       # one load per target

    def test_to_json_normalizes_typed_fields(self):
        # A producer-built op holds typed Table / Mode / Path; to_json emits
        # the string op-log form the loader reads back.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        data = MagicMock()
        data.to_url.return_value.to_string.return_value = "dbfs+volume:/x.parquet"
        payload = json.loads(
            DatabricksTableInsert(target=target, mode="overwrite", data=data).to_json()
        )
        assert payload["target"] == "c.s.t"
        assert payload["mode"] == "overwrite"
        assert payload["data"] == "dbfs+volume:/x.parquet"

    def test_schema_and_maintenance_fields_round_trip(self):
        # The richer keyed-write surface persists in the op-log and rebuilds.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        op = DatabricksTableInsert(
            target="c.s.t", mode="overwrite", data="dbfs+volume:/x.parquet",
            schema=schema, zorder_by=["id"], vacuum_hours=72,
            optimize_after_merge=True, safe_merge=True,
        )
        payload = json.loads(op.to_json())
        assert payload["zorder_by"] == ["id"]
        assert payload["vacuum_hours"] == 72
        assert payload["optimize_after_merge"] is True
        assert payload["safe_merge"] is True
        assert payload["schema"]  # serialized field json present
        parsed = DatabricksTableInsert.from_json(payload)
        assert [f.name for f in parsed.schema.fields] == ["id", "v"]
        assert parsed.zorder_by == ["id"]
        assert parsed.vacuum_hours == 72
        assert parsed.optimize_after_merge is True
        assert parsed.safe_merge is True

    def test_rejects_bad_mode(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        with pytest.raises(ValueError, match="OVERWRITE / APPEND"):
            DatabricksTableInsert(target="c.s.t", mode="merge", data="dbfs+volume:/x.parquet")

    def test_keyed_op_allows_any_mode(self):
        # The sync path builds a richer op (schema / match_by) and supports
        # every mode — the OVERWRITE-only guard is for the bare async shape.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(
            target="c.s.t", mode="merge", data="dbfs+volume:/x.parquet",
            match_by=["id"], schema=_schema(("id", pa.int64())),
        )
        assert op.match_by == ["id"]

    def test_data_path_reconstructs_from_uniform_url(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="dbfs+volume:/x.parquet")
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp:
            op.data_path(client="CL")
        dp.assert_called_once_with("dbfs+volume:/x.parquet", client="CL")


# --------------------------------------------------------------------------- #
# make_sql_select / make_sql_insert — the centralized generator
# --------------------------------------------------------------------------- #
class TestMakeSqlSelect:
    def test_default_renders_warehouse_parquet_ref(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, make_sql_select
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="dbfs+volume:/c/s/t/x.parquet")
        path = MagicMock()
        path.full_path.return_value = "/Volumes/c/s/t/x.parquet"
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", return_value=path):
            sql = make_sql_select(op, client="CL")
        assert sql == "SELECT * FROM parquet.`/Volumes/c/s/t/x.parquet`"

    def test_explicit_source_projects_schema_columns(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, make_sql_select
        op = DatabricksTableInsert(
            target="c.s.t", mode="append", data="dbfs+volume:/x.parquet",
            schema=_schema(("id", pa.int64()), ("v", pa.float64())),
        )
        sql = make_sql_select(op, source="{__tmpsrc__}")
        assert sql == "SELECT `id`, `v` FROM {__tmpsrc__}"

    def test_explicit_source_without_schema_uses_star(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, make_sql_select
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="dbfs+volume:/x.parquet")
        assert make_sql_select(op, source="V") == "SELECT * FROM V"

    def test_select_sql_alias_delegates(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, make_sql_select
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="dbfs+volume:/x.parquet")
        with patch.object(op, "data_path") as dp:
            dp.return_value.full_path.return_value = "/Volumes/x.parquet"
            assert op.select_sql() == make_sql_select(op)


class TestMakeSqlInsertAtomic:
    def _op(self, **kw):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        defaults = dict(
            target="c.s.t", mode="append", data="dbfs+volume:/x.parquet",
            schema=_schema(("id", pa.int64()), ("v", pa.float64())),
        )
        defaults.update(kw)
        return DatabricksTableInsert(**defaults)

    def test_append_no_keys_plain_insert(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(self._op(mode="append"), source_sql="SELECT * FROM src")
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("INSERT INTO c.s.t (`id`, `v`)")
        assert "MERGE" not in sql

    def test_overwrite_no_keys_insert_overwrite(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(self._op(mode="overwrite"), source_sql="SELECT * FROM src")
        assert _normalize_ws(stmts[0]).startswith("INSERT OVERWRITE c.s.t")

    def test_truncate_no_keys_truncate_then_insert(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(self._op(mode="truncate"), source_sql="SELECT * FROM src")
        assert _normalize_ws(stmts[0]).startswith("TRUNCATE TABLE c.s.t")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO c.s.t")

    def test_keyed_append_uses_insert_only_merge(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(
            self._op(mode="append", match_by=["id"]), source_sql="SELECT * FROM src",
        )
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("MERGE INTO c.s.t AS T")
        assert "WHEN NOT MATCHED THEN INSERT" in sql
        assert "WHEN MATCHED" not in sql

    def test_keyed_upsert_full_merge(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(
            self._op(mode="upsert", match_by=["id"]), source_sql="SELECT * FROM src",
        )
        sql = _normalize_ws(stmts[0])
        assert "WHEN MATCHED THEN UPDATE SET" in sql
        assert "WHEN NOT MATCHED THEN INSERT" in sql

    def test_safe_merge_upsert_delete_insert(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(
            self._op(mode="upsert", match_by=["id"], safe_merge=True),
            source_sql="SELECT * FROM src",
        )
        assert _normalize_ws(stmts[0]).startswith("DELETE FROM")
        assert _normalize_ws(stmts[1]).startswith("INSERT INTO")

    def test_maintenance_tail(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        stmts = make_sql_insert(
            self._op(mode="append", match_by=["id"], zorder_by=["id"], vacuum_hours=72),
            source_sql="SELECT * FROM src",
        )
        joined = " ".join(_normalize_ws(s) for s in stmts)
        assert "ZORDER BY (`id`)" in joined
        assert "VACUUM c.s.t RETAIN 72 HOURS" in joined

    def test_columns_derived_from_schema_when_omitted(self):
        from yggdrasil.databricks.table.insert import make_sql_insert
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp:
            dp.return_value.full_path.return_value = "/Volumes/x.parquet"
            stmts = make_sql_insert(self._op(mode="append"))
        sql = _normalize_ws(stmts[0])
        assert "(`id`, `v`)" in sql
        assert "parquet.`/Volumes/x.parquet`" in sql


# --------------------------------------------------------------------------- #
# DatabricksInsertBatch — grouping / supersede / multiselect / MERGE dedup
# --------------------------------------------------------------------------- #
class TestDatabricksInsertBatch:
    @staticmethod
    def _op(data, *, mode="append", ts=0.0, target="c.s.t", match_by=None, schema=None):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(
            target=target, mode=mode, data=data, ts=ts,
            match_by=match_by, schema=schema,
        )
        # render select_sql without touching DatabricksPath
        op.select_sql = lambda client=None, _d=data: f"SELECT * FROM parquet.`{_d}`"  # type: ignore
        return op

    def test_groups_by_target(self):
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        ops = [self._op("a", target="c.s.t1"), self._op("b", target="c.s.t2"),
               self._op("c", target="c.s.t1")]
        batches = {b.logs[0].target_name: b for b in DatabricksInsertBatch.group(ops)}
        assert set(batches) == {"c.s.t1", "c.s.t2"}
        assert len(batches["c.s.t1"].logs) == 2

    def test_appends_union_in_ts_order(self):
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        from yggdrasil.enums.mode import Mode
        [batch] = DatabricksInsertBatch.group(
            [self._op("b", ts=2.0), self._op("a", ts=1.0)]
        )
        assert batch.mode is Mode.APPEND
        assert batch.make_sql() == (
            "SELECT * FROM parquet.`a` UNION ALL SELECT * FROM parquet.`b`"
        )

    def test_overwrite_supersedes_earlier_ops_but_keeps_them_for_cleanup(self):
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        from yggdrasil.enums.mode import Mode
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="append", ts=1.0),
            self._op("b", mode="overwrite", ts=2.0),
            self._op("c", mode="append", ts=3.0),
        ])
        # OVERWRITE wins → mode is OVERWRITE; only b + c feed the load …
        assert batch.mode is Mode.OVERWRITE
        assert batch.make_sql() == (
            "SELECT * FROM parquet.`b` UNION ALL SELECT * FROM parquet.`c`"
        )
        # … but the superseded 'a' is still tracked so its data gets cleaned up
        assert len(batch.logs) == 3

    def test_keyed_overwrite_does_not_supersede_earlier_appends(self):
        # A keyed overwrite (match_by set) only replaces matching rows, so it
        # must NOT discard appends staged before it — they stay in the load.
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        from yggdrasil.enums.mode import Mode
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="append", ts=1.0),
            self._op("b", mode="overwrite", match_by=["id"], schema=schema, ts=2.0),
        ])
        # no keyless overwrite → not a full-table overwrite; both ops feed the load
        assert batch.mode is not Mode.OVERWRITE
        assert len(batch.active) == 2
        assert "parquet.`a`" in batch.make_sql() and "parquet.`b`" in batch.make_sql()

    def test_keyed_merge_batch_deduplicates_union(self):
        # A keyed MERGE batch can carry duplicate keys across staged files;
        # the union is wrapped in a ROW_NUMBER()=1 dedup so Delta's MERGE
        # never sees multiple source rows for one target row.
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="merge", match_by=["id"], schema=schema, ts=1.0),
            self._op("b", mode="merge", match_by=["id"], schema=schema, ts=2.0),
        ])
        body = _normalize_ws(batch.make_sql())
        assert "UNION ALL" in body                       # both files unioned
        assert "ROW_NUMBER() OVER ( PARTITION BY `id`" in body
        assert "__ygg_rn__ = 1" in body

    def test_keyed_merge_dedup_orders_by_arrival_latest_wins(self):
        # Across several drops the dedup must keep the LATEST drop's row per key
        # — tag each file with its arrival ordinal and order by it DESC, so the
        # winner is deterministic (not an arbitrary row).
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="merge", match_by=["id"], schema=schema, ts=1.0),
            self._op("b", mode="merge", match_by=["id"], schema=schema, ts=2.0),
            self._op("c", mode="merge", match_by=["id"], schema=schema, ts=3.0),
        ])
        body = _normalize_ws(batch.make_sql())
        # one arrival ordinal per file (3 drops → seq 0,1,2), union of all three
        assert body.count("UNION ALL") == 2
        assert "0 AS __ygg_seq__" in body
        assert "1 AS __ygg_seq__" in body
        assert "2 AS __ygg_seq__" in body
        # deterministic incoming-wins ordering, and both helpers projected away
        assert "ORDER BY __ygg_seq__ DESC" in body
        assert "EXCEPT (__ygg_rn__, __ygg_seq__)" in body

    def test_append_batch_skips_dedup(self):
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="append", ts=1.0),
            self._op("b", mode="append", ts=2.0),
        ])
        body = batch.make_sql()
        assert "ROW_NUMBER" not in body
        assert "UNION ALL" in body

    def test_make_sql_insert_batch_dispatch(self):
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch, make_sql_insert
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="merge", match_by=["id"], schema=schema, target="c.s.t", ts=1.0),
            self._op("b", mode="merge", match_by=["id"], schema=schema, target="c.s.t", ts=2.0),
        ])
        stmts = make_sql_insert(batch, target_location="c.s.t")
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("MERGE INTO c.s.t AS T")
        assert "ROW_NUMBER()" in sql            # dedup'd source feeds the MERGE


# --------------------------------------------------------------------------- #
# MERGE DML generation aggregated from several inner merges (the loader's
# many-file-arrivals → one MERGE path)
# --------------------------------------------------------------------------- #
class TestBatchedMergeDML:
    @staticmethod
    def _merge_batch(n, *, match_by, mode="merge", schema=None, update_cols=None):
        from yggdrasil.databricks.table.insert import (
            DatabricksInsertBatch, DatabricksTableInsert,
        )
        ops = []
        for i in range(n):
            op = DatabricksTableInsert(
                target="c.s.t", mode=mode, data=f"f{i}", ts=float(i),
                match_by=match_by, schema=schema, update_column_names=update_cols,
            )
            op.select_sql = lambda client=None, _d=f"f{i}": f"SELECT * FROM parquet.`{_d}`"  # type: ignore
            ops.append(op)
        [batch] = DatabricksInsertBatch.group(ops)
        return batch

    def _sql(self, batch):
        from yggdrasil.databricks.table.insert import make_sql_insert
        return _normalize_ws(make_sql_insert(batch, target_location="c.s.t")[0])

    def test_three_merges_render_one_full_merge(self):
        schema = _schema(("id", pa.int64()), ("v", pa.float64()), ("w", pa.string()))
        sql = self._sql(self._merge_batch(3, match_by=["id"], schema=schema))
        # one MERGE INTO over the deduped union of all three drops
        assert sql.startswith("MERGE INTO c.s.t AS T USING (")
        assert sql.count("UNION ALL") == 2
        # null-safe key match
        assert "ON T.`id` <=> S.`id`" in sql
        # update every non-key column, insert the full row
        assert "WHEN MATCHED THEN UPDATE SET T.`v` = S.`v`, T.`w` = S.`w`" in sql
        assert "WHEN NOT MATCHED THEN INSERT (`id`, `v`, `w`)" in sql
        assert "VALUES (S.`id`, S.`v`, S.`w`)" in sql

    def test_composite_key_merge(self):
        schema = _schema(
            ("id", pa.int64()), ("region", pa.string()), ("v", pa.float64()),
        )
        sql = self._sql(self._merge_batch(2, match_by=["id", "region"], schema=schema))
        # both keys in the ON, both excluded from the UPDATE SET, both in PARTITION
        assert "ON T.`id` <=> S.`id` AND T.`region` <=> S.`region`" in sql
        assert "UPDATE SET T.`v` = S.`v`" in sql
        assert "T.`id`" not in sql.split("UPDATE SET")[1].split("WHEN NOT")[0]
        assert "PARTITION BY `id`, `region`" in sql

    def test_update_column_names_override_restricts_set(self):
        schema = _schema(
            ("id", pa.int64()), ("v", pa.float64()), ("w", pa.string()),
        )
        sql = self._sql(
            self._merge_batch(2, match_by=["id"], schema=schema, update_cols=["v"])
        )
        # only the named column is updated; 'w' is insert-only
        assert "UPDATE SET T.`v` = S.`v` WHEN NOT MATCHED" in sql
        assert "T.`w` = S.`w`" not in sql

    def test_keyed_append_is_insert_only_merge(self):
        # APPEND + keys → insert-only MERGE (no UPDATE clause), no dedup CTE.
        schema = _schema(("id", pa.int64()), ("v", pa.float64()))
        sql = self._sql(
            self._merge_batch(2, match_by=["id"], mode="append", schema=schema)
        )
        assert sql.startswith("MERGE INTO c.s.t AS T")
        assert "WHEN MATCHED THEN UPDATE" not in sql
        assert "WHEN NOT MATCHED THEN INSERT" in sql
        assert "ROW_NUMBER()" not in sql            # insert-only doesn't dedup

    def test_partition_filters_added_to_merge_on_as_literals(self):
        # A partitioned target prunes the MERGE scan to the partitions the source
        # touches — literal IN predicates on the ON (no subquery: MERGE search
        # conditions can't contain one).
        from yggdrasil.databricks.table.insert import _build_dml_statements
        from yggdrasil.enums.mode import Mode
        stmts = _build_dml_statements(
            target_location="c.s.t", source_sql="SELECT * FROM src",
            columns=["id", "d", "v"], mode=Mode.MERGE, match_by=["id"],
            update_column_names=None, prune_predicates=[],
            partition_filters=["T.`d` IN ('2024-01-01', '2024-01-02')"],
        )
        sql = _normalize_ws(stmts[0])
        assert sql.startswith("MERGE INTO c.s.t AS T")
        assert "T.`id` <=> S.`id` AND T.`d` IN ('2024-01-01', '2024-01-02')" in sql
        assert "SELECT" in sql.split("WHEN")[0]      # sanity: still a MERGE
        assert "__ygg_part__" not in sql              # never a subquery in ON

    def test_multiple_partition_filters_all_anded_on(self):
        from yggdrasil.databricks.table.insert import _build_dml_statements
        from yggdrasil.enums.mode import Mode
        stmts = _build_dml_statements(
            target_location="c.s.t", source_sql="SELECT * FROM src",
            columns=["id", "y", "m", "v"], mode=Mode.MERGE, match_by=["id"],
            update_column_names=None, prune_predicates=[],
            partition_filters=["T.`y` IN (2024)", "T.`m` IN (1, 2)"],
        )
        sql = _normalize_ws(stmts[0])
        assert "T.`y` IN (2024)" in sql and "T.`m` IN (1, 2)" in sql

    def test_no_partition_filters_leaves_on_clause_clean(self):
        from yggdrasil.databricks.table.insert import _build_dml_statements
        from yggdrasil.enums.mode import Mode
        stmts = _build_dml_statements(
            target_location="c.s.t", source_sql="SELECT * FROM src",
            columns=["id", "v"], mode=Mode.MERGE, match_by=["id"],
            update_column_names=None, prune_predicates=[], partition_filters=[],
        )
        sql = _normalize_ws(stmts[0])
        assert "ON T.`id` <=> S.`id` WHEN" in sql


class TestMergePartitionFilters:
    def test_sql_literal_renders_common_types(self):
        import datetime
        from yggdrasil.databricks.table.table import _sql_literal
        assert _sql_literal("a'b") == "'a''b'"          # quote-escaped string
        assert _sql_literal(5) == "5"
        assert _sql_literal(True) == "TRUE"
        assert _sql_literal(None) is None               # NULL → skip
        assert _sql_literal(datetime.date(2024, 1, 2)) == "'2024-01-02'"
        assert _sql_literal(object()) is None           # unsupported → skip

    @staticmethod
    def _mock_table(part_cols=("d",), distinct=None):
        from types import SimpleNamespace
        mock = MagicMock()
        mock.collect_schema.return_value.fields = [
            SimpleNamespace(name=c, partition_by=True) for c in part_cols
        ] + [SimpleNamespace(name="v", partition_by=False)]
        if distinct is not None:
            mock.sql.execute.return_value.to_arrow_table.return_value = distinct
        return mock

    def test_partition_filters_from_distinct_source_values(self):
        from yggdrasil.databricks.table.table import Table
        mock = self._mock_table(
            distinct=pa.table({"d": ["2024-01-02", "2024-01-01", "2024-01-02"]})
        )
        out = Table.merge_partition_filters(mock, "SELECT * FROM src")
        assert out == ["T.`d` IN ('2024-01-01', '2024-01-02')"]   # distinct + sorted

    def test_unpartitioned_table_returns_no_filters(self):
        from yggdrasil.databricks.table.table import Table
        mock = self._mock_table(part_cols=())     # no partition columns
        assert Table.merge_partition_filters(mock, "SELECT * FROM src") == []
        mock.sql.execute.assert_not_called()      # no distinct query either

    def test_placeholder_source_is_skipped(self):
        from yggdrasil.databricks.table.table import Table
        mock = self._mock_table()
        # the sync arrow path carries a {__tmpsrc__} placeholder — not queryable
        assert Table.merge_partition_filters(mock, "SELECT * FROM {__tmpsrc__}") == []
        mock.collect_schema.assert_not_called()

    def test_null_partition_value_skips_pruning_for_that_column(self):
        from yggdrasil.databricks.table.table import Table
        mock = self._mock_table(distinct=pa.table({"d": ["a", None]}))
        # NULL wouldn't be matched by IN — drop the filter rather than risk a
        # wrong prune.
        assert Table.merge_partition_filters(mock, "src") == []

    def test_empty_source_yields_no_filters(self):
        from yggdrasil.databricks.table.table import Table
        mock = self._mock_table(distinct=pa.table({"d": pa.array([], pa.string())}))
        assert Table.merge_partition_filters(mock, "src") == []


# --------------------------------------------------------------------------- #
# stage_async_insert (producer)
# --------------------------------------------------------------------------- #
class TestStageAsyncInsert:
    def test_rejects_unsupported_mode(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        with pytest.raises(ValueError, match="OVERWRITE / APPEND / MERGE / UPSERT"):
            stage_async_insert(t, object(), mode="truncate")

    def test_merge_requires_keys(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        with pytest.raises(ValueError, match="requires match_by"):
            stage_async_insert(t, object(), mode="merge")

    def test_merge_with_keys_stages_with_match_by(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        data_file = MagicMock()
        data_file.to_url.return_value.to_string.return_value = "dbfs+volume:/x.parquet"
        t.insert_volume_path.return_value = data_file
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file
        with patch("yggdrasil.databricks.table.insert.logs_path", lambda tbl: logs_dir):
            stage_async_insert(t, {"a": [1]}, mode="merge", match_by=["id"])
        payload = json.loads(log_file.write_bytes.call_args[0][0])
        assert payload["mode"] == "merge"
        assert payload["match_by"] == ["id"]

    def test_string_match_by_is_rejected(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        with pytest.raises(ValueError, match="list of key columns"):
            stage_async_insert(t, object(), mode="merge", match_by="id")

    def test_log_does_not_persist_schema_or_partition_columns(self):
        # Partition columns (and the schema) are NEVER baked into the op-log —
        # the loader refetches the live partition layout in the job. The log
        # carries only the keys + staged data location.
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        data_file = MagicMock()
        data_file.to_url.return_value.to_string.return_value = "dbfs+volume:/x.parquet"
        t.insert_volume_path.return_value = data_file
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file
        with patch("yggdrasil.databricks.table.insert.logs_path", lambda tbl: logs_dir):
            stage_async_insert(t, {"a": [1]}, mode="merge", match_by=["id"])
        payload = json.loads(log_file.write_bytes.call_args[0][0])
        assert "schema" not in payload
        assert "partition" not in json.dumps(payload).lower()
        assert payload["match_by"] == ["id"]      # keys are carried, partitions aren't

    def test_writes_parquet_to_staging_and_logs_its_uniform_url(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        # data goes to the default tmp staging path
        data_file = MagicMock()
        data_file.to_url.return_value.to_string.return_value = (
            "dbfs+volume:/c/s/t/.sql/tmp/tmp-1-ab.parquet"
        )
        t.insert_volume_path.return_value = data_file
        # log dir
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file

        with patch("yggdrasil.databricks.table.insert.logs_path", lambda tbl: logs_dir):
            result = stage_async_insert(t, {"a": [1]}, mode="append")

        assert result is log_file
        data_file.write_table.assert_called_once()           # staged Parquet
        log_file.write_bytes.assert_called_once()            # operation log
        payload = json.loads(log_file.write_bytes.call_args[0][0])
        assert payload["target"] == "c.s.t"
        assert payload["mode"] == "append"
        assert payload["data"] == "dbfs+volume:/c/s/t/.sql/tmp/tmp-1-ab.parquet"

    def test_string_source_is_read_then_staged(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        data_file = MagicMock()
        data_file.to_url.return_value.to_string.return_value = (
            "dbfs+volume:/c/s/t/.sql/tmp/x.parquet"
        )
        t.insert_volume_path.return_value = data_file
        logs_dir, log_file = MagicMock(), MagicMock()
        logs_dir.__truediv__.return_value = log_file
        src = MagicMock()
        src.read_arrow_table.return_value = {"a": [1]}
        with patch("yggdrasil.databricks.table.insert.logs_path", lambda tbl: logs_dir), \
             patch("yggdrasil.io.holder.IO.from_", return_value=src) as io_from:
            stage_async_insert(t, "s3://b/data.parquet", mode="append")
        io_from.assert_called_once_with("s3://b/data.parquet")
        src.read_arrow_table.assert_called_once_with()
        data_file.write_table.assert_called_once()


# --------------------------------------------------------------------------- #
# ensure_async_job (get-or-create the file-arrival loader job)
# --------------------------------------------------------------------------- #
class TestEnsureAsyncJob:
    def test_creates_job_with_file_arrival_trigger(self):
        from yggdrasil.databricks.table.insert import ensure_async_job

        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        jobs.get.return_value = None   # no existing job → create path
        created = MagicMock()
        created.job_id = 42
        jobs.create_or_update.return_value = created
        logs = t.staging_volume.path.return_value
        logs.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"
        jobs.list.return_value = []

        # The serverless env (versioned ygg image) is resolved through the
        # client's global ygg_environment() — a get-or-created v5 + wheel bundle.
        from databricks.sdk.service.compute import Environment
        from databricks.sdk.service.jobs import JobEnvironment

        wheels = ["/Workspace/Shared/.ygg/whl/9.9/ygg-9.9-py3-none-any.whl",
                  "/Workspace/Shared/.ygg/whl/9.9/databricks_sdk-1.2-py3-none-any.whl"]
        environment = JobEnvironment(
            environment_key="default",
            spec=Environment(environment_version="5", dependencies=wheels),
        )
        t.client.ygg_environment.return_value = environment

        job = ensure_async_job(t)

        assert job is created
        # the versioned ygg image is resolved once for the create path
        t.client.ygg_environment.assert_called_once_with(
            environment_key="default", rebuild=False,
        )
        # the watched logs dir is created so the trigger URL is valid
        logs.mkdir.assert_called_with(parents=True, exist_ok=True)

        kwargs = jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "[YGG][ASYNC] c.s.t"
        fa = kwargs["trigger"].file_arrival
        assert fa.url == "/Volumes/c/s/t/.sql/async/logs/"
        assert fa.wait_after_last_change_seconds == 120        # 2-min buffering
        assert fa.min_time_between_triggers_seconds == 120
        task = kwargs["tasks"][0]
        # ygg databricks table execute_async_insert --logs <dir> on the cluster
        assert task.python_wheel_task.package_name == "ygg"
        assert task.python_wheel_task.entry_point == "ygg"
        assert task.python_wheel_task.parameters == [
            "databricks", "table", "execute_async_insert",
            "--logs", "/Volumes/c/s/t/.sql/async/logs",
            "--debug", "--prune-partitions", "--spark",
        ]
        # serverless v5 image shipped as the env, wired to the task
        env = kwargs["environments"][0]
        assert env is environment
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == wheels
        assert task.environment_key == env.environment_key

    def test_prunes_stale_jobs_on_same_trigger(self):
        from yggdrasil.databricks.table.insert import ensure_async_job

        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        jobs.get.return_value = None   # no existing job → create path
        created = MagicMock()
        created.job_id = 99
        jobs.create_or_update.return_value = created
        t.staging_volume.path.return_value.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"
        url = "/Volumes/c/s/t/.sql/async/logs/"

        def _job(job_id, trigger_url):
            j = MagicMock()
            j.job_id = job_id
            j.settings.trigger.file_arrival.url = trigger_url
            return j

        keep = _job(99, url)            # the one we just deployed
        stale = _job(7, url)            # an orphan watching the same logs dir
        other = _job(8, "/Volumes/other/.sql/async/logs/")  # unrelated job
        jobs.list.return_value = [keep, stale, other]

        ensure_async_job(t)

        stale.delete.assert_called_once()      # orphan on the shared trigger removed
        keep.delete.assert_not_called()
        other.delete.assert_not_called()

    def test_get_returns_existing_job_without_building(self):
        # Get-or-create: an existing job short-circuits — no image resolve, no
        # upsert, no logs-dir provisioning.
        from yggdrasil.databricks.table.insert import ensure_async_job

        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        existing = MagicMock()
        existing.job_id = 7
        jobs.get.return_value = existing

        job = ensure_async_job(t)

        assert job is existing
        jobs.get.assert_called_once_with(name="[YGG][ASYNC] c.s.t", default=None)
        t.client.ygg_environment.assert_not_called()   # no image resolve on the get path
        jobs.create_or_update.assert_not_called()
        t.staging_volume.path.return_value.mkdir.assert_not_called()

    def test_rebuild_forces_create_path_even_when_job_exists(self):
        # rebuild=True skips the get short-circuit and re-deploys.
        from yggdrasil.databricks.table.insert import ensure_async_job

        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
        jobs.get.return_value = MagicMock(job_id=7)   # would short-circuit if consulted
        created = MagicMock(job_id=8)
        jobs.create_or_update.return_value = created
        t.staging_volume.path.return_value.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"
        jobs.list.return_value = []

        job = ensure_async_job(t, rebuild=True)

        assert job is created
        jobs.get.assert_not_called()           # rebuild bypasses the get
        # rebuild flows through to a forced image rebuild
        t.client.ygg_environment.assert_called_once_with(
            environment_key="default", rebuild=True,
        )
        jobs.create_or_update.assert_called_once()


# --------------------------------------------------------------------------- #
# load_async / dispatch_async — the loader (driven by a log path; groups by table)
# --------------------------------------------------------------------------- #
def _tables_service():
    svc = MagicMock()
    svc.client = MagicMock()
    return svc


def _log(op, *, target="c.s.t", mode="append"):
    f = MagicMock()
    f.name = f"{op}.json"
    # the log records the project's uniform URL for the staged data
    f.read_bytes.return_value = json.dumps(
        {"target": target, "mode": mode,
         "data": f"dbfs+volume:/c/s/t/.sql/tmp/{op}.parquet"}
    ).encode()
    return f


def _logs_dir(*entries):
    d = MagicMock()
    d.exists.return_value = True
    d.is_dir.return_value = True
    d.iterdir.return_value = list(entries)
    return d


def _fake_databricks_from():
    """``DatabricksPath.from_`` stand-in: maps a uniform URL to a mock path
    whose ``full_path()`` is the ``/Volumes/...`` display form. Returns the
    side_effect plus the per-URL cache so cleanup can be asserted."""
    cache: dict = {}

    def _from(url, **_kwargs):
        if url not in cache:
            m = MagicMock()
            # dbfs+volume:/c/s/t/x.parquet → /Volumes/c/s/t/x.parquet
            m.full_path.return_value = "/Volumes" + url.split(":", 1)[1]
            cache[url] = m
        return cache[url]

    return _from, cache


class TestLoadAsync:
    def test_no_logs_returns_zero(self):
        from yggdrasil.databricks.table.insert import load_async
        logs = MagicMock()
        logs.exists.return_value = False
        assert load_async(_tables_service(), logs) == 0

    def test_aggregates_same_group_into_one_insert(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        log_a, log_b = _log("a"), _log("b")
        logs = _logs_dir(log_a, log_b)
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, data_paths = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = load_async(svc, logs, wait=False)

        assert processed == 2
        # one sql_insert load per (target, mode) group, with the union
        target.sql_insert.assert_called_once()
        union = target.sql_insert.call_args.args[0]
        assert "UNION ALL" in union
        # the uniform URL is resolved to the warehouse-facing path for the query
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/b.parquet`" in union
        assert target.sql_insert.call_args.kwargs["mode"] == "append"
        # consumed logs + data (reconstructed from the uniform URL) cleaned up
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/a.parquet"].unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/b.parquet"].unlink.assert_called_once()

    def test_use_spark_resolves_session_and_passes_it(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        session = MagicMock(name="spark-session")
        svc.client.spark.return_value = session
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            load_async(svc, _logs_dir(_log("a")), wait=False, use_spark=True)
        # the loader resolves the cluster Spark session and runs the load on it
        svc.client.spark.assert_called_once()
        assert target.sql_insert.call_args.kwargs["spark_session"] is session

    def test_use_spark_falls_back_to_warehouse_when_unavailable(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        svc.client.spark.side_effect = RuntimeError("no compute reachable")
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            n = load_async(svc, _logs_dir(_log("a")), wait=False, use_spark=True)
        assert n == 1
        # spark unavailable → warehouse path (spark_session=None), load still runs
        assert target.sql_insert.call_args.kwargs["spark_session"] is None

    def test_warehouse_path_does_not_resolve_spark(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            load_async(svc, _logs_dir(_log("a")), wait=False)   # use_spark defaults False
        svc.client.spark.assert_not_called()
        assert target.sql_insert.call_args.kwargs["spark_session"] is None

    def test_debug_prints_batch_and_sql_to_stdout(self, capsys):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        logs = _logs_dir(_log("a"))
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            load_async(svc, logs, wait=False, debug=True)
        out = capsys.readouterr().out
        # the loader announces the target + emits the generated SQL to stdout
        assert "[async-load] c.s.t" in out
        assert "source SQL" in out
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in out

    def test_overwrite_supersedes_earlier_append_for_same_target(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        logs = _logs_dir(
            _log("a", mode="append"), _log("b", mode="overwrite"),
        )
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, data_paths = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = load_async(svc, logs, wait=False)
        assert processed == 2
        target.sql_insert.assert_called_once()
        sql = target.sql_insert.call_args.args[0]
        assert target.sql_insert.call_args.kwargs["mode"] == "overwrite"
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/b.parquet`" in sql
        assert "a.parquet" not in sql            # superseded — not in the load
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/a.parquet"].unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/b.parquet"].unlink.assert_called_once()

    def test_groups_by_target_table_from_logs(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        logs = _logs_dir(
            _log("a", target="c.s.t1"), _log("b", target="c.s.t2"),
        )
        tables: dict[str, MagicMock] = {}
        svc.__getitem__.side_effect = lambda name: tables.setdefault(name, MagicMock())

        with patch("yggdrasil.databricks.path.DatabricksPath.from_"):
            processed = load_async(svc, logs)

        assert processed == 2
        assert set(tables) == {"c.s.t1", "c.s.t2"}
        tables["c.s.t1"].sql_insert.assert_called_once()
        tables["c.s.t2"].sql_insert.assert_called_once()

    def test_single_log_file_path_string(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        log = _log("a")
        log.exists.return_value = True
        log.is_dir.return_value = False
        target = MagicMock()
        svc.__getitem__.return_value = target
        with patch("yggdrasil.databricks.path.DatabricksPath.from_",
                   side_effect=lambda p, **k: log if p == "/logs/a.json" else MagicMock()):
            processed = load_async(svc, "/logs/a.json")
        assert processed == 1
        target.sql_insert.assert_called_once()

    def test_log_files_arg_skips_the_directory_scan(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        log_a, log_b = _log("a"), _log("b")
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = load_async(svc, log_files=[log_a, log_b], wait=False)
        assert processed == 2
        target.sql_insert.assert_called_once()   # one (target, mode) group
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()

    def test_dispatch_async_groups_preparsed_ops(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, dispatch_async
        svc = _tables_service()
        log_a, log_b = MagicMock(), MagicMock()
        ops = [
            DatabricksTableInsert(target="c.s.t", mode="append",
                                  data="dbfs+volume:/c/s/t/.sql/tmp/a.parquet", log_file=log_a),
            DatabricksTableInsert(target="c.s.t", mode="append",
                                  data="dbfs+volume:/c/s/t/.sql/tmp/b.parquet", log_file=log_b),
        ]
        target = MagicMock()
        svc.__getitem__.return_value = target
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn):
            processed = dispatch_async(svc, ops)
        assert processed == 2
        target.sql_insert.assert_called_once()
        union = target.sql_insert.call_args.args[0]
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
