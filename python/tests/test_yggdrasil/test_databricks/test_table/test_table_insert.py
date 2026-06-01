"""Unit tests for the centralized table insert module (no live Databricks).

Covers:
* ``Table.insert(wait=False)`` routing → ``stage_async_insert`` (OVERWRITE /
  APPEND, no match_by), and the sync path otherwise;
* :class:`DatabricksTableInsert` parsing / serialization / validation and the
  per-op + per-batch SQL generators (``make_sql_select`` / ``make_sql_insert``,
  including the keyed-batch MERGE dedup);
* ``stage_async_insert`` staging a Parquet + dropping a JSON operation log;
* ``ensure_async_job`` get-or-create with a file-arrival trigger;
* ``load_async`` / ``dispatch_async`` aggregating logs into one load per
  ``(target, mode)``, then cleaning up consumed logs + data.
"""
from __future__ import annotations

import contextlib
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


async def _drain(awaitable):
    """``await`` an insert op/batch from a sync test."""
    return await awaitable


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

    def test_merge_mode_stays_sync(self):
        t = MagicMock()
        with patch("yggdrasil.databricks.table.insert.stage_async_insert") as stage:
            Table.insert(t, {"a": [1]}, mode="merge", wait=False)
        t.insert_into.assert_called_once()
        stage.assert_not_called()


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
        # A producer-built op holds typed Table / Mode; to_json emits the
        # string op-log form the loader reads back. An already-durable URL
        # string ``data`` is kept as-is by the serialize-time dispatch.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        payload = json.loads(
            DatabricksTableInsert(
                target=target, mode="overwrite", data="dbfs+volume:/x.parquet",
            ).to_json()
        )
        assert payload["target"] == "c.s.t"
        assert payload["mode"] == "overwrite"
        assert payload["data"] == "dbfs+volume:/x.parquet"

    def test_data_url_keeps_string_source(self):
        # An already-durable URL string is kept verbatim (the read-back /
        # pre-staged producer shape).
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="s3://b/k.parquet")
        assert op.data_url == "s3://b/k.parquet"

    def test_data_url_dumps_arrow_source_to_volume(self):
        # A non-spark, non-cloud-path Tabular (Arrow table) is dumped to the
        # target's staging Volume; the log points at the file.
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        op = DatabricksTableInsert(
            target=target, mode="append", data=ArrowTabular(pa.table({"a": [1, 2]})),
        )
        with patch.object(DatabricksTableInsert, "_dump_to_volume") as dump:
            dump.return_value.to_url.return_value.to_string.return_value = (
                "dbfs+volume:/c/s/t/.sql/tmp/x.parquet"
            )
            assert op.data_url == "dbfs+volume:/c/s/t/.sql/tmp/x.parquet"
            dump.assert_called_once()

    def test_data_url_stages_spark_source_as_table(self):
        # A Spark frame can't be serialized — it's staged into a temp Delta
        # table and the log points at the table URL.
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        with patch.object(ArrowTabular, "_native_spark_frame", return_value=object()), \
                patch.object(DatabricksTableInsert, "_stage_spark_table") as stage:
            stage.return_value.to_url.return_value.to_string.return_value = (
                "dbfs+table://h/c/staging/_ygg_stg_x"
            )
            op = DatabricksTableInsert(
                target=target, mode="overwrite", data=ArrowTabular(pa.table({"a": [1]})),
            )
            assert op.data_url == "dbfs+table://h/c/staging/_ygg_stg_x"
            stage.assert_called_once()

    def test_data_url_materialises_once(self):
        # The dispatch (and its staging side-effect) runs at most once.
        from yggdrasil.arrow.tabular import ArrowTabular
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        target = MagicMock()
        op = DatabricksTableInsert(
            target=target, mode="append", data=ArrowTabular(pa.table({"a": [1]})),
        )
        with patch.object(DatabricksTableInsert, "_dump_to_volume") as dump:
            dump.return_value.to_url.return_value.to_string.return_value = "dbfs+volume:/x"
            _ = op.data_url
            _ = op.data_url
            dump.assert_called_once()

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

    def _op_target(self):
        # A Table-ish mock the op's ``_submit`` can build DML against.
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        target.catalog_name, target.schema_name = "c", "s"
        target.collect_schema.return_value.fields = []
        return target

    def test_execute_builds_dml_and_runs_via_execute_many(self):
        # The op renders its own INSERT/MERGE statement list and runs it via
        # the shared ``_run_dml`` (prepare + execute_many) — no sql_insert.
        from yggdrasil.enums.state import State
        from yggdrasil.databricks.table.insert import DatabricksTableInsert

        inner = MagicMock(state=State.SUCCEEDED, is_done=True, error=None, retryable=False)
        target = self._op_target()
        op = DatabricksTableInsert(
            target=target, mode="append",
            data="dbfs+volume:/c/s/t/.sql/tmp/x.parquet", client=MagicMock(),
        )
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp, \
                patch.object(DatabricksTableInsert, "_run_dml", return_value=inner) as run:
            dp.return_value.full_path.return_value = "/Volumes/c/s/t/.sql/tmp/x.parquet"
            ret = op.execute(wait=True)
        assert ret is op and op.is_succeeded and op.result is inner
        run.assert_called_once()
        texts = run.call_args.args[1]               # the statement list
        assert any("INSERT" in t.upper() for t in texts)
        assert any("parquet.`/Volumes/c/s/t/.sql/tmp/x.parquet`" in t for t in texts)
        # the staged file is registered for post-load cleanup
        assert run.call_args.kwargs["staging"] is dp.return_value

    def test_start_without_wait_then_await(self):
        import asyncio
        from yggdrasil.enums.state import State
        from yggdrasil.databricks.table.insert import DatabricksTableInsert

        inner = MagicMock(state=State.SUCCEEDED, is_done=True, error=None, retryable=False)
        op = DatabricksTableInsert(
            target=self._op_target(), mode="append",
            data="dbfs+volume:/c/s/t/.sql/tmp/x.parquet", client=MagicMock(),
        )
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp, \
                patch.object(DatabricksTableInsert, "_run_dml", return_value=inner):
            dp.return_value.full_path.return_value = "/Volumes/x"
            op.start(wait=False)
            assert op.started
            asyncio.run(_drain(op))
        assert op.is_succeeded

    def test_execute_propagates_a_failed_load(self):
        from yggdrasil.enums.state import State
        from yggdrasil.databricks.table.insert import DatabricksTableInsert

        inner = MagicMock(
            state=State.FAILED, is_done=True,
            error=RuntimeError("boom"), retryable=False,
        )
        op = DatabricksTableInsert(
            target=self._op_target(), mode="append",
            data="dbfs+volume:/c/s/t/.sql/tmp/x.parquet", client=MagicMock(),
        )
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp, \
                patch.object(DatabricksTableInsert, "_run_dml", return_value=inner):
            dp.return_value.full_path.return_value = "/Volumes/x"
            with pytest.raises(RuntimeError, match="boom"):
                op.execute(wait=True)

    def test_table_url_round_trips_and_is_a_table_source(self):
        # The Spark dispatch records a ``dbfs+table://`` URL; the op-log keeps
        # it byte-for-byte and the loader recognises it as a table source.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        url = "dbfs+table://host/cat/staging/_ygg_stg_x"
        op = DatabricksTableInsert(target="cat.sch.t", mode="overwrite", data=url)
        assert op.data_url == url                      # serialized verbatim
        assert op.is_table_source is True
        parsed = DatabricksTableInsert.from_json(json.loads(op.to_json()))
        assert parsed.data == url                      # rebuilt without mangling
        assert parsed.is_table_source is True

    def test_file_url_is_not_a_table_source(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(target="c.s.t", mode="append", data="dbfs+volume:/x.parquet")
        assert op.is_table_source is False

    def test_staged_source_rebuilds_table_from_url(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(
            target="c.s.t", mode="overwrite",
            data="dbfs+table://host/cat/staging/tmp",
        )
        with patch("yggdrasil.databricks.table.table.Table.from_url") as furl:
            op.staged_source(client="CL")
        furl.assert_called_once_with("dbfs+table://host/cat/staging/tmp", client="CL")

    def test_cleanup_drops_a_staged_table(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        op = DatabricksTableInsert(
            target="c.s.t", mode="overwrite",
            data="dbfs+table://host/cat/staging/tmp",
        )
        with patch.object(DatabricksTableInsert, "staged_source") as ss:
            tbl = MagicMock()
            ss.return_value = tbl
            op.cleanup_staged_data(client="CL")
            tbl.delete.assert_called_once()


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

    def test_table_source_selects_from_the_table(self):
        # A Spark-staged table source reads straight from the table, not via
        # ``parquet.`<path>```.
        from yggdrasil.databricks.table.insert import DatabricksTableInsert, make_sql_select
        op = DatabricksTableInsert(
            target="c.s.t", mode="overwrite",
            data="dbfs+table://host/cat/staging/tmp",
        )
        table = MagicMock()
        table.full_name.return_value = "`cat`.`staging`.`tmp`"
        with patch.object(DatabricksTableInsert, "staged_source", return_value=table):
            sql = make_sql_select(op, client="CL")
        assert sql == "SELECT * FROM `cat`.`staging`.`tmp`"

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

    def test_execute_runs_the_union_via_execute_many(self):
        # The batch is self-executing: it renders the aggregated INSERT over the
        # UNION ALL body and runs it via ``_run_dml`` (execute_many) — no
        # sql_insert.
        from yggdrasil.enums.state import State
        from yggdrasil.databricks.table.insert import DatabricksInsertBatch
        [batch] = DatabricksInsertBatch.group([
            self._op("a", mode="append", target="c.s.t", ts=1.0),
            self._op("b", mode="append", target="c.s.t", ts=2.0),
        ])
        target = MagicMock()
        target.full_name.return_value = "c.s.t"
        target.catalog_name, target.schema_name = "c", "s"
        target.collect_schema.return_value.fields = []
        inner = MagicMock(state=State.SUCCEEDED, is_done=True, error=None, retryable=False)
        with patch("yggdrasil.databricks.path.DatabricksPath.from_") as dp, \
                patch.object(DatabricksInsertBatch, "_run_dml", return_value=inner) as run:
            dp.side_effect = lambda u, **k: MagicMock(
                full_path=MagicMock(return_value="/Volumes" + u.split(":", 1)[1]),
            )
            batch.execute(target=target, wait=True)
        assert batch.is_succeeded
        run.assert_called_once()
        texts = run.call_args.args[1]
        assert any("UNION ALL" in t for t in texts)
        assert any("INSERT" in t.upper() for t in texts)


# --------------------------------------------------------------------------- #
# stage_async_insert (producer)
# --------------------------------------------------------------------------- #
class TestStageAsyncInsert:
    def test_rejects_non_overwrite_append(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        with pytest.raises(ValueError, match="OVERWRITE / APPEND"):
            stage_async_insert(t, object(), mode="merge")

    def test_rejects_match_by(self):
        from yggdrasil.databricks.table.insert import stage_async_insert
        t = _table_mock()
        with pytest.raises(ValueError, match="match_by"):
            stage_async_insert(t, object(), mode="append", match_by=["id"])

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
        created = MagicMock()
        created.job_id = 42
        jobs.create_or_update.return_value = created
        logs = t.staging_volume.path.return_value
        logs.full_path.return_value = "/Volumes/c/s/t/.sql/async/logs"
        jobs.list.return_value = []

        wheels = ["/Workspace/Shared/.ygg/whl/ygg-9.9-py3-none-any.whl",
                  "/Workspace/Shared/.ygg/whl/databricks_sdk-1.2-py3-none-any.whl"]
        with patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel", return_value=wheels) as ew:
            job = ensure_async_job(t)

        assert job is created
        # the full ygg wheel is built + shipped as the env dependencies
        assert ew.call_count == 1
        # the watched logs dir is created so the trigger URL is valid
        logs.mkdir.assert_called_with(parents=True, exist_ok=True)

        kwargs = jobs.create_or_update.call_args.kwargs
        assert kwargs["name"] == "[YGG][ASYNC] c.s.t"
        fa = kwargs["trigger"].file_arrival
        assert fa.url == "/Volumes/c/s/t/.sql/async/logs/"
        assert fa.wait_after_last_change_seconds == 120        # 2-min buffering
        assert fa.min_time_between_triggers_seconds == 120
        task = kwargs["tasks"][0]
        # ygg databricks table execute_insert --logs <dir> on the cluster
        assert task.python_wheel_task.package_name == "ygg"
        assert task.python_wheel_task.entry_point == "ygg"
        assert task.python_wheel_task.parameters == [
            "databricks", "table", "execute_insert",
            "--logs", "/Volumes/c/s/t/.sql/async/logs",
        ]
        # serverless v5; the built ygg wheel is shipped as the dependencies
        env = kwargs["environments"][0]
        assert env.spec.environment_version == "5"
        assert env.spec.dependencies == wheels
        assert task.environment_key == env.environment_key

    def test_prunes_stale_jobs_on_same_trigger(self):
        from yggdrasil.databricks.table.insert import ensure_async_job

        t = _table_mock()
        jobs = MagicMock()
        t.client.jobs = jobs
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

        with patch("yggdrasil.databricks.job.wheel.ensure_ygg_wheel", return_value=["w.whl"]):
            ensure_async_job(t)

        stale.delete.assert_called_once()      # orphan on the shared trigger removed
        keep.delete.assert_not_called()
        other.delete.assert_not_called()


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


def _loader_target():
    """A resolved-Table mock the batch's ``_submit`` can build DML against."""
    t = MagicMock()
    t.full_name.return_value = "c.s.t"
    t.catalog_name, t.schema_name = "c", "s"
    t.collect_schema.return_value.fields = []   # async ops carry no schema
    return t


@contextlib.contextmanager
def _capture_run_dml():
    """Patch the batch's ``_run_dml`` (prepare + execute_many) and capture the
    rendered statement list per call, returning a SUCCEEDED inner batch."""
    from yggdrasil.databricks.table.insert import DatabricksInsertBatch
    from yggdrasil.enums.state import State

    calls: list = []
    inner = MagicMock(state=State.SUCCEEDED, is_done=True, error=None, retryable=False)

    def _run(self, target, texts, **_kwargs):
        calls.append((target, texts))
        return inner

    with patch.object(DatabricksInsertBatch, "_run_dml", _run):
        yield calls


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
        svc.__getitem__.return_value = _loader_target()
        from_fn, data_paths = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn), \
                _capture_run_dml() as calls:
            processed = load_async(svc, logs, wait=False)

        assert processed == 2
        # one load per (target, mode) group, with the union body in the DML
        assert len(calls) == 1
        union = "\n".join(calls[0][1])
        assert "UNION ALL" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/b.parquet`" in union
        assert "INSERT INTO" in union and "OVERWRITE" not in union   # append
        # consumed logs + data (reconstructed from the uniform URL) cleaned up
        log_a.unlink.assert_called_once()
        log_b.unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/a.parquet"].unlink.assert_called_once()
        data_paths["dbfs+volume:/c/s/t/.sql/tmp/b.parquet"].unlink.assert_called_once()

    def test_overwrite_supersedes_earlier_append_for_same_target(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        logs = _logs_dir(
            _log("a", mode="append"), _log("b", mode="overwrite"),
        )
        svc.__getitem__.return_value = _loader_target()
        from_fn, data_paths = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn), \
                _capture_run_dml() as calls:
            processed = load_async(svc, logs, wait=False)
        assert processed == 2
        assert len(calls) == 1
        sql = "\n".join(calls[0][1])
        assert "INSERT OVERWRITE" in sql
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
        svc.__getitem__.side_effect = lambda name: tables.setdefault(name, _loader_target())

        with patch("yggdrasil.databricks.path.DatabricksPath.from_"), \
                _capture_run_dml() as calls:
            processed = load_async(svc, logs)

        assert processed == 2
        assert set(tables) == {"c.s.t1", "c.s.t2"}
        # one load per target group
        assert len(calls) == 2
        assert {c[0] for c in calls} == set(tables.values())

    def test_single_log_file_path_string(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        log = _log("a")
        log.exists.return_value = True
        log.is_dir.return_value = False
        svc.__getitem__.return_value = _loader_target()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_",
                   side_effect=lambda p, **k: log if p == "/logs/a.json" else MagicMock()), \
                _capture_run_dml() as calls:
            processed = load_async(svc, "/logs/a.json")
        assert processed == 1
        assert len(calls) == 1

    def test_log_files_arg_skips_the_directory_scan(self):
        from yggdrasil.databricks.table.insert import load_async
        svc = _tables_service()
        log_a, log_b = _log("a"), _log("b")
        svc.__getitem__.return_value = _loader_target()
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn), \
                _capture_run_dml() as calls:
            processed = load_async(svc, log_files=[log_a, log_b], wait=False)
        assert processed == 2
        assert len(calls) == 1                    # one (target, mode) group
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
        svc.__getitem__.return_value = _loader_target()
        from_fn, _ = _fake_databricks_from()
        with patch("yggdrasil.databricks.path.DatabricksPath.from_", side_effect=from_fn), \
                _capture_run_dml() as calls:
            processed = dispatch_async(svc, ops)
        assert processed == 2
        assert len(calls) == 1
        union = "\n".join(calls[0][1])
        assert "parquet.`/Volumes/c/s/t/.sql/tmp/a.parquet`" in union
