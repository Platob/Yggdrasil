"""Unit tests for the centralized table insert module (no live Databricks).

Covers:
* :class:`DatabricksTableInsert` validation, staged-source resolution, and
  self-execution (build DML → run via ``execute_many``);
* the SQL generators ``make_sql_select`` / ``make_sql_insert`` and the DML
  builders (plain / overwrite / truncate / keyed MERGE / safe-merge);
* ``Table.insert`` forwarding to ``insert_into`` and ``Table.stage_insert``
  staging Parquet only;
* native-Delta vs warehouse write routing in ``_resolve_engine`` /
  ``_write_arrow_batches``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import pyarrow as pa
import pytest
from databricks.sdk.service.catalog import TableType

from yggdrasil.data.schema import Schema
from yggdrasil.databricks.table.table import Table


def _schema(*pairs):
    return Schema.from_arrow(pa.schema([pa.field(n, t) for n, t in pairs]))


def _normalize_ws(sql: str) -> str:
    return " ".join(sql.split())


async def _drain(awaitable):
    """``await`` an insert op from a sync test."""
    return await awaitable


def _table(catalog_name="cat", schema_name="sch", table_name="tbl") -> Table:
    """Build a :class:`Table` with a mocked service so ``self.client`` resolves
    without touching a real Databricks workspace."""
    service = MagicMock()
    ws = MagicMock()
    service.client.workspace_client.return_value = ws
    service.volumes.client.workspace_client.return_value = ws
    service.client.safe_tag_value.side_effect = lambda v, repl="_": str(v).replace("/", repl)
    return Table(
        service=service,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=table_name,
    )


# --------------------------------------------------------------------------- #
# Table.insert / Table.stage_insert
# --------------------------------------------------------------------------- #
class TestInsertForwarding:
    def test_insert_forwards_to_insert_into(self):
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append")
        t.insert_into.assert_called_once()
        assert t.insert_into.call_args.kwargs["mode"] == "append"

    def test_insert_forwards_with_wait_false(self):
        # wait=False is just forwarded now (no async drop branch).
        t = MagicMock()
        Table.insert(t, {"a": [1]}, mode="append", wait=False)
        t.insert_into.assert_called_once()
        assert t.insert_into.call_args.kwargs["wait"] is False

    def test_stage_insert_writes_parquet_and_returns_path(self):
        t = MagicMock()
        path = MagicMock()
        t.insert_volume_path.return_value = path
        out = Table.stage_insert(t, {"a": [1]})
        t.insert_volume_path.assert_called_once_with(t, temporary=False)
        path.write_table.assert_called_once()
        assert out is path

    def test_stage_insert_passes_cast_options(self):
        from yggdrasil.data.options import CastOptions
        t = MagicMock()
        path = MagicMock()
        t.insert_volume_path.return_value = path
        opts = CastOptions()
        Table.stage_insert(t, {"a": [1]}, cast_options=opts)
        assert path.write_table.call_args.args[1] is opts


# --------------------------------------------------------------------------- #
# DatabricksTableInsert — validation / staged-source / self-execution
# --------------------------------------------------------------------------- #
class TestDatabricksTableInsert:
    def test_rejects_bad_mode_on_bare_op(self):
        from yggdrasil.databricks.table.insert import DatabricksTableInsert
        with pytest.raises(ValueError, match="OVERWRITE / APPEND"):
            DatabricksTableInsert(target="c.s.t", mode="merge", data="dbfs+volume:/x.parquet")

    def test_keyed_op_allows_any_mode(self):
        # A richer op (schema / match_by) supports every mode — the
        # OVERWRITE/APPEND-only guard is for the bare op shape.
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
        # the shared ``_run_dml`` (prepare + execute_many).
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
# Write routing — native DeltaFolder vs warehouse (the size gate)
# --------------------------------------------------------------------------- #
class TestWriteRouting:
    """``_write_arrow_batches`` → ``_resolve_engine`` routing: an EXTERNAL
    Delta table under the (relaxed) external cap commits straight to the
    storage path via DeltaFolder; a managed table (or a too-large external one)
    goes to the warehouse ``arrow_insert`` instead."""

    def _batches(self):
        return [pa.RecordBatch.from_pylist([{"a": 1}], schema=pa.schema([("a", pa.int64())]))]

    def _options(self):
        # No ``engine`` set → ``_resolve_engine`` takes the guess path.
        from yggdrasil.data.options import CastOptions
        return CastOptions()

    def test_external_under_cap_routes_to_native_delta(self):
        tbl = _table()
        folder = MagicMock()
        with patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "infos", new_callable=PropertyMock) as infos, \
                patch.object(Table, "_has_active_spark", return_value=False), \
                patch.object(Table, "_delta_total_bytes", return_value=1024), \
                patch.object(Table, "_native_delta_folder", return_value=folder) as native, \
                patch.object(Table, "arrow_insert") as warehouse:
            infos.return_value.table_type = TableType.EXTERNAL
            tbl._write_arrow_batches(self._batches(), self._options())

        native.assert_called_once_with(write=True)
        folder.write_arrow_batches.assert_called_once()
        warehouse.assert_not_called()

    def test_managed_routes_to_warehouse(self):
        tbl = _table()
        # Managed tables aren't write-capable natively → fall to the warehouse.
        with patch.object(Table, "_delta_capable", return_value=False), \
                patch.object(Table, "infos", new_callable=PropertyMock) as infos, \
                patch.object(Table, "_has_active_spark", return_value=False), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_native_delta_folder") as native, \
                patch.object(Table, "arrow_insert") as warehouse:
            infos.return_value.table_type = TableType.MANAGED
            tbl._write_arrow_batches(self._batches(), self._options())

        native.assert_not_called()
        warehouse.assert_called_once()

    def test_external_above_cap_routes_to_warehouse(self):
        from yggdrasil.databricks.table.table import _NATIVE_DELTA_MAX_BYTES_EXTERNAL
        tbl = _table()
        with patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "infos", new_callable=PropertyMock) as infos, \
                patch.object(Table, "_has_active_spark", return_value=False), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(
                    Table, "_delta_total_bytes",
                    return_value=_NATIVE_DELTA_MAX_BYTES_EXTERNAL + 1,
                ), \
                patch.object(Table, "_native_delta_folder") as native, \
                patch.object(Table, "arrow_insert") as warehouse:
            infos.return_value.table_type = TableType.EXTERNAL
            tbl._write_arrow_batches(self._batches(), self._options())

        native.assert_not_called()
        warehouse.assert_called_once()

    def test_external_above_small_cap_still_native(self):
        # A size between the managed cap and the external cap stays native for
        # an EXTERNAL table — the relaxed gate is what's exercised here.
        from yggdrasil.databricks.table.table import _NATIVE_DELTA_MAX_BYTES
        tbl = _table()
        folder = MagicMock()
        with patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "infos", new_callable=PropertyMock) as infos, \
                patch.object(Table, "_has_active_spark", return_value=False), \
                patch.object(
                    Table, "_delta_total_bytes",
                    return_value=_NATIVE_DELTA_MAX_BYTES * 4,
                ), \
                patch.object(Table, "_native_delta_folder", return_value=folder), \
                patch.object(Table, "arrow_insert") as warehouse:
            infos.return_value.table_type = TableType.EXTERNAL
            tbl._write_arrow_batches(self._batches(), self._options())

        folder.write_arrow_batches.assert_called_once()
        warehouse.assert_not_called()


class TestArrowInsertNativeStoragePath:
    """``arrow_insert`` storage-path fast path: a plain APPEND/OVERWRITE on a
    table that's writable per its existing metadata commits straight to the
    storage path via DeltaFolder (no Volume staging, no warehouse INSERT);
    everything else falls back to the classic staged-Parquet path."""

    def _data(self):
        return pa.table({"a": [1, 2, 3]})

    def test_writable_takes_native_storage_path(self):
        self._tbl = _table()
        folder = MagicMock()
        with patch.object(Table, "create", side_effect=lambda *a, **k: self._tbl), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_delta_capable", return_value=True) as capable, \
                patch.object(Table, "_native_delta_folder", return_value=folder) as native, \
                patch.object(Table, "insert_volume_path") as staging:
            out = self._tbl.arrow_insert(self._data(), mode="append")

        capable.assert_called_once_with(write=True)
        native.assert_called_once_with(write=True)
        folder.write_table.assert_called_once()
        assert folder.write_table.call_args.kwargs["mode"].name == "APPEND"
        staging.assert_not_called()  # no Volume staging
        assert out is None

    def test_not_writable_falls_back_to_classic(self):
        self._tbl = _table()
        with patch.object(Table, "create", side_effect=lambda *a, **k: self._tbl), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_delta_capable", return_value=False), \
                patch.object(Table, "_native_delta_folder") as native, \
                patch.object(Table, "insert_volume_path") as staging, \
                patch("yggdrasil.databricks.table.table.DatabricksTableInsert") as op_cls:
            staging.return_value = MagicMock()
            self._tbl.arrow_insert(self._data(), mode="append")

        native.assert_not_called()             # metadata says not writable
        staging.assert_called_once()           # classic Volume staging ran
        op_cls.assert_called_once()            # classic INSERT op built

    def test_keyed_merge_stays_classic_even_if_writable(self):
        self._tbl = _table()
        with patch.object(Table, "create", side_effect=lambda *a, **k: self._tbl), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "_native_delta_folder") as native, \
                patch.object(Table, "insert_volume_path") as staging, \
                patch("yggdrasil.databricks.table.table.DatabricksTableInsert") as op_cls:
            staging.return_value = MagicMock()
            self._tbl.arrow_insert(self._data(), mode="upsert", match_by=["a"])

        native.assert_not_called()             # keyed MERGE needs the warehouse
        staging.assert_called_once()
        op_cls.assert_called_once()

    def test_explicit_engine_stays_classic(self):
        self._tbl = _table()
        with patch.object(Table, "create", side_effect=lambda *a, **k: self._tbl), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "_native_delta_folder") as native, \
                patch.object(Table, "insert_volume_path") as staging, \
                patch("yggdrasil.databricks.table.table.DatabricksTableInsert") as op_cls:
            staging.return_value = MagicMock()
            self._tbl.arrow_insert(self._data(), mode="append", engine="api")

        native.assert_not_called()             # forced engine → no native guess
        staging.assert_called_once()
        op_cls.assert_called_once()

    def test_probe_failure_falls_back_to_classic(self):
        # Metadata claims writable but the credential probe returns None — the
        # rows must still land, via the classic warehouse path.
        self._tbl = _table()
        with patch.object(Table, "create", side_effect=lambda *a, **k: self._tbl), \
                patch.object(Table, "collect_schema", return_value=_schema(("a", pa.int64()))), \
                patch.object(Table, "_delta_capable", return_value=True), \
                patch.object(Table, "_native_delta_folder", return_value=None) as native, \
                patch.object(Table, "insert_volume_path") as staging, \
                patch("yggdrasil.databricks.table.table.DatabricksTableInsert") as op_cls:
            staging.return_value = MagicMock()
            self._tbl.arrow_insert(self._data(), mode="append")

        native.assert_called_once_with(write=True)  # probe attempted
        staging.assert_called_once()                # then classic fallback
        op_cls.assert_called_once()                 # classic INSERT op built
