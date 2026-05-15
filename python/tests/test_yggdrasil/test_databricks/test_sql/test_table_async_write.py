"""Tests for :class:`AsyncInsert`, :func:`stage_async_insert`, and
:meth:`Table.insert(async_write=True)`.

The whole module is mockable end-to-end: ``stage_async_insert`` is
exercised against a mocked staging folder, the SQL rendering /
execute / cleanup path is exercised against a mock SQLEngine and
mock paths, and the merge logic runs entirely on in-memory dataclass
instances.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from yggdrasil.databricks.fs import VolumePath
from yggdrasil.databricks.table.async_write import (
    AsyncInsert,
    METADATA_VERSION,
    _make_operation_id,
    _normalize_prune_by,
    _normalize_prune_values,
    _path_for_sql,
    _predicate_to_sql,
    stage_async_insert,
)
from yggdrasil.databricks.table.table import Table
from yggdrasil.pickle import json as ygg_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table_with_staging(
    *,
    field_names: list[str] | None = None,
) -> tuple[Table, MagicMock, MagicMock, MagicMock]:
    """Build a :class:`Table` whose ``staging_folder`` / ``collect_schema``
    are mocked so the staging call doesn't touch a workspace."""
    service = MagicMock()
    service.client.workspace_client.return_value = MagicMock()
    tbl = Table(
        service=service,
        catalog_name="cat",
        schema_name="sch",
        table_name="tbl",
    )

    folder = MagicMock(spec=VolumePath)
    parquet_path = MagicMock(spec=VolumePath)
    meta_path = MagicMock(spec=VolumePath)
    parquet_path.full_path.return_value = (
        "/Volumes/cat/sch/stg_tbl/.sql/async/insert/async-1.parquet"
    )
    meta_path.full_path.return_value = (
        "/Volumes/cat/sch/stg_tbl/.sql/async/insert/async-1.json"
    )

    def _join(name: str) -> VolumePath:
        return parquet_path if name.endswith(".parquet") else meta_path

    folder.joinpath.side_effect = _join

    tbl.staging_folder = MagicMock(return_value=folder)  # type: ignore[assignment]

    if field_names is None:
        tbl.collect_schema = MagicMock(  # type: ignore[assignment]
            side_effect=RuntimeError("no schema for unit test"),
        )
    else:
        schema = MagicMock()
        schema.field_names.return_value = field_names
        tbl.collect_schema = MagicMock(return_value=schema)  # type: ignore[assignment]

    return tbl, folder, parquet_path, meta_path


def _make_record(
    *,
    target: str = "cat.sch.tbl",
    parquets: tuple[str, ...] = ("/Volumes/cat/sch/stg_tbl/.sql/async/insert/p.parquet",),
    metas: tuple[str, ...] = ("/Volumes/cat/sch/stg_tbl/.sql/async/insert/p.json",),
    ops: tuple[str, ...] = ("async-1",),
    created_at: str = "2026-05-15T10:00:00+00:00",
    mode: str | None = "append",
    target_field_names: tuple[str, ...] | None = ("id", "value"),
    **overrides: Any,
) -> AsyncInsert:
    return AsyncInsert(
        target_full_name=target,
        parquet_paths=parquets,
        metadata_paths=metas,
        operation_ids=ops,
        created_at=created_at,
        mode=mode,
        target_field_names=target_field_names,
        **overrides,
    )


# ---------------------------------------------------------------------------
# Pure-helper coverage
# ---------------------------------------------------------------------------


class TestMakeOperationId:

    def test_unique_per_call(self):
        assert _make_operation_id() != _make_operation_id()

    def test_async_prefix(self):
        assert _make_operation_id().startswith("async-")


class TestPredicateToSQL:

    def test_none(self):
        assert _predicate_to_sql(None) is None

    def test_string_passthrough(self):
        assert _predicate_to_sql("x = 1") == "x = 1"

    def test_fallback_to_str(self):
        assert _predicate_to_sql(42) == "42"


class TestNormalizers:

    def test_prune_by_string(self):
        assert _normalize_prune_by("dt") == ("dt",)

    def test_prune_by_list(self):
        assert _normalize_prune_by(["a", "b"]) == ("a", "b")

    def test_prune_by_none(self):
        assert _normalize_prune_by(None) is None

    def test_prune_values_filters_nones(self):
        assert _normalize_prune_values({"a": None, "b": [1, 2]}) == {"b": (1, 2)}

    def test_prune_values_empty(self):
        assert _normalize_prune_values({}) is None
        assert _normalize_prune_values(None) is None


class TestPathForSQL:

    def test_uses_full_path_when_available(self):
        p = MagicMock()
        p.full_path.return_value = "/Volumes/a/b/c.parquet"
        assert _path_for_sql(p) == "/Volumes/a/b/c.parquet"

    def test_falls_back_to_url(self):
        p = MagicMock()
        p.full_path.side_effect = RuntimeError("boom")
        p.url = "dbfs+volume:///a/b.parquet"
        assert _path_for_sql(p) == "dbfs+volume:///a/b.parquet"

    def test_string_pass_through(self):
        # Bare strings have no full_path/url — fall through to str().
        assert _path_for_sql("/some/path") == "/some/path"


# ---------------------------------------------------------------------------
# Dataclass round-trip / serialisation
# ---------------------------------------------------------------------------


class TestAsyncInsertSerialization:

    def test_frozen(self):
        rec = _make_record()
        with pytest.raises(Exception):
            rec.mode = "overwrite"  # type: ignore[misc]

    def test_to_dict_then_from_dict_roundtrip(self):
        rec = _make_record(
            mode="append",
            match_by=("id",),
            zorder_by=("dt",),
            prune_by=("dt",),
            prune_values={"dt": ("2026-05-15",)},
            where="dt = '2026-05-15'",
        )
        data = rec.to_dict()
        # Lists, not tuples, in the dict form.
        assert isinstance(data["parquet_paths"], list)
        rebuilt = AsyncInsert.from_dict(data)
        assert rebuilt == rec

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "target_full_name": "cat.sch.tbl",
            "unknown_field": "ignored",
        }
        rec = AsyncInsert.from_dict(data)
        assert rec.target_full_name == "cat.sch.tbl"

    def test_json_bytes_roundtrip(self):
        rec = _make_record(mode="overwrite")
        rebuilt = AsyncInsert.from_json_bytes(rec.to_json_bytes())
        assert rebuilt == rec

    def test_from_file_reads_bytes_from_path(self):
        rec = _make_record(mode="append")
        path = MagicMock()
        path.read_bytes.return_value = rec.to_json_bytes()
        loaded = AsyncInsert.from_file(path)
        assert loaded == rec


class TestAsyncInsertProperties:

    def test_is_overwrite(self):
        assert _make_record(mode="overwrite").is_overwrite
        assert _make_record(mode="OVERWRITE").is_overwrite
        assert not _make_record(mode="append").is_overwrite

    def test_is_append_default(self):
        assert _make_record(mode=None).is_append
        assert _make_record(mode="append").is_append
        assert _make_record(mode="auto").is_append

    def test_operation_id_first_id(self):
        rec = _make_record(ops=("a", "b", "c"))
        assert rec.operation_id == "a"

    def test_operation_id_empty_when_no_ids(self):
        rec = AsyncInsert(target_full_name="x")
        assert rec.operation_id == ""


# ---------------------------------------------------------------------------
# to_sql
# ---------------------------------------------------------------------------


class TestToSQL:

    def test_append_single_parquet(self):
        rec = _make_record(mode="append")
        sql = rec.to_sql()
        assert len(sql) == 1
        assert sql[0].startswith("INSERT INTO cat.sch.tbl (`id`, `value`) ")
        assert "SELECT * FROM parquet.`/Volumes" in sql[0]

    def test_overwrite_emits_insert_overwrite(self):
        rec = _make_record(mode="overwrite")
        sql = rec.to_sql()[0]
        assert sql.startswith("INSERT OVERWRITE cat.sch.tbl")

    def test_multiple_parquets_use_union_all(self):
        rec = _make_record(
            parquets=("/Volumes/a.parquet", "/Volumes/b.parquet"),
            mode="append",
            target_field_names=None,
        )
        sql = rec.to_sql()[0]
        assert "UNION ALL" in sql
        assert sql.count("SELECT * FROM parquet.`") == 2

    def test_no_target_field_names_omits_column_list(self):
        rec = _make_record(target_field_names=None)
        sql = rec.to_sql()[0]
        assert "(`" not in sql.split("SELECT", 1)[0]

    def test_where_clause_appended(self):
        rec = _make_record(where="x > 0", target_field_names=None)
        sql = rec.to_sql()[0]
        assert sql.endswith(" WHERE x > 0")

    def test_empty_when_no_parquets(self):
        rec = AsyncInsert(target_full_name="cat.sch.tbl")
        assert rec.to_sql() == []

    def test_empty_when_no_target(self):
        rec = AsyncInsert(
            target_full_name="",
            parquet_paths=("/Volumes/a.parquet",),
        )
        assert rec.to_sql() == []


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergeWith:

    def test_two_appends_combine_parquets(self):
        a = _make_record(parquets=("/a.parquet",), metas=("/a.json",), ops=("a",),
                         created_at="2026-05-15T10:00:00+00:00", mode="append")
        b = _make_record(parquets=("/b.parquet",), metas=("/b.json",), ops=("b",),
                         created_at="2026-05-15T11:00:00+00:00", mode="append")
        merged = a.merge_with(b)
        assert merged.is_append
        assert merged.parquet_paths == ("/a.parquet", "/b.parquet")
        assert merged.metadata_paths == ("/a.json", "/b.json")
        assert merged.operation_ids == ("a", "b")
        assert merged.created_at == b.created_at

    def test_commutative_for_appends(self):
        a = _make_record(parquets=("/a.parquet",), metas=("/a.json",), ops=("a",),
                         created_at="2026-05-15T10:00:00+00:00", mode="append")
        b = _make_record(parquets=("/b.parquet",), metas=("/b.json",), ops=("b",),
                         created_at="2026-05-15T11:00:00+00:00", mode="append")
        assert a.merge_with(b) == b.merge_with(a)

    def test_newer_overwrite_drops_older_append_data(self):
        older = _make_record(
            parquets=("/older.parquet",), metas=("/older.json",), ops=("older",),
            created_at="2026-05-15T09:00:00+00:00", mode="append",
        )
        newer = _make_record(
            parquets=("/newer.parquet",), metas=("/newer.json",), ops=("newer",),
            created_at="2026-05-15T12:00:00+00:00", mode="overwrite",
        )
        merged = older.merge_with(newer)
        # SQL projection drops older data
        assert merged.parquet_paths == ("/newer.parquet",)
        # Cleanup still removes the older parquet + metas
        assert "/older.parquet" in merged.metadata_paths
        assert "/older.json" in merged.metadata_paths
        assert merged.is_overwrite

    def test_older_overwrite_keeps_newer_append_in_projection(self):
        older = _make_record(
            parquets=("/older.parquet",), metas=("/older.json",), ops=("older",),
            created_at="2026-05-15T09:00:00+00:00", mode="overwrite",
        )
        newer = _make_record(
            parquets=("/newer.parquet",), metas=("/newer.json",), ops=("newer",),
            created_at="2026-05-15T12:00:00+00:00", mode="append",
        )
        merged = older.merge_with(newer)
        # The merged op stays an overwrite — but pulls both parquets in.
        assert merged.is_overwrite
        assert merged.parquet_paths == ("/older.parquet", "/newer.parquet")

    def test_two_overwrites_latest_wins(self):
        older = _make_record(
            parquets=("/older.parquet",), metas=("/older.json",), ops=("older",),
            created_at="2026-05-15T09:00:00+00:00", mode="overwrite",
        )
        newer = _make_record(
            parquets=("/newer.parquet",), metas=("/newer.json",), ops=("newer",),
            created_at="2026-05-15T12:00:00+00:00", mode="overwrite",
        )
        merged = older.merge_with(newer)
        assert merged.parquet_paths == ("/newer.parquet",)
        assert "/older.parquet" in merged.metadata_paths
        assert merged.is_overwrite

    def test_different_targets_raises(self):
        a = _make_record(target="cat.sch.a")
        b = _make_record(target="cat.sch.b")
        with pytest.raises(ValueError):
            a.merge_with(b)


class TestMergeClassmethod:

    def test_groups_by_target_and_combines(self):
        records = [
            _make_record(target="t1", parquets=("/t1-a.parquet",),
                         metas=("/t1-a.json",), ops=("a",),
                         created_at="2026-05-15T10:00:00+00:00"),
            _make_record(target="t1", parquets=("/t1-b.parquet",),
                         metas=("/t1-b.json",), ops=("b",),
                         created_at="2026-05-15T11:00:00+00:00"),
            _make_record(target="t2", parquets=("/t2.parquet",),
                         metas=("/t2.json",), ops=("c",),
                         created_at="2026-05-15T10:30:00+00:00"),
        ]
        merged = AsyncInsert.merge(records)
        merged_by_target = {r.target_full_name: r for r in merged}
        assert merged_by_target["t1"].parquet_paths == ("/t1-a.parquet", "/t1-b.parquet")
        assert merged_by_target["t2"].parquet_paths == ("/t2.parquet",)

    def test_latest_overwrite_wipes_earlier_within_target(self):
        records = [
            _make_record(target="t1", parquets=("/t1-a.parquet",),
                         metas=("/t1-a.json",), ops=("a",),
                         created_at="2026-05-15T10:00:00+00:00", mode="append"),
            _make_record(target="t1", parquets=("/t1-b.parquet",),
                         metas=("/t1-b.json",), ops=("b",),
                         created_at="2026-05-15T11:00:00+00:00", mode="overwrite"),
            _make_record(target="t1", parquets=("/t1-c.parquet",),
                         metas=("/t1-c.json",), ops=("c",),
                         created_at="2026-05-15T12:00:00+00:00", mode="append"),
        ]
        merged = AsyncInsert.merge(records)
        assert len(merged) == 1
        rec = merged[0]
        # The overwrite at 11:00 wipes the 10:00 append. The 12:00
        # append rides on top of the overwrite (merged into the
        # overwrite scope).
        assert rec.is_overwrite
        # Older append's parquet is NOT in the SQL projection.
        assert "/t1-a.parquet" not in rec.parquet_paths
        # but is in metadata_paths for cleanup.
        assert "/t1-a.parquet" in rec.metadata_paths
        assert "/t1-b.parquet" in rec.parquet_paths
        assert "/t1-c.parquet" in rec.parquet_paths

    def test_accepts_folder_volumepath(self):
        rec = _make_record(target="t1")
        json_entry = MagicMock()
        json_entry.name = "async-1.json"
        json_entry.read_bytes.return_value = rec.to_json_bytes()
        non_parquet = MagicMock()
        non_parquet.name = "async-1.parquet"

        folder = MagicMock(spec=VolumePath)
        folder.ls.return_value = [json_entry, non_parquet]

        merged = AsyncInsert.merge(folder)
        assert len(merged) == 1
        assert merged[0].target_full_name == "t1"


# ---------------------------------------------------------------------------
# Execute + cleanup
# ---------------------------------------------------------------------------


class TestExecute:

    def test_runs_each_sql_statement_against_engine(self):
        rec = _make_record(mode="append")
        engine = MagicMock()
        # Single-statement: execute returns whatever engine returns.
        engine.execute.return_value = "result-1"

        # Skip cleanup so this test doesn't touch DatabricksPath.
        result = rec.execute(engine, cleanup=False)

        assert result == "result-1"
        engine.execute.assert_called_once()
        sql_arg = engine.execute.call_args.args[0]
        assert sql_arg.startswith("INSERT INTO cat.sch.tbl")

    def test_empty_op_does_nothing(self):
        rec = AsyncInsert(target_full_name="cat.sch.tbl")
        engine = MagicMock()
        assert rec.execute(engine) is None
        engine.execute.assert_not_called()

    def test_cleanup_invoked_on_success(self):
        rec = _make_record(
            parquets=("/Volumes/x/p.parquet",),
            metas=("/Volumes/x/p.json",),
        )
        engine = MagicMock()

        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
        ) as from_:
            handle = MagicMock()
            from_.return_value = handle

            rec.execute(engine, cleanup=True)

        # One call for the parquet, one for the metadata file.
        assert from_.call_count == 2
        assert handle.remove.call_count == 2

    def test_wait_and_raise_error_forwarded(self):
        rec = _make_record()
        engine = MagicMock()
        rec.execute(engine, wait=False, raise_error=False, cleanup=False)
        _, kwargs = engine.execute.call_args
        assert kwargs["wait"] is False
        assert kwargs["raise_error"] is False


class TestCleanup:

    def test_removes_all_paths(self):
        rec = _make_record(
            parquets=("/Volumes/a/p1.parquet", "/Volumes/a/p2.parquet"),
            metas=("/Volumes/a/p1.json", "/Volumes/a/p2.json"),
        )
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
        ) as from_:
            handle = MagicMock()
            from_.return_value = handle
            rec.cleanup()
        assert from_.call_count == 4

    def test_swallows_per_path_errors(self):
        rec = _make_record(
            parquets=("/Volumes/a/p.parquet",),
            metas=("/Volumes/a/p.json",),
        )
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
        ) as from_:
            from_.side_effect = RuntimeError("boom")
            # Must not raise.
            rec.cleanup()

    def test_skips_empty_path_entries(self):
        rec = AsyncInsert(
            target_full_name="cat.sch.tbl",
            parquet_paths=("",),  # empty entry
            metadata_paths=(),
        )
        with patch(
            "yggdrasil.databricks.path.DatabricksPath.from_",
        ) as from_:
            rec.cleanup()
        from_.assert_not_called()


# ---------------------------------------------------------------------------
# stage_async_insert
# ---------------------------------------------------------------------------


class TestStageAsyncInsert:

    def test_writes_parquet_and_metadata_files(self):
        tbl, folder, parquet_path, meta_path = _make_table_with_staging()
        data = MagicMock(name="arrow_table")

        out = stage_async_insert(tbl, data, mode="append", match_by=["id"])

        assert out is parquet_path
        tbl.staging_folder.assert_called_once_with(temporary=False, async_write=True)
        # parquet + metadata file paths joined off the folder
        joined = [call.args[0] for call in folder.joinpath.call_args_list]
        assert any(n.endswith(".parquet") for n in joined)
        assert any(n.endswith(".json") for n in joined)
        parquet_path.write_table.assert_called_once()
        meta_path.write_bytes.assert_called_once()

    def test_metadata_payload_roundtrips(self):
        tbl, _, _, meta_path = _make_table_with_staging()
        stage_async_insert(
            tbl, MagicMock(),
            mode="append",
            match_by=["id"],
            zorder_by=["dt"],
            where="dt = '2026-05-15'",
        )
        payload = meta_path.write_bytes.call_args.args[0]
        rec = AsyncInsert.from_json_bytes(payload)
        assert rec.mode == "append"
        assert rec.match_by == ("id",)
        assert rec.zorder_by == ("dt",)
        assert rec.where == "dt = '2026-05-15'"
        assert rec.operation_id.startswith("async-")

    def test_explicit_operation_id(self):
        tbl, _, _, meta_path = _make_table_with_staging()
        stage_async_insert(tbl, MagicMock(), operation_id="op-fixed")
        payload = meta_path.write_bytes.call_args.args[0]
        rec = AsyncInsert.from_json_bytes(payload)
        assert rec.operation_id == "op-fixed"

    def test_target_field_names_when_schema_resolved(self, monkeypatch):
        tbl, _, _, meta_path = _make_table_with_staging(field_names=["a", "b"])
        # ``CastOptions.with_target`` coerces ``target`` through
        # ``Field.from_``, which can't sanely accept a MagicMock. Bypass
        # the cast-options machinery for this test — we're verifying the
        # metadata-side schema capture, not the cast path.
        import yggdrasil.databricks.table.async_write as aw

        fake_opts = MagicMock()
        fake_opts.with_target.return_value = fake_opts
        monkeypatch.setattr(aw.CastOptions, "check", classmethod(lambda cls, **kw: fake_opts))

        stage_async_insert(tbl, MagicMock())
        payload = meta_path.write_bytes.call_args.args[0]
        rec = AsyncInsert.from_json_bytes(payload)
        assert rec.target_field_names == ("a", "b")

    def test_version_recorded(self):
        tbl, _, _, meta_path = _make_table_with_staging()
        stage_async_insert(tbl, MagicMock())
        payload = meta_path.write_bytes.call_args.args[0]
        decoded = ygg_json.loads(payload)
        assert decoded["version"] == METADATA_VERSION


# ---------------------------------------------------------------------------
# Table.insert async_write flag
# ---------------------------------------------------------------------------


class TestTableInsertAsyncWriteFlag:

    def test_async_write_routes_through_stage_async_insert(self, monkeypatch):
        tbl, _, parquet_path, _ = _make_table_with_staging()

        seen: dict[str, Any] = {}

        def _fake_stage(table, data, **kwargs):
            seen["table"] = table
            seen["data"] = data
            seen.update(kwargs)
            return parquet_path

        import yggdrasil.databricks.table.async_write as aw
        monkeypatch.setattr(aw, "stage_async_insert", _fake_stage)

        tbl.insert_into = MagicMock(name="should_not_run")  # type: ignore[assignment]

        out = tbl.insert(
            "dummy-data",
            mode="append",
            match_by=["id"],
            async_write=True,
        )

        assert out is parquet_path
        assert seen["table"] is tbl
        assert seen["data"] == "dummy-data"
        assert seen["mode"] == "append"
        assert seen["match_by"] == ["id"]
        tbl.insert_into.assert_not_called()

    def test_default_falls_through_to_insert_into(self):
        tbl, _, _, _ = _make_table_with_staging()
        tbl.insert_into = MagicMock(return_value="sql-result")  # type: ignore[assignment]
        out = tbl.insert("dummy", mode="append")
        assert out == "sql-result"
        tbl.insert_into.assert_called_once()
