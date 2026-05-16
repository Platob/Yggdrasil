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
    ASYNC_INSERT_DATA_SUBDIR,
    ASYNC_INSERT_LOGS_SUBDIR,
    AsyncInsert,
    AsyncWrite,
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
    are mocked so the staging call doesn't touch a workspace.

    The folder mock implements the ``data/`` and ``logs/`` subfolder
    layout that ``stage_async_insert`` writes into — Parquet payloads
    under ``data/<op_id>.parquet`` and JSON metadata under
    ``logs/<op_id>.json``.
    """
    service = MagicMock()
    service.client.workspace_client.return_value = MagicMock()
    tbl = Table(
        service=service,
        catalog_name="cat",
        schema_name="sch",
        table_name="tbl",
    )

    folder = MagicMock(spec=VolumePath)
    data_folder = MagicMock(spec=VolumePath)
    logs_folder = MagicMock(spec=VolumePath)
    parquet_path = MagicMock(spec=VolumePath)
    meta_path = MagicMock(spec=VolumePath)
    parquet_path.full_path.return_value = (
        "/Volumes/cat/sch/stg_tbl/.sql/async/insert/data/async-1.parquet"
    )
    meta_path.full_path.return_value = (
        "/Volumes/cat/sch/stg_tbl/.sql/async/insert/logs/async-1.json"
    )

    def _root_join(name: str) -> VolumePath:
        if name == "data":
            return data_folder
        if name == "logs":
            return logs_folder
        raise AssertionError(f"unexpected root subdir {name!r}")

    folder.joinpath.side_effect = _root_join
    data_folder.joinpath.side_effect = lambda name: parquet_path
    logs_folder.joinpath.side_effect = lambda name: meta_path

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

    def test_to_dict_serializes_path_objects_to_url_strings(self):
        """Path objects on ``parquet_paths`` / ``metadata_paths`` are
        dumped via ``_path_for_sql`` so the dict survives orjson."""
        parquet = MagicMock()
        parquet.full_path.return_value = "/Volumes/cat/sch/stg/data/x.parquet"
        meta = MagicMock()
        meta.full_path.return_value = "/Volumes/cat/sch/stg/logs/x.json"
        rec = AsyncInsert(
            target_full_name="cat.sch.tbl",
            parquet_paths=(parquet,),
            metadata_paths=(meta,),
        )
        data = rec.to_dict()
        assert data["parquet_paths"] == ["/Volumes/cat/sch/stg/data/x.parquet"]
        assert data["metadata_paths"] == ["/Volumes/cat/sch/stg/logs/x.json"]

    def test_pickle_drops_executor_and_results(self):
        """Pickling rehydrates a metadata-only record — the bound
        warehouse, in-flight results, and schema cache are gone."""
        import pickle

        rec = _make_record(mode="append")
        rec.executor = MagicMock(name="warehouse")
        rec.results["a"] = MagicMock(name="result")
        rec._cached_schema = MagicMock(name="schema")

        revived = pickle.loads(pickle.dumps(rec))
        assert revived == rec  # metadata fields preserved
        assert revived.executor is None
        assert revived.results == {}
        assert revived._cached_schema is None
        assert revived.external_volume_paths == {}

    def test_pickle_path_objects_become_url_strings(self):
        """Path objects survive the pickle as URL strings (callers
        coerce them back via ``DatabricksPath.from_`` lazily)."""
        import pickle

        parquet = MagicMock()
        parquet.full_path.return_value = "/Volumes/cat/sch/stg/data/x.parquet"
        meta = MagicMock()
        meta.full_path.return_value = "/Volumes/cat/sch/stg/logs/x.json"
        rec = AsyncInsert(
            target_full_name="cat.sch.tbl",
            parquet_paths=(parquet,),
            metadata_paths=(meta,),
        )
        revived = pickle.loads(pickle.dumps(rec))
        assert revived.parquet_paths == ("/Volumes/cat/sch/stg/data/x.parquet",)
        assert revived.metadata_paths == ("/Volumes/cat/sch/stg/logs/x.json",)


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

    def test_async_insert_root_descends_directly_into_logs(self):
        """``.sql/async/insert/`` as source descends directly into
        ``logs/`` — the parent listing (which would just rediscover
        ``data/`` + ``logs/``) is skipped, saving one round trip."""
        rec = _make_record(target="t1")
        json_entry = MagicMock()
        json_entry.name = "async-1.json"
        json_entry.read_bytes.return_value = rec.to_json_bytes()

        logs_folder = MagicMock(spec=VolumePath, name="logs_folder")
        logs_folder.ls.return_value = [json_entry]

        root = MagicMock(spec=VolumePath)
        root.name = "insert"
        root.joinpath.return_value = logs_folder

        merged = AsyncInsert.merge(root)
        assert len(merged) == 1
        # Parent root was never listed — only ``logs/``.
        root.ls.assert_not_called()
        root.joinpath.assert_called_once_with(ASYNC_INSERT_LOGS_SUBDIR)
        logs_folder.ls.assert_called_once_with(recursive=False)


# ---------------------------------------------------------------------------
# Execute + cleanup
# ---------------------------------------------------------------------------


class TestToStatements:
    """:meth:`AsyncInsert.to_statements` — prepared-statement rendering."""

    def _patched_prepare(self):
        """Build patch contexts for the two SDK boundaries to_statements hits.

        Returns a pair of patches: ``DatabricksPath.from_`` (to avoid
        hitting the live workspace when resolving paths) and
        ``WarehousePreparedStatement.prepare`` (to capture the args
        without actually staging anything).
        """
        path_patch = patch("yggdrasil.databricks.path.DatabricksPath.from_")
        prep_patch = patch(
            "yggdrasil.databricks.warehouse.statement.WarehousePreparedStatement.prepare"
        )
        return path_patch, prep_patch

    def test_empty_record_returns_empty(self):
        rec = AsyncInsert(target_full_name="cat.sch.t")  # no parquets
        assert rec.to_statements() == []

    def test_builds_one_statement_with_alias_substitutions(self):
        rec = _make_record(
            parquets=("/Volumes/a/p1.parquet", "/Volumes/a/p2.parquet"),
            metas=("/Volumes/a/p1.json", "/Volumes/a/p2.json"),
            ops=("op-1", "op-2"),
            target_catalog_name="main",
            target_schema_name="sales",
        )

        # Each DatabricksPath.from_ call returns a fresh MagicMock so we
        # can inspect which paths got attached and whether they were
        # marked temporary.
        path_patch, prep_patch = self._patched_prepare()
        with path_patch, prep_patch as prepare:
            prepare.return_value = MagicMock(name="prepared")
            statements = rec.to_statements()

        assert len(statements) == 1
        prepare.assert_called_once()

        # SQL uses {__p0__}/{__p1__} placeholders, not literal paths.
        sql_arg = prepare.call_args.args[0]
        assert "{__p0__}" in sql_arg
        assert "{__p1__}" in sql_arg
        # Literal paths should NOT leak into the SQL text.
        assert "/Volumes/a/p1.parquet" not in sql_arg
        # external_volume_paths carries one entry per parquet (under __pN__)
        # plus one per metadata file (under __mN__) — four total.
        ext = prepare.call_args.kwargs["external_volume_paths"]
        assert set(ext) == {"__p0__", "__p1__", "__m0__", "__m1__"}
        # Each resolved path was marked temporary so the statement
        # lifecycle unlinks it on success.
        for handle in ext.values():
            assert handle.temporary is True

        # Catalog/schema flowed through to prepare() so the statement
        # binds against the right namespace.
        assert prepare.call_args.kwargs["catalog_name"] == "main"
        assert prepare.call_args.kwargs["schema_name"] == "sales"

    def test_cleanup_false_attaches_paths_without_marking_temporary(self):
        rec = _make_record(
            parquets=("/Volumes/a/p.parquet",),
            metas=("/Volumes/a/p.json",),
            ops=("op-1",),
        )

        path_patch, prep_patch = self._patched_prepare()
        with path_patch as from_, prep_patch as prepare:
            handle = MagicMock()
            handle.temporary = False
            from_.return_value = handle
            prepare.return_value = MagicMock()
            rec.to_statements(cleanup=False)

        # Paths still attached for SQL substitution, but their
        # ``temporary`` flag stays False so clear_temporary_resources
        # skips them.
        ext = prepare.call_args.kwargs["external_volume_paths"]
        for path in ext.values():
            assert path.temporary is False

    def test_overwrite_emits_insert_overwrite(self):
        rec = _make_record(mode="overwrite")
        path_patch, prep_patch = self._patched_prepare()
        with path_patch, prep_patch as prepare:
            prepare.return_value = MagicMock()
            rec.to_statements()

        sql_arg = prepare.call_args.args[0]
        assert sql_arg.startswith("INSERT OVERWRITE ")

    def test_retry_forwarded_to_prepare(self):
        rec = _make_record()
        path_patch, prep_patch = self._patched_prepare()
        with path_patch, prep_patch as prepare:
            prepare.return_value = MagicMock()
            rec.to_statements(retry={"timeout": 30.0})

        assert prepare.call_args.kwargs["retry"] == {"timeout": 30.0}


class TestExecute:
    """:meth:`AsyncInsert.execute` binds an executor and submits.

    AsyncInsert is itself a :class:`WarehouseStatementBatch`, so
    ``execute`` plugs *engine*'s warehouse in as :attr:`executor`,
    renders the prepared statements via :meth:`to_statements`, and
    extends self with them. No intermediate :class:`AsyncWrite` is
    built — the record IS the batch.
    """

    def _patched_to_statements(self, sentinel: Any):
        """Patch :meth:`to_statements` to return a fixed list without
        hitting :class:`WarehousePreparedStatement.prepare`."""
        return patch(
            "yggdrasil.databricks.table.async_write.AsyncInsert.to_statements",
            return_value=sentinel,
        )

    def test_binds_executor_and_submits_statements(self):
        rec = _make_record(mode="append")
        engine = MagicMock()
        warehouse = engine.warehouse.return_value
        stmt = MagicMock(name="prepared")
        with self._patched_to_statements([stmt]):
            rec.extend = MagicMock(name="extend")  # type: ignore[assignment]
            rec.wait = MagicMock(name="wait")  # type: ignore[assignment]
            result = rec.execute(engine, cleanup=False)

        assert result is rec
        assert rec.executor is warehouse
        rec.extend.assert_called_once_with([stmt])
        rec.wait.assert_called_once_with(wait=True, raise_error=True)

    def test_empty_op_does_nothing(self):
        rec = AsyncInsert(target_full_name="cat.sch.tbl")
        engine = MagicMock()
        assert rec.execute(engine) is None
        engine.warehouse.assert_not_called()
        assert rec.executor is None

    def test_no_statements_rendered_returns_none(self):
        rec = _make_record()
        engine = MagicMock()
        with self._patched_to_statements([]):
            assert rec.execute(engine) is None
        # Executor wasn't bound when no statements were ready.
        assert rec.executor is None
        engine.warehouse.assert_not_called()

    def test_wait_and_raise_error_forwarded(self):
        rec = _make_record()
        engine = MagicMock()
        with self._patched_to_statements([MagicMock(name="prepared")]):
            rec.extend = MagicMock()  # type: ignore[assignment]
            rec.wait = MagicMock()  # type: ignore[assignment]
            rec.execute(engine, wait=False, raise_error=False, cleanup=False)

        rec.wait.assert_called_once_with(wait=False, raise_error=False)

    def test_cleanup_flag_forwarded_to_to_statements(self):
        """``cleanup=True/False`` flows through to :meth:`to_statements`,
        which decides whether to mark attached paths temporary."""
        rec = _make_record()
        engine = MagicMock()

        for cleanup in (True, False):
            with patch(
                "yggdrasil.databricks.table.async_write.AsyncInsert.to_statements",
                return_value=[MagicMock(name="prepared")],
            ) as to_stmts:
                rec.extend = MagicMock()  # type: ignore[assignment]
                rec.wait = MagicMock()  # type: ignore[assignment]
                rec.execute(engine, cleanup=cleanup)
                assert to_stmts.call_args.kwargs["cleanup"] is cleanup


class TestConcat:
    """:meth:`AsyncInsert.concat` and the matching ``__call__`` shortcut."""

    def test_concat_records_into_sql_suite(self):
        a = _make_record(target="cat.sch.t1", parquets=("/t1.parquet",),
                         metas=("/t1.json",), ops=("a",), mode="append")
        b = _make_record(target="cat.sch.t2", parquets=("/t2.parquet",),
                         metas=("/t2.json",), ops=("b",), mode="append")

        suite = AsyncInsert.concat([a, b])
        # One statement per target — order follows the merge result.
        assert len(suite) == 2
        assert all(s.startswith("INSERT INTO ") for s in suite)
        # Both targets covered (merge groups by target).
        assert any("t1" in s for s in suite)
        assert any("t2" in s for s in suite)

    def test_concat_merges_appends_for_same_target(self):
        """Two appends on the same target collapse to one UNION-ALL INSERT."""
        a = _make_record(target="cat.sch.t", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",), mode="append",
                         created_at="2026-05-15T10:00:00+00:00")
        b = _make_record(target="cat.sch.t", parquets=("/b.parquet",),
                         metas=("/b.json",), ops=("b",), mode="append",
                         created_at="2026-05-15T11:00:00+00:00")

        suite = AsyncInsert.concat([a, b])
        assert len(suite) == 1
        # Both parquets ride a single statement via UNION ALL.
        assert "UNION ALL" in suite[0]
        assert "/a.parquet" in suite[0]
        assert "/b.parquet" in suite[0]

    def test_concat_empty_returns_empty_list(self):
        assert AsyncInsert.concat([]) == []

    _ASYNC_WRITE = "yggdrasil.databricks.table.async_write.AsyncWrite.from_records"

    def test_concat_with_engine_builds_async_write(self):
        a = _make_record(target="cat.sch.t1", parquets=("/t1.parquet",),
                         metas=("/t1.json",), ops=("a",), mode="append")
        b = _make_record(target="cat.sch.t2", parquets=("/t2.parquet",),
                         metas=("/t2.json",), ops=("b",), mode="append")

        engine = MagicMock()
        sentinel_batch = MagicMock(name="batch")
        with patch(self._ASYNC_WRITE, return_value=sentinel_batch) as from_records:
            result = AsyncInsert.concat([a, b], engine=engine, cleanup=True)

        assert result is sentinel_batch
        from_records.assert_called_once()
        records_arg = list(from_records.call_args.args[0])
        # One merged record per target.
        assert len(records_arg) == 2
        targets = {r.target_full_name for r in records_arg}
        assert targets == {"cat.sch.t1", "cat.sch.t2"}
        assert from_records.call_args.kwargs["executor"] is engine.warehouse.return_value

    def test_concat_with_engine_forwards_cleanup_flag(self):
        a = _make_record(target="cat.sch.t", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",), mode="append")
        engine = MagicMock()

        with patch(self._ASYNC_WRITE, return_value=MagicMock()) as from_records:
            AsyncInsert.concat([a], engine=engine, cleanup=False)
            assert from_records.call_args.kwargs["cleanup"] is False

            AsyncInsert.concat([a], engine=engine, cleanup=True)
            assert from_records.call_args.kwargs["cleanup"] is True

    def test_concat_with_engine_empty_returns_none(self):
        engine = MagicMock()
        with patch(self._ASYNC_WRITE) as from_records:
            assert AsyncInsert.concat([], engine=engine) is None
        from_records.assert_not_called()
        engine.warehouse.assert_not_called()

    def test_concat_forwards_wait_and_raise_error(self):
        a = _make_record(target="cat.sch.t", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",), mode="append")
        engine = MagicMock()

        with patch(self._ASYNC_WRITE, return_value=MagicMock()) as from_records:
            AsyncInsert.concat(
                [a], engine=engine, wait=False, raise_error=False,
            )

        _, kwargs = from_records.call_args
        assert kwargs["wait"] is False
        assert kwargs["raise_error"] is False

    def test_concat_accepts_folder_volumepath(self):
        rec = _make_record(target="cat.sch.t")
        json_entry = MagicMock()
        json_entry.name = "x.json"
        json_entry.read_bytes.return_value = rec.to_json_bytes()
        folder = MagicMock(spec=VolumePath)
        folder.ls.return_value = [json_entry]

        suite = AsyncInsert.concat(folder)
        assert len(suite) == 1
        assert suite[0].startswith("INSERT INTO cat.sch.t")

    def test_call_runs_concat_against_engine(self):
        a = _make_record(target="cat.sch.t", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",), mode="append")
        b = _make_record(target="cat.sch.t", parquets=("/b.parquet",),
                         metas=("/b.json",), ops=("b",), mode="append",
                         created_at="2026-05-15T11:00:00+00:00")

        engine = MagicMock()
        sentinel_batch = MagicMock(name="batch")
        with patch(self._ASYNC_WRITE, return_value=sentinel_batch) as from_records:
            result = a(engine, b)  # __call__

        assert result is sentinel_batch
        # Same target → one merged record handed to AsyncWrite.
        records_arg = list(from_records.call_args.args[0])
        assert len(records_arg) == 1

    def test_call_without_engine_returns_sql_suite(self):
        a = _make_record(target="cat.sch.t1", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",), mode="append")
        result = a()
        assert isinstance(result, list)
        assert result[0].startswith("INSERT INTO cat.sch.t1")


class TestAsyncWrite:
    """:class:`AsyncWrite` — unified WarehouseStatementBatch factory."""

    _BATCH = (
        "yggdrasil.databricks.warehouse.statement.WarehouseStatementBatch"
    )

    def test_from_records_empty_returns_none(self):
        executor = MagicMock(name="warehouse")
        with patch(self._BATCH) as batch_cls:
            assert AsyncWrite.from_records([], executor=executor) is None
        batch_cls.assert_not_called()

    def test_from_records_submits_one_batch_with_every_statement(self):
        a = _make_record(target="cat.sch.t1", parquets=("/a.parquet",),
                         metas=("/a.json",), ops=("a",))
        b = _make_record(target="cat.sch.t2", parquets=("/b.parquet",),
                         metas=("/b.json",), ops=("b",))
        executor = MagicMock(name="warehouse")
        batch = MagicMock(name="batch")

        with patch.object(
            AsyncInsert, "to_statements",
            side_effect=lambda **kw: [MagicMock(name="stmt")],
        ) as to_stmts, patch(self._BATCH, return_value=batch) as batch_cls:
            out = AsyncWrite.from_records([a, b], executor=executor)

        assert out is batch
        # One batch construction covers every record.
        batch_cls.assert_called_once()
        kwargs = batch_cls.call_args.kwargs
        assert kwargs["executor"] is executor
        # Two records → two statements in the unified batch.
        assert len(kwargs["statements"]) == 2
        assert to_stmts.call_count == 2
        # wait() fires from from_records by default.
        batch.wait.assert_called_once()

    def test_from_source_with_engine_resolves_warehouse(self):
        rec = _make_record(target="cat.sch.t")
        engine = MagicMock()
        batch = MagicMock(name="batch")

        with patch.object(
            AsyncInsert, "to_statements",
            return_value=[MagicMock(name="stmt")],
        ), patch(self._BATCH, return_value=batch):
            out = AsyncWrite.from_source([rec], engine=engine)

        assert out is batch
        engine.warehouse.assert_called_once()

    def test_from_source_requires_engine_or_executor(self):
        with pytest.raises(ValueError):
            AsyncWrite.from_source([_make_record()])

    def test_from_source_empty_returns_none(self):
        executor = MagicMock(name="warehouse")
        with patch(self._BATCH) as batch_cls:
            assert AsyncWrite.from_source([], executor=executor) is None
        batch_cls.assert_not_called()


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
        # Parquet payload joined off the ``data/`` subfolder; JSON
        # metadata joined off the ``logs/`` subfolder. The
        # _make_table_with_staging helper enforces the routing.
        root_joins = [call.args[0] for call in folder.joinpath.call_args_list]
        assert ASYNC_INSERT_DATA_SUBDIR in root_joins
        assert ASYNC_INSERT_LOGS_SUBDIR in root_joins
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


class TestAsyncInsertJob:
    """:class:`AsyncInsertJob` CRUD flow against a mock Jobs service.

    The applier job is keyed off ``(catalog, schema, table)`` from the
    bound :class:`Table` — one Databricks Job per target table, fired
    by file arrival in the table's async staging ``data/`` folder.
    """

    @staticmethod
    def _mock_jobs():
        svc = MagicMock(name="Jobs")
        svc.create_or_update.return_value = MagicMock(name="Job")
        svc.create.return_value = MagicMock(name="Job")
        return svc

    @staticmethod
    def _table_with_trigger_path(
        *,
        trigger_path: str = (
            "/Volumes/cat/sch/stg_tbl/.sql/async/insert/data"
        ),
    ):
        """Build a Table whose staging folder yields a known trigger URL."""
        tbl, _, _, _ = _make_table_with_staging()
        data_folder = MagicMock(spec=VolumePath)
        data_folder.full_path.return_value = trigger_path
        async_root = MagicMock(spec=VolumePath)
        async_root.joinpath.return_value = data_folder
        tbl.staging_folder = MagicMock(return_value=async_root)  # type: ignore[assignment]
        return tbl, async_root, data_folder

    def test_job_name_keyed_off_full_table_triple(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        tbl, _, _ = self._table_with_trigger_path()
        assert AsyncInsertJob.job_name(tbl) == "ygg-async-insert-cat-sch-tbl"

    def test_trigger_url_points_at_async_data_folder(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        tbl, _, _ = self._table_with_trigger_path()
        url = AsyncInsertJob.trigger_url(tbl)
        # ``dbfs:`` scheme, trailing slash so the prefix match catches
        # every new file under it.
        assert url == "dbfs:/Volumes/cat/sch/stg_tbl/.sql/async/insert/data/"

    def test_create_or_update_attaches_file_arrival_trigger_by_default(self):
        from databricks.sdk.service.jobs import (
            FileArrivalTriggerConfiguration,
            TriggerSettings,
        )
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(
            tbl, notebook_path="/Workspace/Users/me/apply", jobs=jobs_svc,
        )

        jobs_svc.create_or_update.assert_called_once()
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["name"] == "ygg-async-insert-cat-sch-tbl"

        trigger = kwargs["trigger"]
        assert isinstance(trigger, TriggerSettings)
        assert isinstance(trigger.file_arrival, FileArrivalTriggerConfiguration)
        assert (
            trigger.file_arrival.url
            == "dbfs:/Volumes/cat/sch/stg_tbl/.sql/async/insert/data/"
        )

        tasks = kwargs["tasks"]
        assert len(tasks) == 1
        base = tasks[0].notebook_task.base_parameters
        assert base["catalog_name"] == "cat"
        assert base["schema_name"] == "sch"
        assert base["table_name"] == "tbl"

        param_names = {p.name for p in kwargs["parameters"]}
        assert param_names == {"catalog_name", "schema_name", "table_name"}

    def test_create_or_update_skips_trigger_when_disabled(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(
            tbl,
            notebook_path="/p",
            file_arrival_trigger=False,
            jobs=jobs_svc,
        )
        _, kwargs = jobs_svc.create_or_update.call_args
        assert "trigger" not in kwargs

    def test_trigger_debounce_kwargs_flow_through(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(
            tbl,
            notebook_path="/p",
            min_time_between_triggers_seconds=120,
            wait_after_last_change_seconds=30,
            jobs=jobs_svc,
        )
        _, kwargs = jobs_svc.create_or_update.call_args
        fa = kwargs["trigger"].file_arrival
        assert fa.min_time_between_triggers_seconds == 120
        assert fa.wait_after_last_change_seconds == 30

    def test_accepts_explicit_task(self):
        from databricks.sdk.service.jobs import NotebookTask, Task
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        custom = Task(
            task_key="custom",
            notebook_task=NotebookTask(notebook_path="/Workspace/custom"),
        )
        AsyncInsertJob.create_or_update(tbl, task=custom, jobs=jobs_svc)
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["tasks"] == [custom]

    def test_accepts_list_of_tasks(self):
        from databricks.sdk.service.jobs import NotebookTask, Task
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        t1 = Task(task_key="a", notebook_task=NotebookTask(notebook_path="/a"))
        t2 = Task(task_key="b", notebook_task=NotebookTask(notebook_path="/b"))
        AsyncInsertJob.create_or_update(tbl, task=[t1, t2], jobs=jobs_svc)
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["tasks"] == [t1, t2]

    def test_with_cron_string_builds_cron_schedule(self):
        from databricks.sdk.service.jobs import CronSchedule
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(
            tbl,
            notebook_path="/p",
            schedule="0 0 */1 * * ?",
            schedule_timezone="UTC",
            jobs=jobs_svc,
        )
        _, kwargs = jobs_svc.create_or_update.call_args
        assert isinstance(kwargs["schedule"], CronSchedule)
        assert kwargs["schedule"].quartz_cron_expression == "0 0 */1 * * ?"
        assert kwargs["schedule"].timezone_id == "UTC"

    def test_with_cron_schedule_passes_through(self):
        from databricks.sdk.service.jobs import CronSchedule, PauseStatus
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        existing = CronSchedule(
            quartz_cron_expression="0 0 0 * * ?",
            timezone_id="Europe/Paris",
            pause_status=PauseStatus.UNPAUSED,
        )
        AsyncInsertJob.create_or_update(
            tbl, notebook_path="/p", schedule=existing, jobs=jobs_svc,
        )
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["schedule"] is existing

    def test_no_schedule_passes_none(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(tbl, notebook_path="/p", jobs=jobs_svc)
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["schedule"] is None

    def test_merges_caller_parameters(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(
            tbl,
            notebook_path="/p",
            parameters={"catalog_name": "override", "extra": "yes"},
            jobs=jobs_svc,
        )
        _, kwargs = jobs_svc.create_or_update.call_args
        as_dict = {p.name: p.default for p in kwargs["parameters"]}
        # Caller wins on collision
        assert as_dict["catalog_name"] == "override"
        # New keys appended
        assert as_dict["extra"] == "yes"

    def test_invalid_schedule_type_raises(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        tbl, _, _ = self._table_with_trigger_path()
        with pytest.raises(TypeError):
            AsyncInsertJob.create_or_update(
                tbl, notebook_path="/p", schedule=42, jobs=self._mock_jobs(),
            )

    def test_no_task_logs_warning_and_emits_empty_task_list(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.create_or_update(tbl, jobs=jobs_svc)
        _, kwargs = jobs_svc.create_or_update.call_args
        assert kwargs["tasks"] == []

    def test_requires_full_table_identity(self):
        """Missing catalog / schema / table → can't build job identity."""
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        bare = MagicMock()
        bare.catalog_name = "cat"
        bare.schema_name = "sch"
        bare.table_name = None  # not fully qualified
        with pytest.raises(ValueError):
            AsyncInsertJob.job_name(bare)

    def test_find_returns_wrapper_when_jobs_finds_match(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        underlying = MagicMock(name="Job", job_id=42)
        jobs_svc.find.return_value = underlying

        tbl, _, _ = self._table_with_trigger_path()
        found = AsyncInsertJob.find(tbl, jobs=jobs_svc)
        assert found is not None
        assert found.table is tbl
        assert found.job is underlying
        assert found.job_id == 42
        jobs_svc.find.assert_called_once_with(name="ygg-async-insert-cat-sch-tbl")

    def test_find_returns_none_when_absent(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        jobs_svc.find.return_value = None
        tbl, _, _ = self._table_with_trigger_path()
        assert AsyncInsertJob.find(tbl, jobs=jobs_svc) is None

    def test_get_or_create_returns_existing_when_present(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        jobs_svc.find.return_value = MagicMock(name="Job")
        tbl, _, _ = self._table_with_trigger_path()
        wrapper = AsyncInsertJob.get_or_create(
            tbl, notebook_path="/p", jobs=jobs_svc,
        )
        assert wrapper.job is jobs_svc.find.return_value
        # Existing job → no create call.
        jobs_svc.create.assert_not_called()
        jobs_svc.create_or_update.assert_not_called()

    def test_get_or_create_creates_when_missing(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        jobs_svc.find.return_value = None
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.get_or_create(tbl, notebook_path="/p", jobs=jobs_svc)
        jobs_svc.create.assert_called_once()
        _, kwargs = jobs_svc.create.call_args
        assert kwargs["name"] == "ygg-async-insert-cat-sch-tbl"

    def test_delete_routes_through_jobs_by_name(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        jobs_svc = self._mock_jobs()
        tbl, _, _ = self._table_with_trigger_path()
        AsyncInsertJob.delete(tbl, jobs=jobs_svc)
        jobs_svc.delete.assert_called_once_with(
            name="ygg-async-insert-cat-sch-tbl",
        )

    def test_instance_run_forwards_to_underlying_job(self):
        from yggdrasil.databricks.table.async_job import AsyncInsertJob

        tbl, _, _ = self._table_with_trigger_path()
        underlying = MagicMock(name="Job", job_id=7)
        wrapper = AsyncInsertJob(tbl, underlying)
        wrapper.run(idempotency_token="tok-1")
        underlying.run.assert_called_once_with(idempotency_token="tok-1")


class TestSchemaDiscovery:
    """:meth:`AsyncInsert.merge_schema` / :meth:`apply_schema` walk
    every stg_* volume in a schema and drain its metadata."""

    def _client_with_volumes(self, volumes):
        client = MagicMock()
        client.volumes.list.return_value = iter(volumes)
        return client

    def _make_volume(self, name: str, *, json_records=()):
        vol = MagicMock()
        vol.volume_name = name
        folder = MagicMock()
        # Each json_record becomes a path object the folder's ls yields.
        folder.ls.return_value = [
            self._json_entry(r) for r in json_records
        ]
        vol.path.return_value = folder
        return vol, folder

    def _json_entry(self, record):
        entry = MagicMock()
        entry.name = "async-1.json"
        entry.read_bytes.return_value = record.to_json_bytes()
        return entry

    def test_iter_staging_folders_filters_to_stg_prefix(self):
        rec_a = _make_record(target="cat.sch.a", parquets=("/a.parquet",),
                             metas=("/a.json",), ops=("a",))
        vol_a, folder_a = self._make_volume("stg_a", json_records=[rec_a])
        vol_b, _ = self._make_volume("other_volume", json_records=[])

        client = self._client_with_volumes([vol_a, vol_b])
        folders = list(AsyncInsert.iter_schema_staging_folders(
            "cat", "sch", client=client,
        ))
        assert folders == [folder_a]
        # Volume filtering happens before path resolution.
        vol_b.path.assert_not_called()

    def test_merge_schema_drains_multiple_volumes(self):
        rec_a = _make_record(target="cat.sch.a", parquets=("/a.parquet",),
                             metas=("/a.json",), ops=("a",))
        rec_b = _make_record(target="cat.sch.b", parquets=("/b.parquet",),
                             metas=("/b.json",), ops=("b",))
        vol_a, _ = self._make_volume("stg_a", json_records=[rec_a])
        vol_b, _ = self._make_volume("stg_b", json_records=[rec_b])

        client = self._client_with_volumes([vol_a, vol_b])
        merged = AsyncInsert.merge_schema("cat", "sch", client=client)
        names = sorted(r.target_full_name for r in merged)
        assert names == ["cat.sch.a", "cat.sch.b"]

    def test_merge_schema_returns_empty_for_no_records(self):
        client = self._client_with_volumes([])
        assert AsyncInsert.merge_schema("cat", "sch", client=client) == []

    def test_merge_schema_swallows_per_folder_errors(self):
        rec_a = _make_record(target="cat.sch.a", parquets=("/a.parquet",),
                             metas=("/a.json",), ops=("a",))
        vol_a, folder_a = self._make_volume("stg_a", json_records=[rec_a])
        vol_bad, folder_bad = self._make_volume("stg_bad", json_records=[])
        folder_bad.ls.side_effect = RuntimeError("boom")

        client = self._client_with_volumes([vol_a, vol_bad])
        # Bad folder doesn't blow up the whole walk.
        merged = AsyncInsert.merge_schema("cat", "sch", client=client)
        assert len(merged) == 1
        assert merged[0].target_full_name == "cat.sch.a"

    _ASYNC_WRITE = "yggdrasil.databricks.table.async_write.AsyncWrite.from_records"

    def test_apply_schema_submits_one_unified_batch(self):
        rec_a = _make_record(target="cat.sch.a", parquets=("/a.parquet",),
                             metas=("/a.json",), ops=("a",))
        rec_b = _make_record(target="cat.sch.b", parquets=("/b.parquet",),
                             metas=("/b.json",), ops=("b",))
        vol_a, _ = self._make_volume("stg_a", json_records=[rec_a])
        vol_b, _ = self._make_volume("stg_b", json_records=[rec_b])

        client = self._client_with_volumes([vol_a, vol_b])

        engine = MagicMock()
        sentinel_batch = MagicMock(name="batch")
        with patch(self._ASYNC_WRITE, return_value=sentinel_batch) as from_records:
            result = AsyncInsert.apply_schema(
                engine, "cat", "sch", client=client,
            )

        assert result is sentinel_batch
        # One unified AsyncWrite covers every target — not one batch per record.
        from_records.assert_called_once()
        records_arg = list(from_records.call_args.args[0])
        assert {r.target_full_name for r in records_arg} == {
            "cat.sch.a", "cat.sch.b",
        }

    def test_apply_schema_no_records_returns_none(self):
        client = self._client_with_volumes([])
        engine = MagicMock()
        with patch(self._ASYNC_WRITE) as from_records:
            assert AsyncInsert.apply_schema(engine, "cat", "sch", client=client) is None
        from_records.assert_not_called()
        engine.warehouse.assert_not_called()


class TestTableAsyncInsert:

    def test_async_insert_routes_through_stage_async_insert(self, monkeypatch):
        tbl, _, _, _ = _make_table_with_staging()

        sentinel_record = AsyncInsert(target_full_name="cat.sch.tbl")
        seen: dict[str, Any] = {}

        def _fake_stage(table, data, **kwargs):
            seen["table"] = table
            seen["data"] = data
            seen.update(kwargs)
            return sentinel_record

        import yggdrasil.databricks.table.async_write as aw
        monkeypatch.setattr(aw, "stage_async_insert", _fake_stage)

        tbl.insert_into = MagicMock(name="should_not_run")  # type: ignore[assignment]

        out = tbl.async_insert(
            "dummy-data",
            mode="append",
            match_by=["id"],
        )

        assert out is sentinel_record
        assert seen["table"] is tbl
        assert seen["data"] == "dummy-data"
        assert seen["mode"] == "append"
        assert seen["match_by"] == ["id"]
        assert seen["lazy"] is True
        tbl.insert_into.assert_not_called()

    def test_insert_falls_through_to_insert_into(self):
        tbl, _, _, _ = _make_table_with_staging()
        tbl.insert_into = MagicMock(return_value="sql-result")  # type: ignore[assignment]
        out = tbl.insert("dummy", mode="append")
        assert out == "sql-result"
        tbl.insert_into.assert_called_once()
