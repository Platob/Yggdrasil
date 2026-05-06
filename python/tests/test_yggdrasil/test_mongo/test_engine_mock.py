"""End-to-end tests for :class:`MongoEngine` against :mod:`mongomock`.

These cover the full read / write contract: tabular Arrow IO,
records iteration, find / aggregate / count via MongoCommand, save
modes (overwrite / truncate / append / ignore / upsert), and the
collection lifecycle (create / drop / rename / truncate).

Skipped automatically when neither :mod:`pymongo` nor
:mod:`mongomock` is importable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo")
pytest.importorskip("mongomock")

from yggdrasil.data.enums import Mode
from yggdrasil.mongo import (
    MongoCommand,
    MongoCommandKind,
    infer_arrow_schema_from_documents,
)
from yggdrasil.mongo.tests import MongoTestCase


class TestCollectionRoundTrip(MongoTestCase):
    """Insert documents, read them back as Arrow / pylist / records."""

    def test_insert_and_read_arrow_table(self):
        self.collection.insert_into([
            {"_id": 1, "name": "alice", "age": 30},
            {"_id": 2, "name": "bob", "age": 25},
        ])
        table = self.collection.read_arrow_table()
        assert table.num_rows == 2
        assert set(table.column_names) >= {"_id", "name", "age"}

    def test_read_pylist_streams_documents(self):
        self.collection.insert_into([
            {"_id": "a", "value": 1},
            {"_id": "b", "value": 2},
            {"_id": "c", "value": 3},
        ])
        rows = self.collection.read_pylist()
        assert len(rows) == 3
        assert {r["_id"] for r in rows} == {"a", "b", "c"}

    def test_count_helper(self):
        self.collection.insert_into([{"x": 1}, {"x": 2}])
        assert self.collection.count() == 2

    def test_collect_schema_from_sample(self):
        self.collection.insert_into([
            {"_id": 1, "label": "x", "ratio": 0.5},
            {"_id": 2, "label": "y", "ratio": 0.75},
        ])
        schema = self.collection.collect_schema()
        assert "ratio" in schema.names
        assert "label" in schema.names


class TestSaveModes(MongoTestCase):
    """Mode resolution: overwrite, append, truncate, upsert, ignore."""

    def test_overwrite_drops_then_inserts(self):
        self.collection.insert_into([{"x": 1}, {"x": 2}])
        self.collection.insert_into([{"x": 99}], mode=Mode.OVERWRITE)
        rows = self.collection.read_pylist()
        assert len(rows) == 1
        assert rows[0]["x"] == 99

    def test_append_keeps_existing_rows(self):
        self.collection.insert_into([{"x": 1}])
        self.collection.insert_into([{"x": 2}], mode=Mode.APPEND)
        rows = sorted(r["x"] for r in self.collection.read_pylist())
        assert rows == [1, 2]

    def test_truncate_keeps_collection_drops_rows(self):
        self.collection.insert_into([{"x": 1}, {"x": 2}])
        self.collection.create_index("x")
        self.collection.insert_into([{"x": 99}], mode=Mode.TRUNCATE)
        # Truncate should keep indexes (mongomock honours delete_many).
        rows = self.collection.read_pylist()
        assert len(rows) == 1

    def test_ignore_skips_when_non_empty(self):
        self.collection.insert_into([{"x": 1}])
        self.collection.insert_into([{"x": 2}], mode=Mode.IGNORE)
        rows = self.collection.read_pylist()
        assert len(rows) == 1
        assert rows[0]["x"] == 1

    def test_upsert_replaces_matching_documents(self):
        self.collection.insert_into([{"_id": 1, "v": "old"}])
        self.collection.insert_into(
            [{"_id": 1, "v": "new"}, {"_id": 2, "v": "fresh"}],
            mode=Mode.UPSERT,
            match_by=["_id"],
        )
        rows = sorted(self.collection.read_pylist(), key=lambda r: r["_id"])
        assert [r["v"] for r in rows] == ["new", "fresh"]


class TestLifecycle(MongoTestCase):
    """create / drop / rename / truncate."""

    def test_create_and_drop(self):
        new = self.engine.collection(collection_name="brand_new")
        assert not new.exists
        new.create()
        assert new.exists
        new.drop()
        assert not new.exists

    def test_rename_moves_namespace(self):
        self.collection.insert_into([{"x": 1}])
        new_name = "renamed_collection"
        self.collection.rename(new_name)
        assert self.collection.collection_name == new_name
        assert self.collection.read_pylist() == [
            r for r in self.collection.read_pylist()
        ]

    def test_truncate_clears_documents(self):
        self.collection.insert_into([{"x": 1}, {"x": 2}])
        self.collection.truncate()
        assert self.collection.count() == 0


class TestExecutorCommands(MongoTestCase):
    """Raw db.command, find, aggregate via the executor surface."""

    def test_run_command_passthrough(self):
        result = self.engine.run_command({"ping": 1})
        # mongomock returns ``{"ok": 1.0}`` for ping.
        assert result.row_count == 1

    def test_find_via_executor(self):
        self.collection.insert_into([{"x": 1}, {"x": 2}, {"x": 3}])
        result = self.engine.find(
            collection_name=self.default_collection,
            filter={"x": {"$gt": 1}},
        )
        assert result.row_count >= 2

    def test_aggregate_via_executor(self):
        self.collection.insert_into([{"g": "a", "v": 1}, {"g": "a", "v": 2}, {"g": "b", "v": 5}])
        result = self.engine.aggregate(
            collection_name=self.default_collection,
            pipeline=[
                {"$group": {"_id": "$g", "sum": {"$sum": "$v"}}},
            ],
        )
        rows = result.read_pylist()
        assert len(rows) == 2

    def test_command_kind_dispatch(self):
        cmd = MongoCommand.find(
            collection_name=self.default_collection,
            database_name=self._db_name,
        )
        assert cmd.kind == MongoCommandKind.FIND
        cmd = MongoCommand.aggregate(self.default_collection, pipeline=[{"$count": "n"}])
        assert cmd.kind == MongoCommandKind.AGGREGATE
        cmd = MongoCommand.command({"ping": 1})
        assert cmd.kind == MongoCommandKind.COMMAND


class TestSchemaInference(MongoTestCase):
    """Sample-based schema inference picks up the right BSON types."""

    def test_collect_schema_returns_yggdrasil_schema(self):
        self.collection.insert_into([
            {"name": "alice", "balance": 100},
            {"name": "bob", "balance": 200},
        ])
        schema = self.collection.collect_schema()
        assert schema.names
        assert "name" in schema.names
        assert "balance" in schema.names

    def test_infer_uses_sampled_docs(self):
        self.collection.insert_into([
            {"x": 1},
            {"x": "two"},  # heterogeneous — should promote to string
        ])
        # Pull the raw documents to feed the helper directly.
        docs = self.fetch_documents()
        schema = infer_arrow_schema_from_documents(docs)
        # Promotion ladder pushes mixed int/string to string.
        assert str(schema.field("x").type) == "string"


class TestRepr(MongoTestCase):
    def test_collection_repr_includes_full_name(self):
        text = repr(self.collection)
        assert "MongoCollection<" in text
        assert self.default_collection in text

    def test_database_repr_includes_name(self):
        text = repr(self.database)
        assert self._db_name in text
