"""Tests for :class:`MongoConnection` URI parsing and lazy client.

Requires :mod:`pymongo` for the URI normalisation tests; everything
else is pure-Python.
"""

from __future__ import annotations

import pytest

from yggdrasil.mongo.connection import (
    DEFAULT_URI_ENVS,
    MongoConnection,
    normalize_mongo_uri,
)


class TestNormalizeMongoUri:
    def test_valid_uri_passes_through(self):
        assert normalize_mongo_uri("mongodb://localhost:27017") == "mongodb://localhost:27017"

    def test_srv_uri_passes_through(self):
        assert normalize_mongo_uri("mongodb+srv://x.example") == "mongodb+srv://x.example"

    def test_strips_whitespace(self):
        assert normalize_mongo_uri("  mongodb://h  ") == "mongodb://h"

    def test_promotes_bare_host(self):
        # No scheme → assume mongodb://.
        assert normalize_mongo_uri("localhost:27017") == "mongodb://localhost:27017"

    def test_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="Unsupported"):
            normalize_mongo_uri("postgres://h")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_mongo_uri("")


class TestMongoConnectionEnvironment:
    def test_picks_first_env_var(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)
        monkeypatch.setenv("MONGODB_URI", "mongodb://from-env:27017/somedb")
        conn = MongoConnection()
        assert conn.uri == "mongodb://from-env:27017/somedb"

    def test_default_database_from_uri_path(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)
        conn = MongoConnection("mongodb://localhost:27017/mydb")
        assert conn.default_database == "mydb"

    def test_explicit_default_db_overrides_uri(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)
        conn = MongoConnection("mongodb://localhost:27017/from-uri", default_database="explicit")
        assert conn.default_database == "explicit"

    def test_no_uri_no_env_raises(self, monkeypatch):
        for env in DEFAULT_URI_ENVS:
            monkeypatch.delenv(env, raising=False)
        with pytest.raises(ValueError, match="requires a URI"):
            MongoConnection()


class TestMongoConnectionFromHelpers:
    def test_passes_through_existing(self, monkeypatch):
        monkeypatch.delenv("MONGO_URI", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)
        conn = MongoConnection("mongodb://h:27017/db")
        assert MongoConnection.from_(conn) is conn

    def test_string_passes_to_constructor(self):
        conn = MongoConnection.from_("mongodb://h:1234/db")
        assert conn.uri == "mongodb://h:1234/db"

    def test_mapping_passes_to_constructor(self):
        conn = MongoConnection.from_({"uri": "mongodb://h:1/db", "default_database": "x"})
        assert conn.uri == "mongodb://h:1/db"
        assert conn.default_database == "x"

    def test_unknown_input_raises(self):
        with pytest.raises(TypeError, match="Cannot build MongoConnection"):
            MongoConnection.from_(42)

    def test_database_requires_explicit_when_no_default(self):
        conn = MongoConnection("mongodb://h:27017")
        # No default db → explicit name required.
        with pytest.raises(ValueError, match="explicit name"):
            conn.database()


class TestRepr:
    def test_repr_strips_password(self):
        conn = MongoConnection("mongodb://user:secret@h:27017/db")
        text = repr(conn)
        assert "secret" not in text
        assert "***" in text
