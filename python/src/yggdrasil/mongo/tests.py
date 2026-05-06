"""Unittest base class for MongoDB tests.

Quick start
-----------
::

    from yggdrasil.mongo.tests import MongoTestCase

    class TestUserSync(MongoTestCase):
        def test_round_trip(self):
            self.collection.insert_into([{"name": "alice"}])
            tbl = self.collection.read_arrow_table()
            self.assertEqual(tbl.num_rows, 1)

Resolution
----------
The base class is happy to use either:

* a real MongoDB instance reachable via ``MONGO_URI`` /
  ``MONGODB_URI`` (the integration path); or
* the in-process :mod:`mongomock` driver, when installed (the unit-
  test path).

Without either, every test in the class is skipped with a clear hint.
The choice is exposed as :attr:`mongo_kind` (``"real"`` or
``"mock"``) so individual tests can branch when they exercise driver-
specific behaviour.

The class allocates a temporary database name per :meth:`setUp` and
drops it on :meth:`tearDown`, so tests don't bleed into each other.
"""
from __future__ import annotations

import os
import unittest
import uuid
from typing import Any, ClassVar, Optional

from yggdrasil.environ import runtime_import_module

__all__ = ["MongoTestCase"]


def _env_uri() -> Optional[str]:
    for env in ("MONGO_URI", "MONGODB_URI"):
        value = os.environ.get(env)
        if value:
            return value
    return None


class MongoTestCase(unittest.TestCase):
    """Base class for MongoDB integration / unit tests.

    Attributes
    ----------
    pymongo : module
        The imported ``pymongo`` module. Always populated.
    bson : module
        The imported ``bson`` module (ships with pymongo).
    engine : MongoEngine
        Engine bound to the test database.
    database : MongoDatabase
        Test-scoped database; freshly minted in :meth:`setUp`.
    collection : MongoCollection
        Default test-scoped collection (``"test_collection"``);
        override :attr:`default_collection` to change.
    mongo_kind : ClassVar[str]
        ``"real"`` when the test runs against a live cluster, ``"mock"``
        when :mod:`mongomock` provided the client.
    """

    auto_install: ClassVar[bool | None] = None
    require_real_mongo: ClassVar[bool] = False
    default_collection: ClassVar[str] = "test_collection"

    pymongo: ClassVar[Any]
    bson: ClassVar[Any]
    mongo_kind: ClassVar[str] = "mock"
    _client: ClassVar[Any] = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            cls.pymongo = runtime_import_module("pymongo", install=False)
            cls.bson = runtime_import_module("bson", install=False)
        except ImportError:
            raise unittest.SkipTest(
                "'pymongo' is not installed. Install it with "
                "`pip install pymongo` or `pip install 'ygg[mongo]'`."
            )

        uri = _env_uri()
        if uri:
            cls._client = cls.pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
            try:
                cls._client.admin.command("ping")
            except Exception as exc:
                cls._client.close()
                cls._client = None
                if cls.require_real_mongo:
                    raise unittest.SkipTest(
                        f"Could not reach MongoDB at {uri!r}: {exc}."
                    ) from exc
                # else fall through to mongomock
            else:
                cls.mongo_kind = "real"
                return

        if cls.require_real_mongo:
            raise unittest.SkipTest(
                "MONGO_URI / MONGODB_URI not set; this test requires a real MongoDB."
            )

        try:
            mongomock = runtime_import_module("mongomock", install=False)
        except ImportError:
            raise unittest.SkipTest(
                "Neither MONGO_URI / MONGODB_URI is set nor 'mongomock' is "
                "installed. Install mongomock with `pip install mongomock` "
                "or set MONGO_URI to a reachable MongoDB instance."
            )
        cls._client = mongomock.MongoClient()
        cls.mongo_kind = "mock"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._client is not None:
            try:
                cls._client.close()
            except Exception:
                pass
        super().tearDownClass()

    def setUp(self) -> None:
        super().setUp()
        from .connection import MongoConnection
        from .engine import MongoEngine

        self._db_name = f"ygg_test_{uuid.uuid4().hex[:10]}"
        self._connection = MongoConnection(client=self._client, default_database=self._db_name)
        self.engine = MongoEngine(connection=self._connection)
        self.database = self.engine.database(self._db_name)
        self.collection = self.engine.collection(
            database_name=self._db_name,
            collection_name=self.default_collection,
        )

    def tearDown(self) -> None:
        try:
            self._client.drop_database(self._db_name)
        except Exception:
            pass
        super().tearDown()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def insert_documents(self, documents: list[dict], collection: Optional[str] = None) -> None:
        """Insert raw BSON documents through the underlying pymongo client."""
        target = self._client[self._db_name][collection or self.default_collection]
        if documents:
            target.insert_many(documents)

    def fetch_documents(self, collection: Optional[str] = None) -> list[dict]:
        """Return raw documents under a test collection (debugging helper)."""
        target = self._client[self._db_name][collection or self.default_collection]
        return list(target.find())
