"""MongoDB database — collection-of-collections navigator.

A :class:`MongoDatabase` is the equivalent of a Postgres ``Schema`` —
the container that owns collections. Lifecycle DDL is intentionally
narrower than Postgres because Mongo databases are auto-materialised
on first write; the public surface is therefore:

* :meth:`exists` / :meth:`delete` (i.e. ``dropDatabase``);
* :meth:`collections` / :meth:`collection` for navigation;
* :meth:`stats` / :meth:`server_info` for inspection.

There's no ``CREATE DATABASE`` because creating a database in MongoDB
just means inserting into a collection — :meth:`MongoCollection.create`
or any plain ``insert_many`` does the right thing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from .collection import MongoCollection, MongoCollections
    from .executor import MongoExecutor

logger = logging.getLogger(__name__)

__all__ = ["MongoDatabase", "MongoDatabases"]


class MongoDatabase:
    """A single MongoDB database, addressed by name."""

    def __init__(
        self,
        executor: Optional["MongoExecutor"] = None,
        database_name: Optional[str] = None,
        *,
        service: Optional["MongoDatabases"] = None,
    ):
        if executor is None and service is not None:
            executor = service.executor
        if executor is None:
            raise ValueError(
                "MongoDatabase requires an executor (or a service that "
                "carries one)."
            )
        self.executor = executor
        self.service = service
        self.database_name = database_name or executor.connection.default_database
        if not self.database_name:
            raise ValueError(
                "MongoDatabase could not resolve a database name; pass "
                "database_name= or set MongoConnection.default_database."
            )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def full_name(self) -> str:
        return self.database_name

    def __repr__(self) -> str:
        return f"MongoDatabase<{self.full_name()!r}>"

    def __str__(self) -> str:
        return self.database_name

    def __getitem__(self, name: str) -> "MongoCollection":
        return self.collection(name)

    def __iter__(self) -> Iterator["MongoCollection"]:
        return iter(self.collections)

    # ------------------------------------------------------------------
    # Existence / lifecycle
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """``True`` iff the database is reported by ``listDatabases``."""
        names = self.executor.connection.client.list_database_names()
        return self.database_name in names

    def delete(self) -> "MongoDatabase":
        """``dropDatabase()`` — wipes every collection in this database."""
        self.executor.connection.client.drop_database(self.database_name)
        return self

    drop = delete

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @property
    def collections(self) -> "MongoCollections":
        from .collection import MongoCollections
        return MongoCollections(executor=self.executor, database_name=self.database_name)

    def collection(self, name: str) -> "MongoCollection":
        return self.collections.collection(collection_name=name)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """``db.stats()`` — size, indexSize, collectionCount, …"""
        return dict(
            self.executor.connection.client[self.database_name].command("dbStats")
        )

    def server_info(self) -> dict[str, Any]:
        """``buildInfo`` — version, gitVersion, modules."""
        return dict(self.executor.connection.client.server_info())

    @property
    def native(self) -> Any:
        """The underlying :class:`pymongo.database.Database`."""
        return self.executor.connection.client[self.database_name]


# ---------------------------------------------------------------------------
# Database registry / navigation
# ---------------------------------------------------------------------------


class MongoDatabases:
    """Iterator + factory for databases under an executor."""

    def __init__(self, executor: "MongoExecutor"):
        self.executor = executor

    def __repr__(self) -> str:
        return f"MongoDatabases(connection={self.executor.connection!r})"

    def __iter__(self) -> Iterator[MongoDatabase]:
        return self.iter()

    def __getitem__(self, name: str) -> MongoDatabase:
        return self.database(name)

    def database(self, name: Optional[str] = None) -> MongoDatabase:
        return MongoDatabase(executor=self.executor, database_name=name, service=self)

    def iter(self) -> Iterator[MongoDatabase]:
        for name in sorted(self.executor.connection.client.list_database_names()):
            yield self.database(name)

    def names(self) -> list[str]:
        return sorted(self.executor.connection.client.list_database_names())
