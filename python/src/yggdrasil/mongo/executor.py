""":class:`MongoExecutor` — :class:`StatementExecutor` for MongoDB.

A thin shim around :class:`MongoConnection` that pins the typed
:class:`MongoCommand` / :class:`MongoStatementResult` pair onto the
base :class:`StatementExecutor`. Mongo execution is synchronous, so
the executor's only job is:

1. Coerce the incoming statement / payload into :class:`MongoCommand`.
2. Build a :class:`MongoStatementResult` bound to the connection.
3. Call :meth:`MongoStatementResult.start` so any eager backend
   rejection surfaces immediately.

Sub-services
------------
:meth:`databases` / :meth:`database` / :meth:`collections` /
:meth:`collection` are exposed here too — same pattern as
:class:`yggdrasil.postgres.PostgresExecutor` — so a pure-statement
caller doesn't have to walk through :class:`MongoEngine` to reach
the resource hierarchy.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from yggdrasil.data.executor import StatementExecutor

from .connection import MongoConnection
from .statement import (
    MongoCommand,
    MongoStatementBatch,
    MongoStatementResult,
)

logger = logging.getLogger(__name__)

__all__ = ["MongoExecutor"]


class MongoExecutor(
    StatementExecutor[
        MongoCommand,
        MongoStatementResult,
        MongoStatementBatch,
    ]
):
    """Run typed :class:`MongoCommand` payloads against a :class:`MongoConnection`."""

    _PREPARED_STATEMENT_CLASS: ClassVar[type[MongoCommand]] = MongoCommand
    _STATEMENT_RESULT_CLASS: ClassVar[type[MongoStatementResult]] = MongoStatementResult
    _STATEMENT_BATCH_CLASS: ClassVar[type[MongoStatementBatch]] = MongoStatementBatch

    def __init__(
        self,
        connection: "MongoConnection | str | Any | None" = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.connection: MongoConnection = MongoConnection.from_(connection)

    # ------------------------------------------------------------------
    # Sub-services
    # ------------------------------------------------------------------

    @property
    def databases(self):
        from .database import MongoDatabases
        return MongoDatabases(executor=self)

    def database(self, name: Optional[str] = None):
        return self.databases.database(name)

    @property
    def collections(self):
        from .collection import MongoCollections
        return MongoCollections(executor=self)

    def collection(
        self,
        location: Optional[str] = None,
        *,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        return self.collections.collection(
            location=location,
            database_name=database_name,
            collection_name=collection_name,
        )

    # ------------------------------------------------------------------
    # Executor contract
    # ------------------------------------------------------------------

    def _submit_statement(
        self,
        statement: MongoCommand,
        start: bool = True
    ) -> MongoStatementResult:
        result = self._STATEMENT_RESULT_CLASS(
            statement=statement,
            executor=self,
            connection=self.connection,
        )
        # raise_error=False — the base executor's _execute calls
        # raise_for_status afterwards if the caller's options say so.
        result.start(wait=False, raise_error=False)
        return result

    # ------------------------------------------------------------------
    # Conveniences
    # ------------------------------------------------------------------

    def run_command(
        self,
        payload: Any,
        *,
        database_name: Optional[str] = None,
    ) -> MongoStatementResult:
        """Run a raw ``db.command(payload)``."""
        cmd = MongoCommand.command(payload, database_name=database_name)
        return self.execute(cmd, wait=False, raise_error=True)

    def find(
        self,
        collection_name: str,
        filter: Optional[Any] = None,
        *,
        database_name: Optional[str] = None,
        projection: Optional[Any] = None,
        sort: Optional[Any] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        batch_size: Optional[int] = None,
        prefer_arrow: bool = True,
    ) -> MongoStatementResult:
        """Run a ``find()`` and return the materialised result handle."""
        cmd = MongoCommand.find(
            collection_name=collection_name,
            database_name=database_name,
            filter=filter,
            projection=projection,
            sort=sort,
            limit=limit,
            skip=skip,
            batch_size=batch_size,
            prefer_arrow=prefer_arrow,
        )
        return self.execute(cmd, wait=False, raise_error=True)

    def aggregate(
        self,
        collection_name: str,
        pipeline: Any,
        *,
        database_name: Optional[str] = None,
        allow_disk_use: bool = True,
        batch_size: Optional[int] = None,
        prefer_arrow: bool = True,
    ) -> MongoStatementResult:
        """Run an aggregation pipeline."""
        cmd = MongoCommand.aggregate(
            collection_name=collection_name,
            pipeline=pipeline,
            database_name=database_name,
            allow_disk_use=allow_disk_use,
            batch_size=batch_size,
            prefer_arrow=prefer_arrow,
        )
        return self.execute(cmd, wait=False, raise_error=True)

    # ------------------------------------------------------------------
    # Disposable
    # ------------------------------------------------------------------

    def _release(self, committed: bool = False) -> None:
        super()._release(committed=committed)
        try:
            self.connection.close()
        except Exception:
            logger.exception("Closing MongoConnection failed; continuing.")
