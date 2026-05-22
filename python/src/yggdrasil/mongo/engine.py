""":class:`MongoEngine` — top-level façade around the MongoDB backend.

Composes the database/collection navigator with the
:class:`MongoExecutor` so end users have a single entry point::

    from yggdrasil.mongo import MongoEngine

    with MongoEngine("mongodb://localhost:27017/mydb") as eng:
        coll = eng.collection("users")            # MongoCollection
        df = coll.read_polars_frame()             # find() → Polars
        coll.insert_into({"name": "alice"})       # bulk insert
        eng.run_command({"ping": 1})              # raw db.command

The engine itself **is** a :class:`MongoExecutor` (which is a
:class:`StatementExecutor`), so every cross-backend statement helper —
``execute``, ``execute_many``, ``batch`` — works out of the box. The
hierarchy navigation (``engine.databases`` / ``engine.collections``)
shadows the executor's own properties to thread default scope (the
implicit database) through.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Optional

from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import StatementResult
from yggdrasil.data.enums import Mode

from .connection import MongoConnection
from .executor import MongoExecutor
from .statement import MongoCommand

if TYPE_CHECKING:
    from .collection import MongoCollection, MongoCollections
    from .database import MongoDatabase, MongoDatabases

logger = logging.getLogger(__name__)

__all__ = ["MongoEngine"]


class MongoEngine(MongoExecutor):
    """Top-level MongoDB backend façade.

    Construction
    ------------
    Pass a URI, an existing :class:`MongoConnection` / :class:`pymongo.MongoClient`,
    or rely on the ``MONGO_URI`` / ``MONGODB_URI`` environment variables.
    """

    def __init__(
        self,
        connection: "MongoConnection | str | Any | None" = None,
        *,
        database_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(connection=connection, **kwargs)
        self.database_name = database_name or self.connection.default_database

    # ------------------------------------------------------------------
    # Scope rebind
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        database_name: Optional[str] = None,
    ) -> "MongoEngine":
        """Return a re-scoped engine that shares the same connection."""
        if database_name is None or database_name == self.database_name:
            return self
        eng = MongoEngine.__new__(MongoEngine)
        # Bypass __init__ so we don't open a second client — the
        # rebind reuses the live connection verbatim.
        MongoExecutor.__init__(eng, connection=self.connection)
        eng.database_name = database_name
        return eng

    # ------------------------------------------------------------------
    # Hierarchy navigation
    # ------------------------------------------------------------------

    @property
    def databases(self) -> "MongoDatabases":
        from .database import MongoDatabases
        return MongoDatabases(executor=self)

    def database(self, name: Optional[str] = None) -> "MongoDatabase":
        from .database import MongoDatabase
        return MongoDatabase(
            executor=self,
            database_name=name or self.database_name,
        )

    @property
    def collections(self) -> "MongoCollections":
        from .collection import MongoCollections
        return MongoCollections(
            executor=self,
            database_name=self.database_name,
        )

    def collection(
        self,
        location: Optional[str] = None,
        *,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> "MongoCollection":
        return self.collections.collection(
            location=location,
            database_name=database_name or self.database_name,
            collection_name=collection_name,
        )

    # ------------------------------------------------------------------
    # High-level conveniences
    # ------------------------------------------------------------------

    def insert_into(
        self,
        data: Any,
        *,
        location: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        mode: Mode | str | None = None,
        match_by: Optional[Iterable[str]] = None,
        update_column_names: Optional[Iterable[str]] = None,
        cast_options: Optional[CastOptions] = None,
        collection: Optional["MongoCollection"] = None,
    ) -> "MongoCollection":
        """Resolve target + delegate to :meth:`MongoCollection.insert_into`."""
        if collection is None:
            collection = self.collection(
                location=location,
                database_name=database_name,
                collection_name=collection_name,
            )
        return collection.insert_into(
            data,
            mode=mode,
            match_by=list(match_by) if match_by else None,
            update_column_names=list(update_column_names) if update_column_names else None,
            cast_options=cast_options,
        )

    def create_collection(
        self,
        *,
        location: Optional[str] = None,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        capped: bool = False,
        size: Optional[int] = None,
        max_documents: Optional[int] = None,
        validator: Optional[Mapping[str, Any]] = None,
        missing_ok: bool = True,
    ) -> "MongoCollection":
        target = self.collection(
            location=location,
            database_name=database_name,
            collection_name=collection_name,
        )
        return target.create(
            capped=capped,
            size=size,
            max_documents=max_documents,
            validator=validator,
            missing_ok=missing_ok,
        )

    def drop_collection(
        self,
        location: Optional[str] = None,
        *,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        if_exists: bool = True,
    ) -> None:
        target = self.collection(
            location=location,
            database_name=database_name,
            collection_name=collection_name,
        )
        target.delete(if_exists=if_exists)

    # ------------------------------------------------------------------
    # Public execute — fold per-call routing knobs into the typed cmd.
    # ------------------------------------------------------------------

    def execute(
        self,
        statement: "MongoCommand | StatementResult | str | Mapping[str, Any]",
        *,
        database_name: Optional[str] = None,
        wait: Any = True,
        raise_error: bool = True,
        retry: Any = None,
    ) -> StatementResult:
        """Execute a Mongo command with per-call routing kwargs.

        Already-started results pass through untouched. ``str`` /
        :class:`Mapping` inputs are coerced through
        :meth:`MongoCommand.from_`.
        """
        if isinstance(statement, StatementResult):
            already_running = (
                getattr(statement, "started", statement.done)
            )
            if already_running:
                return statement.wait(wait=wait, raise_error=raise_error)
            statement = statement.statement

        prepared = MongoCommand.from_(statement)
        if database_name is not None:
            prepared.database_name = database_name
        elif prepared.database_name is None:
            prepared.database_name = self.database_name
        if retry is not None:
            prepared = prepared.with_retry(retry)
        return super().execute(prepared, wait=wait, raise_error=raise_error)
