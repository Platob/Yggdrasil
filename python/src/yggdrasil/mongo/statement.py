"""MongoDB-backed :class:`PreparedStatement` / :class:`StatementResult`.

MongoDB doesn't speak SQL — there is no string statement to plan.
Instead, every command is one of:

* a raw ``runCommand`` payload (``{"ping": 1}``, ``{"listCollections": 1}``);
* an aggregation pipeline (``[{"$match": ...}, {"$group": ...}, ...]``);
* a ``find()`` (filter + projection + cursor knobs).

We model these uniformly via :class:`MongoCommand`, the prepared
statement that the executor submits. The ``text`` field carries a
JSON-rendered version of the command (so the inherited
``looks_like_query`` / repr / log behaviour works); the canonical
representation is the ``payload`` / ``pipeline`` attributes.

Result handling is synchronous from the client's perspective —
pymongo returns either a single :class:`pymongo.command_cursor.CommandCursor`
or a result document. We materialise it into a :class:`pa.Table` once
on :meth:`MongoStatementResult.start` (preferring pymongoarrow's
``aggregate_arrow_all`` / ``find_arrow_all`` when installed) and stash
it on a :class:`ArrowTabular` so every Tabular read method works.
"""

from __future__ import annotations

import json
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

import pyarrow as pa

from yggdrasil.data import Schema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.statement import (
    PreparedStatement,
    StatementBatch,
    StatementResult,
)
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.tabular.base import O
from yggdrasil.data.enums import MimeType

from .lib import has_pymongoarrow, pymongoarrow_api_module
from .types import documents_to_arrow_table

if TYPE_CHECKING:
    from .connection import MongoConnection
    from .executor import MongoExecutor

logger = logging.getLogger(__name__)

__all__ = [
    "MongoCommand",
    "MongoStatementResult",
    "MongoStatementBatch",
    "MONGO_COMMAND_MIME",
    "MONGO_COLLECTION_MIME",
    "MongoCommandKind",
]


# Mongo-specific mime types — keep them defined here so the executor
# / collection use the same registry slots.
MONGO_COMMAND_MIME = MimeType.define(
    MimeType("MONGO_COMMAND", "application/vnd.mongodb.command"),
)
MONGO_COLLECTION_MIME = MimeType.define(
    MimeType("MONGO_COLLECTION", "application/vnd.mongodb.collection"),
)


class MongoCommandKind:
    """Enum-ish discriminator for :class:`MongoCommand`."""

    FIND = "find"
    AGGREGATE = "aggregate"
    COMMAND = "command"
    COUNT = "count"
    DISTINCT = "distinct"

    ALL = (FIND, AGGREGATE, COMMAND, COUNT, DISTINCT)


def _safe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, default=str, sort_keys=True, ensure_ascii=False)
    except Exception:
        return repr(payload)


class MongoCommand(PreparedStatement):
    """A prepared MongoDB command.

    Three high-level shapes that drive :meth:`MongoStatementResult.start`:

    * ``kind=FIND`` — a ``find()`` against ``collection_name`` with
      ``filter`` / ``projection`` / ``sort`` / ``limit`` / ``skip``.
    * ``kind=AGGREGATE`` — an aggregation pipeline against ``collection_name``.
    * ``kind=COMMAND`` — a raw ``db.command(payload)`` against the
      database (no collection scope).

    The ``text`` slot inherited from :class:`PreparedStatement` holds
    a JSON-rendered representation so log lines / batch keys work as
    expected — the *canonical* representation is the typed kwargs.
    """

    def __init__(
        self,
        *,
        kind: str = MongoCommandKind.COMMAND,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        projection: Optional[Mapping[str, Any] | Sequence[str]] = None,
        sort: Optional[Sequence[tuple[str, int]] | Mapping[str, int]] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        pipeline: Optional[Sequence[Mapping[str, Any]]] = None,
        payload: Optional[Mapping[str, Any]] = None,
        allow_disk_use: Optional[bool] = None,
        batch_size: Optional[int] = None,
        comment: Optional[str] = None,
        prefer_arrow: bool = True,
        text: str = "",
        key: Optional[str] = None,
        retry: Optional[WaitingConfigArg] = None,
        **kwargs: Any,
    ):
        super().__init__(text=text, key=key, retry=retry)
        if kind not in MongoCommandKind.ALL:
            raise ValueError(
                f"Unknown MongoCommand kind={kind!r}; expected one of "
                f"{MongoCommandKind.ALL}."
            )
        self.kind = kind
        self.database_name = database_name
        self.collection_name = collection_name
        self.filter = dict(filter) if filter else {}
        self.projection = projection
        self.sort = list(sort.items()) if isinstance(sort, Mapping) else (list(sort) if sort else None)
        self.limit = limit
        self.skip = skip
        self.pipeline = [dict(stage) for stage in pipeline] if pipeline else None
        self.payload = dict(payload) if payload else None
        self.allow_disk_use = allow_disk_use
        self.batch_size = batch_size
        self.comment = comment
        self.prefer_arrow = bool(prefer_arrow)
        if not self.text:
            self.text = self._render_text()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        statement: "MongoCommand | PreparedStatement | str | Mapping[str, Any]",
    ) -> "MongoCommand":
        """Coerce raw inputs into a :class:`MongoCommand`.

        - existing :class:`MongoCommand` → passed through.
        - ``Mapping`` → treated as a raw command payload.
        - ``str`` → JSON-decoded if possible (treated as a raw command),
          else stored verbatim under ``text`` with ``kind=COMMAND``.
        - any other :class:`PreparedStatement` → text carried over.
        """
        if isinstance(statement, cls):
            return statement
        if isinstance(statement, Mapping):
            return cls(payload=statement, kind=MongoCommandKind.COMMAND)
        if isinstance(statement, str):
            try:
                payload = json.loads(statement)
                if isinstance(payload, Mapping):
                    return cls(payload=payload, kind=MongoCommandKind.COMMAND, text=statement)
            except json.JSONDecodeError:
                pass
            return cls(text=statement, kind=MongoCommandKind.COMMAND)
        if isinstance(statement, PreparedStatement):
            return cls(text=statement.text, key=statement.key, retry=statement.retry)
        raise TypeError(f"Cannot prepare {statement!r} as MongoCommand.")

    @classmethod
    def find(
        cls,
        collection_name: str,
        *,
        database_name: Optional[str] = None,
        filter: Optional[Mapping[str, Any]] = None,
        projection: Optional[Mapping[str, Any] | Sequence[str]] = None,
        sort: Optional[Any] = None,
        limit: Optional[int] = None,
        skip: Optional[int] = None,
        batch_size: Optional[int] = None,
        comment: Optional[str] = None,
        prefer_arrow: bool = True,
    ) -> "MongoCommand":
        """Build a ``find()`` command."""
        return cls(
            kind=MongoCommandKind.FIND,
            database_name=database_name,
            collection_name=collection_name,
            filter=filter,
            projection=projection,
            sort=sort,
            limit=limit,
            skip=skip,
            batch_size=batch_size,
            comment=comment,
            prefer_arrow=prefer_arrow,
        )

    @classmethod
    def aggregate(
        cls,
        collection_name: str,
        pipeline: Sequence[Mapping[str, Any]],
        *,
        database_name: Optional[str] = None,
        allow_disk_use: bool = True,
        batch_size: Optional[int] = None,
        comment: Optional[str] = None,
        prefer_arrow: bool = True,
    ) -> "MongoCommand":
        """Build an aggregation pipeline command."""
        return cls(
            kind=MongoCommandKind.AGGREGATE,
            database_name=database_name,
            collection_name=collection_name,
            pipeline=pipeline,
            allow_disk_use=allow_disk_use,
            batch_size=batch_size,
            comment=comment,
            prefer_arrow=prefer_arrow,
        )

    @classmethod
    def command(
        cls,
        payload: Mapping[str, Any],
        *,
        database_name: Optional[str] = None,
    ) -> "MongoCommand":
        """Build a raw ``db.command()`` payload."""
        return cls(
            kind=MongoCommandKind.COMMAND,
            database_name=database_name,
            payload=payload,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_text(self) -> str:
        if self.kind == MongoCommandKind.FIND:
            payload = {
                "find": self.collection_name,
                "filter": self.filter,
                "projection": self.projection,
                "sort": self.sort,
                "limit": self.limit,
                "skip": self.skip,
            }
        elif self.kind == MongoCommandKind.AGGREGATE:
            payload = {
                "aggregate": self.collection_name,
                "pipeline": self.pipeline,
            }
        elif self.kind == MongoCommandKind.COMMAND:
            payload = self.payload
        else:
            payload = {self.kind: self.collection_name, "filter": self.filter}
        return _safe_json({k: v for k, v in payload.items() if v is not None}) if payload else ""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class MongoStatementResult(StatementResult[MongoCommand]):
    """Synchronous Mongo result handle wrapping a fetched Arrow table.

    Materialises the underlying cursor / command response on
    :meth:`start` and stashes the table on a :class:`ArrowTabular` so
    every :class:`Tabular` reader / records iterator just works.

    The polling loop on :class:`StatementBatch.wait` degenerates into
    a no-op for Mongo because the result is terminal as soon as
    :meth:`start` returns.
    """

    _PREPARED_STATEMENT_CLASS: ClassVar[type[MongoCommand]] = MongoCommand

    _TRANSIENT_ERROR_PATTERNS = (
        r"NotPrimary",
        r"NetworkError",
        r"ConnectionFailure",
        r"AutoReconnect",
        r"ExceededTimeLimit",
        r"WriteConflict",
    )

    def __init__(
        self,
        statement: MongoCommand,
        *,
        executor: Optional["MongoExecutor"] = None,
        connection: Optional["MongoConnection"] = None,
        target_schema: Optional[pa.Schema] = None,
        **kwargs: Any,
    ):
        super().__init__(statement=statement, executor=executor, **kwargs)
        self._connection = connection
        self._target_schema = target_schema
        self._started: bool = False
        self._failure: Optional[BaseException] = None
        self._row_count: int = -1
        self._raw_response: Any = None

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return MONGO_COMMAND_MIME

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def done(self) -> bool:
        return self._started

    @property
    def failed(self) -> bool:
        return self._failure is not None

    def refresh_status(self) -> None:
        return None

    def _failure_message(self) -> str:
        if self._failure is None:
            return ""
        return f"{type(self._failure).__name__}: {self._failure}"

    def _raise_for_status(self) -> None:
        if self._failure is not None:
            raise self._failure

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def raw_response(self) -> Any:
        """The raw command response (before Arrow materialisation).

        Useful for callers that need to inspect ``ok`` / ``errmsg`` /
        write-concern data on a non-result-bearing command.
        """
        return self._raw_response

    # ------------------------------------------------------------------
    # Start / cancel
    # ------------------------------------------------------------------

    def start(
        self,
        reset: bool = False,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        **kwargs: Any,
    ) -> "MongoStatementResult":
        if self._started and not reset:
            if raise_error:
                self.raise_for_status()
            return self
        if reset:
            self._failure = None
            self._persisted_data = None
            self._cached_schema = None
            self._row_count = -1
            self._started = False
            self._raw_response = None

        connection = self._resolve_connection()
        try:
            table, row_count, raw = self._execute(connection)
        except BaseException as exc:
            self._failure = exc
            self._started = True
            if raise_error:
                raise
            return self

        from yggdrasil.io.tabular import ArrowTabular
        self._persisted_data = ArrowTabular(table)
        self._row_count = row_count
        self._raw_response = raw
        self._started = True
        return self

    def cancel(self) -> "MongoStatementResult":
        # pymongo cursors are cancellable via ``cursor.close()`` but
        # we materialise eagerly inside ``start`` so there's nothing
        # left to cancel from the caller's perspective.
        return self

    # ------------------------------------------------------------------
    # Tabular hooks — read only
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        self._require_started()
        yield from self._persisted_data._read_arrow_batches(options)

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions,
    ) -> None:
        raise NotImplementedError(
            "Cannot write to a MongoStatementResult; use "
            "MongoCollection.write_arrow_table or MongoExecutor.execute "
            "with an insert/update payload instead."
        )

    def _read_records(self, options: O) -> Iterator[Any]:
        self._require_started()
        yield from self._persisted_data._read_records(options)

    def _collect_schema(self, options: CastOptions) -> Schema:
        if self._cached_schema is None:
            if self._persisted_data is not None:
                self._cached_schema = self._persisted_data.collect_schema(options)
            else:
                self._cached_schema = super()._collect_schema(options)
        return self._cached_schema

    def _require_started(self) -> None:
        if self._persisted_data is None:
            raise RuntimeError(
                "Cannot read from a non-started Mongo statement; call "
                "start() first."
            )

    # ------------------------------------------------------------------
    # Connection / driver dispatch
    # ------------------------------------------------------------------

    def _resolve_connection(self) -> "MongoConnection":
        if self._connection is not None:
            return self._connection
        if self.executor is not None and hasattr(self.executor, "connection"):
            return self.executor.connection
        raise RuntimeError(
            "MongoStatementResult has no bound connection; pass connection= "
            "at construction or run via a MongoExecutor."
        )

    def _execute(self, connection: "MongoConnection") -> tuple[pa.Table, int, Any]:
        cmd = self.statement
        database = connection.database(cmd.database_name)
        if cmd.kind == MongoCommandKind.FIND:
            return self._execute_find(database, cmd)
        if cmd.kind == MongoCommandKind.AGGREGATE:
            return self._execute_aggregate(database, cmd)
        if cmd.kind == MongoCommandKind.COUNT:
            return self._execute_count(database, cmd)
        if cmd.kind == MongoCommandKind.DISTINCT:
            return self._execute_distinct(database, cmd)
        return self._execute_command(database, cmd)

    # ------------------------------------------------------------------
    # find()
    # ------------------------------------------------------------------

    def _execute_find(
        self,
        database: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        if not cmd.collection_name:
            raise ValueError("MongoCommand.find requires collection_name")
        collection = database[cmd.collection_name]
        if cmd.prefer_arrow and has_pymongoarrow():
            return self._execute_find_arrow(collection, cmd)
        return self._execute_find_rows(collection, cmd)

    def _execute_find_arrow(
        self,
        collection: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        api = pymongoarrow_api_module()
        schema = self._target_pma_schema(cmd, collection)
        kwargs = self._find_kwargs(cmd)
        if schema is not None:
            kwargs["schema"] = schema
        table = api.find_arrow_all(collection, cmd.filter, **kwargs)
        return table, table.num_rows, None

    def _execute_find_rows(
        self,
        collection: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        cursor = collection.find(
            filter=cmd.filter or {},
            projection=cmd.projection,
        )
        for setter, value in (
            ("sort", cmd.sort),
            ("limit", cmd.limit),
            ("skip", cmd.skip),
            ("batch_size", cmd.batch_size),
            ("comment", cmd.comment),
        ):
            if value is None:
                continue
            apply = getattr(cursor, setter, None)
            if apply is None:
                continue
            try:
                cursor = apply(value)
            except TypeError:
                cursor = apply(*value) if isinstance(value, (list, tuple)) else cursor
        documents = list(cursor)
        target = self._target_arrow_schema(cmd)
        table = documents_to_arrow_table(documents, schema=target)
        return table, table.num_rows, None

    # ------------------------------------------------------------------
    # aggregate()
    # ------------------------------------------------------------------

    def _execute_aggregate(
        self,
        database: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        if not cmd.collection_name:
            raise ValueError("MongoCommand.aggregate requires collection_name")
        collection = database[cmd.collection_name]
        if cmd.prefer_arrow and has_pymongoarrow():
            api = pymongoarrow_api_module()
            schema = self._target_pma_schema(cmd, collection)
            kwargs: dict[str, Any] = {
                "allowDiskUse": True if cmd.allow_disk_use is None else bool(cmd.allow_disk_use),
            }
            if cmd.batch_size is not None:
                kwargs["batchSize"] = int(cmd.batch_size)
            if schema is not None:
                kwargs["schema"] = schema
            table = api.aggregate_arrow_all(collection, list(cmd.pipeline or []), **kwargs)
            return table, table.num_rows, None
        cursor_kwargs: dict[str, Any] = {}
        if cmd.allow_disk_use is not None:
            cursor_kwargs["allowDiskUse"] = bool(cmd.allow_disk_use)
        if cmd.batch_size is not None:
            cursor_kwargs["batchSize"] = int(cmd.batch_size)
        cursor = collection.aggregate(list(cmd.pipeline or []), **cursor_kwargs)
        documents = list(cursor)
        target = self._target_arrow_schema(cmd)
        table = documents_to_arrow_table(documents, schema=target)
        return table, table.num_rows, None

    # ------------------------------------------------------------------
    # raw db.command()
    # ------------------------------------------------------------------

    def _execute_command(
        self,
        database: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        payload = cmd.payload or self._payload_from_text(cmd.text)
        if not payload:
            raise ValueError(
                "MongoCommand of kind=COMMAND requires either payload= or "
                "JSON-decodable text."
            )
        response = database.command(payload)
        cursor = (response or {}).get("cursor") if isinstance(response, Mapping) else None
        if cursor and isinstance(cursor.get("firstBatch"), list):
            documents = list(cursor["firstBatch"])
            target = self._target_arrow_schema(cmd)
            table = documents_to_arrow_table(documents, schema=target)
            return table, table.num_rows, response
        # Non-result-bearing command — emit a single-row table that
        # carries the response so callers can inspect `ok` / `n` /
        # `nModified` without losing the payload.
        if isinstance(response, Mapping):
            table = pa.Table.from_pylist([dict(response)])
            return table, 1, response
        return pa.table({}), 0, response

    @staticmethod
    def _payload_from_text(text: str) -> Optional[Mapping[str, Any]]:
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, Mapping) else None

    # ------------------------------------------------------------------
    # count / distinct
    # ------------------------------------------------------------------

    def _execute_count(
        self,
        database: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        collection = database[cmd.collection_name]
        n = int(collection.count_documents(cmd.filter or {}))
        table = pa.table({"n": [n]})
        return table, 1, {"ok": 1, "n": n}

    def _execute_distinct(
        self,
        database: Any,
        cmd: MongoCommand,
    ) -> tuple[pa.Table, int, Any]:
        collection = database[cmd.collection_name]
        payload = cmd.payload or {}
        key = payload.get("key") if isinstance(payload, Mapping) else None
        if not key:
            raise ValueError("MongoCommand.distinct requires payload={'key': ...}")
        values = list(collection.distinct(key, cmd.filter or {}))
        table = pa.table({key: values})
        return table, len(values), {"ok": 1, "values": values}

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _target_arrow_schema(self, cmd: MongoCommand) -> Optional[pa.Schema]:
        if self._target_schema is not None:
            return self._target_schema
        return None

    def _target_pma_schema(self, cmd: MongoCommand, collection: Any) -> Any:
        """Return a ``pymongoarrow.schema.Schema`` if a target was bound."""
        target = self._target_arrow_schema(cmd)
        if target is None:
            return None
        try:
            from pymongoarrow.schema import Schema as PmaSchema  # type: ignore[import-not-found]
        except ImportError:
            return None
        # pymongoarrow's Schema accepts ``{name: pa.DataType | python type}``
        # — pa types are honored verbatim.
        return PmaSchema({field.name: field.type for field in target})

    @staticmethod
    def _find_kwargs(cmd: MongoCommand) -> dict[str, Any]:
        """Translate a :class:`MongoCommand` into pymongo find()/find_arrow_all kwargs."""
        kwargs: dict[str, Any] = {}
        if cmd.projection is not None:
            kwargs["projection"] = (
                {f: 1 for f in cmd.projection}
                if isinstance(cmd.projection, (list, tuple, set))
                else dict(cmd.projection)
            )
        if cmd.sort is not None:
            kwargs["sort"] = cmd.sort
        if cmd.limit is not None:
            kwargs["limit"] = int(cmd.limit)
        if cmd.skip is not None:
            kwargs["skip"] = int(cmd.skip)
        if cmd.batch_size is not None:
            kwargs["batch_size"] = int(cmd.batch_size)
        return kwargs


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class MongoStatementBatch(StatementBatch[MongoCommand, MongoStatementResult]):
    """A batch of Mongo commands — the base behaviour fits."""

    def _coerce(self, statement: "MongoCommand | str") -> MongoCommand:
        return MongoCommand.from_(statement)


