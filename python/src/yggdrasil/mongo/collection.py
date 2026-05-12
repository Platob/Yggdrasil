"""MongoDB collection — :class:`Tabular` over a single namespace.

A :class:`MongoCollection` is the resource handle for ``db.coll``:

* navigation / lifecycle: :meth:`exists`, :meth:`create`, :meth:`drop`,
  :meth:`rename`, :meth:`truncate`, :meth:`count`, :meth:`indexes`;
* tabular IO: implements the two :class:`Tabular` hooks
  (:meth:`_read_arrow_batches` / :meth:`_write_arrow_batches`) so the
  caller can lift a collection through any of the cross-engine readers
  and writers (Arrow / Polars / pandas / Spark / records / pylist /
  pydict). The fast path is pymongoarrow (``find_arrow_all`` and
  ``write``); when that's not installed we fall back to a pymongo
  cursor + Arrow lift / pymongo bulk-write.

Save modes
----------
:class:`Mode` resolves to the matching MongoDB write strategy:

* :data:`Mode.AUTO` / :data:`Mode.APPEND` → ``insert_many``
  (unordered, ``ordered=False`` so a duplicate-key in one document
  doesn't abort the whole batch).
* :data:`Mode.OVERWRITE` → ``drop()`` then ``insert_many``.
* :data:`Mode.TRUNCATE` → ``delete_many({})`` then ``insert_many`` —
  preserves indexes / sharding.
* :data:`Mode.IGNORE` → no-op when the collection is non-empty,
  otherwise ``insert_many``.
* :data:`Mode.ERROR_IF_EXISTS` → raise when non-empty, otherwise
  ``insert_many``.
* :data:`Mode.UPSERT` / :data:`Mode.MERGE` → ``bulk_write`` of
  ``ReplaceOne`` upserts keyed on ``options.match_by`` (or
  ``["_id"]`` by default).
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

import pyarrow as pa

from yggdrasil.data import Schema as DataSchema
from yggdrasil.data.options import CastOptions
from yggdrasil.data.enums import MimeType, Mode
from yggdrasil.io.tabular.base import O, Tabular

from yggdrasil.lazy_imports import has_pymongoarrow, pymongoarrow_api_module
from .statement import (
    MONGO_COLLECTION_MIME,
    MongoCommand,
    MongoStatementResult,
)
from .types import (
    arrow_table_to_documents,
    documents_to_arrow_table,
    infer_arrow_schema_from_documents,
)

if TYPE_CHECKING:
    from .connection import MongoConnection
    from .database import MongoDatabase
    from .executor import MongoExecutor

logger = logging.getLogger(__name__)

__all__ = ["MongoCollection", "MongoCollections"]


class MongoCollection(Tabular):
    """A single MongoDB collection — DDL, DML, and Arrow IO."""

    _FINAL_TABULAR_IO: ClassVar[bool] = True

    @classmethod
    def default_media_type(cls) -> "MimeType | None":
        return MONGO_COLLECTION_MIME

    def __init__(
        self,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        *,
        service: Optional["MongoCollections"] = None,
        executor: Optional["MongoExecutor"] = None,
        connection: Optional["MongoConnection"] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if executor is None and service is not None:
            executor = service.executor
        if executor is None:
            raise ValueError(
                "MongoCollection requires an executor (or a service that "
                "carries one)."
            )
        if not collection_name:
            raise ValueError("MongoCollection requires a non-empty collection_name")
        self.service = service
        self.executor = executor
        self._connection = connection
        self.database_name = database_name or executor.connection.default_database
        if not self.database_name:
            raise ValueError(
                "MongoCollection could not resolve a database; pass database_name= "
                "or set MongoConnection.default_database."
            )
        self.collection_name = collection_name
        self._cached_schema: Optional[DataSchema] = None
        self._cached_schema_arrow: Optional[pa.Schema] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_(
        cls,
        obj: "MongoCollection | str",
        *,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        service: Optional["MongoCollections"] = None,
        executor: Optional["MongoExecutor"] = None,
    ) -> "MongoCollection":
        if isinstance(obj, cls):
            if database_name is None and collection_name is None:
                return obj
            return cls(
                database_name=database_name or obj.database_name,
                collection_name=collection_name or obj.collection_name,
                service=service or obj.service,
                executor=executor or obj.executor,
                connection=obj._connection,
            )
        if isinstance(obj, str):
            db, coll = _split_namespace(obj, database_name, collection_name)
        else:
            raise TypeError(f"Cannot resolve MongoCollection from {obj!r}")
        return cls(
            database_name=db,
            collection_name=coll,
            service=service,
            executor=executor,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def full_name(self) -> str:
        return f"{self.database_name}.{self.collection_name}"

    def __repr__(self) -> str:
        return f"MongoCollection<{self.full_name()!r}>"

    def __str__(self) -> str:
        return self.full_name()

    # ------------------------------------------------------------------
    # Connection routing
    # ------------------------------------------------------------------

    @property
    def connection(self) -> "MongoConnection":
        if self._connection is not None:
            return self._connection
        return self.executor.connection

    @property
    def database(self) -> "MongoDatabase":
        from .database import MongoDatabase
        return MongoDatabase(executor=self.executor, database_name=self.database_name)

    @property
    def collection(self) -> Any:
        """The underlying pymongo Collection — escape hatch for unsupported ops."""
        return self.connection.client[self.database_name][self.collection_name]

    # ------------------------------------------------------------------
    # Existence / lifecycle
    # ------------------------------------------------------------------

    @property
    def exists(self) -> bool:
        """``True`` iff the collection appears in ``db.list_collection_names()``."""
        return self.collection_name in self.connection.client[self.database_name].list_collection_names()

    def create(
        self,
        *,
        capped: bool = False,
        size: Optional[int] = None,
        max_documents: Optional[int] = None,
        validator: Optional[Mapping[str, Any]] = None,
        if_not_exists: bool = True,
        comment: Optional[str] = None,
    ) -> "MongoCollection":
        """``createCollection`` — collections are otherwise auto-materialised on first insert.

        Calling :meth:`create` explicitly is what you want for capped /
        validated / time-series collections, where MongoDB requires
        the create options up-front.
        """
        if if_not_exists and self.exists:
            return self
        kwargs: dict[str, Any] = {}
        if capped:
            kwargs["capped"] = True
            if size is not None:
                kwargs["size"] = int(size)
            if max_documents is not None:
                kwargs["max"] = int(max_documents)
        if validator is not None:
            kwargs["validator"] = dict(validator)
        if comment is not None:
            kwargs["comment"] = comment
        self.connection.client[self.database_name].create_collection(
            self.collection_name,
            **kwargs,
        )
        self._cached_schema = None
        self._cached_schema_arrow = None
        return self

    def ensure_created(self, **kwargs: Any) -> "MongoCollection":
        if not self.exists:
            self.create(if_not_exists=True, **kwargs)
        return self

    def delete(self, *, if_exists: bool = True) -> "MongoCollection":
        """``drop()`` — idempotent by default."""
        if if_exists and not self.exists:
            return self
        self.collection.drop()
        self._cached_schema = None
        self._cached_schema_arrow = None
        return self

    drop = delete

    def truncate(self) -> "MongoCollection":
        """``delete_many({})`` — wipe documents while keeping indexes."""
        self.collection.delete_many({})
        return self

    def rename(self, new_name: str) -> "MongoCollection":
        new_name = (new_name or "").strip()
        if not new_name:
            raise ValueError("Cannot rename collection to an empty name")
        if new_name == self.collection_name:
            return self
        self.collection.rename(new_name)
        self.collection_name = new_name
        self._cached_schema = None
        self._cached_schema_arrow = None
        return self

    def count(self, filter: Optional[Mapping[str, Any]] = None) -> int:
        """``count_documents`` — exact count, optionally filtered."""
        return int(self.collection.count_documents(dict(filter or {})))

    def indexes(self) -> list[Mapping[str, Any]]:
        """List of index spec dictionaries from ``listIndexes``."""
        return list(self.collection.list_indexes())

    def create_index(
        self,
        keys: Sequence[tuple[str, int]] | Mapping[str, int] | str,
        *,
        unique: bool = False,
        name: Optional[str] = None,
        sparse: bool = False,
        **kwargs: Any,
    ) -> str:
        """Convenience wrapper around ``create_index``."""
        index_spec: Any
        if isinstance(keys, str):
            index_spec = [(keys, 1)]
        elif isinstance(keys, Mapping):
            index_spec = list(keys.items())
        else:
            index_spec = list(keys)
        if unique:
            kwargs["unique"] = True
        if sparse:
            kwargs["sparse"] = True
        if name:
            kwargs["name"] = name
        return self.collection.create_index(index_spec, **kwargs)

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def _collect_schema(self, options: O) -> DataSchema:
        """Build a yggdrasil :class:`Schema` by sampling documents.

        Mongo has no fixed schema, so the answer is approximate. We
        sample up to 100 documents (override with
        ``options.row_size`` when the caller already bounded the
        scan), unify per-key BSON types via the rank ladder in
        :mod:`yggdrasil.mongo.types`, and cache the result.
        """
        if self._cached_schema is not None:
            return self._cached_schema
        if options.target is not None:
            self._cached_schema = options.target
            self._cached_schema_arrow = options.target.to_arrow_schema()
            return self._cached_schema
        sample_size = options.row_size or 100
        cursor = self.collection.find(limit=int(sample_size))
        documents = list(cursor)
        arrow_schema = infer_arrow_schema_from_documents(documents, sample_size=sample_size)
        self._cached_schema_arrow = arrow_schema
        self._cached_schema = (
            DataSchema.from_arrow(arrow_schema)
            if len(arrow_schema) > 0
            else DataSchema.empty()
        )
        return self._cached_schema

    # ------------------------------------------------------------------
    # Tabular — read
    # ------------------------------------------------------------------

    def _read_arrow_batches(self, options: O) -> Iterator[pa.RecordBatch]:
        """Stream ``find()`` results as Arrow record batches."""
        cmd = self._build_find_command(options)
        result = MongoStatementResult(
            statement=cmd,
            executor=self.executor,
            connection=self.connection,
            target_schema=self._target_arrow_schema(options),
        )
        result.start(wait=True, raise_error=True)
        row_size = getattr(options, "row_size", None) or None
        for batch in result.read_arrow_batches(options=options):
            if row_size:
                yield from _rechunk_batch(batch, row_size)
            else:
                yield batch

    def _build_find_command(self, options: O) -> MongoCommand:
        column_names = options.select_source_column_names() or None
        projection = (
            {name: 1 for name in column_names} if column_names else None
        )
        return MongoCommand.find(
            collection_name=self.collection_name,
            database_name=self.database_name,
            filter=None,
            projection=projection,
            limit=None,
            skip=None,
            batch_size=options.row_size,
            prefer_arrow=True,
        )

    def _target_arrow_schema(self, options: O) -> Optional[pa.Schema]:
        target = options.target or options.merged
        if target is None:
            return None
        return target.to_arrow_schema()

    # ------------------------------------------------------------------
    # Tabular — write
    # ------------------------------------------------------------------

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: O,
    ) -> None:
        """Bulk-write Arrow batches into this collection.

        Picks the matching strategy from :attr:`CastOptions.mode`.
        UPSERT / MERGE go through a key-based ``bulk_write`` of
        ``ReplaceOne`` upserts; everything else funnels through
        ``insert_many`` after honoring the disposition (drop /
        delete / no-op).
        """
        mode = options.mode
        action = self._resolve_save_mode(mode)
        if action == "ignore":
            return
        if action == "error":
            raise FileExistsError(
                f"MongoCollection write with Mode.ERROR_IF_EXISTS but "
                f"{self.full_name()!r} is non-empty."
            )
        if action == "drop":
            self.delete(if_exists=True)
        elif action == "truncate":
            if self.exists:
                self.truncate()

        # Materialise the iterable into an Arrow table — the writer
        # paths below want random access (pymongoarrow.write streams
        # the table; the row fallback chunks by row_size).
        batches_list = list(batches)
        if not batches_list:
            self._cached_schema = None
            self._cached_schema_arrow = None
            return
        table = pa.Table.from_batches(batches_list)

        if mode in (Mode.UPSERT, Mode.MERGE):
            self._write_upsert(table, options)
        else:
            self._write_insert(table, options)

        self._cached_schema = None
        self._cached_schema_arrow = None

    def _resolve_save_mode(self, mode: Mode) -> str:
        if mode in (Mode.AUTO, Mode.APPEND, Mode.UPSERT, Mode.MERGE):
            return "append"
        if mode == Mode.OVERWRITE:
            return "drop"
        if mode == Mode.TRUNCATE:
            return "truncate"
        if mode == Mode.IGNORE:
            return "ignore" if self.exists and self.count() > 0 else "append"
        if mode == Mode.ERROR_IF_EXISTS:
            return "error" if self.exists and self.count() > 0 else "append"
        return "append"

    def _write_insert(self, table: pa.Table, options: O) -> None:
        """``insert_many`` — Arrow-fast via pymongoarrow when available."""
        if has_pymongoarrow():
            api = pymongoarrow_api_module()
            try:
                api.write(self.collection, table)
                return
            except Exception:
                logger.exception(
                    "pymongoarrow.write failed for %r; falling back to pymongo "
                    "insert_many.",
                    self.full_name(),
                )
        self._pymongo_insert(table, options)

    def _pymongo_insert(self, table: pa.Table, options: O) -> None:
        """Row-fallback insert path."""
        chunk_size = options.row_size or 1000
        documents = arrow_table_to_documents(table)
        if not documents:
            return
        for offset in range(0, len(documents), int(chunk_size)):
            chunk = documents[offset:offset + int(chunk_size)]
            self.collection.insert_many(chunk, ordered=False)

    def _write_upsert(self, table: pa.Table, options: O) -> None:
        """Keyed upsert via ``bulk_write([UpdateOne(filter, $set, upsert=True)])``.

        We standardise on ``UpdateOne`` with a ``$set`` payload (instead
        of ``ReplaceOne``) so the match keys aren't accidentally replaced
        on update — and so ``update_column_names`` can scope the
        update to a subset of fields without losing the rest.
        """
        from pymongo import UpdateOne  # type: ignore[import-not-found]

        match_by = options.match_by_keys or ["_id"]
        update_cols = options.update_column_names or None
        documents = arrow_table_to_documents(table)
        if not documents:
            return

        operations: list[Any] = []
        for doc in documents:
            filter_ = {key: doc[key] for key in match_by if key in doc}
            if not filter_:
                raise ValueError(
                    f"UPSERT into {self.full_name()!r} requires every row to "
                    f"carry the match keys {match_by!r}; got document with "
                    f"keys {sorted(doc.keys())}."
                )
            if update_cols:
                set_payload = {key: doc[key] for key in update_cols if key in doc}
            else:
                set_payload = {key: value for key, value in doc.items() if key not in match_by}
            update_payload: dict[str, Any] = {}
            if set_payload:
                update_payload["$set"] = set_payload
            # On insert, also seed the match keys themselves so the
            # newly-created document carries the discriminator.
            update_payload["$setOnInsert"] = filter_
            operations.append(UpdateOne(filter_, update_payload, upsert=True))

        chunk_size = options.row_size or 1000
        for offset in range(0, len(operations), int(chunk_size)):
            chunk = operations[offset:offset + int(chunk_size)]
            try:
                self.collection.bulk_write(chunk, ordered=False)
            except TypeError:
                # Fallback for backends (e.g. older mongomock) whose
                # bulk-write builder rejects newer pymongo kwargs.
                # Per-document ``update_one`` produces the same result
                # — slower but correct.
                for op, doc in zip(chunk, documents[offset:offset + int(chunk_size)]):
                    filter_ = {key: doc[key] for key in match_by if key in doc}
                    self.collection.update_one(
                        filter_, op._doc, upsert=True,
                    )

    # ------------------------------------------------------------------
    # Pylist override — short-circuit through the docs path when no Arrow target.
    # ------------------------------------------------------------------

    def _read_pylist(self, options: O) -> List[dict]:
        target = self._target_arrow_schema(options)
        if target is not None:
            return self._read_arrow_table(options).to_pylist()
        cursor = self.collection.find(filter={})
        from .types import decode_value_from_bson
        return [
            {k: decode_value_from_bson(v) for k, v in doc.items()}
            for doc in cursor
        ]

    def _write_pylist(self, data: Iterable[dict], options: O) -> None:
        rows = self._normalize_records(data)
        if not rows:
            return
        action = self._resolve_save_mode(options.mode)
        if action == "ignore":
            return
        if action == "error":
            raise FileExistsError(
                f"MongoCollection write with Mode.ERROR_IF_EXISTS but "
                f"{self.full_name()!r} is non-empty."
            )
        if action == "drop":
            self.delete(if_exists=True)
        elif action == "truncate":
            if self.exists:
                self.truncate()
        if options.mode in (Mode.UPSERT, Mode.MERGE):
            table = documents_to_arrow_table(rows)
            self._write_upsert(table, options)
            return
        chunk_size = options.row_size or 1000
        for offset in range(0, len(rows), int(chunk_size)):
            chunk = rows[offset:offset + int(chunk_size)]
            self.collection.insert_many(chunk, ordered=False)

    # ------------------------------------------------------------------
    # Insert convenience
    # ------------------------------------------------------------------

    def insert_into(
        self,
        data: Any,
        *,
        mode: Mode | str | None = None,
        match_by: Optional[Sequence[str]] = None,
        update_column_names: Optional[Sequence[str]] = None,
        cast_options: Optional[CastOptions] = None,
    ) -> "MongoCollection":
        """High-level insert — arrow / polars / pandas / dict / list → collection."""
        from yggdrasil.arrow.cast import any_to_arrow_table

        options = self.check_options(
            cast_options,
            mode=mode if mode is not None else Mode.AUTO,
            match_by=list(match_by) if match_by else None,
            update_column_names=list(update_column_names) if update_column_names else None,
        )
        if isinstance(data, list) and (not data or isinstance(data[0], Mapping)):
            self._write_pylist(data, options)
            return self
        table = any_to_arrow_table(data, options)
        self.write_arrow_table(table, options=options)
        return self


# ---------------------------------------------------------------------------
# Collection registry / navigation
# ---------------------------------------------------------------------------


class MongoCollections:
    """Iterator + factory for collections under an executor / database scope.

    Same shape as :class:`yggdrasil.postgres.Tables` — the executor /
    engine carries the implicit scope (database) and ``collection(...)``
    is the resolution entry point.
    """

    def __init__(
        self,
        executor: "MongoExecutor",
        *,
        database_name: Optional[str] = None,
    ):
        self.executor = executor
        self.database_name = database_name or executor.connection.default_database

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MongoCollections(database={self.database_name!r})"

    def __iter__(self) -> Iterator[MongoCollection]:
        return self.iter()

    def __getitem__(self, name: str) -> MongoCollection:
        return self.collection(collection_name=name)

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def collection(
        self,
        location: Optional[str] = None,
        *,
        database_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> MongoCollection:
        """Resolve a single collection.

        Acceptable input shapes:

        - ``location="db.coll"`` — canonical dotted form.
        - ``database_name=`` + ``collection_name=`` — both explicit.
        - ``collection_name=`` only — picks the navigator's database.
        """
        db = database_name or self.database_name
        coll: Optional[str] = collection_name
        if location:
            db_from_loc, coll_from_loc = _split_namespace(location, db, collection_name)
            db = db_from_loc
            coll = coll_from_loc
        if not db:
            raise ValueError(
                "MongoCollections requires a database name; pass database_name= or "
                "set MongoConnection.default_database."
            )
        if not coll:
            raise ValueError(
                "MongoCollections requires a collection name; pass collection_name= "
                "or use a dotted ``database.collection`` location."
            )
        return MongoCollection(
            database_name=db,
            collection_name=coll,
            executor=self.executor,
            service=self,
        )

    def iter(self) -> Iterator[MongoCollection]:
        """Iterate collections under :attr:`database_name`."""
        if not self.database_name:
            raise ValueError(
                "MongoCollections.iter requires a database; pass database_name= "
                "or set MongoConnection.default_database."
            )
        names = self.executor.connection.client[self.database_name].list_collection_names()
        for name in sorted(names):
            yield self.collection(collection_name=name)

    def names(self) -> list[str]:
        """Sorted list of collection names under the configured database."""
        return sorted(
            self.executor.connection.client[self.database_name].list_collection_names()
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_namespace(
    location: str,
    database_name: Optional[str],
    collection_name: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """Split ``"db.coll"`` (or ``"coll"``) into ``(db, coll)``.

    Explicit kwargs win on conflict — passing ``location="a.b"`` plus
    ``collection_name="c"`` keeps ``c`` as the collection.
    """
    if "." in location:
        head, _, tail = location.partition(".")
        return database_name or head, collection_name or tail
    return database_name, collection_name or location


def _rechunk_batch(batch: pa.RecordBatch, row_size: int) -> Iterator[pa.RecordBatch]:
    if batch.num_rows <= row_size:
        yield batch
        return
    offset = 0
    while offset < batch.num_rows:
        yield batch.slice(offset, row_size)
        offset += row_size
