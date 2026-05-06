"""MongoDB backend — Arrow-native collection navigator + executor.

Public surface
--------------

::

    from yggdrasil.mongo import (
        MongoEngine,           # top-level façade
        MongoConnection,       # pymongo client wrapper
        MongoExecutor,         # StatementExecutor implementation
        MongoDatabase, MongoDatabases,
        MongoCollection, MongoCollections,
        MongoCommand, MongoStatementResult, MongoStatementBatch,
        infer_arrow_schema_from_documents,
    )

The engine is a :class:`yggdrasil.data.executor.StatementExecutor`
that composes the database/collection hierarchy with the Arrow-native
read/write path. :class:`MongoCollection` implements
:class:`yggdrasil.io.tabular.base.Tabular`, so every cross-engine
conversion (Arrow / Polars / pandas / Spark / records / pylist /
pydict) lights up out-of-the-box.

Optional dependencies live behind :mod:`yggdrasil.mongo.lib`:

* :mod:`pymongo` (required) — canonical driver, used for metadata,
  lifecycle, raw commands, and the row-fallback read/write path.
* :mod:`pymongoarrow` (preferred, optional) — Arrow-native fast path
  for ``find_arrow_all`` / ``aggregate_arrow_all`` / ``write``.
"""

from .collection import MongoCollection, MongoCollections
from .connection import MongoConnection, normalize_mongo_uri, DEFAULT_URI_ENVS
from .database import MongoDatabase, MongoDatabases
from .engine import MongoEngine
from .executor import MongoExecutor
from .statement import (
    MONGO_COLLECTION_MIME,
    MONGO_COMMAND_MIME,
    MongoCommand,
    MongoCommandKind,
    MongoStatementBatch,
    MongoStatementResult,
)
from .types import (
    BSON_METADATA_KEY,
    BSON_SUBTYPE_METADATA_KEY,
    OBJECT_ID_BYTES,
    arrow_field_to_bson_extra,
    arrow_table_to_documents,
    arrow_to_bson_type_name,
    bson_to_arrow_type,
    decode_value_from_bson,
    documents_to_arrow_table,
    encode_value_for_bson,
    infer_arrow_schema_from_documents,
    infer_schema_from_documents,
)

__all__ = [
    # Engine / executor / connection
    "MongoEngine",
    "MongoExecutor",
    "MongoConnection",
    "normalize_mongo_uri",
    "DEFAULT_URI_ENVS",
    # Hierarchy
    "MongoDatabase",
    "MongoDatabases",
    "MongoCollection",
    "MongoCollections",
    # Statement / batch
    "MongoCommand",
    "MongoCommandKind",
    "MongoStatementResult",
    "MongoStatementBatch",
    "MONGO_COMMAND_MIME",
    "MONGO_COLLECTION_MIME",
    # Type bridging
    "BSON_METADATA_KEY",
    "BSON_SUBTYPE_METADATA_KEY",
    "OBJECT_ID_BYTES",
    "bson_to_arrow_type",
    "arrow_to_bson_type_name",
    "arrow_field_to_bson_extra",
    "infer_arrow_schema_from_documents",
    "infer_schema_from_documents",
    "documents_to_arrow_table",
    "arrow_table_to_documents",
    "encode_value_for_bson",
    "decode_value_from_bson",
]
