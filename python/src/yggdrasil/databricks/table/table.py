"""
Per-table resource: DDL, DML, schema introspection and storage helpers.

The :class:`Table` dataclass wraps a single Unity Catalog table and exposes
instance-level methods only.  Collection operations (``find_table``,
``list_tables``) live in :mod:`~yggdrasil.databricks.table.tables`.

Caching strategy
----------------
``TableInfo`` (and the derived columns list) is cached on the instance
with a shared TTL and loaded lazily on first access.

Entity-tag assignments — both table-level and per-column — are *not*
cached on the instance.  They route through
:attr:`DatabricksClient.entity_tags`, whose module-level
:class:`ExpiringDict` is host-scoped and authoritative; surgical patches
on write keep the cache fresh without fan-out invalidation.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
import time
import uuid
from collections.abc import MutableMapping
from typing import Any, Dict, Optional, Union, TYPE_CHECKING, Mapping, Iterable, Iterator, Literal, ClassVar

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    ColumnInfo,
    ColumnTypeName,
    DataSourceFormat,
    TableInfo,
    TableOperation,
    TableType, EntityTagAssignment,
)
from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Field
from yggdrasil.data.data_utils import safe_constraint_name
from yggdrasil.data.options import CastOptions
from yggdrasil.databricks.table.options import TableOptions
from yggdrasil.data.schema import Schema as DataSchema, Schema
from yggdrasil.data.statement import PreparedStatement, StatementResult
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.column.column import Column
from yggdrasil.databricks.path import DatabricksPath
from yggdrasil.databricks.sql.sql_utils import (
    MAX_TABLE_NAME_LEN,
    quote_ident,
    quote_principal,
    quote_qualified_ident,
    requalify_table_refs,
    safe_table_name,
    sql_literal, escape_sql_string,
)
from yggdrasil.dataclasses import Singleton
from yggdrasil.dataclasses.waiting import WaitingConfig, WaitingConfigArg
from yggdrasil.enums import MimeTypes, MimeType, MediaType, MediaTypes, ModeLike, Mode, Scheme
from yggdrasil.enums.engine_type import EngineType
from yggdrasil.execution.expr import (
    Predicate,
)
from yggdrasil.execution.expr.backends.sql import Dialect, to_sql as expr_to_sql
from yggdrasil.io.holder import IO
from yggdrasil.io.io_stats import IOKind, IOStats
from yggdrasil.io.tabular import Tabular, O
from yggdrasil.path import Path
from yggdrasil.url import URL

from ..fs import VolumePath
from ..volume import Volume

if TYPE_CHECKING:
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.table.tables import Tables
    from yggdrasil.databricks.catalog.catalog import UCCatalog
    from yggdrasil.databricks.column.columns import Columns
    from yggdrasil.databricks.schema.schema import UCSchema
    from yggdrasil.aws.client import AWSClient
    from yggdrasil.databricks.aws import AWSDatabricksTableCredentials
    from yggdrasil.data.statement import StatementBatch

_READ_ONLY_MODES = frozenset({Mode.AUTO})

# Unity Catalog ``table_type`` tokens that identify view-shaped securables.
# Used by ``Table`` to dispatch view-specific DDL (``ALTER VIEW`` /
# ``DROP VIEW`` / ``CREATE VIEW``) and by ``Tables`` to filter list output.
_VIEW_TABLE_TYPES: frozenset[TableType] = frozenset({
    TableType.VIEW,
    TableType.MATERIALIZED_VIEW,
    TableType.METRIC_VIEW,
})

# Below this on-disk size, the ``engine=None`` guess reads/writes a Delta table
# natively (DeltaFolder); at or above it the SQL warehouse parallelises the
# scan/commit better. Matches Delta's default target file size.
_NATIVE_DELTA_MAX_BYTES: int = 128 * 1024 * 1024


def _coerce_tag_str(value: Any) -> str:
    """Coerce a tag key/value to a UTF-8 string.

    PyArrow stores schema/field metadata as ``bytes``, so the
    auto-tags propagated from :data:`yggdrasil.data.Schema` flow
    through here as ``b"primary_key"`` / ``b"true"``. ``str(b"x")``
    would render the literal ``"b'x'"`` — what the Databricks API
    actually receives — so decode bytes-shaped tags before forwarding.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def _resolve_table_operation(
    operation: "TableOperation | ModeLike | None",
    table_type: "TableType | None",
) -> TableOperation:
    """Resolve a user-facing operation hint into a UC :class:`TableOperation`.

    ``None`` defaults to READ for managed tables, READ_WRITE for
    external. A :class:`TableOperation` passes through. A
    :class:`Mode` / mode-like string is mapped read-only-or-not, then
    collapsed to READ for managed tables (UC won't vend write creds
    for those).
    """
    if isinstance(operation, TableOperation):
        op = operation
    elif operation is None:
        op = (
            TableOperation.READ if table_type == TableType.MANAGED
            else TableOperation.READ_WRITE
        )
    else:
        mode = Mode.from_(operation, default=Mode.AUTO)
        op = (
            TableOperation.READ if mode in _READ_ONLY_MODES
            else TableOperation.READ_WRITE
        )

    if op == TableOperation.READ_WRITE and table_type == TableType.MANAGED:
        op = TableOperation.READ
    return op

__all__ = [
    "Table",
    "YGG_SCHEMA_FIELD_PREFIX",
    "YGG_SCHEMA_FIELD_SUFFIX",
]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")

# URL path / free-text → identifier: collapse anything outside ``[0-9A-Za-z]``
# to ``_``. Compiled once so ``Table.safe_name`` is a single ``re.sub`` call.
_PATH_TO_IDENT_RE: re.Pattern[str] = re.compile(r"[^0-9A-Za-z]+")


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


INFOS_TTL: float = 300.0

# The insert DML generator lives in one place — :mod:`insert`. The three sync
# paths (arrow / spark / sql) build a source reference and feed it through the
# same builders re-exported here; they differ only in how the *source* is
# prepared:
#
#   arrow → stage Parquet to a UC Volume; reference it via ``{__tmpsrc__}``
#   spark → register a temp view; reference it by quoted name
#   sql   → wrap caller's query + CAST projection
#
# Retry semantics: caller-supplied ``retry`` (a ``WaitingConfig`` arg) is
# applied only to DML statements (INSERT/MERGE/DELETE/UPDATE).
# TRUNCATE/OPTIMIZE/VACUUM stay non-retryable on purpose: re-running TRUNCATE
# after a successful INSERT is dangerous, and OPTIMIZE/VACUUM are best-effort.
from yggdrasil.databricks.table.insert import (  # noqa: E402
    DatabricksTableInsert,
    _append_maintenance_statements,
    _build_dml_statements,
    _build_where_predicates,
)


# ---------------------------------------------------------------------------
# DML retry / execution helpers — kept local to the table layer.
# ---------------------------------------------------------------------------


def _resolve_retry(retry: Optional[WaitingConfigArg]) -> Optional[WaitingConfig]:
    """Normalize a caller-supplied retry arg to a :class:`WaitingConfig`.

    ``None`` and ``False`` both disable explicit pre-installation — the
    statement-level auto-promote on transient failures still runs.  Any
    other value is coerced through :meth:`WaitingConfig.from_`.
    """
    if retry is None or retry is False:
        return None
    return WaitingConfig.from_(retry)


def _execute_dml(
    sql_engine: "SQLEngine",
    *,
    statements: list,
    wait: WaitingConfigArg,
    raise_error: bool,
    engine: Literal["api", "spark"],
):
    """Submit DML statements through *sql_engine* and surface failures.

    Replaces the legacy MERGE-fallback funnel: there's no fallback
    factory, no per-batch retry shuffle, no auto-promote dance —
    statement-level retry policies (set by the caller via
    :class:`WaitingConfig`) still fire inside
    :class:`SparkPreparedStatement` / :class:`WarehousePreparedStatement`,
    but the table layer no longer second-guesses them.

    On a failed batch we route through :meth:`StatementBatch.retry`
    rather than :meth:`raise_for_status` so a transient Delta
    concurrent-append (a race between sibling MERGE / DELETE + INSERT
    writers on overlapping keys) gets auto-promoted and retried
    instead of bubbling straight up.  Non-transient failures still
    surface — ``batch.retry`` re-raises through ``raise_for_status``
    once the budget is exhausted or the failure isn't retryable.
    """
    batch = sql_engine.execute_many(
        statements, wait=wait, raise_error=False, engine=engine,
    )
    if raise_error and batch.failed:
        batch.retry(wait=wait, raise_error=True)
    elif raise_error:
        batch.raise_for_status()
    return batch


def _coalesce_predicate(
    cast_options: "CastOptions | None",
    predicate: "Predicate | None",
) -> "CastOptions":
    """Fold *predicate* into :attr:`CastOptions.predicate`.

    Used at insert-method boundaries so callers can pass a top-level
    ``predicate=`` kwarg without juggling :class:`CastOptions`
    manually. When both the kwarg and ``cast_options.predicate`` carry
    a value, they're combined with ``&`` (logical AND) so the
    downstream SQL prune and source-row filter both see the merged
    expression.
    """
    opts = CastOptions.check(options=cast_options)
    if predicate is None:
        return opts
    if opts.predicate is None:
        return opts.copy(predicate=predicate)
    return opts.copy(predicate=opts.predicate & predicate)


def _spark_filter_existing_keys(
    *,
    session: Any,
    data_df: Any,
    target_location: str,
    match_by: list[str],
):
    """Drop rows from *data_df* whose ``match_by`` tuple already exists in target.

    The Spark fast path for keyed APPEND. Reads only the
    ``match_by`` columns from the target via
    ``session.table(target_location).select(*match_by).distinct()``
    and left-anti-joins them against the incoming DataFrame.
    Catalyst pushes the join down to the Delta files, so the
    target side reads only its key columns — much cheaper than
    the SQL ``NOT EXISTS`` shape used on the warehouse path.

    Returns a tuple ``(filtered_df, ok)``:

    * ``ok=True`` — the anti-join succeeded; ``filtered_df`` is
      the survivor DataFrame ready for a plain INSERT.
    * ``ok=False`` — target doesn't exist yet (first write) or the
      session can't see it; caller falls through to the SQL
      ``NOT EXISTS`` path which handles empty / missing targets.
    """
    try:
        target_df = session.table(target_location)
        key_df = target_df.select(*match_by).distinct()
        return data_df.join(key_df, list(match_by), "left_anti"), True
    except Exception:
        return data_df, False


def _delta_conf_for(
    overwrite_schema: bool | None,
    spark_options: Optional[Dict[str, Any]],
) -> dict[str, str]:
    """Translate caller-facing knobs into Spark session conf keys."""
    out: dict[str, str] = {}
    if overwrite_schema or (spark_options and spark_options.get("overwriteSchema")):
        out["spark.databricks.delta.schema.autoMerge.enabled"] = "true"
    return out


# Mapping from yggdrasil-recognised type-id keywords to UC SDK
# ``ColumnTypeName`` enum entries.  Used by :meth:`Table.api_create` to
# build ``ColumnInfo`` objects from a ``DataSchema``.
_DDL_TO_COLUMN_TYPE_NAME: dict[str, ColumnTypeName] = {
    "BIGINT": ColumnTypeName.LONG,
    "LONG": ColumnTypeName.LONG,
    "INT": ColumnTypeName.INT,
    "INTEGER": ColumnTypeName.INT,
    "SMALLINT": ColumnTypeName.SHORT,
    "SHORT": ColumnTypeName.SHORT,
    "TINYINT": ColumnTypeName.BYTE,
    "BYTE": ColumnTypeName.BYTE,
    "FLOAT": ColumnTypeName.FLOAT,
    "DOUBLE": ColumnTypeName.DOUBLE,
    "DECIMAL": ColumnTypeName.DECIMAL,
    "BOOLEAN": ColumnTypeName.BOOLEAN,
    "BOOL": ColumnTypeName.BOOLEAN,
    "STRING": ColumnTypeName.STRING,
    "BINARY": ColumnTypeName.BINARY,
    "DATE": ColumnTypeName.DATE,
    "TIMESTAMP": ColumnTypeName.TIMESTAMP,
    "TIMESTAMP_NTZ": ColumnTypeName.TIMESTAMP_NTZ,
    "INTERVAL": ColumnTypeName.INTERVAL,
    "ARRAY": ColumnTypeName.ARRAY,
    "MAP": ColumnTypeName.MAP,
    "STRUCT": ColumnTypeName.STRUCT,
    "VARIANT": ColumnTypeName.VARIANT,
    "NULL": ColumnTypeName.NULL,
}


def _column_type_name_from_ddl(ddl: str) -> ColumnTypeName:
    """Pick the UC ``ColumnTypeName`` enum for a Databricks DDL fragment."""
    head = ddl.strip().split("(", 1)[0].split("<", 1)[0].strip().upper()
    return _DDL_TO_COLUMN_TYPE_NAME.get(head, ColumnTypeName.STRING)


# ---------------------------------------------------------------------------
# ``ygg.schema[<field_name>]`` TBLPROPERTIES — shared by sql_create and
# api_create. Brackets wrap the field name so identifiers containing
# ``.`` don't collide with the property-namespace separator
# (``ygg.schema.user.first_name`` would otherwise be ambiguous between
# a field named ``user.first_name`` and a nested ``user`` field with a
# ``first_name`` child). Everything else a reader could want
# (table_type, storage_location, partition/cluster/primary keys,
# created_at, data_source_format) is already first-class on UC's
# ``TableInfo`` — re-stamping it on TBLPROPERTIES is dead weight.
# ---------------------------------------------------------------------------

YGG_SCHEMA_FIELD_PREFIX = "ygg.schema["
YGG_SCHEMA_FIELD_SUFFIX = "]"


def _sql_literal(value: Any) -> "str | None":
    """Render *value* as a SQL literal for an ``IN`` list, or ``None`` when it
    can't be safely rendered (``NULL`` — which ``IN`` wouldn't match anyway — or
    an unsupported type). Used to inline partition values into a MERGE filter."""
    import datetime
    import decimal

    if value is None:
        return None
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (datetime.date, datetime.datetime)):
        return "'" + value.isoformat() + "'"
    return None


def _ygg_schema_key(name: str) -> str:
    """Build the ``ygg.schema[<name>]`` TBLPROPERTIES key for a field."""
    return f"{YGG_SCHEMA_FIELD_PREFIX}{name}{YGG_SCHEMA_FIELD_SUFFIX}"


def _build_ygg_properties(schema_info: DataSchema) -> dict[str, str]:
    """Build the ``ygg.schema[<field>]`` TBLPROPERTIES yggdrasil stamps on create.

    Emitted by both create paths (:meth:`Table.sql_create` and
    :meth:`Table.api_create`) so the two surfaces stay symmetric.

    Top-level data fields are dumped one-per-property under
    ``ygg.schema[<field_name>]`` (each value is a JSON document for
    that field) rather than as a single ``ygg.schema_json`` blob.
    Per-field keys keep individual TBLPROPERTIES values comfortably
    under Databricks' per-property size budget on wide schemas, and
    let readers fetch only the columns they care about. Constraint-only
    fields (FK/CHECK rows on ``schema.constraints``) are skipped: they're
    applied via the SDK constraints API and aren't columns the table
    actually carries.
    """
    props: dict[str, str] = {}
    seen: set[str] = set()
    for f in schema_info.children:
        if getattr(f, "constraint_key", False):
            continue
        name = f.name
        # Defensive de-dup: schemas constructed by hand can repeat names;
        # later definitions would silently shadow earlier ones in a dict.
        if not name or name in seen:
            continue
        seen.add(name)
        props[_ygg_schema_key(name)] = f.to_json(to_bytes=False)
    return props


class TableProperties(MutableMapping):
    """Live, mutable view of a table's Unity Catalog ``TBLPROPERTIES``.

    A ``dict``-like façade bound to a :class:`Table`. Reads resolve the
    table's cached :attr:`Table.infos` (a remote fetch only when the cache is
    cold/stale); writes issue ``ALTER TABLE|VIEW … SET/UNSET TBLPROPERTIES``
    immediately so the catalog is always the source of truth — there's no
    local copy to drift.

    Every mutation diffs against the current value first, so a useless remote
    call is skipped when:

    - assigning a key the value it already holds (``props['k'] = 'v'`` where
      ``props['k'] == 'v'``), and
    - :meth:`update` is handed only no-op pairs (it batches the *changed*
      keys into a single ``SET TBLPROPERTIES`` and does nothing if none
      changed).

    Deleting an absent key raises ``KeyError`` without a round trip.
    """

    __slots__ = ("_table",)

    def __init__(self, table: "Table") -> None:
        self._table = table

    def _current(self) -> Dict[str, str]:
        """A snapshot copy of the catalog's current properties."""
        return dict(self._table.infos.properties or {})

    def _keyword(self) -> str:
        """``VIEW`` for view-shaped securables, else ``TABLE`` — for the DDL."""
        return "VIEW" if self._table.infos.table_type in _VIEW_TABLE_TYPES else "TABLE"

    def _set(self, items: Dict[str, str]) -> None:
        assignments = ", ".join(
            f"'{escape_sql_string(k)}' = '{escape_sql_string(v)}'"
            for k, v in items.items()
        )
        self._table.sql.execute(
            f"ALTER {self._keyword()} {self._table.full_name(safe=True)} "
            f"SET TBLPROPERTIES ({assignments})",
            wait=True,
        )
        self._table.invalidate_singleton(remove_global=True)

    # ── MutableMapping read protocol ─────────────────────────────────────
    def __getitem__(self, key: str) -> str:
        return self._current()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())

    def __contains__(self, key: object) -> bool:
        return key in self._current()

    # ── mutation — value-diff guarded ────────────────────────────────────
    def __setitem__(self, key: str, value: Any) -> None:
        value = _coerce_tag_str(value)
        if self._current().get(key) == value:
            return  # unchanged — don't pay for a remote ALTER
        self._set({key: value})

    def __delitem__(self, key: str) -> None:
        if key not in self._current():
            raise KeyError(key)
        self._table.sql.execute(
            f"ALTER {self._keyword()} {self._table.full_name(safe=True)} "
            f"UNSET TBLPROPERTIES IF EXISTS ('{escape_sql_string(key)}')",
            wait=True,
        )
        self._table.invalidate_singleton(remove_global=True)

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Batch-apply only the keys whose value actually changes.

        Coerces ``args``/``kwargs`` like ``dict.update``, drops keys already
        at the requested value, and emits a single ``SET TBLPROPERTIES`` for
        whatever remains (nothing at all when every pair is a no-op).
        """
        incoming: Dict[str, Any] = {}
        incoming.update(*args, **kwargs)
        current = self._current()
        changed = {
            k: _coerce_tag_str(v)
            for k, v in incoming.items()
            if current.get(k) != _coerce_tag_str(v)
        }
        if changed:
            self._set(changed)

    def __repr__(self) -> str:
        return f"TableProperties({self._current()!r})"


# ===========================================================================
# Table — per-table resource
# ===========================================================================

class Table(DatabricksPath):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers.

    Registers under :attr:`Scheme.DATABRICKS_TABLE` (``dbfs+table://``)
    so a URL of the shape
    ``dbfs+table://[creds@]host/<catalog>/<schema>/<table>?…`` round-trips
    a Table through :meth:`from_url` / :meth:`to_url`. Reads and writes
    flow through the active :class:`SQLEngine` via the existing
    :class:`Tabular` hooks (``_read_arrow_batches`` /
    ``_write_arrow_batches``); the byte-level :class:`Holder`
    primitives are intentionally not implemented because a SQL table
    is not a positional byte buffer — callers should use the Tabular
    surface (``read_arrow_table`` / ``write_arrow_table`` / …).

    Identity is ``(client, catalog_name, schema_name, table_name)``:
    two callers asking for the same fully-qualified table under the
    same client collapse onto one instance via the :class:`Singleton`
    cache, so the cached :class:`TableInfo` / columns / staging
    volume slot are shared across views into the same UC resource.
    """

    NAMESPACE_PREFIX: ClassVar[str] = "/Tables/"
    _INSTANCES: ClassVar = Singleton._INSTANCES.__class__(default_ttl=None)
    _SINGLETON_TTL: ClassVar[Any] = None

    @classmethod
    def _singleton_key(
        cls,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        **_kwargs: Any,
    ) -> Any:
        # Key on the bound :class:`DatabricksClient` *instance* + the
        # three-part name. Same convention as :class:`Catalog` /
        # :class:`Schema`. ``safe_table_name`` is applied here so two
        # callers with semantically-equivalent but textually-distinct
        # names (long names, suffix-trimmed forms) collapse correctly.
        client = None
        try:
            client = service.client if service is not None else None
        except Exception:
            client = None
        if catalog_name is None and service is not None:
            catalog_name = getattr(service, "catalog_name", None)
        if schema_name is None and service is not None:
            schema_name = getattr(service, "schema_name", None)
        return (cls, client, catalog_name, schema_name, safe_table_name(table_name))

    def __new__(
        cls,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        *,
        singleton_ttl: "int | None" = ...,
        **kwargs: Any,
    ):
        # Mirror :class:`Catalog` / :class:`Schema`'s opt-in cache
        # contract: per-call ``singleton_ttl`` overrides
        # ``_SINGLETON_TTL``; ``...`` on both sides means "don't
        # register" and every call allocates a fresh instance. Cache
        # lookup runs BEFORE the :class:`RemotePath` /
        # :class:`Holder` construction chain so a hit skips
        # allocation entirely; ``object.__new__`` keeps the MRO's
        # :class:`Singleton.__new__` from re-keying with empty args.
        if singleton_ttl is ...:
            singleton_ttl = cls._SINGLETON_TTL

        def _allocate() -> "Table":
            return object.__new__(cls)

        if singleton_ttl is ...:
            return _allocate()

        key = cls._singleton_key(
            service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )
        ttl_arg = (
            float(singleton_ttl)
            if isinstance(singleton_ttl, int) and not isinstance(singleton_ttl, bool)
            else singleton_ttl
        )

        def _build() -> "Table":
            inst = _allocate()
            try:
                object.__setattr__(inst, "_singleton_key_", key)
            except AttributeError:
                pass
            return inst

        return cls._INSTANCES.get_or_set(key, _build, ttl=ttl_arg)

    @property
    def parent(self):
        return self.schema

    @property
    def parents(self) -> "Iterator[DatabricksPath]":
        yield self.schema
        yield self.catalog

    def _stat_uncached(self) -> IOStats:
        infos = self.read_infos(default=None)
        kind = IOKind.MISSING if infos is None else IOKind.DIRECTORY

        return IOStats(
            kind=kind,
            media_type=MediaTypes.DATABRICKS_UNITY_CATALOG_TABLE
        )

    def _from_url(self, url: URL) -> "DatabricksPath":
        parts = url.parts
        n = len(parts)

        if n <= 1:
            return self
        elif n == 2:
            # /<catalog>
            return self.catalog
        elif n == 3:
            # /<catalog>/<schema>
            return self.schema
        elif n == 4:
            # /<catalog>/<schema>/<table> — this table itself
            return self
        else:
            raise ValueError(
                f"URL {url} has too many parts to resolve against a Table "
                f"(got {n}, expected 1-4)."
            )

    def _mkdir(self, parents: bool, exist_ok: bool) -> None:
        del parents, exist_ok
        if not self.exists():
            raise NotImplementedError("Table is a read-only resource")

    def _ls(
        self,
        recursive: bool = False,
        *,
        singleton_ttl: Any = False,
    ) -> Iterator["Path"]:
        del recursive, singleton_ttl
        return iter(())

    def _remove_file(self, missing_ok: bool, wait: WaitingConfig) -> None:
        self.delete(wait=wait, missing_ok=missing_ok)

    def _remove_dir(self, recursive: bool, missing_ok: bool, wait: WaitingConfig) -> None:
        del recursive
        self.delete(wait=wait, missing_ok=missing_ok)

    def full_path(self) -> str:
        return f"{self.NAMESPACE_PREFIX}{self.catalog_name}/{self.schema_name}/{self.table_name}"

    scheme: ClassVar[Scheme] = Scheme.DATABRICKS_TABLE

    @classmethod
    def options_class(cls) -> type[CastOptions]:
        return TableOptions

    @classmethod
    def safe_name(cls, raw: str | None) -> str:
        """Build a Unity-Catalog-safe table name from any raw string.

        Centralized "raw string → table name" builder so every caller
        (URL paths, free-text in user code, composed names from upstream
        metadata) lands on the same identifier without duplicating the
        sanitization logic.

        Pipeline:

        1. Lowercase the input, collapse every run of non-alphanumeric
           characters to a single ``_`` (``/``, ``.``, query-string
           punctuation, whitespace, non-ASCII all fold to the same
           separator).
        2. Strip surrounding ``_``; substitute ``"root"`` for the empty
           result so ``"/"`` / ``""`` / ``None`` still yield a legal
           identifier.
        3. Hand off to :func:`safe_table_name` for the 255-char UC
           ceiling — overflow tokens are joined and BLAKE2b-hashed
           into a 32-char suffix so distinct overflows stay distinct.

        When the returned name differs from *raw* (sanitization or
        truncation kicked in), a :class:`logging.WARNING` is emitted
        on this module's logger so the rewrite is visible in the wall
        of logs that any pipeline already collects. An identifier
        that's already safe round-trips silently — no warning churn
        for the steady-state case.
        """
        original = raw or ""
        cleaned = _PATH_TO_IDENT_RE.sub("_", original.lower()).strip("_")
        if not cleaned:
            cleaned = "root"
        name = safe_table_name(cleaned)
        assert name is not None and len(name) <= MAX_TABLE_NAME_LEN, (
            f"Table.safe_name: derived name {name!r} "
            f"({len(name) if name else 0} chars) exceeds Unity Catalog's "
            f"{MAX_TABLE_NAME_LEN}-char limit — safe_table_name contract broken."
        )
        if original and name != original:
            logger.warning(
                "Sanitized table name %r -> %r (reason=%s)",
                original, name,
                "truncated" if len(original) > MAX_TABLE_NAME_LEN else "non-identifier-chars",
            )
        return name

    def __init__(
        self,
        service: "Tables | None" = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        *,
        infos: TableInfo | None = None,
        infos_fetched_at: float | None = None,
        columns: list[Column] | None = None,
        url: URL | None = None,
        temporary: bool = False,
        singleton_ttl: "int | None" = ...,
    ):
        # ``singleton_ttl`` is consumed by ``__new__``; accept it here
        # too so Python's auto-call after ``__new__`` doesn't trip on
        # an unexpected kwarg.
        del singleton_ttl
        # Singleton-cached re-entry: a second ``Table(service=…,
        # catalog_name=…, schema_name=…, table_name=…)`` call returns
        # the live instance via ``__new__``; skip the second pass so
        # the cached ``_infos`` / columns / staging volume don't get
        # reset under the caller.
        if getattr(self, "_initialized", False):
            return

        if service is None:
            from .tables import Tables
            service = Tables.current()

        # Build a canonical ``dbfs+table://...`` URL so :class:`Holder`
        # has a real URL to bind ``self._url`` to. The host comes from
        # the underlying client (when available) so the URL alone
        # round-trips through :meth:`from_url`.
        if url is None:
            host = ""
            try:
                base = service.client.base_url
                host = base.host or ""
            except Exception:
                host = ""
            path_parts = [
                p for p in (catalog_name, schema_name, table_name) if p
            ]
            url = URL(
                scheme=type(self).scheme.value,
                host=host,
                path="/" + "/".join(path_parts) if path_parts else "/",
            )

        super().__init__(service=service, url=url, temporary=temporary)
        self.catalog_name = catalog_name
        self.schema_name = schema_name
        # Unity Catalog caps identifiers at 255 chars. Normalize once at the
        # boundary so every downstream SQL/URL/cache key sees the safe form.
        self.table_name = safe_table_name(table_name)
        self._infos = infos
        self._infos_fetched_at = infos_fetched_at
        self._columns = columns
        self._staging_volume: Volume | None = None
        self._initialized = True

    # ------------------------------------
    # Tabular
    # ------------------------------------

    @classmethod
    def default_media_type(cls) -> MimeType:
        return MimeTypes.DATABRICKS_UNITY_CATALOG_TABLE

    # ------------------------------------------------------------------
    # Holder primitives — Table is *logical*, not byte-shaped.
    # The Tabular surface (``read_arrow_table`` / ``write_arrow_table``
    # / …) is the supported way to move rows; the byte-level
    # primitives raise so a misuse fails loudly with a hint at the
    # right surface.
    # ------------------------------------------------------------------

    @property
    def is_memory(self) -> bool:
        return False

    @property
    def is_local_path(self) -> bool:
        return False

    @property
    def is_remote_path(self) -> bool:
        # The table is *logical* — neither local nor remote in the
        # filesystem sense. The Databricks-side identity lives in the
        # warehouse, not at a file URL we can hand to ``is_remote_path``.
        return False

    @property
    def size(self) -> int:
        # A SQL table has no positional byte size. Return 0 so
        # ``IO(holder=table)``-style code sees an empty buffer
        # instead of crashing; the byte primitives still raise on
        # actual read/write attempts.
        return 0

    def stat(self) -> IOStats:
        return self._stat()

    def _stat(self) -> IOStats:
        return IOStats(
            size=0, mtime=0.0, kind=IOKind.MISSING,
            media_type=type(self).default_media_type(),
        )

    def _read_mv(self, n: int, pos: int) -> memoryview:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog table, "
            f"not a positional byte buffer. Use the Tabular surface "
            f"(``read_arrow_table()``, ``read_pandas_frame()``, etc.) "
            f"to materialize rows."
        )

    def _write_mv(self, data: memoryview, pos: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__} is a logical Unity Catalog table, "
            f"not a positional byte buffer. Use ``insert(...)`` / "
            f"``write_arrow_table(...)`` to write rows."
        )

    def reserve(self, n: int) -> None:
        # No capacity layer to pre-grow; honor the contract by
        # rejecting only nonsense inputs.
        if n < 0:
            raise ValueError(f"reserve size must be >= 0, got {n!r}")

    def truncate(self, n: int) -> int:
        raise NotImplementedError(
            f"{type(self).__name__}.truncate is byte-shaped and does "
            f"not apply to a SQL table. Use ``insert(..., mode='overwrite')`` "
            f"or ``execute('TRUNCATE TABLE ...')`` for the SQL equivalent."
        )

    def _clear(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__}._clear is byte-shaped and does "
            f"not apply to a SQL table. Use ``execute('DROP TABLE ...')`` "
            f"or ``insert(..., mode='overwrite')`` for the SQL equivalent."
        )

    # ------------------------------------------------------------------
    # URLBased — ``dbfs+table://[creds@]host/<cat>/<sch>/<tbl>``
    # ------------------------------------------------------------------

    @classmethod
    def from_url(cls, url: "URL | str", **kwargs: Any) -> "Table":
        """Build a :class:`Table` from a ``dbfs+table://...`` URL.

        Reads the catalog / schema / table from the URL path
        (``/catalog/schema/table``) and, when ``service`` is not
        passed in *kwargs*, infers the underlying
        :class:`DatabricksClient` from the URL via
        :meth:`DatabricksClient.from_url` — userinfo carries the PAT
        / OAuth secret, the URL host is the workspace, and remaining
        query items are forwarded as DatabricksClient init kwargs.
        Then a :class:`Tables` service is built on top of that
        client.

        Caller-supplied ``service`` / ``catalog_name`` /
        ``schema_name`` / ``table_name`` overrides anything the URL
        provided.
        """
        u = URL.from_(url)
        path = (u.path or "/").strip("/")
        parts = path.split("/") if path else []
        cat = parts[0] if len(parts) > 0 else None
        sch = parts[1] if len(parts) > 1 else None
        tbl = parts[2] if len(parts) > 2 else None

        service = kwargs.pop("service", None)
        if service is None:
            client = kwargs.pop("client", None)
            if client is None:
                # Coerce the URL through ``DatabricksClient.from_url``
                # so the same userinfo / host / query knobs that work
                # on ``dbks://`` work on ``dbfs+table://`` too.
                client = DatabricksClient.from_url(u)
            from .tables import Tables
            service = Tables(client=client)
        else:
            kwargs.pop("client", None)

        return cls(
            service=service,
            catalog_name=kwargs.pop("catalog_name", None) or cat,
            schema_name=kwargs.pop("schema_name", None) or sch,
            table_name=kwargs.pop("table_name", None) or tbl,
            **kwargs,
        )

    def to_url(self) -> URL:
        """Render this Table as a ``dbfs+table://...`` URL.

        Layers the table's ``/catalog/schema/table`` path on top of
        :meth:`DatabricksClient.to_url` so credentials / profile /
        account_id ride along the same URL — symmetric with
        :meth:`from_url`.
        """
        try:
            client_url = self.client.to_url(scheme=type(self).scheme.value)
        except Exception:
            # No usable client — fall back to a bare logical URL.
            client_url = URL(scheme=type(self).scheme.value)
        path_parts = [
            p for p in (self.catalog_name, self.schema_name, self.table_name) if p
        ]
        return client_url.with_path("/" + "/".join(path_parts) if path_parts else "/")

    def _options_to_sql(self, options: CastOptions):
        safe_char = "`"
        names = ",".join(
            safe_char + name + safe_char
            for name in options.column_names
        )
        query = f"SELECT {names} FROM {self.full_name(safe=True)}"

        if options.predicate is not None:
            query += (
                f" WHERE "
                f"{expr_to_sql(options.predicate, dialect=Dialect.DATABRICKS)}"
            )

        if options.row_limit:
            query += f" LIMIT {options.row_limit}"

        return query

    def _delta_capable(self, *, write: bool) -> bool:
        """True when a native :meth:`delta` DeltaFolder read (or write) is
        possible: a Delta table with resolvable storage — and, for a write, an
        *external* one (UC vends READ_WRITE creds only for external; a managed
        commit would 403)."""
        try:
            infos = self.infos
        except Exception:
            return False
        if (
            infos.table_type in _VIEW_TABLE_TYPES
            or infos.data_source_format != DataSourceFormat.DELTA
            or not infos.storage_location
        ):
            return False
        if write and infos.table_type != TableType.EXTERNAL:
            return False
        return True

    def _resolve_engine(self, options: CastOptions, *, write: bool) -> "EngineType":
        """Resolve the compute engine for this read / write.

        ``options.engine`` selects explicitly (an :class:`EngineType` or alias);
        ``YGGDRASIL`` on a table that can't take the native path degrades to the
        warehouse rather than erroring. ``None`` **guesses best**: an active
        Spark session → ``SPARK``; else a small Delta table
        (``< _NATIVE_DELTA_MAX_BYTES`` on disk) → ``YGGDRASIL``; else →
        ``DATABRICKS_SQL_WAREHOUSE`` (it parallelises big scans/writes better).
        """
        engine = EngineType.from_(getattr(options, "engine", None))
        if engine is not None:
            if engine == EngineType.YGGDRASIL and not self._delta_capable(write=write):
                return EngineType.DATABRICKS_SQL_WAREHOUSE
            return engine

        if self._has_active_spark(options):
            return EngineType.SPARK
        if self._delta_capable(write=write):
            size = self._delta_total_bytes()
            if size is not None and size < _NATIVE_DELTA_MAX_BYTES:
                return EngineType.YGGDRASIL
        return EngineType.DATABRICKS_SQL_WAREHOUSE

    @staticmethod
    def _has_active_spark(options: CastOptions) -> bool:
        """True when a Spark session is bound on *options* or active in-process."""
        if getattr(options, "spark_session", None) is not None:
            return True
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession() is not None
        except Exception:
            return False

    def _delta_total_bytes(self) -> "int | None":
        """The Delta table's on-disk byte size (sum of active ``AddFile``
        sizes from the ``_delta_log``), or ``None`` if it can't be resolved.

        Sizes with *read* credentials — this only informs the routing guess,
        so it must not need write access."""
        try:
            return self.delta(write=False).snapshot().total_bytes
        except Exception:
            return None

    def _native_delta_folder(self, *, write: bool) -> "DeltaFolder | None":
        """Build the :meth:`delta` DeltaFolder for a read or a write and verify
        the actual access, so any failure surfaces here — *before* a write
        consumes its batch stream.

        The credential scope follows the operation (``write=False`` → ``READ``,
        ``write=True`` → ``READ_WRITE``) because a principal can hold read but
        not write on a table. Verification:

        - **read** — read the ``_delta_log`` snapshot (a GetObject), which also
          forces the credential vend;
        - **write** — additionally touch + remove a tiny probe object under the
          table root (a real PutObject). UC sometimes vends ``READ_WRITE``
          credentials whose underlying S3 session policy still *denies* writes,
          and reading the log only exercises GetObject — so the probe is the
          only way to learn the write would 403.

        Returns ``None`` on any failure, so the caller transparently falls back
        to Databricks (the SQL warehouse)."""
        try:
            folder = self.delta(write=write)
            folder.snapshot(fresh=True)  # vend + verify READ (GetObject)
            if write:
                # Verify PutObject actually works. ``_ygg/`` is outside Delta's
                # ``_delta_log`` / data-file scan, so the transient marker is
                # invisible to readers; remove it best-effort afterwards.
                probe = folder.path / ("_ygg/.write_probe_%s" % uuid.uuid4().hex)
                probe.write_bytes(b"")
                try:
                    probe.remove(missing_ok=True)
                except Exception:
                    pass
            return folder
        except Exception as exc:  # noqa: BLE001 — any vend/IO failure → fall back
            logger.warning(
                "Native Delta %s path unavailable for %r (%s: %s); "
                "falling back to Databricks.",
                "write" if write else "read", self, type(exc).__name__, exc,
            )
            return None

    def _read_arrow_batches(self, options: CastOptions) -> Iterator[pa.RecordBatch]:
        engine = self._resolve_engine(options, write=False)

        # YGGDRASIL → read straight off the ``_delta_log`` + parquet via our
        # DeltaFolder. ``_native_delta_folder`` probes UC credentials first,
        # returning None (and falling through to the warehouse) on a vend error.
        if engine == EngineType.YGGDRASIL:
            folder = self._native_delta_folder(write=False)
            if folder is not None:
                yield from folder._read_arrow_batches(folder.check_options(options))
                return
            engine = EngineType.DATABRICKS_SQL_WAREHOUSE

        options = options.with_source(source=self.collect_schema())
        query = self._options_to_sql(options)
        sql_engine = "spark" if engine == EngineType.SPARK else "api"

        try:
            execution = self.sql.execute(query, engine=sql_engine)
        except Exception:
            if not self.exists() and options.target:
                self.create(options.target)
                s: pa.Schema = options.target.to_arrow_schema()
                yield pa.RecordBatch.from_pylist([], schema=s)
                return
            else:
                raise

        for batch in execution.read_arrow_batches(options=options):
            yield batch

    def _write_arrow_batches(
        self,
        batches: Iterable[pa.RecordBatch],
        options: CastOptions
    ) -> None:
        engine = self._resolve_engine(options, write=True)

        # YGGDRASIL → commit straight to the ``_delta_log`` via our DeltaFolder.
        # The credential probe runs *before* ``batches`` is consumed, so a vend
        # failure falls back to the SQL insert below with the stream intact.
        if engine == EngineType.YGGDRASIL:
            folder = self._native_delta_folder(write=True)
            if folder is not None:
                folder.write_arrow_batches(batches, options=options)
                return
            engine = EngineType.DATABRICKS_SQL_WAREHOUSE

        options = options.with_target(self.collect_schema(options))

        # SPARK vs the SQL warehouse, via the staged-Parquet insert path which
        # takes an explicit ``engine`` (so the choice is honoured even when a
        # Spark session is reachable in-process).
        return self.arrow_insert(
            batches,
            engine="spark" if engine == EngineType.SPARK else "api",
            mode=options.mode,
            match_by=options.match_by_keys,
            update_column_names=options.update_column_names,
            wait=options.wait,
            zorder_by=options.zorder_by,
            optimize_after_merge=options.optimize_after_merge,
            vacuum_hours=options.vacuum_hours,
            predicate=options.predicate,
            retry=options.retry,
            return_data=options.return_data,
            safe_merge=options.safe_merge,
        )

    def _delete(
        self,
        predicate: "Any" = None,
        *,
        wait: "Any" = True,
        missing_ok: bool = False,
        delete_staging: bool = True,
        **kwargs: "Any",
    ) -> int:
        """Row-level delete that avoids rewriting the (potentially huge) table.

        * **No predicate** → drop the table through the Unity Catalog tables
          API (:meth:`delete`). Emptying the whole table needs no row
          filtering, so there's no reason to spin up a SQL warehouse.
        * **With a predicate** → issue a server-side ``DELETE FROM <t> WHERE …``
          so the warehouse does the work in place, instead of streaming every
          batch back to the client to filter and rewrite (the generic
          :meth:`~yggdrasil.io.tabular.base.Tabular._delete_rewrite` path).

        Returns ``0`` — the affected-row count isn't surfaced by the
        execution result; the public :meth:`delete` returns the table itself.
        """
        if predicate is None:
            # Whole-table removal — drop the asset through the UC tables
            # API, no SQL warehouse. Implemented inline (not via
            # ``self.delete``): the public ``delete`` dispatches back here,
            # so delegating would recurse infinitely.
            #
            # ``delete_staging=False`` keeps the staging volume around for
            # internal drop-and-recreate flows (OVERWRITE) where the very
            # next step uploads a fresh parquet to the same volume — the
            # background ``VolumesAPI.delete`` would otherwise race the
            # upload and surface as PATH_NOT_FOUND on the warehouse INSERT.
            uc = self.client.workspace_client().tables
            logger.debug(
                "Deleting table %r (wait=%s, delete_staging=%s)",
                self, bool(wait), delete_staging,
            )
            if wait:
                try:
                    uc.delete(full_name=self.full_name())
                    if delete_staging and self._staging_volume:
                        self._staging_volume.delete(wait=False)
                except DatabricksError:
                    if not missing_ok:
                        raise
            else:
                Job.make(
                    self._delete,
                    wait=True,
                    missing_ok=missing_ok,
                    delete_staging=delete_staging,
                ).fire_and_forget()
            self.invalidate_singleton(remove_global=True)
            logger.info("Deleted table %r", self)
            return 0
        where = predicate if isinstance(predicate, str) else expr_to_sql(
            predicate, dialect=Dialect.DATABRICKS,
        )
        self.sql.execute(f"DELETE FROM {self.full_name(safe=True)} WHERE {where}", wait=wait)
        self.invalidate_singleton(remove_global=True)
        return 0

    def _read_spark_frame(self, options: O) -> "SparkDataFrame":
        options = options.with_source(source=self.collect_schema(options))
        query = self._options_to_sql(options)

        try:
            execution = self.sql.execute(query)
        except Exception:
            if not self.exists() and options.target:
                self.create(options.target)
                s: pa.Schema = options.target.to_spark_schema()
                return options.get_spark_session(
                    create=True
                ).createDataFrame([], schema=s)
            else:
                raise

        return execution.read_spark_frame(options)

    def _write_spark_frame(
        self,
        frame: "SparkDataFrame",
        options: O,
    ) -> None:
        return self.spark_insert(
            frame,
            mode=options.mode,
            match_by=options.match_by_keys,
            wait=options.wait,
            return_data=options.return_data,
            safe_merge=options.safe_merge,
            spark_session=getattr(options, "spark_session", None),
        )

    # Properties
    
    @property
    def name(self):
        return self.table_name

    @property
    def explore_url(self) -> URL:
        """Workspace UI deep-link for this table (``/explore/data/...``).

        Mirrors :attr:`Catalog.explore_url` / :attr:`Schema.explore_url`.
        The canonical addressable URL for this table lives on
        :attr:`url` (inherited from :class:`Holder`); ``explore_url``
        is the human-friendly Catalog Explorer link.
        """
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.table_name}")
        )

    @classmethod
    def from_(
        cls,
        obj: Any,
        *,
        media_type: MediaType | None = None,
        default: Any = ...,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        service: "Tables",
        **kwargs,
    ):
        if isinstance(obj, cls):
            return obj

        return cls.from_str(
            location=str(obj) if obj is not None else None,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=service
        )

    @classmethod
    def from_str(
        cls,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        service: "Tables",
    ):
        _, catalog_name, schema_name, table_name = service.parse_check_location_params(
            location=location,
            catalog_name=catalog_name or service.catalog_name,
            schema_name=schema_name or service.schema_name,
            table_name=table_name,
        )

        return Table(
            service=service,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name
        )

    # =========================================================================
    # Convenience shorthand — service delegates
    # =========================================================================

    @property
    def sql(self) -> "SQLEngine":
        return self.client.sql(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name
        )
    
    def execute(
        self,
        statement: str | PreparedStatement,
        *args,
        **kwargs
    ):
        return self.sql.execute(
            statement=statement,
            *args,
            **kwargs
        )

    def lazy(
        self,
        sql: str | PreparedStatement | None = None,
        **kwargs,
    ) -> "Tabular":
        """Return a deferred :class:`Tabular` for *sql* against this table.

        When *sql* is provided, submits the query via
        :attr:`sql.execute` and returns the resulting
        :class:`StatementResult` — itself a :class:`Tabular`. The
        warehouse executes the query eagerly so the result handle is
        ready, but the rows aren't materialised until the caller
        invokes a Tabular hook (``read_arrow_table`` /
        ``read_arrow_batches`` / ``read_pandas_frame`` …)::

            handle = tbl.lazy(sql="SELECT id, val FROM {self} WHERE id > 5")
            arrow = handle.read_arrow_table()

        ``{self}`` in the query string is substituted with the
        backtick-quoted full name of this table — saves the caller
        from concatenating ``tbl.full_name(safe=True)`` into every
        query. When no ``{self}`` placeholder is present, the SQL
        flows through verbatim.

        Calling ``lazy()`` with ``sql=None`` returns the table itself
        (already a :class:`Tabular`) so callers that want to chain on
        the table's own data hand back the same object.
        """
        if sql is None:
            return self
        if isinstance(sql, str) and "{self}" in sql:
            sql = sql.format(self=self.full_name(safe=True))
        return self.sql.execute(statement=sql, **kwargs)
    
    @property
    def catalog(self) -> "UCCatalog":
        """Navigate up to the parent :class:`UCCatalog`.

        Returns the singleton-cached :class:`UCCatalog` for this
        client + catalog name — repeated calls hand back the same
        instance with shared :class:`CatalogInfo` cache.
        """
        from yggdrasil.databricks.catalog.catalog import UCCatalog as _Catalog
        return _Catalog(
            service=self.client.catalogs,
            catalog_name=self.catalog_name,
        )

    @property
    def schema(self) -> "UCSchema":
        """Navigate up to the parent :class:`UCSchema`.

        Returns the singleton-cached :class:`UCSchema` for this
        client + (catalog, schema) — repeated calls hand back the
        same instance with shared :class:`SchemaInfo` cache.
        """
        from yggdrasil.databricks.schema.schema import UCSchema as _Schema
        return _Schema(
            service=self.client.schemas,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    # =========================================================================
    # Identity / repr
    # =========================================================================

    def schema_full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def full_name(self, safe: str | bool | None = None) -> str:
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return (
                f"{q}{self.catalog_name}{q}"
                f".{q}{self.schema_name}{q}"
                f".{q}{self.table_name}{q}"
            )
        return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    def column_full_name(self, column_name: str) -> str:
        """Fully-qualified column name suitable for ``entity_tag_assignments``."""
        return f"{self.full_name()}.{column_name}"

    def __repr__(self) -> str:
        return f"Table({self.url.to_string()!r})"

    def __str__(self):
        return self.full_name(safe=True)

    def __getitem__(self, item: str) -> Column:
        return self.column(item)

    def __setitem__(self, item: str, new_name: str) -> None:
        """``table["old_col"] = "new_col"`` renames a column."""
        self.column(item).rename(new_name)

    def __iter__(self) -> Iterable[Column]:
        """Iterate over the columns of this table."""
        return iter(self.columns)

    # =========================================================================
    # Cache management
    # =========================================================================

    def invalidate_singleton(self, remove_global: bool = False) -> None:
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)
        self._invalidate_entity_tag_cache()
        super().invalidate_singleton(remove_global=remove_global)

    def _invalidate_entity_tag_cache(self) -> None:
        """Drop cached tag lists for this table and every cached column."""
        tags = self.client.entity_tags
        tags.invalidate_cached_tags("tables", self.full_name())
        # Use the still-cached columns list (if any) — refusing to refetch
        # ``infos`` here keeps invalidation cheap and safe inside teardown.
        for col in (self._columns or ()):
            tags.invalidate_cached_tags("columns", self.column_full_name(col.name))

    # =========================================================================
    # Databricks SDK — lazy-loaded properties
    # =========================================================================

    def exists(self) -> bool:
        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def table_id(self) -> str:
        return self.infos.table_id

    @staticmethod
    def _is_fresh(fetched_at: float | None) -> bool:
        if fetched_at is None:
            return False
        return (time.time() - fetched_at) < INFOS_TTL

    def _store_infos(self, infos: TableInfo) -> TableInfo:
        """Populate the ``infos`` + ``columns`` caches."""
        self._infos_fetched_at = time.time()
        self._infos = infos
        self._columns = [
            Column.from_api(table=self, infos=col_info)
            for col_info in (infos.columns or [])
        ]
        # A successful info fetch proves the table exists — seed the stat
        # cache in the same beat (built inline to avoid recursing through
        # ``_stat_uncached`` → ``read_infos`` → here) so a follow-up
        # ``exists`` / ``stat`` reuses it.
        self._persist_stat_cache(
            IOStats(
                kind=IOKind.DIRECTORY,
                media_type=MediaTypes.DATABRICKS_UNITY_CATALOG_TABLE,
            )
        )
        logger.debug(
            "Stored info for table %r (id=%s, columns=%d, type=%s)",
            self, getattr(infos, "table_id", None),
            len(self._columns), getattr(infos, "table_type", None),
        )
        return infos

    def read_infos(self, default: Any = ...):
        """Basic :class:`TableInfo` — TTL-cached."""
        if self._infos is not None and self._is_fresh(self._infos_fetched_at):
            return self._infos

        info = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
            default=None
        )

        if info is None:
            if default is ...:
                raise NotFound(
                    f"Volume {self.full_name(safe=True)} not found"
                )
            return None

        self._store_infos(info)
        return info

    @property
    def infos(self) -> TableInfo:
        """Basic :class:`TableInfo` — TTL-cached."""
        if self._infos is not None and self._is_fresh(self._infos_fetched_at):
            return self._infos

        info = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )
        self._store_infos(info)
        return info

    # =========================================================================
    # View-shaped tables — Unity Catalog stores views in the same ``tables``
    # API as managed/external tables, distinguished by ``table_type``.
    # =========================================================================

    @property
    def table_type(self) -> Optional[TableType]:
        """:class:`TableType` from the cached ``infos``.

        Returns ``None`` when the table hasn't been resolved against
        Unity Catalog yet — the property never triggers a network
        round trip on its own. Callers that need a guaranteed-fresh
        answer should access ``self.infos.table_type`` directly.
        """
        cached = self._infos
        return cached.table_type if cached is not None else None

    @property
    def is_view(self) -> bool:
        """True for ``VIEW`` / ``MATERIALIZED_VIEW`` / ``METRIC_VIEW`` securables.

        Reads the cached :attr:`table_type`; returns ``False`` until
        the table's ``infos`` has been resolved at least once.
        """
        return self.table_type in _VIEW_TABLE_TYPES

    @property
    def is_delta(self) -> bool:
        """True for a Delta-backed table (``USING DELTA``), from cached infos.

        Reads the cached ``infos`` only — never a network round trip; returns
        ``False`` until the table has been resolved at least once. Views are
        never Delta.
        """
        cached = self._infos
        if cached is None or self.is_view:
            return False
        return cached.data_source_format == DataSourceFormat.DELTA

    @property
    def is_materialized_view(self) -> bool:
        return self.table_type == TableType.MATERIALIZED_VIEW

    @property
    def is_metric_view(self) -> bool:
        return self.table_type == TableType.METRIC_VIEW

    @property
    def view_definition(self) -> Optional[str]:
        """The SQL ``SELECT`` text for a view; ``None`` for non-views.

        Reads the cached ``infos``; does not trigger a remote fetch.
        """
        cached = self._infos
        return cached.view_definition if cached is not None else None

    @property
    def view_dependencies(self):
        """Upstream dependencies declared by a view (cached only)."""
        cached = self._infos
        return cached.view_dependencies if cached is not None else None

    @property
    def owner(self) -> Optional[str]:
        """The table's Unity Catalog owner principal (user / group / SP).

        Resolves ``infos`` (a remote read if not cached), mirroring
        :attr:`Catalog.owner` / :attr:`Schema.owner`. Assigning re-owners the
        securable via ``ALTER TABLE|VIEW … OWNER TO``.
        """
        return self.infos.owner

    @owner.setter
    def owner(self, principal: str) -> None:
        if not principal:
            raise ValueError("owner must be a non-empty principal name")
        # ALTER VIEW for view-shaped securables, ALTER TABLE otherwise — resolve
        # ``infos`` so the keyword is correct even on a never-inspected handle.
        keyword = "VIEW" if self.infos.table_type in _VIEW_TABLE_TYPES else "TABLE"
        logger.debug("Re-owning %s %r → %s", keyword, self, principal)
        self.sql.execute(
            f"ALTER {keyword} {self.full_name(safe=True)} "
            f"OWNER TO {quote_principal(principal)}"
        )
        # Drop the cached infos so a follow-up ``owner`` read re-fetches.
        self.invalidate_singleton(remove_global=True)

    @property
    def properties(self) -> TableProperties:
        """Live, mutable view of the table's Unity Catalog ``TBLPROPERTIES``.

        Returns a :class:`TableProperties` (a ``MutableMapping``): reads resolve
        cached :attr:`infos`, while item assignment / deletion / :meth:`dict.update`
        transparently issue ``ALTER … SET/UNSET TBLPROPERTIES`` — skipping the
        remote call whenever the value is already what's requested::

            t.properties["delta.appendOnly"] = "true"   # one ALTER
            t.properties["delta.appendOnly"] = "true"   # no-op, no network
            del t.properties["stale.key"]               # UNSET … IF EXISTS
        """
        return TableProperties(self)

    # ── view name aliases — old ``view_name`` callers stay working ───────────

    @property
    def view_name(self) -> str:
        """Alias for :attr:`table_name` so view-style call sites keep working."""
        return self.table_name

    @view_name.setter
    def view_name(self, value: str) -> None:
        self.table_name = value

    # =========================================================================
    # Entity-tag assignments — delegated to client.entity_tags
    # =========================================================================

    @property
    def tags(self) -> tuple[EntityTagAssignment, ...]:
        """Table-level entity-tag assignments — served from ``client.entity_tags``."""
        return tuple(
            self.client.entity_tags.entity_tags(
                "tables", self.full_name(), default=()
            ) or ()
        )

    @property
    def column_tags(self) -> Mapping[str, tuple[EntityTagAssignment, ...]]:
        """Per-column entity-tag assignments.

        Fan-out is parallelised so wide tables pay one aggregate wall-clock
        round trip rather than N sequential ones; cache hits inside
        ``client.entity_tags`` short-circuit each leg.
        """
        tags = self.client.entity_tags
        full = self.full_name()
        jobs: dict[str, Any] = {}
        for col_info in (self.infos.columns or []):
            col_name = col_info.name
            if not col_name:
                continue
            jobs[col_name] = Job.make(
                tags.entity_tags,
                "columns",
                f"{full}.{col_name}",
                default=(),
            ).fire_and_forget()

        result: dict[str, tuple[Any, ...]] = {}
        for col_name, job in jobs.items():
            assignments = tuple(job.wait() or ())
            if assignments:
                result[col_name] = assignments
        return result

    # =========================================================================
    # Arrow schema introspection
    # =========================================================================

    @property
    def columns(self) -> list[Column]:
        if self._columns is None:
            _ = self.infos  # populates _columns as a side effect
        return self._columns

    def column(
        self,
        name: str,
        safe: bool = False,
        raise_error: bool = True
    ) -> Column:
        columns = self.columns

        for col in columns:
            if col.name == name:
                return col

        if not safe:
            case_folded = name.casefold()
            for col in columns:
                if col.name.casefold() == case_folded:
                    return col

        if raise_error:
            raise ValueError(f"Column {name!r} not found in {self!r}")
        return None

    def _collect_schema(self, options: CastOptions) -> DataSchema:
        """Return the field schema, optionally enriched with UC metadata."""
        logger.debug(
            "Collecting schema for table %r (columns=%d)", self, len(self.columns),
        )
        metadata: dict[bytes, bytes] = {
            b"engine": b"databricks",
            b"catalog_name": self.catalog_name.encode(),
            b"schema_name": self.schema_name.encode(),
            b"table_name": self.table_name.encode(),
        }

        col_tags = self.column_tags

        fields: list[Field] = []
        for column in self.columns:
            base = column.field
            extra_tags: dict[bytes, bytes] = {}

            for assignment in col_tags.get(column.name, ()):
                key = getattr(assignment, "tag_key", None)
                if not key:
                    continue
                value = getattr(assignment, "tag_value", None) or ""
                extra_tags[key.encode("utf-8")] = str(value).encode("utf-8")

            if extra_tags:
                fields.append(
                    base.copy(
                        metadata=dict(base.metadata or {}),
                        tags=extra_tags,
                    )
                )
            else:
                fields.append(base)

        for assignment in self.tags:
            key = getattr(assignment, "tag_key", None)
            if not key:
                continue
            value = getattr(assignment, "tag_value", None) or ""
            metadata[f"tag:{key}".encode("utf-8")] = str(value).encode("utf-8")

        schema = DataSchema.from_fields(fields, metadata=metadata, name=self.table_name, nullable=False)
        self._persist_schema(schema)
        logger.debug(
            "Built schema for table %r (fields=%d, metadata_keys=%d)",
            self, len(fields), len(metadata),
        )
        return schema

    def collect_data_field(self, safe: bool = False) -> Field:
        return self.collect_schema(safe=safe).to_field()

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [c.field.to_arrow_field() for c in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.collect_schema().to_arrow_schema()

    @property
    def arrow_field(self) -> pa.Field:
        return self.collect_data_field().to_arrow_field()

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
    ) -> "Table":
        """Apply table-level tags via the UC ``entity_tag_assignments`` API.

        ``tag_collation`` is accepted for API compatibility and ignored —
        collations only matter for the legacy DDL literal form.
        """
        if not tags:
            return self

        self.client.entity_tags.update_entity_tags(
            tags=tags,
            entity_type="tables",
            entity_name=self.full_name(),
        )
        return self

    def unset_tags(
        self,
        tag_keys: Iterable[str],
        *,
        if_exists: bool = True,
    ) -> "Table":
        """Delete table-level tag assignments by key."""
        self.client.entity_tags.delete_entity_tags(
            entity_type="tables",
            entity_name=self.full_name(),
            tag_keys=tag_keys,
            if_exists=if_exists,
        )
        return self

    # =========================================================================
    # Lifecycle — create / ensure / delete
    # =========================================================================

    def ensure_created(
        self,
        definition: Union[pa.Schema, Any, None],
        *,
        mode: Mode | str | None = None,
        **options
    ) -> "Table":
        return self.create(
            definition=definition,
            mode=mode,
            **options,
        )

    def _columns_service(self) -> "Columns":
        """Columns service scoped to this table's catalog/schema/table defaults."""
        from yggdrasil.databricks.column.columns import Columns

        return Columns(
            client=self.client,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )

    def with_column(
        self,
        column: Field,
        *,
        mode: Mode | str | None = None,
    ):
        return self.with_columns([column], mode=mode)

    def with_columns(
        self,
        columns: Iterable[Field],
        *,
        mode: Mode | str | None = None,
    ):
        mode = Mode.from_(mode, default=Mode.AUTO)
        alter_table = f"ALTER TABLE {self.full_name(safe=True)}"
        update_dtype = mode in (Mode.UPSERT, Mode.MERGE, Mode.OVERWRITE)
        drop_missing = mode == Mode.OVERWRITE

        rename_statements: list[str] = []
        type_statements: list[str] = []
        add_columns: list[str] = []
        matched_existing: set[str] = set()

        for column in columns:
            data_field = Field.from_any(column)
            existing = self.column(name=data_field.name, safe=False, raise_error=False)

            if existing is None:
                add_columns.append(
                    f"`{data_field.name}` {data_field.dtype.to_spark_name()}"
                )
                continue

            matched_existing.add(existing.name)
            current_name = existing.name

            if existing.name != data_field.name:
                rename_statements.append(
                    f"{alter_table} RENAME COLUMN `{existing.name}` "
                    f"TO `{data_field.name}`"
                )
                current_name = data_field.name

            if update_dtype:
                existing_ddl = existing.field.dtype.to_spark_name()
                new_ddl = data_field.dtype.to_spark_name()
                if existing_ddl != new_ddl:
                    type_statements.append(
                        f"{alter_table} ALTER COLUMN `{current_name}` "
                        f"TYPE {new_ddl}"
                    )

        drop_names: list[str] = []
        if drop_missing:
            drop_names = [
                col.name for col in self.columns
                if col.name not in matched_existing
            ]

        add_col_statement: str | None = None
        if add_columns:
            add_col_statement = (
                f"{alter_table} ADD COLUMNS ({', '.join(add_columns)})"
            )

        needs_phase_split = bool(rename_statements and type_statements)

        first_phase: list[str] = []
        second_phase: list[str] = []

        if needs_phase_split:
            first_phase.extend(rename_statements)
        else:
            second_phase.extend(rename_statements)

        second_phase.extend(type_statements)
        if drop_names:
            second_phase.append(
                f"{alter_table} DROP COLUMNS "
                + "(" + ", ".join(f"`{n}`" for n in drop_names) + ")"
            )
        if add_col_statement is not None:
            second_phase.append(add_col_statement)

        executed = False
        if first_phase:
            self.sql.execute_many(first_phase, parallel=True)
            executed = True
        if second_phase:
            self.sql.execute_many(second_phase, parallel=True)
            executed = True

        if executed:
            self.invalidate_singleton(remove_global=True)

        return self

    def create(
        self,
        definition: Schema,
        *,
        mode: Mode | str | None = None,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat | None = None,
        missing_ok: bool = True,
        wait: WaitingConfigArg = True,
        or_replace: bool = False,
        record_ygg_properties: bool = True,
    ) -> "Table":
        mode = Mode.from_(mode, default=Mode.AUTO)

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if data_source_format is None:
            data_source_format = DataSourceFormat.DELTA

        if self.exists():
            data_source_format = self.infos.data_source_format

            if mode == Mode.OVERWRITE and data_source_format == DataSourceFormat.DELTA:
                pass
            elif mode == Mode.OVERWRITE:
                self.delete(wait=True, missing_ok=True)
            else:
                if mode == Mode.ERROR_IF_EXISTS:
                    raise ValueError(f"Table {self!r} already exists")
                elif mode in (Mode.IGNORE, Mode.AUTO):
                    return self

                schema = DataSchema.from_(definition)
                return self.with_columns(schema.fields, mode=mode)

        if table_type == TableType.MANAGED:
            result = self.sql_create(
                definition,
                comment=comment,
                missing_ok=missing_ok,
                properties=properties,
                or_replace=mode == Mode.OVERWRITE,
                data_source_format=data_source_format,
                record_ygg_properties=record_ygg_properties,
            )
        elif table_type == TableType.EXTERNAL:
            # An external table is created via DDL — ``CREATE EXTERNAL TABLE
            # … USING <fmt> LOCATION '…'`` — so the LOCATION is recorded and
            # (for Delta) the ``_delta_log`` is initialised at that path.
            # Default the location to the catalog's governed storage root
            # when the caller didn't pin one.
            if not storage_location:
                storage_location = self._default_external_location()
            result = self.sql_create(
                definition,
                storage_location=storage_location,
                comment=comment,
                missing_ok=missing_ok,
                properties=properties,
                or_replace=mode == Mode.OVERWRITE,
                data_source_format=data_source_format,
                record_ygg_properties=record_ygg_properties,
            )
        else:
            result = self.api_create(
                definition=definition,
                storage_location=storage_location,
                comment=comment,
                properties=properties,
                table_type=table_type,
                data_source_format=data_source_format,
                missing_ok=missing_ok,
                record_ygg_properties=record_ygg_properties,
            )

        return result

    def _default_external_location(self) -> str:
        """Governed default ``LOCATION`` for an external table created without
        one: the catalog's UC ``storage_root`` + ``<schema>/<table>``.

        Prefers a schema-scoped storage root when the schema advertises one
        (Databricks' ``__unitystorage`` layout), else falls back to the
        catalog storage root via
        :meth:`DatabricksClient.default_storage_location`.
        """
        try:
            return "%s/tables/%s" % (
                self.schema_storage_location(table_type=TableType.EXTERNAL),
                self.table_name,
            )
        except NotImplementedError:
            return self.client.default_storage_location(
                suffix="%s/%s" % (self.schema_name, self.table_name),
                catalog_name=self.catalog_name,
            )

    def sql_create(
        self,
        description: DataSchema,
        *,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, Any]] = None,
        missing_ok: bool = True,
        or_replace: bool = False,
        wait: WaitingConfigArg = True,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: bool | None = None,
        enable_deletion_vectors: bool | None = None,
        target_file_size: int | None = None,
        column_mapping_mode: str | None = None,
        auto_tag: bool = False,
        record_ygg_properties: bool = True,
    ) -> "Table":
        schema_info = DataSchema.from_any(description)
        if auto_tag:
            schema_info = schema_info.autotag()
        comment = comment or schema_info.comment
        effective_fields: list[Field] = []
        column_definitions: list[str] = []
        partition_by = schema_info.partition_fields
        cluster_by = schema_info.cluster_fields
        primary_keys = schema_info.primary_fields

        for f in schema_info.children:
            effective_fields.append(f)
            column_definitions.append(f.to_spark_name())

        any_invalid = any(_needs_column_mapping(f.name) for f in effective_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        # Inline-PK constraint: a single named PRIMARY KEY clause covering
        # every primary-key field.  Delta requires PK columns to be NOT
        # NULL — already enforced by the with_nullable(False) loop above.
        # FK / CHECK constraints can't be expressed inline against an
        # arbitrary parent table (the SDK ``table_constraints`` API does
        # the cross-table reference); they're applied post-create below.
        constraint_clauses = self._build_inline_constraints(
            self.full_name(safe=False), primary_keys
        )

        table_definitions = column_definitions + constraint_clauses

        # A bound ``storage_location`` makes this external. We don't emit the
        # ``EXTERNAL`` keyword — the ``LOCATION`` clause below is what makes the
        # table external (this is the form Databricks itself generates), and it
        # composes cleanly with ``CLUSTER BY`` / ``OR REPLACE``.
        external = storage_location is not None
        if or_replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif missing_ok:
            create_kw = "CREATE TABLE IF NOT EXISTS"
        else:
            create_kw = "CREATE TABLE"

        sql_parts: list[str] = [
            f"{create_kw} {self.full_name(safe=True)} (",
            "  " + ",\n  ".join(table_definitions),
            ")",
            f"USING {data_source_format.value}",
        ]

        if partition_by:
            sql_parts.append(
                "PARTITIONED BY (" + ", ".join(quote_ident(c.name) for c in partition_by) + ")"
            )
        elif cluster_by:
            sql_parts.append(
                "CLUSTER BY (" + ", ".join(quote_ident(c.name) for c in cluster_by) + ")"
            )
        elif external:
            # CLUSTER BY AUTO is UC *managed*-only (Databricks rejects it on an
            # external table with CLUSTER_BY_AUTO_UNSUPPORTED_TABLE_TYPE_ERROR),
            # so specify explicit liquid-clustering columns instead — the
            # primary key when present, else the first column (liquid clustering
            # caps at 4 keys). Callers can override via partition / cluster tags.
            default_cluster = (primary_keys or effective_fields[:1])[:4]
            if default_cluster:
                sql_parts.append(
                    "CLUSTER BY ("
                    + ", ".join(quote_ident(c.name) for c in default_cluster)
                    + ")"
                )
        else:
            sql_parts.append("CLUSTER BY AUTO")

        if comment:
            sql_parts.append(f"COMMENT '{escape_sql_string(comment)}'")

        if storage_location:
            sql_parts.append(f"LOCATION '{escape_sql_string(storage_location)}'")

        props: dict[str, Any] = {
            "delta.autoOptimize.optimizeWrite": bool(optimize_write),
            "delta.autoOptimize.autoCompact": bool(auto_compact),
        }
        if enable_cdf is not None:
            props["delta.enableChangeDataFeed"] = bool(enable_cdf)
        if enable_deletion_vectors is not None:
            props["delta.enableDeletionVectors"] = bool(enable_deletion_vectors)
        if target_file_size is not None:
            props["delta.targetFileSize"] = int(target_file_size)
        if column_mapping_mode != "none":
            props["delta.columnMapping.mode"] = column_mapping_mode
            props["delta.minReaderVersion"] = 2
            props["delta.minWriterVersion"] = 5
        if record_ygg_properties:
            props.update(_build_ygg_properties(schema_info))
        if properties:
            props.update(properties)

        if props:
            def _fmt(k: str, v: Any) -> str:
                if isinstance(v, str):
                    return f"'{k}' = '{escape_sql_string(v)}'"
                if isinstance(v, bool):
                    return f"'{k}' = '{'true' if v else 'false'}'"
                return f"'{k}' = {v}"

            sql_parts.append(
                "TBLPROPERTIES (" + ", ".join(_fmt(k, v) for k, v in props.items()) + ")"
            )

        statement = "\n".join(sql_parts)
        logger.debug(
            "Creating table %r via SQL (or_replace=%s, missing_ok=%s, "
            "columns=%d, partition_by=%d, cluster_by=%d, primary_keys=%d, "
            "data_source_format=%s, column_mapping_mode=%s)",
            self, or_replace, missing_ok,
            len(column_definitions), len(partition_by or ()),
            len(cluster_by or ()), len(primary_keys or ()),
            data_source_format, column_mapping_mode,
        )

        try:
            self.sql.execute(statement, wait=wait)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                logger.debug(
                    "Parent schema missing for table %r — auto-creating %s.%s and retrying",
                    self, self.catalog_name, self.schema_name,
                )
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait)
            elif "CONSTRAINT_ALREADY_EXISTS_IN_SCHEMA" in str(exc):
                logger.debug(
                    "Constraint already exists on table %r — ignoring", self,
                )
            else:
                raise

        # Apply remaining constraints (FK / CHECK) via the SDK post-create.
        # Inline PK was already emitted in DDL — skip it here.
        self._apply_post_create_constraints(schema_info)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

            # Per-column tags in one parallelised pass rather than N sequential
            # round-trips. validate=False: column names are authoritative here
            # (we just emitted the DDL from these same fields).
        column_tag_batches = {
            f.name: f.tags for f in effective_fields if f.tags
        }
        if column_tag_batches:
            self.update_columns_tags(column_tag_batches, validate=False)

        return self

    def update_columns_tags(
        self,
        tags_by_column: Mapping[str, Mapping[str, str] | list[EntityTagAssignment]] | None,
        *,
        mode: ModeLike | None = None,
        parallel_columns: int | bool | None = None,
        parallel_per_column: int | bool | None = None,
        cache_ttl: float | None = 300.0,
        continue_on_error: bool = True,
        validate: bool = True,
    ) -> dict[str, BaseException | None]:
        """Apply tag batches to many columns of this table in parallel.

        Per-column counterpart of :meth:`set_tags`. Each column's batch is
        routed through :meth:`EntityTags.update_entities_tags` with the same
        *mode* and *cache_ttl*; columns are processed concurrently up to
        *parallel_columns*.

        Args:
            tags_by_column:
                Mapping of column name to its tag batch. Each batch may be a
                ``{tag_key: tag_value}`` dict or a list of
                :class:`EntityTagAssignment` (entity addressing on the
                assignments is filled in here — callers don't need to set it).
            mode:
                Batch mode applied per column. See
                :meth:`EntityTags.update_entity_tags` for semantics.
            parallel_columns:
                Outer concurrency — columns processed at once. Defaults to 4.
            parallel_per_column:
                Inner concurrency — writes within a single column's batch.
                Defaults to 1; bump only when the workspace can absorb the
                extra load (rate limits are workspace-wide).
            cache_ttl:
                TTL for the per-column tag-list cache reads used to diff
                before writing. ``None`` bypasses the cache.
            continue_on_error:
                When ``True`` (default), per-column failures are returned in
                the result rather than aborting the whole call. With
                ``False``, the first exception propagates.
            validate:
                When ``True`` (default), unknown column names raise
                :class:`ValueError` before any write goes out. Turn off when
                applying tags speculatively against a partially-known schema.

        Returns:
            ``{column_name: None | BaseException}``. ``None`` denotes success.
        """
        if not tags_by_column:
            return {}

        # ---- validate column names against the table schema --------------
        # Cheap local check — saves a round trip on the typo case where the
        # API would otherwise return an opaque "entity not found".
        if validate:
            known = {c.name for c in self.columns}
            unknown = [name for name in tags_by_column if name not in known]
            if unknown:
                raise ValueError(
                    f"Unknown column(s) on {self.full_name()}: {sorted(unknown)}. "
                    f"Pass validate=False to apply tags anyway."
                )

        # ---- normalise into the {(et, en): batch} shape ------------------
        # Each column is its own UC entity; we build entity names eagerly so
        # the assignments carry the right identity and update_entities_tags
        # can group/dispatch directly without re-deriving them.
        full = self.full_name()
        grouped: dict[tuple[str, str], list[EntityTagAssignment]] = {}

        for col_name, batch in tags_by_column.items():
            entity_name = f"{full}.{col_name}"
            key = ("columns", entity_name)

            if not batch:
                # OVERWRITE with an empty batch clears all tags for that
                # column; other modes drop the entry. update_entities_tags
                # does this filter itself, but normalising here keeps the
                # column→entity mapping symmetric on the way back out.
                grouped[key] = []
                continue

            if isinstance(batch, Mapping):
                assignments = [
                    EntityTagAssignment(
                        entity_type="columns",
                        entity_name=entity_name,
                        tag_key=_coerce_tag_str(k),
                        tag_value=_coerce_tag_str(v) if v is not None else "",
                    )
                    for k, v in batch.items()
                ]
            else:
                # List of EntityTagAssignment — stamp our entity addressing
                # over whatever the caller put on them, since we own the
                # routing here. Copying via from_dict/to_dict keeps the
                # frozen-ness intact without touching SDK internals.
                assignments = []
                for a in batch:
                    if not isinstance(a, EntityTagAssignment):
                        raise TypeError(
                            f"update_columns_tags: expected EntityTagAssignment "
                            f"in list batch for column {col_name!r}, got {type(a)}"
                        )
                    d = a.as_dict() if hasattr(a, "as_dict") else dict(a.__dict__)
                    d["entity_type"] = "columns"
                    d["entity_name"] = entity_name
                    assignments.append(EntityTagAssignment.from_dict(d))

            grouped[key] = assignments

        # ---- dispatch via the multi-entity service -----------------------
        raw_results = self.client.entity_tags.update_entities_tags(
            tags_by_entity=grouped,
            mode=mode,
            parallel_entities=parallel_columns,
            parallel_per_entity=parallel_per_column,
            cache_ttl=cache_ttl,
            continue_on_error=continue_on_error,
        )

        # ---- pivot results back to column-name keyspace ------------------
        # Strip the ``"columns"`` entity_type and the table prefix from the
        # entity_name so callers don't have to know we routed through the
        # multi-entity API underneath.
        prefix = f"{full}."
        out: dict[str, BaseException | None] = {}
        for (entity_type, entity_name), err in raw_results.items():
            col_name = (
                entity_name[len(prefix):]
                if entity_name.startswith(prefix) else entity_name
            )
            out[col_name] = err
        return out

    @staticmethod
    def _build_inline_constraints(prefix: str, primary_keys: Iterable[Field]) -> list[str]:
        """Render inline DDL ``CONSTRAINT … PRIMARY KEY(…)`` clauses.

        FK / CHECK aren't emitted inline: FK needs a parent reference that
        only the constraint :class:`Field` (or the SDK call) carries, and
        CHECK predicates aren't part of this layer.  Those go through
        :meth:`_apply_post_create_constraints`.
        """
        pk_fields = [f for f in primary_keys if f and f.primary_key]
        if not pk_fields:
            return []

        col_names = [f.name for f in pk_fields]
        constraint_name = safe_constraint_name(col_names, prefix="pk_" + prefix)
        cols = ", ".join(quote_ident(n) for n in col_names)
        return [f"CONSTRAINT {quote_ident(constraint_name)} PRIMARY KEY ({cols}) RELY"]

    def _apply_post_create_constraints(self, schema_info: DataSchema) -> None:
        """Push FK / CHECK constraint Fields through the SDK constraints API.

        Inline-PK constraints already landed in the CREATE TABLE DDL — the
        primary-key fields on ``schema_info`` are intentionally skipped
        here to avoid a duplicate-name collision.
        """
        constraint_fields = [
            f for f in (schema_info.constraints or [])
            if f.foreign_key or (f.constraint_key and not f.primary_key)
        ]
        if not constraint_fields:
            return

        try:
            from yggdrasil.databricks.constraints.service import TableConstraints
        except ImportError:
            logger.debug(
                "yggdrasil.databricks.constraints not available — "
                "skipping post-create constraints on table %r", self,
            )
            return

        constraints_service = TableConstraints(client=self.client)
        for cf in constraint_fields:
            try:
                constraints_service.create_constraint(self, cf)
            except Exception:
                logger.warning(
                    "Failed to create constraint %r on table %r",
                    cf.name, self, exc_info=True,
                )

    def api_create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        missing_ok: bool = False,
        record_ygg_properties: bool = True,
    ) -> "Table":
        """Create the table via the Unity Catalog ``tables.create`` REST API.

        Targets EXTERNAL tables — the SDK ``tables.create`` endpoint
        requires an explicit ``storage_location``.  For MANAGED tables,
        prefer :meth:`sql_create`, which is also the only path that
        exposes Delta-specific knobs (``CLUSTER BY``, ``OPTIMIZE``,
        ``TBLPROPERTIES``, column mapping mode, …).

        ``comment`` and constraints (PK / FK / CHECK) carried by the
        schema are applied post-create — the SDK call itself only takes
        columns + storage + properties — so the behaviour ends up
        symmetric with :meth:`sql_create`.
        """
        if missing_ok and self.exists():
            return self

        schema_info = DataSchema.from_any(definition).autotag()
        comment = comment or schema_info.comment

        effective_fields: list[Field] = []
        column_infos: list[ColumnInfo] = []
        for position, f in enumerate(schema_info.children):
            if f.constraint_key:
                continue
            effective_fields.append(f)
            column_infos.append(self._field_to_column_info(f, position=position))

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if not storage_location:
            raise ValueError(
                "api_create requires an explicit storage_location — the UC "
                "tables.create endpoint won't materialise a MANAGED table for you. "
                "Use sql_create for managed Delta tables."
            )

        merged_properties: dict[str, str] = {}
        if record_ygg_properties:
            merged_properties.update(_build_ygg_properties(schema_info))
        if properties:
            merged_properties.update({str(k): str(v) for k, v in properties.items()})

        logger.debug(
            "Creating table %r via API (table_type=%s, data_source_format=%s, "
            "storage_location=%s, columns=%d, properties=%d)",
            self, table_type, data_source_format,
            storage_location, len(column_infos), len(merged_properties),
        )
        try:
            self.client.workspace_client().tables.create(
                name=self.table_name,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_type=table_type,
                data_source_format=data_source_format,
                storage_location=storage_location,
                columns=column_infos,
                properties=merged_properties or None,
            )
        except DatabricksError as exc:
            if missing_ok and "already exists" in str(exc).lower():
                logger.debug(
                    "Table %r already exists — soft-resetting cache", self,
                )
                return self
            raise

        # The SDK endpoint doesn't accept a comment — set it via ALTER
        # TABLE so the behaviour matches sql_create (which embeds COMMENT
        # in the CREATE DDL).
        if comment:
            self.sql.execute(
                f"ALTER TABLE {self.full_name(safe=True)} "
                f"SET TBLPROPERTIES ('comment' = '{escape_sql_string(comment)}')",
                wait=True,
            )

        self._apply_post_create_constraints(schema_info)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

        for f in effective_fields:
            if f.tags:
                col = self.column(f.name, raise_error=False)
                if col is not None:
                    col.set_tags(f.tags)

        return self

    @staticmethod
    def _field_to_column_info(f: Field, *, position: int) -> ColumnInfo:
        """Translate a :class:`Field` into a UC SDK :class:`ColumnInfo`."""
        ddl = f.dtype.to_spark_name()
        type_name = _column_type_name_from_ddl(ddl)
        comment_bytes = (f.metadata or {}).get(b"comment") if f.metadata else None
        comment = comment_bytes.decode("utf-8") if isinstance(comment_bytes, bytes) else None
        return ColumnInfo(
            name=f.name,
            type_text=ddl,
            type_name=type_name,
            type_json=None,
            nullable=bool(f.nullable),
            position=position,
            comment=comment,
            partition_index=position if f.partition_by else None,
        )

    # =========================================================================
    # View DDL — same securable family, different create / drop keywords
    # =========================================================================

    def create_view_ddl(
        self,
        query: str,
        *,
        or_replace: bool = False,
        missing_ok: bool = False,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Render a ``CREATE [OR REPLACE] VIEW [IF NOT EXISTS]`` DDL statement.

        Mirrors the legacy :meth:`View.create_ddl` shape; ``or_replace``
        and ``missing_ok`` are mutually exclusive, and the SELECT
        text is required.
        """
        if or_replace and missing_ok:
            raise ValueError("Use either or_replace or missing_ok, not both.")

        select_text = (query or "").strip().rstrip(";").strip()
        if not select_text:
            raise ValueError("View query (SELECT text) cannot be empty")

        if or_replace:
            create_kw = "CREATE OR REPLACE VIEW"
        elif missing_ok:
            create_kw = "CREATE VIEW IF NOT EXISTS"
        else:
            create_kw = "CREATE VIEW"

        parts: list[str] = [f"{create_kw} {self.full_name(safe=True)}"]

        if columns:
            parts.append("(" + ", ".join(quote_ident(c) for c in columns) + ")")

        if comment:
            parts.append(f"COMMENT '{escape_sql_string(comment)}'")

        if properties:
            def _fmt(k: str, v: Any) -> str:
                if isinstance(v, bool):
                    return f"'{k}' = '{'true' if v else 'false'}'"
                if isinstance(v, str):
                    return f"'{k}' = '{escape_sql_string(v)}'"
                return f"'{k}' = {v}"

            parts.append(
                "TBLPROPERTIES ("
                + ", ".join(_fmt(k, v) for k, v in properties.items())
                + ")"
            )

        parts.append(f"AS {select_text}")
        return "\n".join(parts)

    def create_view(
        self,
        query: str,
        *,
        mode: ModeLike = None,
        or_replace: bool | None = None,
        missing_ok: bool | None = None,
        columns: Iterable[str] | None = None,
        comment: str | None = None,
        properties: Optional[Mapping[str, Any]] = None,
        tags: Mapping[str, str] | None = None,
        wait: WaitingConfigArg = True,
    ) -> "Table":
        """Create (or replace) this Table as a Unity Catalog view.

        When neither ``or_replace`` nor ``missing_ok`` is provided
        the keywords are derived from ``mode``:

        * :data:`Mode.OVERWRITE` → ``or_replace=True``
        * :data:`Mode.AUTO` / :data:`Mode.APPEND` / :data:`Mode.UPSERT`
          / :data:`Mode.IGNORE` → ``missing_ok=True``
        * :data:`Mode.ERROR_IF_EXISTS` → plain ``CREATE VIEW``
        """
        parsed_mode = Mode.from_(mode, default=Mode.AUTO)

        if or_replace is None and missing_ok is None:
            if parsed_mode == Mode.OVERWRITE:
                or_replace = True
                missing_ok = False
            elif parsed_mode == Mode.ERROR_IF_EXISTS:
                or_replace = False
                missing_ok = False
            else:
                or_replace = False
                missing_ok = True

        statement = self.create_view_ddl(
            query,
            or_replace=bool(or_replace),
            missing_ok=bool(missing_ok),
            columns=columns,
            comment=comment,
            properties=properties,
        )

        logger.debug(
            "Creating view %r (or_replace=%s, missing_ok=%s, mode=%s)",
            self, bool(or_replace), bool(missing_ok), parsed_mode.name,
        )
        try:
            self.sql.execute(statement, wait=wait)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                self.schema.get_or_create()
            else:
                raise

        if tags:
            self.set_tags(tags)

        return self

    def concat_tables(
        self,
        tables: Iterable["Table"],
        *,
        by_name: bool = True,
        cast: bool = True,
        comment: str | None = None,
        mode: ModeLike = Mode.OVERWRITE,
    ) -> "Table":
        """Create or replace this Table as the ``UNION ALL`` of *tables*.

        When ``cast`` is ``True`` (default), the union is "smart": column
        names are aligned across inputs, types are promoted to the widest
        compatible :class:`DataType` via ``merge_with(upcast=True)``,
        each input projects the unified column list in order, and any
        column missing from a given input is emitted as
        ``CAST(NULL AS <ddl>)`` so the unified schema is preserved.

        When ``cast`` is ``False`` the method falls back to a plain
        ``SELECT * FROM <table> UNION ALL [BY NAME] ...`` and lets
        Databricks reconcile the schemas at query time.
        """
        tables_list = list(tables)
        if not tables_list:
            raise ValueError("concat_tables requires at least one table")

        if cast:
            query = self._build_smart_union_query(tables_list)
        else:
            separator = "\nUNION ALL BY NAME\n" if by_name else "\nUNION ALL\n"
            query = separator.join(
                f"SELECT * FROM {t.full_name(safe=True)}"
                for t in tables_list
            )

        return self.create_view(query, mode=mode, comment=comment)

    @staticmethod
    def _build_smart_union_query(tables_list: list["Table"]) -> str:
        """Render a ``UNION ALL`` query projecting each input to a unified schema.

        Walks every input's ``columns``, accumulates a unified schema
        (first-seen column order, types promoted via
        ``merge_with(upcast=True)``), then projects each input to that
        column order — selecting present columns as-is and substituting
        ``CAST(NULL AS <ddl>)`` for absent ones.
        """
        from yggdrasil.enums.mode import Mode as _Mode

        column_order: list[str] = []
        unified: dict[str, Any] = {}
        per_table: list[dict[str, Any]] = []

        for tbl in tables_list:
            cols: dict[str, Any] = {}
            for c in tbl.columns:
                cols[c.name] = c.field.dtype
                if c.name not in unified:
                    column_order.append(c.name)
                    unified[c.name] = c.field.dtype
                else:
                    unified[c.name] = unified[c.name].merge_with(
                        c.field.dtype, mode=_Mode.UPSERT, upcast=True,
                    )
            per_table.append(cols)

        if not column_order:
            raise ValueError(
                "concat_tables: input tables have no columns to union; "
                "ensure each input has been resolved against the catalog"
            )

        select_blocks: list[str] = []
        for tbl, cols in zip(tables_list, per_table):
            exprs: list[str] = []
            for name in column_order:
                qname = quote_ident(name)
                if name in cols:
                    exprs.append(qname)
                else:
                    ddl = unified[name].to_spark_name()
                    exprs.append(f"CAST(NULL AS {ddl}) AS {qname}")

            select_blocks.append(
                "SELECT\n  " + ",\n  ".join(exprs)
                + f"\nFROM {tbl.full_name(safe=True)}"
            )

        return "\nUNION ALL\n".join(select_blocks)

    # =========================================================================
    # Rename
    # =========================================================================

    def rename(
        self,
        new_name: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> "Table":
        """Rename this table in-place (``ALTER TABLE … RENAME TO …``).

        Accepts an unqualified name (``"new_orders"``), a two-part name
        (``"sales.new_orders"`` → cross-schema move within the same catalog),
        or a three-part name (``"main.sales.new_orders"``). Catalog/schema
        keyword overrides win over parts parsed from *new_name*.

        Unity Catalog allows cross-schema renames within the same catalog;
        moves across catalogs are rejected here with a clear error rather
        than letting the server return a generic failure.
        """
        if new_name is not None:
            parsed_c, parsed_s, parsed_t = self.sql.tables.parse_catalog_schema_table_names(new_name)
        else:
            parsed_c = parsed_s = parsed_t = None

        target_catalog = (catalog_name or parsed_c or self.catalog_name or "").strip().strip("`")
        target_schema = (schema_name or parsed_s or self.schema_name or "").strip().strip("`")
        target_table = (table_name or parsed_t or "").strip().strip("`")

        if not target_table:
            raise ValueError("Cannot rename table to an empty name")
        if not target_catalog or not target_schema:
            raise ValueError(
                f"Cannot rename {self.full_name()} — target needs a catalog and"
                f" schema (got catalog={target_catalog!r} schema={target_schema!r})"
            )
        if target_catalog != self.catalog_name:
            raise ValueError(
                f"Unity Catalog ALTER TABLE RENAME TO cannot move a table across"
                f" catalogs ({self.catalog_name!r} → {target_catalog!r}). Use"
                f" Table.clone(...) to copy across catalogs instead."
            )
        if target_schema == self.schema_name and target_table == self.table_name:
            logger.debug(
                "Skipping rename of table %r — new name matches current", self,
            )
            return self

        if target_schema == self.schema_name:
            rename_to = quote_ident(target_table)
        else:
            rename_to = f"{quote_ident(target_schema)}.{quote_ident(target_table)}"

        keyword = "VIEW" if self.is_view else "TABLE"
        logger.debug(
            "Renaming %s %r → %s.%s.%s",
            keyword, self, target_catalog, target_schema, target_table,
        )
        self.sql.execute(
            f"ALTER {keyword} {self.full_name(safe=True)} RENAME TO {rename_to}"
        )
        self.invalidate_singleton(remove_global=True)
        self.schema_name = target_schema
        self.table_name = target_table
        return self

    # =========================================================================
    # Clone
    # =========================================================================

    def clone(
        self,
        target: "str | Table | None" = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        deep: bool = True,
        replace: bool = False,
        missing_ok: bool = False,
        mode: "ModeLike | None" = None,
        properties: Mapping[str, Any] | None = None,
        location: str | None = None,
        version: int | None = None,
        timestamp: "str | _dt.datetime | _dt.date | None" = None,
    ) -> "Table":
        """Clone this table to *target* via Delta ``CREATE TABLE … CLONE``.

        Emits one of::

            CREATE TABLE [IF NOT EXISTS] <target> [SHALLOW|DEEP] CLONE <source>
                [TBLPROPERTIES (...)] [LOCATION '...']
            CREATE OR REPLACE TABLE <target> [SHALLOW|DEEP] CLONE <source> ...

        Args:
            target:        Target location — :class:`Table`, a 1/2/3-part dotted
                           name, or ``None`` when *catalog_name* / *schema_name*
                           / *table_name* are passed explicitly.
            deep:          ``True`` (default) → DEEP CLONE (independent copy);
                           ``False`` → SHALLOW CLONE (metadata only, shares files).
            replace:       Emit ``CREATE OR REPLACE TABLE``.
            missing_ok: Emit ``CREATE TABLE IF NOT EXISTS``. Mutually
                           exclusive with *replace*.
            mode:          Existence policy as a :class:`Mode` (overrides
                           *replace* / *missing_ok* when set): ``OVERWRITE`` /
                           ``TRUNCATE`` → ``CREATE OR REPLACE``, ``IGNORE`` →
                           ``CREATE … IF NOT EXISTS``, ``ERROR_IF_EXISTS`` →
                           plain ``CREATE`` (fails if the target exists).
            properties:    Optional ``TBLPROPERTIES`` overrides.
            location:      External storage path for the target.
            version:       Delta source version (``VERSION AS OF``).
            timestamp:     Delta source timestamp (``TIMESTAMP AS OF``).

        Returns:
            A :class:`Table` bound to this service pointing at the target.
        """
        if mode is not None:
            resolved = Mode.from_(mode)
            if resolved in (Mode.OVERWRITE, Mode.TRUNCATE):
                replace, missing_ok = True, False
            elif resolved is Mode.IGNORE:
                replace, missing_ok = False, True
            elif resolved is Mode.ERROR_IF_EXISTS:
                replace, missing_ok = False, False
            else:
                raise ValueError(
                    f"clone mode must be OVERWRITE/TRUNCATE, IGNORE, or "
                    f"ERROR_IF_EXISTS — got {resolved.name}. Clone creates the "
                    f"target; APPEND/MERGE/UPSERT/AUTO don't apply."
                )
        if replace and missing_ok:
            raise ValueError("Use either replace=True or missing_ok=True, not both.")
        if version is not None and timestamp is not None:
            raise ValueError(
                "Pass either version or timestamp to clone, not both — Delta"
                " accepts one temporal anchor on the source."
            )

        tables = self.sql.tables
        if isinstance(target, Table):
            target_catalog = target.catalog_name
            target_schema = target.schema_name
            target_table = target.table_name
        else:
            parsed_c, parsed_s, parsed_t = (
                tables.parse_catalog_schema_table_names(target) if target else (None, None, None)
            )
            target_catalog = catalog_name or parsed_c or self.catalog_name
            target_schema = schema_name or parsed_s or self.schema_name
            target_table = table_name or parsed_t

        if not (target_catalog and target_schema and target_table):
            raise ValueError(
                f"Cannot clone {self.full_name()} — target needs catalog +"
                f" schema + table (got catalog={target_catalog!r}"
                f" schema={target_schema!r} table={target_table!r})"
            )
        if (
            target_catalog == self.catalog_name
            and target_schema == self.schema_name
            and target_table == self.table_name
        ):
            raise ValueError(
                f"Cannot clone {self.full_name()} onto itself — choose a"
                f" different target catalog/schema/table."
            )

        # Resolve the source's type so a view is detected even on a fresh
        # handle whose infos haven't been read yet — ``is_view`` reads the
        # cache only, so without this a never-inspected view would report
        # ``False`` and wrongly take the Delta ``CLONE`` path below.
        if self._infos is None:
            try:
                _ = self.infos
            except Exception:
                pass

        # Views can't ride the Delta ``CLONE`` path — re-emit the
        # source's ``view_definition`` as a fresh ``CREATE [OR REPLACE]
        # VIEW [IF NOT EXISTS]`` against the target, mirroring the
        # legacy :meth:`View.clone` shape.
        if self.is_view:
            select_text = (self.view_definition or "").strip().rstrip(";").strip()
            if not select_text:
                raise ValueError(
                    f"Cannot clone {self.full_name()} — source has no"
                    f" view_definition. Run ``create_view(query=...)``"
                    f" against the target directly with explicit SQL."
                )
            # Re-point the inner query at the target: the stored
            # ``view_definition`` references the *source* catalog/schema, so a
            # cross-schema clone must requalify those prefixes or the cloned
            # view would still read the source's tables.
            select_text = requalify_table_refs(
                select_text,
                source=(self.catalog_name, self.schema_name),
                target=(target_catalog, target_schema),
            )
            cloned = Table(
                service=tables,
                catalog_name=target_catalog,
                schema_name=target_schema,
                table_name=target_table,
            )
            statement = cloned.create_view_ddl(
                select_text,
                or_replace=replace,
                missing_ok=missing_ok,
                properties=properties,
            )
            logger.debug(
                "Cloning view %r → %s.%s.%s (replace=%s, missing_ok=%s)",
                self, target_catalog, target_schema, target_table,
                replace, missing_ok,
            )
            self.sql.execute(statement)
            return cloned

        target_full = (
            f"{quote_ident(target_catalog)}.{quote_ident(target_schema)}."
            f"{quote_ident(target_table)}"
        )

        if replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif missing_ok:
            create_kw = "CREATE TABLE IF NOT EXISTS"
        else:
            create_kw = "CREATE TABLE"

        source_full = self.full_name(safe=True)
        if version is not None:
            source_clause = f"{source_full} VERSION AS OF {int(version)}"
        elif timestamp is not None:
            if isinstance(timestamp, (_dt.datetime, _dt.date)):
                ts_lit = f"'{timestamp.isoformat()}'"
            else:
                ts_lit = f"'{escape_sql_string(str(timestamp))}'"
            source_clause = f"{source_full} TIMESTAMP AS OF {ts_lit}"
        else:
            source_clause = source_full

        clone_kw = "DEEP CLONE" if deep else "SHALLOW CLONE"
        sql_parts: list[str] = [
            f"{create_kw} {target_full} {clone_kw} {source_clause}",
        ]
        if properties:
            sql_parts.append(
                "TBLPROPERTIES ("
                + ", ".join(
                    f"'{escape_sql_string(str(k))}' = {sql_literal(v)}"
                    for k, v in properties.items()
                )
                + ")"
            )
        if location:
            sql_parts.append(f"LOCATION '{escape_sql_string(location)}'")

        statement = " ".join(sql_parts)
        logger.debug(
            "Cloning table %r → %s.%s.%s (deep=%s, replace=%s, missing_ok=%s)",
            self, target_catalog, target_schema, target_table,
            deep, replace, missing_ok,
        )
        self.sql.execute(statement)

        cloned = Table(
            service=tables,
            catalog_name=target_catalog,
            schema_name=target_schema,
            table_name=target_table,
        )

        return cloned

    def insert(
        self,
        data: Any,
        *,
        mode: ModeLike = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
        return_data: bool = False,
        **kwargs
    ) -> "Tabular | None":
        """Insert *data* into this table — thin wrapper over :meth:`insert_into`.

        ``wait=False`` switches to the **async drop** path
        (:func:`~yggdrasil.databricks.table.insert.stage_async_insert`): the
        staged Parquet + a JSON operation log are written under the table's
        ``.sql/async`` area and a file-arrival job (:meth:`async_job`)
        aggregates and loads them later — no warehouse statement runs here.
        ``OVERWRITE`` / ``APPEND`` (no keys) and ``MERGE`` / ``UPSERT`` (with
        ``match_by`` keys) qualify; anything else (or a query / Spark source)
        falls through to the normal synchronous path.

        On the ``wait=False`` path, ``check_job`` (forwarded via ``**kwargs``):
        if True, get-or-create the async file-arrival loader job for this table
        so the staged data actually gets picked up — same as calling
        :meth:`stage_insert` directly.
        """
        from yggdrasil.databricks.table.insert import ASYNC_MODES, stage_async_insert

        mode_enum = Mode.from_(mode, default=Mode.APPEND)
        keyed = mode_enum in (Mode.MERGE, Mode.UPSERT)
        async_supported = (mode_enum in ASYNC_MODES and not match_by) or (
            keyed and match_by
        )
        if (
            wait is False
            and async_supported
            and not isinstance(data, (PreparedStatement, StatementResult))
            and not PreparedStatement.looks_like_query(data)
        ):
            return stage_async_insert(self, data, mode=mode, match_by=match_by, **kwargs)
        return self.insert_into(
            data,
            mode=mode,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
            return_data=return_data,
            **kwargs,
        )

    def stage_insert(
        self,
        data: Any,
        *,
        mode: ModeLike = None,
        match_by: Optional[list[str]] = None,
        cast_options: Optional[CastOptions] = None,
        check_job: bool = False,
    ) -> VolumePath:
        """Stage an async insert and return the op-log path — no warehouse run.

        Writes *data* as Parquet to this table's staging area and drops a JSON
        operation log under ``.sql/async/logs``; no SQL statement runs on a
        warehouse here. The file-arrival loader job (:meth:`async_job`)
        aggregates the staged logs and loads them into the table later.

        ``check_job``: if True, get-or-create the async file-arrival loader job
        for this table so the staged data will actually be picked up. Thin
        wrapper over
        :func:`~yggdrasil.databricks.table.insert.stage_async_insert`.
        """
        from yggdrasil.databricks.table.insert import stage_async_insert

        return stage_async_insert(
            self,
            data,
            mode=mode,
            match_by=match_by,
            cast_options=cast_options,
            check_job=check_job,
        )

    # =========================================================================
    # insert_into — top-level dispatcher (arrow / spark / sql paths)
    # =========================================================================

    def insert_into(
        self,
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
            dict, list, str,
            PreparedStatement, StatementResult,
            "pandas.DataFrame", "polars.DataFrame", "pyspark.sql.DataFrame",
        ],
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        spark_options: Optional[Dict[str, Any]] = None,
        predicate: Predicate | None = None,
        retry: Optional[WaitingConfigArg] = None,
        return_data: bool = False,
        safe_merge: bool = False,
    ) -> "StatementBatch | Tabular | None":
        """Insert *data* into this table using the most appropriate backend.

        Routing:

        - Spark DataFrame (or anything when a ``SparkSession`` is reachable)
          → :meth:`spark_insert`
        - Otherwise → :meth:`arrow_insert` (the specialized warehouse path
          with Volume staging, which delegates to
          :class:`~yggdrasil.databricks.table.insert.DatabricksTableInsert`).

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True`` the backend that ran the write hands back its
        source payload as a :class:`Tabular` — :class:`ArrowTabular` from
        :meth:`arrow_insert`, :class:`Dataset` from :meth:`spark_insert` —
        for downstream chaining without re-querying the target.
        """
        # Fold the top-level predicate into cast_options so the
        # downstream backends read a single source of truth.
        cast_options = _coalesce_predicate(cast_options, predicate)
        common = dict(
            mode=mode,
            match_by=match_by,
            update_column_names=update_column_names,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            retry=retry,
            return_data=return_data,
            safe_merge=safe_merge,
        )

        if spark_session is None:
            session_attr = getattr(data, "sparkSession", None)
            spark_session = session_attr if session_attr is not None else self.sql.spark.resolve_session(create=False)

        if spark_session is not None:
            return self.spark_insert(
                data=data,
                schema_mode=schema_mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                spark_options=spark_options,
                spark_session=spark_session,
                **common,
            )

        return self.arrow_insert(
            data=data,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            **common,
        )

    # =========================================================================
    # arrow_insert — warehouse path, Volume staging
    # =========================================================================

    @property
    def staging_volume(self):
        if self._staging_volume is None:
            if not self.catalog_name or not self.schema_name or not self.table_name:
                raise ValueError(f"Table {self} is missing required catalog, schema, or table name")

            self._staging_volume = Volume(
                service=self.service.volumes,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                volume_name=self.client.safe_tag_value(self.table_name, repl="_").lower()
            )
        return self._staging_volume

    @staging_volume.setter
    def staging_volume(self, value):
        self._staging_volume = self.client.volumes(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        ).volume(value)

    def ensure_staging_volume(self) -> "Volume":
        """Get-or-create this table's staging :class:`Volume` and return it.

        For an **external** table the staging volume is created *external*
        too, rooted on the schema's storage location — the segment before
        ``__unitystorage`` — and keyed by the table's UC id:
        ``<schema_root>/uc/tables/<table_id>``. That keeps staged Parquet on
        the same governed external location as the table instead of a
        fabricated client default location. A managed table keeps the default
        (managed) create path, materialised lazily on first write.

        Kept off the :attr:`staging_volume` property on purpose — resolving
        ``infos`` / creating a volume on a bare handle read is too surprising;
        creation belongs at the staging-folder boundary where a write is
        actually imminent.
        """
        volume = self.staging_volume
        if self.infos.table_type == TableType.EXTERNAL:
            root = self.schema_storage_location().split("/__unitystorage")[0].rstrip("/")
            volume.get_or_create(
                volume_type="EXTERNAL",
                storage_location=f"{root}/uc/tables/{self.table_id}",
            )
        return volume

    def staging_folder(
        self,
        temporary: bool = False,
    ) -> VolumePath:
        """Return the staging folder for this table.

        Ensures the staging volume exists first (external for an external
        table — see :meth:`ensure_staging_volume`).
        """
        return self.ensure_staging_volume().path(".sql/tmp", temporary=temporary)

    def insert_volume_path(
        self,
        target: "Table | None" = None,
        *,
        staging_volume: "Volume | None" = None,
        temporary: bool = True,
    ) -> VolumePath:
        """Mint a fresh Parquet staging path under the target table's
        :attr:`staging_volume`.

        Roots the file at ``<staging_volume>/.sql/tmp/tmp-<epoch_ms>-<seed>.parquet``
        (same shape as :meth:`staging_folder` but with a unique leaf
        per call). ``target`` defaults to ``self``; pass another
        :class:`Table` when the staging hierarchy needs to live next
        to a different table (e.g. dispatch fan-out). Lifted out of
        :meth:`arrow_insert` so callers — and tests — can pre-mint or
        swap the staging location without driving the full insert.
        """
        target = target if target is not None else self
        seed = uuid.uuid4().hex[:8]
        leaf = f"tmp-{int(time.time() * 1000)}-{seed}.parquet"
        staging_volume = target.ensure_staging_volume() if staging_volume is None else staging_volume

        return staging_volume.path(
            f".sql/tmp/{leaf}",
            temporary=temporary,
        )

    # =========================================================================
    # async insert — drop Parquet + an operation log; a file-arrival job loads
    # =========================================================================

    def async_job(self, *, rebuild: bool = False) -> Any:
        """Get-or-create the file-arrival loader job for this table; return it.

        An existing job is returned as-is; otherwise one is created — resolving
        the full ygg wheel bundle for the current version (reusing an
        already-deployed bundle when present, else building + uploading it) and
        upserting a serverless job that watches ``.sql/async/logs`` and
        aggregates the operation logs ``insert(..., wait=False)`` drops into one
        load per target (an OVERWRITE supersedes everything staged before it).
        Pass ``rebuild=True`` to force a redeploy.
        See :func:`~yggdrasil.databricks.table.insert.ensure_async_job`.
        """
        from yggdrasil.databricks.table.insert import ensure_async_job

        return ensure_async_job(self, rebuild=rebuild)

    def arrow_insert(
        self,
        data,
        *,
        engine: Literal["api", "spark"] | None = None,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        predicate: Predicate | None = None,
        retry: Optional[WaitingConfigArg] = None,
        return_data: bool = False,
        safe_merge: bool = False,
        staging_volume: "Volume | None" = None
    ) -> "StatementBatch | Tabular | None":
        """Insert through the warehouse SQL path with staged Parquet.

        ``safe_merge`` controls keyed-write strategy:

        * ``safe_merge=False`` (default) — emits a single ``MERGE
          INTO`` statement. Databricks / Delta plans the keyed dedup
          once.
        * ``safe_merge=True`` — sidesteps MERGE: keyed APPEND becomes
          ``INSERT ... WHERE NOT EXISTS (...)``, keyed UPSERT becomes
          ``DELETE`` matching keys then ``INSERT``. Useful for
          backends without native MERGE or callers that want explicit
          dedup semantics.

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True``, returns an :class:`ArrowTabular` wrapping
        the staged source rows so callers can chain on the payload
        without re-reading from the target.
        """
        cast_options = _coalesce_predicate(cast_options, predicate)

        mode_enum = Mode.from_(mode, default=Mode.AUTO)
        # Data-level OVERWRITE replaces rows (the write below), not the schema:
        # keep the target's columns so a narrower source still lands with the
        # missing columns filled as NULL. Schema evolution stays opt-in through
        # ``schema_mode`` / ``overwrite_schema`` — don't force a CREATE OR
        # REPLACE that would shrink the table to the source's shape.
        target = self.create(
            data,
            mode=schema_mode,
        )
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).with_target(existing_schema)

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None

        wait = WaitingConfig.from_(wait)
        staging = self.insert_volume_path(target, temporary=bool(wait), staging_volume=staging_volume)
        output_data: "Tabular | None" = None
        staging.write_table(data, cast_options, mode=Mode.OVERWRITE)
        if return_data:
            output_data = staging.read_arrow_table()

        # Build the op and let it run its own DML — the staged Parquet is
        # referenced as ``parquet.`<path>``` and registered for post-load
        # cleanup, the INSERT/MERGE statements are prepared + executed via
        # ``execute_many``, all inside :meth:`DatabricksTableInsert.execute`.
        op = DatabricksTableInsert(
            target=target,
            mode=mode_enum,
            data=staging,
            client=self.client,
            schema=existing_schema,
            predicate=cast_options.predicate,
            match_by=match_by,
            update_column_names=update_column_names,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            safe_merge=safe_merge,
        )
        op.execute(wait=wait, raise_error=raise_error, engine=engine, retry=retry)

        return output_data if return_data else op.result

    # =========================================================================
    # spark_insert — Spark path, temp-view source
    # =========================================================================

    def spark_insert(
        self,
        data: Any,
        *,
        mode: Mode | str | None = None,
        schema_mode: Mode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_column_names: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
        predicate: Predicate | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        retry: Optional[WaitingConfigArg] = None,
        return_data: bool = False,
        safe_merge: bool = False,
        partition_filters: Optional[list[str]] = None,
    ) -> "StatementBatch | Tabular | None":
        """Insert into this table using Spark.

        ``retry`` is applied to DML statements (INSERT/MERGE/DELETE/UPDATE)
        only — TRUNCATE/OPTIMIZE/VACUUM stay non-retryable.
        :class:`SparkStatementResult` already auto-promotes transient
        Delta failures (``ConcurrentAppendException``, …) to retryable;
        passing ``retry=True`` (or any :class:`WaitingConfig` arg) makes
        the policy explicit instead of relying on auto-promote.

        Returns the submitted :class:`StatementBatch` by default. With
        ``return_data=True``, returns a :class:`Dataset` wrapping
        the materialised source DataFrame — handy for chaining
        downstream transforms without re-querying the target.
        """
        cast_options = _coalesce_predicate(cast_options, predicate)

        from yggdrasil.spark.cast import any_to_spark_dataframe
        from yggdrasil.spark.statement import SparkPreparedStatement

        mode_enum = Mode.from_(mode, default=Mode.AUTO)
        # Data-level OVERWRITE replaces rows (the write below), not the schema:
        # keep the target's columns so a narrower source still lands with the
        # missing columns filled as NULL. Schema evolution stays opt-in through
        # ``schema_mode`` / ``overwrite_schema`` — don't force a CREATE OR
        # REPLACE that would shrink the table to the source's shape.
        target = self.create(
            data,
            mode=schema_mode,
        )
        target_location = target.full_name(safe=True)
        existing_schema = target.collect_schema()
        cast_options = CastOptions.check(options=cast_options).check_target(
            target.collect_data_field(),
        )

        sql_engine = self.sql
        session = spark_session or sql_engine.spark.resolve_session(create=True)
        data_df = any_to_spark_dataframe(data, cast_options)

        if match_by == "auto":
            match_by = [f.name for f in existing_schema.primary_fields] or None
        prune_predicates = _build_where_predicates(
            cast_options.predicate, target_alias="T",
        )

        # Spark fast path for keyed APPEND under ``safe_merge=True``
        # (see :func:`_spark_filter_existing_keys`). Catalyst's
        # anti-join only reads the target's key columns from disk,
        # so this is dramatically cheaper than the SQL NOT EXISTS
        # shape used on the warehouse path. ``safe_merge=False``
        # leaves the work to a native MERGE INTO statement.
        anti_join_handled = False
        if (
            safe_merge
            and match_by
            and mode_enum in (Mode.APPEND, Mode.AUTO)
        ):
            data_df, anti_join_handled = _spark_filter_existing_keys(
                session=session,
                data_df=data_df,
                target_location=target_location,
                match_by=list(match_by),
            )

        view_name = f"_yg_src_{uuid.uuid4().hex}"
        data_df.createOrReplaceTempView(view_name)

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)
        # Plain column projection — :func:`any_to_spark_dataframe`
        # already aligned the DataFrame to the target schema, and the
        # INSERT itself applies the column-boundary coercion, so the
        # SQL stays free of per-column CASTs.
        source_sql = f"SELECT {cols_quoted} FROM {quote_ident(view_name)}"

        # The DataFrame anti-join already dedup'd; emit a plain INSERT
        # so we don't pay for the SQL-side NOT EXISTS twice.
        effective_mode = (
            Mode.OVERWRITE if anti_join_handled else mode_enum
        )
        effective_match_by = None if anti_join_handled else match_by
        # OVERWRITE-with-no-match_by would normally trigger a target
        # delete up front; skip that — the fast path is *append*.
        if anti_join_handled:
            sql_texts = [
                f"INSERT INTO {target_location} ({cols_quoted})\n{source_sql}"
            ]
            _append_maintenance_statements(
                sql_texts,
                target_location=target_location,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                keyed=True,
                vacuum_hours=vacuum_hours,
            )
        else:
            sql_texts = _build_dml_statements(
                target_location=target_location,
                source_sql=source_sql,
                columns=columns,
                mode=effective_mode,
                match_by=effective_match_by,
                update_column_names=update_column_names,
                prune_predicates=prune_predicates,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                safe_merge=safe_merge,
                partition_filters=partition_filters,
            )

        retry_cfg = _resolve_retry(retry)

        def _prepare_spark_batch(texts: list[str]) -> list[SparkPreparedStatement]:
            return [
                SparkPreparedStatement(text=sql, spark_session=session)
                for sql in texts
            ]

        prepared = _prepare_spark_batch(sql_texts)

        logger.debug(
            "Inserting via Spark into table %r (mode=%s, match_by=%s, "
            "statements=%d, retry=%s, anti_join=%s)",
            target_location, mode_enum.name, match_by, len(prepared),
            retry_cfg is not None, anti_join_handled,
        )

        applied_conf = _delta_conf_for(overwrite_schema, spark_options)

        primary_batch = None
        try:
            with sql_engine.spark.scoped_spark_conf(session, applied_conf):
                primary_batch = _execute_dml(
                    sql_engine,
                    statements=prepared,
                    wait=wait,
                    raise_error=raise_error,
                    engine="spark",
                )
            logger.info(
                "Inserted via Spark into table %r (mode=%s, match_by=%s, "
                "statements=%d, anti_join=%s)",
                target_location, mode_enum.name, match_by, len(prepared),
                anti_join_handled,
            )
        finally:
            try:
                session.catalog.dropTempView(view_name)
            except Exception:
                logger.debug("Failed to drop temp view %r; continuing.", view_name, exc_info=True)
            if not return_data:
                try:
                    data_df.unpersist()
                except Exception:
                    logger.debug("Failed to unpersist cached source; continuing.", exc_info=True)

        if return_data:
            from yggdrasil.spark.tabular import SparkDataset
            return SparkDataset(data_df)
        return primary_batch

    # =========================================================================
    # Storage & credentials
    # =========================================================================

    def schema_storage_location(
        self,
        table_type: Optional[TableType] = None,
    ) -> str:
        infos = self.client.workspace_client().schemas.get(
            full_name=self.schema_full_name()
        )
        if not infos.storage_location:
            raise NotImplementedError

        if table_type == TableType.EXTERNAL and "/__unitystorage" in infos.storage_location:
            root = infos.storage_location.split("/__unitystorage")[0]
            return root + "/catalogs/%s/schemas/%s" % (
                self.catalog_name or "default",
                self.schema_name or "default",
            )

        return infos.storage_location

    def storage_location(self) -> str | None:
        """Return the raw storage-location URL string for this table, or
        ``None`` when the table has no resolvable metadata.

        For a Delta table this is the cloud-object root that contains
        the parquet data files plus the ``_delta_log`` directory.
        :meth:`storage_path` wraps the same URL in an
        :class:`AWSClient`-backed Path so callers can ``iterdir()`` /
        ``read_bytes()`` it directly.
        """
        infos = self.read_infos(default=None)

        if infos is None:
            return None

        return infos.storage_location

    def storage_path(self, *, write: "bool | None" = None) -> "Path | None":
        """Return the table's backing storage as an addressable :class:`Path`.

        For a Delta table, ``tbl.storage_path()`` yields a Path that
        contains the parquet data files plus the ``_delta_log``
        transaction directory — ``list(tbl.storage_path().iterdir())``
        is the natural way to inspect the on-disk layout.

        ``write`` picks the UC temporary-credential scope the Path's
        :class:`AWSClient` vends — important because a principal can hold
        read but not write on a table:

        - ``None`` (default) — the operation default for the table type
          (``READ`` for managed, ``READ_WRITE`` for external);
        - ``False`` — ``READ`` (least-privilege; reads a table you can't write);
        - ``True`` — ``READ_WRITE`` (collapses to ``READ`` for managed, which
          UC never vends write creds for).
        """
        location = self.storage_location()
        if location is None:
            return None
        if write is None:
            aws = self.aws()
        else:
            aws = self.aws(TableOperation.READ_WRITE if write else TableOperation.READ)
        return aws.s3.path(location)

    def delta(self, *, write: "bool | None" = None) -> "DeltaFolder":
        """Return a :class:`~yggdrasil.io.delta.DeltaFolder` over this table's
        backing storage — the native Delta read/write surface.

        Built from :meth:`storage_path` so the folder (and every parquet /
        ``_delta_log`` child it resolves) inherits the table's
        temporary-credential :class:`AWSClient`; constructing a DeltaFolder
        from the bare URI string would drop those creds. Lets callers read
        (``tbl.delta().read_arrow_table()``) or commit
        (``tbl.delta().write_arrow_table(t, mode=Mode.APPEND)``) straight
        against the transaction log, bypassing the warehouse.

        ``write`` flows to :meth:`storage_path` to scope the vended
        credentials — ``write=False`` for a read-only handle (works even when
        the caller can't write the table), ``write=True`` for a commit.
        """
        from yggdrasil.io.delta import DeltaFolder

        path = self.storage_path(write=write)
        if path is None:
            raise FileNotFoundError(
                f"{self!r} has no resolvable storage_location for a DeltaFolder."
            )
        return DeltaFolder(path=path)

    def aws(
        self,
        operation: "TableOperation | ModeLike | None" = None,
        *,
        region: Optional[str] = None,
    ) -> "AWSClient":
        """Return an :class:`AWSClient` whose credentials self-refresh
        from Unity Catalog's ``temporary_table_credentials`` API.

        Routes through :meth:`credentials_refresher` — every
        :class:`Table` instance pointing at the same UC table id
        collapses to one provider that handles both read and write
        modes internally. The provider caches its :class:`AWSClient`
        per ``(mode, region)`` so the boto session,
        :class:`RefreshableCredentials`, connection pool, and STS
        vending are shared across every caller on the same scope.

        ``operation`` accepts a :class:`TableOperation`, a
        :class:`Mode` / mode-like string, or ``None`` (defaults to the
        right operation for this table's type).
        """
        op = _resolve_table_operation(operation, self.infos.table_type)
        mode = Mode.READ_ONLY if op == TableOperation.READ else Mode.OVERWRITE
        return self.credentials_refresher().aws_client(mode=mode, region=region)

    def credentials_refresher(self) -> "AWSDatabricksTableCredentials":
        """Return the process-wide singleton credentials provider for
        this table.

        Keyed by ``table_id``; handles both read and write modes
        internally via :meth:`AWSDatabricksTableCredentials.get_credentials`.
        """
        from yggdrasil.databricks.aws import AWSDatabricksTableCredentials

        return AWSDatabricksTableCredentials(
            table_id=self.table_id,
            client=self.client,
        )

    def temporary_credentials(self, operation: TableOperation = TableOperation.READ):
        return (
            self.client.workspace_client()
            .temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=operation,
            )
        )


# ===========================================================================
# SQL filter helpers  (used by to_arrow_dataset)
# ===========================================================================

def _build_predicate(col: str, op: str, val: Any) -> str:
    """Build a single SQL WHERE predicate from a ``(col, op, val)`` tuple."""
    _ALLOWED_OPS = {
        "=", "!=", "<>", ">", ">=", "<", "<=",
        "LIKE", "NOT LIKE", "IS", "IS NOT", "IN", "NOT IN",
    }
    op_norm = op.strip().upper()
    if op_norm == "==":
        op_norm = "="
    elif op_norm not in _ALLOWED_OPS:
        raise ValueError(f"Unsupported filter operator: {op!r}")

    col_sql = quote_qualified_ident(col)

    if val is None:
        return f"{col_sql} IS NULL"

    if op_norm in ("IS", "IS NOT"):
        return f"{col_sql} {op_norm} {sql_literal(val)}"

    if op_norm in ("IN", "NOT IN"):
        raw = str(val).strip()
        if raw.lower().startswith("sql:"):
            return f"{col_sql} {op_norm} {sql_literal(raw)}"
        inner = raw.strip("()")
        items = [x.strip() for x in inner.split(",") if x.strip()]
        if not items:
            return "FALSE" if op_norm == "IN" else "TRUE"
        return f"{col_sql} {op_norm} ({', '.join(sql_literal(x) for x in items)})"

    return f"{col_sql} {op_norm} {sql_literal(val)}"
