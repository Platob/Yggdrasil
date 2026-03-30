"""
Per-table resource: DDL, DML, schema introspection and storage helpers.

The :class:`Table` dataclass wraps a single Unity Catalog table and exposes
instance-level methods only.  Collection operations (``find_table``,
``list_tables``) live in :mod:`~yggdrasil.databricks.sql.tables`.

Caching strategy
----------------
``_infos`` and ``_columns`` are instance-level one-shot caches.

    1. **Local** — return the cached value immediately if set.
    2. **Remote** — call ``_fetch_infos_remote`` / ``_fetch_columns_remote``
       only on a miss.
    3. **Update** — store the fetched value so the next access is free.

Debug-level log lines are emitted on every cache hit so you can trace the
access pattern without grep-ing for API calls.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING, Mapping, AnyStr

import pyarrow as pa
from databricks.sdk.client_types import HostType
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    TableInfo,
    TableOperation,
    TableType,
    DataSourceFormat,
)
from pyarrow.fs import FileSystem, S3FileSystem

from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Schema
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import URL
from yggdrasil.io.enums.save_mode import SaveModeArg
from .column import Column
from .types import (
    arrow_field_to_column_info, arrow_field_to_ddl, quote_ident,
    escape_sql_string,
)

if TYPE_CHECKING:
    import delta
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.sql.tables import Tables

__all__ = ["Table"]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


# ===========================================================================
# Table — per-table resource
# ===========================================================================

@dataclass
class Table(DatabricksResource):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers."""

    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None

    # TTL for the _infos cache (seconds).  Set to None to disable expiry.
    _infos_ttl: Optional[float] = field(default=1800.0, repr=False, compare=False, hash=False)

    _infos: Optional[TableInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: Optional[float] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _columns: Optional[list[Column]] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )

    @property
    def url(self) -> URL:
        return (
            self.client.base_url
            .with_path(f"/explore/data/{self.catalog_name}/{self.schema_name}/{self.table_name}")
        )

    @classmethod
    def parse(
        cls,
        obj: Any,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        service: "Tables"
    ):
        if isinstance(obj, cls):
            return obj

        return cls.parse_str(
            location=str(obj) if obj is not None else None,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=service
        )

    @classmethod
    def parse_str(
        cls,
        location: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        service: "Tables",
    ):
        _, catalog_name, schema_name, table_name = service.parse_check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
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
        """Shorthand for the :class:`SQLEngine` attached to this table's client."""
        return self.client.sql

    # =========================================================================
    # Identity / repr
    # =========================================================================

    def schema_full_name(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"

    def full_name(self, safe: str | bool | None = None) -> str:
        """Return the three-part table name, optionally backtick-quoted."""
        if safe:
            q = safe if isinstance(safe, str) else "`"
            return (
                f"{q}{self.catalog_name}{q}"
                f".{q}{self.schema_name}{q}"
                f".{q}{self.table_name}{q}"
            )
        return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    def __repr__(self) -> str:
        return f"Table<{self.url.to_string()!r}>"

    def __str__(self):
        return self.full_name(safe=True)

    def __getitem__(self, item: str) -> Column:
        return self.column(item)

    # =========================================================================
    # Cache management
    # =========================================================================

    def _reset_cache(self) -> None:
        """Evict all cached instance state (``_infos``, ``_columns``, ``_infos_fetched_at``)."""
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)

    def clear(self) -> "Table":
        """Public alias for :meth:`_reset_cache`; returns ``self``."""
        self._reset_cache()
        return self

    # =========================================================================
    # Remote fetchers — single responsibility, no caching inside
    # =========================================================================

    def _fetch_columns_remote(self) -> list[Column]:
        """Derive :class:`Column` objects from current ``infos``."""
        return [
            Column.from_api(table=self, infos=col_info)
            for col_info in self.infos.columns
        ]

    # =========================================================================
    # Databricks SDK — lazy-loaded properties
    # =========================================================================

    @property
    def exists(self) -> bool:
        try:
            _ = self.infos
            return True
        except NotFound:
            return False

    @property
    def table_id(self) -> str:
        return self.infos.table_id

    @property
    def infos(self) -> TableInfo:
        """TableInfo — local cache first (TTL-guarded), then remote on miss."""
        now = time.time()

        # 1. Local cache — valid when present and within TTL
        if self._infos is not None:
            age = now - (self._infos_fetched_at or 0.0)
            if self._infos_ttl is None or age < self._infos_ttl:
                logger.debug(
                    "Cache hit [Table._infos] table=%s table_id=%s age=%.0fs",
                    self.full_name(), self._infos.table_id, age,
                )
                return self._infos
            logger.debug(
                "Cache expired [Table._infos] table=%s age=%.0fs ttl=%.0fs — refreshing",
                self.full_name(), age, self._infos_ttl,
            )

        # 2. Remote fetch
        infos = self.client.tables.find_table_remote(
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
        )

        # 3. Update cache + record fetch timestamp
        object.__setattr__(self, "_infos", infos)
        object.__setattr__(self, "_infos_fetched_at", now)
        return self._infos

    # =========================================================================
    # Arrow schema introspection
    # =========================================================================

    def _schema_metadata(self) -> dict[bytes, bytes]:
        return {
            b"engine":       b"databricks",
            b"name":         self.full_name().encode(),
            b"catalog_name": self.catalog_name.encode(),
            b"schema_name":  self.schema_name.encode(),
            b"table_name":   self.table_name.encode(),
        }

    @property
    def columns(self) -> list[Column]:
        """Column list — local cache first, then remote on miss."""
        # 1. Local cache
        if self._columns is not None:
            logger.debug(
                "Cache hit [Table._columns] table=%s count=%d",
                self.full_name(), len(self._columns),
            )
            return self._columns

        # 2. Remote fetch + update cache
        columns = self._fetch_columns_remote()
        object.__setattr__(self, "_columns", columns)
        return self._columns

    def column(self, name: str) -> Column:
        """Return a :class:`Column` by name (case-sensitive)."""
        for col in self.columns:
            if col.name == name:
                return col
        raise ValueError(f"Column {name!r} not found in {self!r}")

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [col.arrow_field for col in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return pa.schema(self.arrow_fields, metadata=self._schema_metadata())

    @property
    def arrow_field(self) -> pa.Field:
        """This table as a single Arrow struct field (useful for nested schemas)."""
        return pa.field(
            self.full_name(),
            pa.struct(self.arrow_fields),
            metadata=self._schema_metadata(),
        )

    # =========================================================================
    # Tag helpers
    # =========================================================================

    @staticmethod
    def _sql_str(value: str) -> str:
        """Escape a string for use as a SQL string literal."""
        return "'" + Table._safe_str(value).replace("'", "''") + "'"

    @staticmethod
    def _safe_str(value: AnyStr) -> str:
        """Escape a string for use as an unquoted SQL identifier."""
        if not value:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, (bytes, memoryview, bytearray)):
            return value.decode()

        return str(value)

    def set_tags_ddl(self, tags: Mapping[str, str] | None) -> str:
        pairs: list[str] = []
        for k, v in (tags or {}).items():
            key = self._safe_str(k).strip() if k is not None else ""
            val = self._safe_str(v).strip() if v is not None else ""
            if key and val:
                pairs.append(f"{self._sql_str(key)} = {self._sql_str(val)}")

        if not pairs:
            raise ValueError(f"Cannot set empty tags on {self!r}")

        return f"ALTER TABLE {self.full_name(safe=True)} SET TAGS ({', '.join(pairs)})"

    def set_tags(self, tags: Mapping[str, str] | None) -> "Table":
        if not tags:
            return self
        self.sql.execute(self.set_tags_ddl(tags))
        return self

    # =========================================================================
    # Lifecycle — create / ensure / delete
    # =========================================================================

    def ensure_created(
        self,
        definition: Union[pa.Schema, Any, None],
        *,
        storage_location: Optional[str] = None,
        comment: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
    ) -> "Table":
        """Create the table if it does not exist, then return ``self``."""
        if not self.exists:
            if definition is None:
                _ = self.infos  # surface a meaningful error
            self.create(
                definition=definition,
                storage_location=storage_location,
                comment=comment,
                properties=properties,
                table_type=table_type,
                data_source_format=data_source_format,
                if_not_exists=True,
            )
        return self

    def create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        partition_by: Optional[list[str]] = None,
        storage_location: Optional[str] = None,
        comment: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = True,
    ) -> "Table":
        """Create the table, routing to SQL DDL or the Unity Catalog REST API."""
        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.MANAGED:
            self.sql_create(
                definition,
                comment=comment,
                partition_by=partition_by,
                if_not_exists=if_not_exists,
            )
        else:
            if table_type == TableType.EXTERNAL and not storage_location:
                storage_location = (
                    self.schema_storage_location(table_type=table_type)
                    + "/tables/%s" % self.table_name
                )
            return self.api_create(
                definition=definition,
                storage_location=storage_location,
                comment=comment,
                properties=properties,
                table_type=table_type,
                data_source_format=data_source_format,
                if_not_exists=if_not_exists,
            )

        self._reset_cache()
        return self

    def sql_create(
        self,
        description: Union[pa.Field, pa.Schema, Any],
        *,
        storage_location: Optional[str] = None,
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[list[str]] = None,
        comment: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        if_not_exists: bool = True,
        or_replace: bool = False,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: Optional[bool] = None,
        enable_deletion_vectors: Optional[bool] = None,
        target_file_size: Optional[int] = None,
        column_mapping_mode: Optional[str] = None,
        wait_result: bool = True,
        auto_tag: bool = True,
    ) -> "Table":
        """Generate and execute a ``CREATE TABLE`` DDL statement."""
        schema = Schema.from_any(description)
        arrow_fields = schema.arrow_fields
        partition_by = partition_by or schema.partition_by
        cluster_by = cluster_by or schema.cluster_by
        comment = comment or schema.comment

        any_invalid = any(_needs_column_mapping(f.name) for f in arrow_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        column_definitions = [arrow_field_to_ddl(child) for child in arrow_fields]

        if or_replace and if_not_exists:
            raise ValueError("Use either or_replace or if_not_exists, not both.")

        if or_replace:
            create_kw = "CREATE OR REPLACE TABLE"
        elif if_not_exists:
            create_kw = "CREATE TABLE IF NOT EXISTS"
        else:
            create_kw = "CREATE TABLE"

        sql_parts: list[str] = [
            f"{create_kw} {self.full_name(safe=True)} (",
            "  " + ",\n  ".join(column_definitions),
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

        try:
            self.sql.execute(statement, wait=wait_result)
        except Exception as exc:
            if "SCHEMA_NOT_FOUND" in str(exc):
                self.sql.execute(
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.schema_name)}", wait=True
                )
                self.sql.execute(statement, wait=wait_result)
            else:
                raise

        self._reset_cache()

        if auto_tag:
            schema.autotag(tags={
                b"format": data_source_format.value
            })

        self.set_tags(schema.tags)

        for f in schema.fields:
            if f.tags:
                self.column(f.name).set_tags(f.tags)

        return self

    def api_create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        storage_location: Optional[str] = None,
        comment: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = False,
    ) -> "Table":
        """Create table via the Unity Catalog REST API (supports EXTERNAL tables)."""
        definition = Schema.from_any(definition).to_arrow_schema()

        if not comment and definition.metadata:
            raw = definition.metadata.get(b"comment") or definition.metadata.get(b"description")
            comment = raw.decode("utf-8") if isinstance(raw, bytes) else raw

        columns = [
            arrow_field_to_column_info(field=f, position=pos)
            for pos, f in enumerate(definition)
        ]

        body: dict[str, Any] = {
            "catalog_name":       self.catalog_name,
            "schema_name":        self.schema_name,
            "name":               self.table_name,
            "table_type":         table_type.value,
            "data_source_format": data_source_format.value,
            "columns":            [c.as_dict() for c in columns],
        }
        if storage_location is not None:
            body["storage_location"] = storage_location
        if comment is not None:
            body["comment"] = comment
        if properties:
            body["properties"] = properties

        headers = {
            "Accept":       "application/json",
            "Content-Type": "application/json",
        }

        uc = self.client.workspace_client().tables
        cfg = uc._api._cfg
        if cfg.host_type == HostType.UNIFIED and cfg.workspace_id:
            headers["X-Databricks-Org-Id"] = cfg.workspace_id

        try:
            res = uc._api.do(
                "POST", "/api/2.1/unity-catalog/tables",
                body=body, headers=headers,
            )
            info = TableInfo.from_dict(res)
            object.__setattr__(self, "_infos", info)
        except DatabricksError as exc:
            if "already exists" in str(exc):
                if not if_not_exists:
                    raise
            else:
                raise

        self._reset_cache()
        return self

    def delete(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Table":
        """Delete this table from Unity Catalog.

        Args:
            wait:        Block until deletion completes (``True``) or fire-and-
                         forget in a background :class:`Job` (``False``).
            raise_error: Re-raise :exc:`DatabricksError` on failure.
        """
        uc = self.client.workspace_client().tables

        if wait:
            try:
                uc.delete(full_name=self.full_name())
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()

        self._reset_cache()
        return self

    # =========================================================================
    # Spark / Delta integration
    # =========================================================================

    def delta_spark(
        self,
        spark_session: "SparkSession | None" = None,
    ) -> "delta.tables.DeltaTable":
        """Return a Delta ``DeltaTable`` handle for this table."""
        try:
            from delta.tables import DeltaTable
        except ImportError:
            from yggdrasil.environ import runtime_import_module
            m = runtime_import_module(
                module_name="delta.tables", pip_name="delta-spark", install=True,
            )
            DeltaTable = m.DeltaTable

        session = spark_session or PyEnv.spark_session(
            create=True, import_error=True, install_spark=True,
        )
        return DeltaTable.forName(sparkSession=session, tableOrViewName=self.full_name(safe=True))

    def to_spark(
        self,
        spark_session: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
        """Return the table as a Spark DataFrame via Delta."""
        return self.delta_spark(spark_session=spark_session).toDF()

    # =========================================================================
    # Data I/O
    # =========================================================================

    def to_arrow_dataset(
        self,
        *,
        filters: Optional[list[tuple[str, str, str]]] = None,
        wait: WaitingConfigArg = True,
        cache_for: WaitingConfigArg = None,
    ):
        """Execute ``SELECT *`` (optionally filtered) and return an Arrow dataset."""
        statement = f"SELECT * FROM {self.full_name(safe=True)}"
        if filters:
            predicates = [_build_predicate(c, o, v) for c, o, v in filters]
            statement += " WHERE " + " AND ".join(predicates)

        result = self.sql.execute(
            statement,
            wait=wait,
            cache_for=cache_for,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )
        return result.to_arrow_dataset()

    def insert(
        self,
        data: Any,
        *,
        mode: SaveModeArg = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        spark_session: Optional["SparkSession"] = None,
    ) -> None:
        """Insert data into this table.

        Delegates to :meth:`SQLEngine.insert_into` which routes between the
        Spark and warehouse-SQL paths automatically.
        """
        return self.sql.insert_into(
            data,
            mode=mode,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
            table=self,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error,
            spark_session=spark_session,
        )

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

    @property
    def storage_location(self) -> str:
        return self.infos.storage_location

    def credentials(self, operation: TableOperation = TableOperation.READ):
        """Generate temporary table credentials for direct S3/GCS access."""
        return (
            self.client.workspace_client()
            .temporary_table_credentials
            .generate_temporary_table_credentials(
                table_id=self.table_id,
                operation=operation,
            )
        )

    def arrow_filesystem(
        self,
        *,
        operation: TableOperation = TableOperation.READ_WRITE,
        expiring: bool = False,
    ) -> Union[FileSystem, "TableFilesystem"]:
        """Return a credentialed :class:`S3FileSystem` for direct Parquet I/O."""
        created_at = time.time_ns()
        creds = self.credentials(operation=operation)
        assert creds.aws_temp_credentials, "Cannot get AWS credentials"
        aws = creds.aws_temp_credentials

        base = S3FileSystem(
            access_key=aws.access_key_id,
            secret_key=aws.secret_access_key,
            session_token=aws.session_token,
            region="eu-west-1",
        )

        if not expiring:
            return base

        ttl_ns = 3_600_000_000_000
        return TableFilesystem.create(
            value=base,
            created_at=created_at,
            ttl=ttl_ns,
            expires_at=created_at + ttl_ns,
            table=self,
            operation=operation,
        )


# ===========================================================================
# TableFilesystem — auto-refreshing S3 filesystem credentials
# ===========================================================================

@dataclass
class TableFilesystem(Expiring[FileSystem]):
    """Expiring wrapper around S3FileSystem that auto-refreshes UC credentials."""

    table: Optional[Table] = field(default=None)
    operation: TableOperation = TableOperation.READ

    def _refresh(self) -> RefreshResult[FileSystem]:
        value = self.table.arrow_filesystem(operation=self.operation, expiring=False)
        created_ns = time.time_ns()
        ttl_ns = 3_600_000_000_000  # 1 hour
        return RefreshResult(
            value=value,
            created_at_ns=created_ns,
            ttl_ns=ttl_ns,
            expires_at_ns=created_ns + ttl_ns,
        )


# ===========================================================================
# SQL filter helpers  (used by to_arrow_dataset)
# ===========================================================================

def _quote_ident(name: str) -> str:
    """Backtick-quote each segment of a dotted identifier."""
    parts = [p.strip() for p in name.split(".") if p.strip()]
    return ".".join(f"`{p.replace('`', '``')}`" for p in parts)


def _sql_literal(v: Any) -> str:
    """Convert a Python value to a SQL literal string."""
    if isinstance(v, (bytes, bytearray, memoryview)):
        b64 = base64.b64encode(bytes(v)).decode("ascii")
        return f"'{b64}'"

    s = str(v).strip()
    if s.lower().startswith("sql:"):
        return s[4:].strip()
    if s.lower() in ("null", "none"):
        return "NULL"
    if s.lower() in ("true", "false"):
        return s.upper()
    try:
        float(s)
        return s
    except ValueError:
        pass
    return "'" + s.replace("'", "''") + "'"


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

    col_sql = _quote_ident(col)

    if val is None:
        return f"{col_sql} IS NULL"

    if op_norm in ("IS", "IS NOT"):
        return f"{col_sql} {op_norm} {_sql_literal(val)}"

    if op_norm in ("IN", "NOT IN"):
        raw = str(val).strip()
        if raw.lower().startswith("sql:"):
            return f"{col_sql} {op_norm} {_sql_literal(raw)}"
        inner = raw.strip("()")
        items = [x.strip() for x in inner.split(",") if x.strip()]
        if not items:
            return "FALSE" if op_norm == "IN" else "TRUE"
        return f"{col_sql} {op_norm} ({', '.join(_sql_literal(x) for x in items)})"

    return f"{col_sql} {op_norm} {_sql_literal(val)}"

