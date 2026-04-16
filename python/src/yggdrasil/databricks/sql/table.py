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

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, TYPE_CHECKING, Mapping, Iterable

import pyarrow as pa
from databricks.sdk.errors import DatabricksError, NotFound
from databricks.sdk.service.catalog import (
    TableInfo,
    TableOperation,
    TableType,
    DataSourceFormat,
)
from pyarrow.fs import FileSystem, S3FileSystem

from yggdrasil.concurrent.threading import Job
from yggdrasil.data import Field
from yggdrasil.data.schema import Schema as DataSchema
from yggdrasil.databricks.client import DatabricksResource
from yggdrasil.databricks.iam import IAMUser, IAMGroup
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import URL
from yggdrasil.io.enums.save_mode import SaveModeArg, SaveMode
from .column import Column
from .sql_utils import (
    DEFAULT_TAG_COLLATION,
    _build_table_constraints_sql,
    _safe_constraint_name,
    _safe_str,
    databricks_tag_literal,
    quote_ident,
    quote_principal,
    quote_qualified_ident,
    sql_literal, escape_sql_string,
)
from .types import PrimaryKeySpec, ForeignKeySpec

if TYPE_CHECKING:
    import delta
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from yggdrasil.databricks.sql.engine import SQLEngine
    from yggdrasil.databricks.sql.tables import Tables
    from yggdrasil.databricks.sql.catalog import Catalog
    from yggdrasil.databricks.sql.schema import Schema as UCSchema, Schema

__all__ = ["Table"]

logger = logging.getLogger(__name__)

_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


INFOS_TTL: float = 300.0


# ===========================================================================
# Table — per-table resource
# ===========================================================================

@dataclass
class Table(DatabricksResource):
    """A single Unity Catalog table — DDL, DML, schema, storage helpers."""

    catalog_name: str = "default"
    schema_name: str = "default"
    table_name: str = "default"

    @property
    def name(self):
        return self.table_name

    _infos: Optional[TableInfo] = field(
        default=None, init=False, repr=False, compare=False, hash=False,
    )
    _infos_fetched_at: float | None = field(
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
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
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

    @property
    def catalog(self) -> "Catalog":
        from .catalog import Catalog as _Catalog
        from .catalogs import Catalogs
        return _Catalog(
            service=Catalogs(client=self.client),
            catalog_name=self.catalog_name,
        )

    @property
    def schema(self) -> "UCSchema":
        from .schema import Schema as _Schema
        from .catalogs import Catalogs
        return _Schema(
            service=Catalogs(client=self.client),
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

    def __repr__(self) -> str:
        return f"Table({self.url.to_string()!r})"

    def __str__(self):
        return self.full_name(safe=True)

    def __getitem__(self, item: str) -> Column:
        return self.column(item)

    # =========================================================================
    # Cache management
    # =========================================================================

    def _reset_cache(self, invalidate_cache: bool = False) -> None:
        if invalidate_cache:
            self.sql.tables.invalidate_cached_table(table=self)

        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_infos_fetched_at", None)
        object.__setattr__(self, "_columns", None)

    def clear(self) -> "Table":
        self._reset_cache()
        return self

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

    def _cache_expired(self) -> bool:
        if self._infos is None:
            return True
        now = time.time()
        age = now - (self._infos_fetched_at or 0.0)
        return age >= INFOS_TTL

    @property
    def infos(self) -> TableInfo:
        if self._cache_expired():
            infos = self.client.tables.find_table_remote(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=self.table_name,
            )

            self._reset_cache()
            object.__setattr__(self, "_infos", infos)
            object.__setattr__(self, "_infos_fetched_at", time.time())
            object.__setattr__(self, "_columns", [
                Column.from_api(table=self, infos=col_info)
                for col_info in self.infos.columns
            ])
        return self._infos

    # =========================================================================
    # Arrow schema introspection
    # =========================================================================

    @property
    def columns(self) -> list[Column]:
        if self._columns is None:
            # Refresh the cache if needed to ensure we have the latest column infos.
            _ = self.infos
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

    @property
    def data_schema(self) -> DataSchema:
        return DataSchema.from_any_fields(
            [c.field for c in self.columns],
            metadata={
                b"name": self.table_name.encode(),
                b"engine": b"databricks",
                b"catalog_name": self.catalog_name.encode(),
                b"schema_name": self.schema_name.encode(),
                b"table_name": self.table_name.encode(),
            }
        )

    @property
    def data_field(self):
        return self.data_schema.to_field()

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [c.field.to_arrow_field() for c in self.columns]

    @property
    def arrow_schema(self) -> pa.Schema:
        return self.data_schema.to_arrow_schema()

    @property
    def arrow_field(self) -> pa.Field:
        return self.data_field.to_arrow_field()

    # =========================================================================
    # Constraint helpers — public ALTER TABLE DDL builders
    # =========================================================================

    def add_primary_key_ddl(
        self,
        columns: "list[str] | str",
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: str | None = None,
    ) -> str:
        if isinstance(columns, str):
            columns = [columns]

        col_exprs = [
            f"`{c}` TIMESERIES" if timeseries == c else f"`{c}`"
            for c in columns
        ]
        cname = _safe_constraint_name(
            constraint_name or f"{self.table_name}_{'_'.join(columns)}_pk"
        )
        rely_clause = " RELY" if rely else ""
        return (
            f"ALTER TABLE {self.full_name(safe=True)} "
            f"ADD CONSTRAINT `{cname}` "
            f"PRIMARY KEY ({', '.join(col_exprs)}) NOT ENFORCED"
            f"{rely_clause}"
        )

    def set_primary_key(
        self,
        columns: "list[str] | str",
        *,
        constraint_name: str | None = None,
        rely: bool = False,
        timeseries: str | None = None,
    ) -> "Table":
        self.sql.execute(
            self.add_primary_key_ddl(
                columns,
                constraint_name=constraint_name,
                rely=rely,
                timeseries=timeseries,
            )
        )
        return self

    def drop_primary_key_ddl(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> str:
        if_exists_clause = " IF EXISTS" if if_exists else ""
        cascade_clause = " CASCADE" if cascade else ""
        return (
            f"ALTER TABLE {self.full_name(safe=True)} "
            f"DROP PRIMARY KEY{if_exists_clause}{cascade_clause}"
        )

    def drop_primary_key(
        self,
        *,
        if_exists: bool = True,
        cascade: bool = False,
    ) -> "Table":
        self.sql.execute(self.drop_primary_key_ddl(if_exists=if_exists, cascade=cascade))
        return self

    def set_tags_ddl(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> str:
        pairs: list[str] = []
        for k, v in (tags or {}).items():
            key = _safe_str(k).strip() if k is not None else ""
            val = _safe_str(v).strip() if v is not None else ""
            if key and val:
                pairs.append(
                    f"{databricks_tag_literal(key, collation=tag_collation)} = "
                    f"{databricks_tag_literal(val, collation=tag_collation)}"
                )

        if not pairs:
            raise ValueError(f"Cannot set empty tags on {self!r}")

        return f"ALTER TABLE {self.full_name(safe=True)} SET TAGS ({', '.join(pairs)})"

    def set_tags(
        self,
        tags: Mapping[str, str] | None,
        *,
        tag_collation: str | None = DEFAULT_TAG_COLLATION,
    ) -> "Table":
        if not tags:
            return self
        self.sql.execute(self.set_tags_ddl(tags, tag_collation=tag_collation))
        return self

    # =========================================================================
    # Constraint helpers — inline CREATE TABLE constraint rendering
    # =========================================================================

    # =========================================================================
    # Lifecycle — create / ensure / delete
    # =========================================================================

    def ensure_created(
        self,
        definition: Union[pa.Schema, Any, None],
        *,
        mode: SaveMode | str | None = None,
        **options
    ) -> "Table":
        return self.create(
            definition=definition,
            mode=mode,
            **options,
        )

    def update_column(
        self,
        column: Field,
        *,
        mode: SaveMode | str | None = None,
    ):
        return self.update_columns(
            [column],
            mode=mode,
        )

    def update_columns(
        self,
        columns: Iterable[Field],
        *,
        mode: SaveMode | str | None = None,
    ):
        mode = SaveMode.parse(mode, SaveMode.AUTO)
        statements: list[str] = []
        add_columns: list[str] = []
        alter_table = f"ALTER TABLE {self.full_name(safe=True)}"

        for column in columns:
            data_field = Field.from_any(column)

            existing = self.column(name=data_field.name, safe=False, raise_error=False)

            if existing is None:
                add_columns.append(
                    f"`{data_field.name}` {data_field.dtype.to_databricks_ddl()}"
                )
            elif mode in (SaveMode.APPEND, SaveMode.UPSERT, SaveMode.AUTO):
                if existing.name != data_field.name:
                    statements.append(
                        f"{alter_table} RENAME COLUMN `{existing.name}` TO `{data_field.name}`"
                    )

        if add_columns:
            statements.append(f"{alter_table} ADD COLUMNS ({', '.join(add_columns)})")

        if statements:
            self.sql.execute_many(statements)
            self._reset_cache(invalidate_cache=True)

        return self

    def create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        mode: SaveMode | str | None = None,
        partition_by: Optional[list[str]] = None,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = True,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        """Create the table, routing to SQL DDL or the Unity Catalog REST API.

        SQL path:
            - inlines PK/FK constraints directly in CREATE TABLE

        API path:
            - creates the table first
            - then applies PK/FK via ALTER TABLE fallback
        """
        mode = SaveMode.parse(mode, SaveMode.AUTO)
        schema = DataSchema.from_(definition)

        if self.exists:
            if mode == SaveMode.ERROR_IF_EXISTS:
                raise ValueError(f"Table {self!r} already exists")
            elif mode == SaveMode.IGNORE:
                return self
            return self.update_columns(
                schema.fields,
                mode=mode,
            )

        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.MANAGED:
            return self.sql_create(
                definition,
                comment=comment,
                partition_by=partition_by,
                if_not_exists=if_not_exists,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
            )

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
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
        )

    def sql_create(
        self,
        description: Union[pa.Field, pa.Schema, Any],
        *,
        storage_location: str | None = None,
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[list[str]] = None,
        comment: str | None = None,
        properties: Optional[dict[str, Any]] = None,
        if_not_exists: bool = True,
        or_replace: bool = False,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: bool | None = None,
        enable_deletion_vectors: bool | None = None,
        target_file_size: int | None = None,
        column_mapping_mode: str | None = None,
        wait_result: bool = True,
        auto_tag: bool = True,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        """Generate and execute a CREATE TABLE DDL statement.

        PK/FK constraints are rendered directly inside CREATE TABLE for the SQL path.
        PK columns are forced to NOT NULL in the column definition to match
        Databricks requirements for primary keys.
        """
        schema_info = DataSchema.from_any(description).autotag()
        partition_by = partition_by or schema_info.partition_by
        cluster_by = cluster_by or schema_info.cluster_by
        primary_keys = primary_keys or schema_info.primary_key_names
        foreign_keys = foreign_keys or schema_info.foreign_key_names
        comment = comment or schema_info.comment

        pk_spec = PrimaryKeySpec.from_any(primary_keys, schema=schema_info)
        fk_specs = ForeignKeySpec.from_any(foreign_keys, schema=schema_info)

        effective_fields: list[Field] = []
        column_definitions: list[str] = []
        for f in schema_info.children_fields:
            if f.primary_key and f.nullable:
                f = f.with_nullable(False)
            effective_fields.append(f)
            column_definitions.append(f.to_databricks_ddl())

        any_invalid = any(_needs_column_mapping(f.name) for f in effective_fields)
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        constraint_definitions = _build_table_constraints_sql(
            self.table_name,
            pk_spec,
            fk_specs,
        )
        table_definitions = column_definitions + constraint_definitions

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
                    f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.catalog_name)}.{quote_ident(self.schema_name)}",
                    wait=True,
                )
                self.sql.execute(statement, wait=wait_result)
            else:
                raise

        self._reset_cache(invalidate_cache=True)

        if schema_info.tags:
            self.set_tags(schema_info.tags)

        for f in schema_info.values():
            self.column(f.name).set_tags(f.tags)

        return self

    def api_create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        storage_location: str | None = None,
        comment: str | None = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = False,
        primary_keys: "list[str] | str | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> "Table":
        # TODO: support for table_type=TableType.EXTERNAL
        raise NotImplementedError("API path not implemented for SQL tables.")

    def _apply_constraints(
        self,
        pk_spec: "PrimaryKeySpec | None",
        fk_specs: "list[ForeignKeySpec]",
    ) -> None:
        """Apply PK then FK constraints via ALTER TABLE.

        Used as the fallback path for REST/API-based creation.
        Failures are logged and do not abort the successful table create.
        """
        if pk_spec and pk_spec.columns:
            try:
                self.set_primary_key(
                    pk_spec.columns,
                    constraint_name=pk_spec.constraint_name,
                    rely=pk_spec.rely,
                    timeseries=pk_spec.timeseries,
                )
                logger.debug(
                    "Applied PRIMARY KEY %r on %s",
                    pk_spec.columns, self.full_name(),
                )
            except Exception:
                logger.warning(
                    "Failed to apply PRIMARY KEY %r on %s — "
                    "the table was created; add the constraint manually.",
                    pk_spec.columns, self.full_name(),
                    exc_info=True,
                )

        for fk in fk_specs:
            try:
                self.column(fk.column).set_foreign_key(
                    fk.ref,
                    constraint_name=fk.constraint_name,
                    rely=fk.rely,
                    match_full=fk.match_full,
                    on_update_no_action=fk.on_update_no_action,
                    on_delete_no_action=fk.on_delete_no_action,
                )
                logger.debug(
                    "Applied FOREIGN KEY %r → %r on %s",
                    fk.column, fk.ref, self.full_name(),
                )
            except Exception:
                logger.warning(
                    "Failed to apply FOREIGN KEY %r → %r on %s — "
                    "add the constraint manually.",
                    fk.column, fk.ref, self.full_name(),
                    exc_info=True,
                )

    def delete(
        self,
        *,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> "Table":
        uc = self.client.workspace_client().tables

        if wait:
            try:
                uc.delete(full_name=self.full_name())
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(uc.delete, self.full_name()).fire_and_forget()

        self._reset_cache(invalidate_cache=True)
        return self

    # =========================================================================
    # Spark / Delta integration
    # =========================================================================

    def delta_spark(
        self,
        spark_session: "SparkSession | None" = None,
    ) -> "delta.tables.DeltaTable":  # noqa
        from delta.tables import DeltaTable  # noqa

        session = spark_session or PyEnv.spark_session(
            create=True, import_error=True, install_spark=False,
        )
        return DeltaTable.forName(sparkSession=session, tableOrViewName=self.full_name(safe=True))

    def to_spark(
        self,
        spark_session: Optional["SparkSession"] = None,
    ) -> "SparkDataFrame":
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

    def add_permissions(
        self,
        iam_id: str | list[str] | None = None,
        *,
        users: Iterable[IAMUser | str] | None = None,
        groups: Iterable[IAMGroup | str] | None = None,
        group: Iterable[IAMGroup | str] | None = None,
        privileges: str | Iterable[str] | None = None,
    ) -> "Table":
        principals = self._normalize_permission_principals(
            iam_id=iam_id,
            users=users,
            groups=groups,
            group=group,
        )
        grant_privileges = self._normalize_table_privileges(privileges)

        if not principals:
            raise ValueError("add_permissions requires at least one user, group, or iam_id")

        for principal in principals:
            self.sql.execute(self.grant_permissions_ddl(principal, grant_privileges))

        return self

    @staticmethod
    def _normalize_table_privileges(
        privileges: str | Iterable[str] | None,
    ) -> tuple[str, ...]:
        """Normalize table privilege names to Databricks SQL GRANT syntax."""
        if privileges is None:
            privileges = ("SELECT",)
        elif isinstance(privileges, str):
            privileges = (privileges,)

        normalized: list[str] = []
        for privilege in privileges:
            value = str(privilege).strip()
            if not value:
                continue
            value = " ".join(value.replace("_", " ").replace("-", " ").upper().split())
            if value not in normalized:
                normalized.append(value)

        if not normalized:
            raise ValueError("add_permissions requires at least one privilege")

        return tuple(normalized)

    @staticmethod
    def _principal_from_user(user: IAMUser | str) -> str:
        if isinstance(user, str):
            principal = user.strip()
        else:
            principal = user.username or user.email or user.name or user.id or ""

        if not principal:
            raise ValueError("User principal must have a username, email, name, or id")

        return principal

    @staticmethod
    def _principal_from_group(group: IAMGroup | str) -> str:
        if isinstance(group, str):
            principal = group.strip()
        else:
            principal = group.name or group.id or ""

        if not principal:
            raise ValueError("Group principal must have a name or id")

        return principal

    def _normalize_permission_principals(
        self,
        *,
        iam_id: str | list[str] | None,
        users: Iterable[IAMUser | str] | None,
        groups: Iterable[IAMGroup | str] | None,
        group: Iterable[IAMGroup | str] | None,
    ) -> tuple[str, ...]:
        """Collect principals from supported user/group inputs preserving order."""
        seen: set[str] = set()
        out: list[str] = []

        def add(v: str) -> None:
            principal = v.strip()
            if principal and principal not in seen:
                seen.add(principal)
                out.append(principal)

        if isinstance(iam_id, str):
            add(iam_id)
        elif iam_id:
            for value in iam_id:
                add(str(value))

        for user in users or ():
            add(self._principal_from_user(user))

        for entry in groups or ():
            add(self._principal_from_group(entry))

        for entry in group or ():
            add(self._principal_from_group(entry))

        return tuple(out)

    def grant_permissions_ddl(
        self,
        principal: str,
        privileges: str | Iterable[str],
    ) -> str:
        """Build a ``GRANT`` DDL statement for one principal on this table."""
        grant_privileges = self._normalize_table_privileges(privileges)
        return (
            f"GRANT {', '.join(grant_privileges)} "
            f"ON TABLE {self.full_name(safe=True)} "
            f"TO {quote_principal(principal)}"
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
        ttl_ns = 3_600_000_000_000
        return RefreshResult(
            value=value,
            created_at_ns=created_ns,
            ttl_ns=ttl_ns,
            expires_at_ns=created_ns + ttl_ns,
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
