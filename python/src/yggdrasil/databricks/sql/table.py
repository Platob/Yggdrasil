import base64
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union, Iterator

import pyarrow as pa
from databricks.sdk.client_types import HostType
from databricks.sdk.errors import DatabricksError, ResourceDoesNotExist
from databricks.sdk.service.catalog import (
    TableInfo,
    TableOperation,
    TableType,
    DataSourceFormat,
)
from pyarrow.fs import FileSystem, S3FileSystem

from yggdrasil.arrow.cast import any_to_arrow_schema
from yggdrasil.concurrent.threading import Job
from yggdrasil.dataclasses.expiring import Expiring, RefreshResult
from yggdrasil.dataclasses.waiting import WaitingConfigArg
from yggdrasil.io.enums import SaveMode
from .types import arrow_field_to_column_info, column_info_to_arrow_field
from ..client import DatabricksService

__all__ = ["Table"]


@dataclass(frozen=True)
class Table(DatabricksService):
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None
    table_name: Optional[str] = None

    _infos: Optional[TableInfo] = field(default=None, repr=False, compare=False, hash=False)
    _arrow_fields: Optional[list[pa.Field]] = field(default=None, repr=False, compare=False, hash=False)

    @classmethod
    def service_name(cls):
        return "uctable"

    # -------------------------------------------------------------------------
    # Identity
    # -------------------------------------------------------------------------

    def schema_full_name(self):
        return f"{self.catalog_name}.{self.schema_name}"

    def full_name(
        self,
        safe: str | bool | None = None
    ) -> str:
        if safe:
            if isinstance(safe, bool):
                safe = "`"
            else:
                safe = safe or "`"

            return "%s%s%s.%s%s%s.%s%s%s" % (
                safe, self.catalog_name, safe,
                safe, self.schema_name, safe,
                safe, self.table_name, safe
            )
        return f"{self.catalog_name}.{self.schema_name}.{self.table_name}"

    def __repr__(self) -> str:
        return f"Table({self.full_name()})"

    # -------------------------------------------------------------------------
    # Databricks SDK plumbing
    # -------------------------------------------------------------------------

    @property
    def table_id(self) -> str:
        return self.infos.table_id

    @property
    def infos(self) -> TableInfo:
        if self._infos is not None:
            return self._infos

        infos = self.client.workspace_client().tables.get(self.full_name())

        object.__setattr__(self, "_infos", infos)
        return self._infos

    def find_table(
        self,
        table_name: Optional[str] = None,
        *,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_id: Optional[str] = None,
        raise_error: bool = True
    ) -> Optional["Table"]:
        """
        Find a table in Unity Catalog and return a Table object with cached TableInfo.

        Lookup priority:
        1) table_id (if provided) via list scan (requires catalog+schema or defaults)
        2) table_name / full name via GET
        3) table_name via list scan (fallback)

        Args:
            table_name:   Either bare table name ("my_table") or full name
                         ("catalog.schema.my_table"). If None, uses self.table_name.
            catalog_name: Optional override for catalog. Defaults to self.catalog_name.
            schema_name:  Optional override for schema. Defaults to self.schema_name.
            table_id:     Optional UC table_id to search for.
            raise_error:  If True (default), raise ResourceDoesNotExist when not found / on errors.
                          If False, return None.

        Returns:
            Table instance with _infos populated, or None if not found (raise_error=False).
        """
        client = self.client.workspace_client().tables

        # Defaults
        catalog = catalog_name or self.catalog_name
        schema = schema_name or self.schema_name
        name = table_name or self.table_name

        def _return(i: TableInfo) -> "Table":
            return Table(
                client=self.client,
                catalog_name=i.catalog_name,
                schema_name=i.schema_name,
                table_name=i.name,
                _infos=i,
            )

        # ------------------------------------------------------------
        # 1) Search by table_id (best-effort; requires listing scope)
        # ------------------------------------------------------------
        if table_id:
            try:
                for info in client.list(catalog_name=catalog, schema_name=schema):
                    if info.table_id == table_id:
                        return _return(info)
            except DatabricksError as e:
                if raise_error:
                    raise ResourceDoesNotExist(
                        f"Failed to search table_id={table_id} in {catalog}.{schema}"
                    ) from e
                return None

            if raise_error:
                raise ResourceDoesNotExist(f"Table with id {table_id} not found in {catalog}.{schema}")
            return None

        # ------------------------------------------------------------
        # 2) If provided a full name (a.b.c), use GET directly
        # ------------------------------------------------------------
        # Be conservative: only treat as full name if it has exactly 2 dots.
        if isinstance(name, str) and name.count(".") == 2:
            full_name = name
        else:
            full_name = f"{catalog}.{schema}.{name}"

        try:
            info = client.get(full_name=full_name)
            return _return(info)
        except DatabricksError:
            # We'll fallback to list scan below
            pass
        except Exception as e:
            if raise_error:
                raise ResourceDoesNotExist(f"Failed to get table {full_name}") from e
            return None

        # ------------------------------------------------------------
        # 3) Fallback: list scan in schema (handles weird quoting / casing)
        # ------------------------------------------------------------
        try:
            for info in client.list(catalog_name=catalog, schema_name=schema):
                # UC names are case-insensitive-ish, but keep it safe:
                if info.name == name or info.name.lower() == str(name).lower():
                    return _return(info)
        except DatabricksError as e:
            if raise_error:
                raise ResourceDoesNotExist(f"Failed to list tables in {catalog}.{schema}") from e
            return None

        if raise_error:
            raise ResourceDoesNotExist(f"Table {full_name} not found")
        return None

    def list_tables(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> Iterator["Table"]:
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        client = self.client.workspace_client().tables

        for info in client.list(
            catalog_name=catalog_name,
            schema_name=schema_name
        ):
            yield Table(
                client=self.client,
                catalog_name=info.catalog_name,
                schema_name=info.schema_name,
                table_name=info.name,
                _infos=info,
            )

    def _schema_metadata(self) -> dict[bytes, bytes]:
        return {
            b"engine":       b"databricks",
            b"name":         self.full_name().encode(),
            b"catalog_name": self.catalog_name.encode(),
            b"schema_name":  self.schema_name.encode(),
            b"table_name":   self.table_name.encode(),
        }

    # -------------------------------------------------------------------------
    # Arrow schema
    # -------------------------------------------------------------------------

    @property
    def arrow_fields(self) -> list[pa.Field]:
        if self._arrow_fields is None:
            arrow_fields = [
                column_info_to_arrow_field(c) for c in self.infos.columns
            ]
            object.__setattr__(self, "_arrow_fields", arrow_fields)
        return self._arrow_fields

    @property
    def arrow_schema(self) -> pa.Schema:
        return pa.schema(self.arrow_fields, metadata=self._schema_metadata())

    @property
    def arrow_field(self) -> pa.Field:
        """This table represented as a single Arrow struct field.

        Useful for embedding this table's schema as a nested field inside a
        parent schema (e.g. a join result or a schema registry).
        """
        return pa.field(
            self.full_name(),
            pa.struct(self.arrow_fields),
            metadata=self._schema_metadata(),
        )

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------

    def create(
        self,
        definition: Union[pa.Schema, Any],
        *,
        storage_location: Optional[str] = None,
        comment: Optional[str] = None,
        properties: Optional[dict[str, str]] = None,
        table_type: TableType | None = None,
        data_source_format: DataSourceFormat = DataSourceFormat.DELTA,
        if_not_exists: bool = False
    ) -> "Table":
        """Create this table in Unity Catalog via the Databricks Tables API.

        Builds the request body from the Arrow schema and instance attributes,
        then POSTs to ``/api/2.1/unity-catalog/tables``. On success, ``_infos``
        is populated from the API response so subsequent property accesses
        (``table_id``, ``storage_location``, ``arrow_schema``, …) reflect the
        server-assigned metadata without an extra round-trip.

        Note:
            The UC Tables API currently only supports ``EXTERNAL`` + ``DELTA``
            for external clients.  ``MANAGED`` tables and non-Delta formats must
            be created via SQL / the DeltaTableBuilder API.

        Args:
            definition:            PyArrow schema defining columns.  Any type
                               accepted by ``any_to_arrow_schema`` is valid
                               (``pa.Schema``, ``list[pa.Field]``, dict, …).
            storage_location:  S3 / ADLS URI for the table data directory, e.g.
                               ``"s3://my-bucket/trading/raw/crude_oil/"``.
                               Required for ``EXTERNAL`` tables.
            comment:           Optional human-readable table description.
                               Falls back to ``schema.metadata[b"comment"]``
                               or ``schema.metadata[b"description"]`` when
                               not supplied explicitly.
            properties:        Delta / UC table properties dict, e.g.
                               ``{"delta.autoOptimize.optimizeWrite": "true"}``.
            table_type:        ``MANAGED`` (default) or ``EXTERNAL``.
            data_source_format: ``DELTA`` (default) or another supported format.

        Returns:
            ``self`` — enables fluent chaining::

                ds = (
                    Table(workspace=ws, catalog_name="trading",
                          schema_name="raw", table_name="crude_oil_trades")
                    .create(schema, storage_location="s3://…/crude_oil_trades/")
                    .read_arrow_dataset()
                )

        Raises:
            ValueError: Wraps any ``DatabricksError`` raised by the API with
                        the full table name for easier debugging in multi-table
                        pipelines.
        """
        if table_type is None:
            table_type = TableType.EXTERNAL if storage_location else TableType.MANAGED

        if table_type == TableType.EXTERNAL and not storage_location:
            storage_location = self.schema_storage_location(
                table_type=table_type
            ) + "/tables/%s" % self.table_name

        if table_type == TableType.MANAGED:
            self.sql.create_table(
                definition,
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                table_name=self.table_name,
                comment=comment,
                if_not_exists=if_not_exists
            )

            object.__setattr__(self, "_infos", None)  # fetch fresh infos on next access
        else:
            if not isinstance(definition, pa.Schema):
                definition = any_to_arrow_schema(definition)

            # Resolve comment: explicit arg > schema metadata > None
            if not comment and definition.metadata:
                raw = definition.metadata.get(b"comment") or definition.metadata.get(b"description")
                comment = raw.decode("utf-8") if isinstance(raw, bytes) else raw

            columns = [
                arrow_field_to_column_info(field=f, position=pos)
                for pos, f in enumerate(definition)
            ]

            body: dict[str, Any] = {
                "catalog_name":      self.catalog_name,
                "schema_name":       self.schema_name,
                "name":              self.table_name,
                "table_type":        table_type.value,
                "data_source_format": data_source_format.value,
                "columns":           [c.as_dict() for c in columns],
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

            client = self.client.workspace_client().tables
            cfg = client._api._cfg
            if cfg.host_type == HostType.UNIFIED and cfg.workspace_id:
                headers["X-Databricks-Org-Id"] = cfg.workspace_id

            try:
                res = client._api.do(
                    "POST", "/api/2.1/unity-catalog/tables",
                    body=body, headers=headers,
                )

                info = TableInfo.from_dict(res)
                object.__setattr__(self, "_infos", info)
            except DatabricksError as e:
                if "already exists" in str(e):
                    if not if_not_exists:
                        raise
                else:
                    raise

        object.__setattr__(self, "_arrow_fields", None)  # reset cached fields

        return self

    def delete(
        self,
        *,
        wait: WaitingConfigArg | None = True,
        raise_error: bool = True,
    ) -> "Table":
        """Delete this table from Unity Catalog.

        Args:
            wait:        ``True`` (default) — block until deletion completes.
                         ``False`` / ``None`` — fire-and-forget via a background
                         ``Job``.
            raise_error: Re-raise ``DatabricksError`` on failure.  Only
                         relevant for the synchronous path; background jobs
                         handle errors internally.

        Returns:
            ``self`` with ``_infos`` and ``_arrow_fields`` cleared.
        """
        client = self.client.workspace_client().tables

        if wait:
            try:
                client.delete(full_name=self.full_name())
            except DatabricksError:
                if raise_error:
                    raise
        else:
            Job.make(client.delete, self.full_name()).fire_and_forget()

        self.clear()

        return self

    def clear(self) -> "Table":
        """Evict all cached API state, forcing a fresh fetch on next access."""
        object.__setattr__(self, "_infos", None)
        object.__setattr__(self, "_arrow_fields", None)
        return self

    # -------------------------------------------------------------------------
    # Storage & credentials
    # -------------------------------------------------------------------------

    def schema_storage_location(
        self,
        table_type: Optional[TableType] = None
    ) -> str:
        infos = self.client.workspace_client().schemas.get(full_name=self.schema_full_name())

        if not infos.storage_location:
            raise NotImplementedError

        if table_type == TableType.EXTERNAL:
            if "/__unitystorage" in infos.storage_location:
                result = infos.storage_location.split("/__unitystorage")[0]

                return result + "/catalogs/%s/schemas/%s" % (
                    self.catalog_name or "default",
                    self.schema_name or "default"
                )

        return infos.storage_location

    @property
    def storage_location(self) -> str:
        return self.infos.storage_location

    def credentials(self, operation: TableOperation = TableOperation.READ):
        client = self.client.workspace_client().temporary_table_credentials
        return client.generate_temporary_table_credentials(
            table_id=self.table_id,
            operation=operation,
        )

    def arrow_filesystem(
        self,
        *,
        operation: TableOperation = TableOperation.READ_WRITE,
        expiring: bool = False
    ) -> Union[FileSystem, "TableFilesystem"]:
        """Return a credentialed S3FileSystem for direct Parquet I/O."""
        created_at = time.time_ns()
        creds = self.credentials(operation=operation)

        assert creds.aws_temp_credentials, "Cannot get AWS credentials"

        aws = creds.aws_temp_credentials

        base = S3FileSystem(
            access_key=aws.access_key_id,
            secret_key=aws.secret_access_key,
            session_token=aws.session_token,
            region="eu-west-1",  # S3FileSystem requires a region, but it’s not actually used for UC tables
        )

        if not expiring:
            return base

        ttl = 3_600_000_000

        return TableFilesystem.create(
            value=base,
            created_at=created_at,
            ttl=ttl,
            expires_at=created_at + ttl,
            table=self,
            operation=operation,
        )

    def to_arrow_dataset(
        self,
        *,
        filters: Optional[list[tuple[str, str, str]]] = None,
        wait: WaitingConfigArg = True,
        cache_for: WaitingConfigArg = None
    ):
        statement = f"SELECT * FROM {self.full_name(safe=True)}"

        if filters:
            predicates = [_build_predicate(c, o, v) for (c, o, v) in filters]
            statement += " WHERE " + " AND ".join(predicates)

        statement = self.sql.execute(
            statement,
            wait=wait,
            cache_for=cache_for,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

        return statement.to_arrow_dataset()

    def insert(
        self,
        data: Any,
        *,
        mode: SaveMode | str = None,
        match_by: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True
    ):
        mode = SaveMode.parse(mode, SaveMode.AUTO)
        engine = self.sql

        return engine.insert_into(
            data,
            mode=mode,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
            table_name=self.table_name,
            table=self,
            match_by=match_by,
            wait=wait,
            raise_error=raise_error
        )


@dataclass
class TableFilesystem(Expiring[FileSystem]):
    table: Optional[Table] = field(default=None)
    operation: TableOperation = TableOperation.READ

    def _refresh(self) -> RefreshResult[FileSystem]:
        value = self.table.arrow_filesystem(
            operation=self.operation,
            expiring=False
        )

        created_at_ns = time.time_ns()
        ttl_ns = 3_600_000_000_000

        return RefreshResult(
            value=value,
            created_at_ns=created_at_ns,
            ttl_ns=ttl_ns,
            expires_at_ns=created_at_ns + ttl_ns
        )


def _quote_ident(name: str) -> str:
    # Support dotted paths like "a.b.c" or struct access "col.field"
    # Quote each token with backticks.
    parts = [p.strip() for p in name.split(".") if p.strip()]
    return ".".join(f"`{p.replace('`','``')}`" for p in parts)

def _sql_literal(v) -> str:
    # bytes -> base64 string literal
    if isinstance(v, (bytes, bytearray, memoryview)):
        b64 = base64.b64encode(bytes(v)).decode("ascii")
        return "'" + b64 + "'"  # store/compare as base64 text

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

def _build_predicate(col: str, op: str, val: str) -> str:
    op_norm = op.strip().upper()

    # Whitelist ops (expand as needed)
    allowed = {
        "=", "!=", "<>", ">", ">=", "<", "<=",
        "LIKE", "NOT LIKE",
        "IS", "IS NOT",
        "IN", "NOT IN",
    }
    if op_norm == "==":
        op_norm = "="
    elif op_norm not in allowed:
        raise ValueError(f"Unsupported filter operator: {op!r}")

    col_sql = _quote_ident(col)

    if val is None:
        return f"{col_sql} IS NULL"

    if op_norm in ("IS", "IS NOT"):
        # IS (NOT) expects NULL/TRUE/FALSE typically
        return f"{col_sql} {op_norm} {_sql_literal(val)}"

    if op_norm in ("IN", "NOT IN"):
        raw = val.strip()

        # Accept "a,b,c" OR "(a,b,c)" OR "['a','b']" (lightweight)
        if raw.startswith("(") and raw.endswith(")"):
            inner = raw[1:-1].strip()
            items = [x.strip() for x in inner.split(",") if x.strip()]
        else:
            # If user passes a single token like "sql:(select ...)" allow it
            if raw.lower().startswith("sql:"):
                return f"{col_sql} {op_norm} {_sql_literal(raw)}"
            items = [x.strip() for x in raw.split(",") if x.strip()]

        if not items:
            # IN () is invalid SQL — choose a predicate that’s always false/true
            return "FALSE" if op_norm == "IN" else "TRUE"

        items_sql = ", ".join(_sql_literal(x) for x in items)
        return f"{col_sql} {op_norm} ({items_sql})"

    # Normal binary ops
    return f"{col_sql} {op_norm} {_sql_literal(val)}"