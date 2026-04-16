"""
Databricks SQL engine utilities.

This module provides a thin execution and table-management layer across two
Databricks runtimes:

- Spark SQL / Delta Lake, when a SparkSession is available
- Databricks SQL Statement Execution API, when running outside Spark

It is designed to keep read and write behavior as consistent as possible
between both paths, especially for Delta insert and merge workflows.

Core capabilities
-----------------
- Resolve catalog/schema-scoped table handles
- Execute SQL through Spark or Databricks SQL warehouses
- Insert Arrow, Spark, pandas, polars, dict/list, and similar tabular inputs
  into Delta tables
- Support append, overwrite, and merge-style upsert semantics
- Create tables from schemas or input data when needed

Execution model
---------------
The engine chooses the execution backend as follows:

1. If `engine="spark"` is explicitly requested, Spark SQL is used
2. If `engine="api"` is explicitly requested, the Databricks SQL API is used
3. If no engine is provided:
   - Spark is used when an active SparkSession is available
   - otherwise the Databricks SQL warehouse API is used

Write paths
-----------
Two write strategies are supported:

Spark path
    Used when a SparkSession is available or when the input is already a Spark
    DataFrame. Data is written directly into Delta tables and MERGE semantics
    are implemented through Delta APIs.

Warehouse SQL path
    Used when Spark is not available. Input data is converted to Parquet,
    staged into a temporary Databricks volume, and then inserted or merged into
    the target Delta table through SQL.

Save modes
----------
- `append`
    Insert-only behavior. When `match_by` is provided, only non-matching rows
    are inserted.

- `overwrite`
    Full replacement behavior. The target table is dropped first, then data is
    written back with plain insert/append logic.

- `truncate`
    In-place replacement behavior.  The table structure is preserved.

    * Without ``match_by``: ``TRUNCATE TABLE`` empties the table, then all
      rows from the input are inserted.
    * With ``match_by``: a targeted ``DELETE`` removes every existing row
      whose key appears in the input, then all rows from the input are
      inserted.  This avoids a full table scan while keeping the schema intact.

- `auto`
    Default behavior. When `match_by` is provided, matching rows are updated
    and new rows are inserted.

Merge semantics
---------------
When `match_by` is provided, merge conditions are built using Databricks
null-safe equality (`<=>`) by default so NULL matches NULL.

- `append` + `match_by`
    Insert-only merge

- `auto` + `match_by`
    Upsert merge

- `overwrite`
    Table is dropped first, so merge logic is skipped

Safety and consistency
----------------------
This module is intended to be safe by default:

- SQL identifiers are quoted
- merge conditions are generated from explicit key columns
- schemas are aligned through cast options before writing
- Spark and SQL paths follow the same overwrite and merge rules
"""


from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union, Iterable, OrderedDict, Mapping

import pyarrow as pa
from databricks.sdk.service.sql import Disposition

from yggdrasil.concurrent.threading import Job
from yggdrasil.data.cast import CastOptions
from yggdrasil.databricks.sql.sql_utils import quote_ident
from yggdrasil.dataclasses import ExpiringDict, WaitingConfig, WaitingConfigArg
from yggdrasil.environ import PyEnv
from yggdrasil.io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.parquet_io import ParquetOptions
from yggdrasil.io.enums import SaveMode
from yggdrasil.io.enums.media_type import MediaTypes
from .catalogs import Catalogs
from .grants import Grants
from .schemas import Schemas
from .service import DEFAULT_ALL_PURPOSE_SERVERLESS_NAME
from .staging import StagingPath
from .statement_result import StatementResult
from .table import Table
from .tables import Tables
from .types import PrimaryKeySpec, ForeignKeySpec
from .warehouse import SQLWarehouse
from ..client import DatabricksService
from ..fs.path import DatabricksPath
from ...data.statement_result import StatementResultBatch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas
    import polars
    import pyspark
    from pyspark.sql import SparkSession

__all__ = [
    "SQLEngine",
    "StatementResult",
]


def _build_match_condition(
    match_by: list[str],
    *,
    left_alias: str,
    right_alias: str,
    null_safe: bool = True,
) -> str:
    op = "<=>" if null_safe else "="
    return " AND ".join(
        f"{left_alias}.{quote_ident(k)} {op} {right_alias}.{quote_ident(k)}"
        for k in match_by
    )


@dataclass(frozen=True)
class SQLEngine(DatabricksService):
    """
    Unified SQL execution and Delta table write engine for Databricks.

    `SQLEngine` provides a single interface for:

    - executing SQL through Spark or Databricks SQL warehouses
    - resolving catalog/schema-scoped tables
    - inserting tabular data into Delta tables
    - applying append, overwrite, or merge/upsert semantics
    - performing optional post-write maintenance such as OPTIMIZE or VACUUM

    The engine prefers Spark execution when a SparkSession is available and
    falls back to the Databricks SQL Statement Execution API otherwise.

    Scope
    -----
    An engine instance may be bound to a default catalog, schema, and warehouse.
    The instance is lightweight and can be re-scoped by calling it:

        engine(catalog_name="main", schema_name="analytics")

    This returns a new engine sharing the same client and query cache.

    Notes
    -----
    - Spark and warehouse paths are intended to behave consistently
    - overwrite mode always drops the table before reinserting
    - merge behavior is enabled only when `match_by` is provided
    """

    catalog_name: str | None = None
    schema_name: str | None = None
    default_warehouse: Optional[SQLWarehouse] = field(
        default=None,
        repr=False,
        hash=False,
        compare=False,
    )

    _last_default_wh_check: int = field(
        default=0,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )
    _cached_queries: Optional[ExpiringDict[str, StatementResult]] = field(
        default=ExpiringDict,
        init=False,
        repr=False,
        hash=False,
        compare=False,
    )

    @property
    def catalogs(self) -> "Catalogs":
        """
        Return the `Catalogs` service scoped to this engine's catalog and schema.
        """
        return Catalogs(
            client=self.client,
        )

    @property
    def schemas(self) -> "Schemas":
        """
        Return the `Schemas` service scoped to this engine's catalog and schema.
        """
        return Schemas(
            client=self.client,
            catalog_name=self.catalog_name,
        )

    @property
    def tables(self) -> Tables:
        """
        Return the `Tables` service scoped to this engine's catalog and schema.
        """
        return Tables(
            client=self.client,
            catalog_name=self.catalog_name,
            schema_name=self.schema_name,
        )

    @property
    def grants(self) -> Grants:
        """
        Return the `Grants` service scoped to this engine's catalog and schema.
        """
        return Grants(
            client=self.client,
        )

    def __call__(
        self,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        warehouse: Optional[SQLWarehouse | str] = None,
    ) -> SQLEngine:
        """
        Return a re-scoped engine.

        If no scope changes are provided, the current instance is returned.
        If the requested scope is identical to the current scope, the current
        instance is also returned.

        Args:
            catalog_name:
                Catalog override for the returned engine.
            schema_name:
                Schema override for the returned engine.
            warehouse:
                Default warehouse override. May be a warehouse instance or a
                warehouse name.

        Returns:
            A new `SQLEngine` sharing the same client and query cache, or the
            current instance if no change is needed.
        """
        if catalog_name is None and schema_name is None and warehouse is None:
            return self

        if (
            catalog_name == self.catalog_name
            and schema_name == self.schema_name
            and warehouse == self.default_warehouse
        ):
            return self

        if isinstance(warehouse, str):
            warehouse = self.warehouses.find_warehouse(warehouse_name=warehouse)

        built = SQLEngine(
            client=self.client,
            catalog_name=catalog_name,
            schema_name=schema_name,
            default_warehouse=warehouse,
        )

        object.__setattr__(built, "_cached_queries", self._cached_queries)
        return built

    def warehouse(
        self,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
    ) -> SQLWarehouse:
        """
        Resolve the warehouse used by this engine.

        Behavior:
        - If no default warehouse is cached, resolve one and cache it
        - If a warehouse override is provided, resolve that warehouse directly
        - If the cached default is the all-purpose serverless placeholder,
          periodically refresh it

        Args:
            warehouse_id:
                Explicit warehouse ID to resolve.
            warehouse_name:
                Explicit warehouse name to resolve.

        Returns:
            A resolved `SQLWarehouse` instance.
        """
        if self.default_warehouse is None:
            object.__setattr__(self, "_last_default_wh_check", time.time())
            object.__setattr__(
                self,
                "default_warehouse",
                self.warehouses.find_warehouse(
                    warehouse_id=warehouse_id,
                    warehouse_name=warehouse_name,
                    find_default=True,
                    raise_error=True,
                ),
            )
            return self.default_warehouse

        if warehouse_id or warehouse_name:
            return self.warehouses.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

        if self.default_warehouse.warehouse_name == DEFAULT_ALL_PURPOSE_SERVERLESS_NAME:
            now_s = time.time()
            if (now_s - self._last_default_wh_check) > 30:
                object.__setattr__(
                    self,
                    "default_warehouse",
                    self.warehouses.find_default(),
                )
                object.__setattr__(self, "_last_default_wh_check", now_s)

        return self.default_warehouse

    def execute_many(
        self,
        statements: Iterable[str] | Mapping[str, str],
        *,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        byte_limit: int | None = None,
        cache_for: WaitingConfigArg = None,
        spark_session: Optional["SparkSession"] = None,
        parallel: bool = False,
    ) -> StatementResultBatch:
        """
        Execute multiple SQL statements.

        Behavior
        --------
        - Statements are normalized with ``strip()``
        - Empty statements are ignored
        - Execution order is preserved in the returned batch
        - When ``parallel=False``:
          - all statements except the last are executed with ``wait=True``
          - the final statement uses the caller-provided ``wait`` value
        - When ``parallel=True``:
          - all statements are submitted immediately
          - each statement uses the caller-provided ``wait`` value
          - results are returned in input order, but execution order is not enforced

        Args:
            statements:
                SQL statements to execute.

                Accepts either:
                - an iterable of SQL strings, keyed as ``"0"``, ``"1"``, ...
                - a mapping of ``{name: statement}``, preserving mapping order
            row_limit:
                Optional row limit forwarded to each statement execution.
            catalog_name:
                Catalog override for warehouse API execution context.
            schema_name:
                Schema override for warehouse API execution context.
            wait:
                Waiting configuration.
            raise_error:
                Whether execution errors should be raised.
            engine:
                Explicit engine override: ``"spark"`` or ``"api"``.
            warehouse_id:
                Warehouse ID override for API execution.
            warehouse_name:
                Warehouse name override for API execution.
            byte_limit:
                Optional response byte limit for API execution.
            cache_for:
                Optional TTL for statement result caching.
            spark_session:
                Explicit SparkSession override.
            parallel:
                When ``True``, submit all statements without sequential dependency
                waiting. Use only when statements are independent.

        Returns:
            A :class:`StatementResultBatch` containing results in input order.

        Raises:
            ValueError:
                If no non-empty SQL statements were provided.
        """
        items: OrderedDict[str, str] = OrderedDict()

        if isinstance(statements, Mapping):
            for key, statement in statements.items():
                stmt = statement.strip()
                if stmt:
                    items[str(key)] = stmt
        else:
            for i, statement in enumerate(statements):
                stmt = statement.strip()
                if stmt:
                    items[str(i)] = stmt

        if not items:
            raise ValueError("No non-empty SQL statements were provided.")

        results: OrderedDict[str, StatementResult] = OrderedDict()

        if parallel:
            for key, statement in items.items():
                results[key] = self.execute(
                    statement,
                    row_limit=row_limit,
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    wait=False,
                    raise_error=raise_error,
                    engine=engine,
                    warehouse_id=warehouse_id,
                    warehouse_name=warehouse_name,
                    byte_limit=byte_limit,
                    cache_for=cache_for,
                    spark_session=spark_session,
                )

            return StatementResultBatch(results=results).wait(wait=wait, raise_error=raise_error)

        keys = list(items.keys())

        for key in keys[:-1]:
            results[key] = self.execute(
                items[key],
                row_limit=row_limit,
                catalog_name=catalog_name,
                schema_name=schema_name,
                wait=True,
                raise_error=raise_error,
                engine=engine,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                byte_limit=byte_limit,
                cache_for=cache_for,
                spark_session=spark_session,
            )

        last_key = keys[-1]
        results[last_key] = self.execute(
            items[last_key],
            row_limit=row_limit,
            catalog_name=catalog_name,
            schema_name=schema_name,
            wait=wait,
            raise_error=raise_error,
            engine=engine,
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            byte_limit=byte_limit,
            cache_for=cache_for,
            spark_session=spark_session,
        )

        batch = StatementResultBatch(results=results)

        return batch

    def execute(
        self,
        statement: str,
        *,
        row_limit: int | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: str | None = None,
        warehouse_name: str | None = None,
        byte_limit: int | None = None,
        cache_for: WaitingConfigArg = None,
        spark_session: Optional["SparkSession"] = None,
    ) -> StatementResult:
        """
        Execute a SQL statement through Spark or the Databricks SQL API.

        Engine selection
        ----------------
        - `engine="spark"` forces Spark SQL
        - `engine="api"` forces warehouse API execution
        - `engine=None` auto-selects Spark when a SparkSession is available,
          otherwise falls back to the warehouse API

        Cache behavior
        --------------
        When `cache_for` is provided, results are cached by the normalized SQL
        statement text for the specified TTL.

        Args:
            statement:
                SQL text to execute.
            row_limit:
                Optional row limit. Applied through `limit()` on Spark results
                or forwarded to the SQL API.
            catalog_name:
                Catalog override for warehouse API execution context.
            schema_name:
                Schema override for warehouse API execution context.
            wait:
                Waiting configuration for API execution.
            raise_error:
                Whether execution errors should be raised.
            engine:
                Explicit engine override: `"spark"` or `"api"`.
            warehouse_id:
                Warehouse ID override for API execution.
            warehouse_name:
                Warehouse name override for API execution.
            byte_limit:
                Optional response byte limit for API execution.
            cache_for:
                Optional TTL for statement result caching.
            spark_session:
                Explicit SparkSession override.

        Returns:
            A `StatementResult` wrapping either a Spark result or a warehouse API
            statement execution result.

        Raises:
            ValueError:
                If Spark execution is requested and no SparkSession can be
                resolved.
        """
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if not engine:
            spark_session = (
                PyEnv.spark_session(
                    create=False,
                    import_error=False,
                    install_spark=False,
                )
                if spark_session is None
                else spark_session
            )

            if spark_session is not None:
                engine = "spark"
            else:
                engine = "api"

        if spark_session is not None:
            engine = "spark"

        statement = statement.strip()

        if cache_for is not None:
            cache_for = WaitingConfig.check_arg(cache_for)
            existing = self._cached_queries.get(statement)
            if existing is not None:
                return existing

        if engine == "spark":
            spark_session = (
                PyEnv.spark_session(
                    create=True,
                    install_spark=False,
                    import_error=True,
                )
                if spark_session is None
                else spark_session
            )

            df = spark_session.sql(statement)
            if row_limit:
                df = df.limit(row_limit)

            result = StatementResult(
                client=self.client,
                warehouse_id="SparkSQL",
                statement_id="SparkSQL",
                disposition=Disposition.EXTERNAL_LINKS,
            )
            result.persist(data=df)
        else:
            wh = self.warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

            result = wh.execute(
                statement=statement,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                byte_limit=byte_limit,
                catalog_name=catalog_name,
                schema_name=schema_name,
                wait=wait,
                raise_error=raise_error,
                row_limit=row_limit,
            )

        if cache_for is not None:
            self._cached_queries.set(
                key=statement,
                value=result,
                ttl=cache_for.timeout_total_seconds,
            )

        return result

    def table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
    ) -> Table:
        """
        Resolve a table handle.

        Args:
            location:
                Fully qualified table name.
            catalog_name:
                Catalog override when `location` is not fully specified.
            schema_name:
                Schema override when `location` is not fully specified.
            table_name:
                Table name override when `location` is not provided.

        Returns:
            A `Table` handle.
        """
        return self.tables.table(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

    def insert_into(
        self,
        data: Union[
            pa.Table,
            pa.RecordBatch,
            pa.RecordBatchReader,
            dict,
            list,
            str,
            "pandas.DataFrame",
            "polars.DataFrame",
            "pyspark.sql.DataFrame",
        ],
        *,
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        spark_options: Optional[Dict[str, Any]] = None,
        table: Optional[Table] = None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> None:
        """
        Insert data into a Delta table using the most appropriate backend.

        Routing behavior
        ----------------
        - If a SparkSession is available, use the Spark write path
        - Otherwise, use the warehouse SQL path with staged Parquet

        Args:
            data:
                Input data. Supported inputs include Arrow objects, Spark
                DataFrames, pandas/polars DataFrames, dict/list tabular values,
                and other project-supported convertible types.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            cast_options:
                Casting rules used to align input data to the destination schema.
            overwrite_schema:
                Spark writer option. When True, writes with
                `overwriteSchema=true`.
            match_by:
                Merge key columns enabling key-based insert or upsert behavior.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely (insert-only
                for matched rows).
            wait:
                Waiting configuration for the warehouse SQL path.
            raise_error:
                Whether write errors should be raised on the warehouse SQL path.
            zorder_by:
                Optional ZORDER columns for post-write optimization.
            optimize_after_merge:
                Whether to run optimize after merge-related writes.
            vacuum_hours:
                Optional retention window for VACUUM.
            spark_session:
                Explicit SparkSession override.
            spark_options:
                Additional Spark DataFrameWriter options.
            table:
                Optional pre-resolved table handle.
            primary_keys:
                Column name(s) to set as primary key when the table is created
                by this call.  Composite keys: ``["trade_date", "instrument_id"]``.
                ``None`` reads from field metadata automatically.
            foreign_keys:
                FK constraints to apply when the table is created.
                Accepts a ``{col: "cat.sch.tbl.col"}`` dict or a list of
                :class:`~yggdrasil.databricks.sql.types.ForeignKeySpec`.

        Returns:
            None.
        """
        if spark_session is None:
            if hasattr(data, "sparkSession"):
                spark_session = data.sparkSession
            else:
                spark_session = PyEnv.spark_session(
                    create=False,
                    import_error=False,
                    install_spark=False,
                )

        if spark_session is not None:
            return self.spark_insert_into(
                data=data,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                mode=mode,
                schema_mode=schema_mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                match_by=match_by,
                update_cols=update_cols,
                wait=wait,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                spark_options=spark_options,
                table=table,
                primary_keys=primary_keys,
                foreign_keys=foreign_keys,
            )

        return self.arrow_insert_into(
            data=data,
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            mode=mode,
            schema_mode=schema_mode,
            cast_options=cast_options,
            overwrite_schema=overwrite_schema,
            match_by=match_by,
            update_cols=update_cols,
            wait=wait,
            raise_error=raise_error,
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
            table=table,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
        )

    def arrow_insert_into(
        self,
        data,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        table: Optional[Table] = None,
        temp_volume_path=None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> None:
        """
        Insert data through the warehouse SQL path.

        Implementation
        --------------
        - Ensure the destination table exists
        - Convert input data to Parquet
        - Stage the Parquet file to a temporary Databricks volume
        - Execute INSERT INTO or MERGE INTO against the target table
        - Optionally run OPTIMIZE, ZORDER, and VACUUM

        Behavior
        --------
        - `overwrite` drops the target table before inserting
        - `truncate` (no ``match_by``): ``TRUNCATE TABLE`` then ``INSERT INTO``
        - `truncate` (with ``match_by``): ``DELETE`` rows whose keys appear in
          the input, then ``INSERT INTO`` — keeps the schema intact
        - `match_by` without overwrite/truncate enables MERGE semantics:
          ``append`` → insert-only; ``auto`` → upsert
        - rows with NULL in match columns are expected to follow the same merge
          behavior as the Spark path

        Args:
            data:
                Arrow or Arrow-convertible tabular data.
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            cast_options:
                Casting rules used to align staged data to the destination
                schema.
            overwrite_schema:
                Reserved for API parity with the Spark path.
            match_by:
                Merge key columns.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely.
            wait:
                Waiting configuration for statement execution.
            raise_error:
                Whether statement execution errors should be raised.
            zorder_by:
                Columns used for `OPTIMIZE ... ZORDER BY (...)`.
            optimize_after_merge:
                Whether to run `OPTIMIZE` after merge operations.
            vacuum_hours:
                Optional retention window for `VACUUM`.
            table:
                Optional pre-resolved table handle.
            temp_volume_path:
                Optional explicit staging volume path.

        Returns:
            None.
        """
        mode = SaveMode.parse(mode, default=SaveMode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        table = Table.parse(
            obj=location if table is None else table,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            service=self.tables,
        )

        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        table = table.create(
            data,
            mode=schema_mode,
            primary_keys=primary_keys, foreign_keys=foreign_keys
        )
        location = table.full_name(safe=True)
        cast_options = CastOptions.check(options=cast_options).check_target(table.data_field)
        existing_schema = table.data_schema

        logger.debug("Inserting %s into %s", type(data), location)

        staging: Optional[StagingPath] = None
        if temp_volume_path is None:
            staging = StagingPath.for_table(
                client=self.client,
                catalog_name=table.catalog_name,
                schema_name=table.schema_name,
                table_name=table.table_name,
                max_lifetime=3600,
            )
            staging.register_shutdown_cleanup()
            temp_volume_path = staging.path
        else:
            temp_volume_path = DatabricksPath.parse(obj=temp_volume_path, client=self.client)

        temp_volume_path.parent.mkdir(parents=True, exist_ok=True)

        with BytesIO() as buffer:
            mio = MediaIO.make(buffer=buffer, media=MediaTypes.PARQUET)
            mio.write_table(data, options=ParquetOptions(cast=cast_options))
            mio.buffer.seek(0)
            temp_volume_path.write_bytes(mio.buffer.memoryview())

        columns = list(existing_schema.field_names())
        cols_quoted = ", ".join(quote_ident(c) for c in columns)

        statements: list[str] = []

        if mode == SaveMode.TRUNCATE:
            source_sql = (
                f"SELECT {cols_quoted} "
                f"FROM parquet.{quote_ident(str(temp_volume_path))}"
            )
            insert_sql = f"""INSERT INTO {location} ({cols_quoted})
{source_sql}""".strip()

            if match_by:
                # Delete every existing row whose key appears in the incoming
                # batch, then insert all rows from that batch.
                key_cols = ", ".join(quote_ident(k) for k in match_by)
                on_condition = _build_match_condition(
                    match_by,
                    left_alias="T",
                    right_alias="S",
                    null_safe=True,
                )
                delete_sql = f"""DELETE FROM {location} AS T
USING (
SELECT DISTINCT {key_cols}
FROM parquet.{quote_ident(str(temp_volume_path))}
) AS S
ON {on_condition}""".strip()
                statements.extend([delete_sql, insert_sql])
            else:
                # Wipe the table in-place (schema kept), then insert all rows.
                statements.extend([
                    f"TRUNCATE TABLE {location}",
                    insert_sql,
                ])

        elif match_by and mode != SaveMode.OVERWRITE:
            source_sql = (
                f"SELECT {cols_quoted} "
                f"FROM parquet.{quote_ident(str(temp_volume_path))}"
            )
            on_condition = _build_match_condition(
                match_by,
                left_alias="T",
                right_alias="S",
                null_safe=True,
            )

            if mode == SaveMode.APPEND:
                insert_clause = (
                    f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                    f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
                )

                merge_sql = f"""MERGE INTO {location} AS T
USING (
{source_sql}
) AS S
ON {on_condition}
{insert_clause}""".strip()

                statements.append(merge_sql)
            else:
                update_cols_effective = (
                    update_cols
                    if update_cols is not None
                    else [c for c in columns if c not in match_by]
                )
                update_clause = ""
                if update_cols_effective:
                    update_set = ", ".join(
                        f"T.{quote_ident(c)} = S.{quote_ident(c)}"
                        for c in update_cols_effective
                    )
                    update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"

                insert_clause = (
                    f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                    f"VALUES ({', '.join(f'S.{quote_ident(c)}' for c in columns)})"
                )

                merge_sql = f"""MERGE INTO {location} AS T
USING (
{source_sql}
) AS S
ON {on_condition}
{update_clause}
{insert_clause}""".strip()

                statements.append(merge_sql)
        else:
            insert_sql = f"""INSERT INTO {location} ({cols_quoted})
SELECT {cols_quoted}
FROM parquet.{quote_ident(str(temp_volume_path))}"""
            statements.append(insert_sql)

        try:
            if statements:
                if len(statements) == 1:
                    self.execute(statements[0], wait=wait, raise_error=raise_error)
                else:
                    for stmt in statements[:-1]:
                        self.execute(stmt, wait=True, raise_error=raise_error)
                    self.execute(statements[-1], wait=wait, raise_error=raise_error)
        finally:
            if wait:
                if staging is not None:
                    staging.cleanup(allow_not_found=True, unregister=True)
                else:
                    temp_volume_path.remove()

        logger.info("Arrow inserted into %s", location)

        if zorder_by:
            zorder_cols = ", ".join(quote_ident(c) for c in zorder_by)
            self.execute(f"OPTIMIZE {location} ZORDER BY ({zorder_cols})")

        if optimize_after_merge and match_by:
            self.execute(f"OPTIMIZE {location}")

        if vacuum_hours is not None:
            self.execute(f"VACUUM {location} RETAIN {int(vacuum_hours)} HOURS")

        return None

    def spark_insert_into(
        self,
        data: Any,
        *,
        mode: SaveMode | str | None = None,
        schema_mode: SaveMode | str | None = None,
        location: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        update_cols: Optional[list[str]] = None,
        wait: WaitingConfigArg = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
        table: Optional[Table] = None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
    ) -> None:
        """
        Insert data into a Delta table using Spark.

        Implementation
        --------------
        - Convert input to a Spark DataFrame
        - Ensure the destination table exists
        - Write using append or Delta MERGE semantics
        - Optionally run optimize / ZORDER / vacuum maintenance

        Behavior
        --------
        - `overwrite` drops the target table before writing
        - `truncate` (no ``match_by``): ``DeltaTable.delete()`` empties the
          table in-place, then data is appended
        - `truncate` (with ``match_by``): Delta MERGE deletes rows whose keys
          appear in the input, then data is appended
        - `append` + `match_by` performs insert-only merge
        - `auto` + `match_by` performs upsert merge; ``update_cols`` controls
          which columns are updated (default: all non-key columns)
        - without `match_by`, data is appended directly
        - ``wait=False`` submits the entire write to a background thread and
          returns immediately; errors are logged but not re-raised

        Args:
            data:
                Input data convertible to a Spark DataFrame.
            mode:
                Save mode controlling append / overwrite / merge semantics.
            schema_mode:
                Schema mode to merge with current target schema
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            cast_options:
                Casting rules used to align input data to the destination
                schema.
            overwrite_schema:
                When True, passes `overwriteSchema=true` to the Spark writer.
            match_by:
                Merge key columns.
            update_cols:
                Columns to update when a merge key match is found (``mode=auto``
                only).  ``None`` updates all non-key columns (default).
                An empty list disables the UPDATE clause entirely.
            wait:
                When ``True`` (default) the call blocks until the write
                completes.  When ``False`` the write is submitted to a
                background thread and the method returns immediately.
            zorder_by:
                Columns used for Delta ZORDER optimization.
            optimize_after_merge:
                Whether to run optimize after merge-style writes.
            vacuum_hours:
                Optional retention window for Delta vacuum.
            spark_options:
                Additional Spark writer options.
            table:
                Optional pre-resolved table handle.

        Returns:
            None.
        """
        from yggdrasil.spark.cast import any_to_spark_dataframe
        import pyspark.sql.functions as F

        logger.info(
            "Spark insert into %s (mode=%s, match_by=%s, overwrite_schema=%s, wait=%s)",
            location,
            mode,
            match_by,
            overwrite_schema,
            wait,
        )

        mode = SaveMode.parse(mode, default=SaveMode.AUTO)
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        if table is None:
            table = self.table(
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
            )

        # TODO: Fix async databricks notebook
        wait = True if PyEnv.in_databricks() else wait

        # OVERWRITE: drop before the background thread so the slot is freed
        # synchronously and callers can re-create the table immediately.
        if mode == SaveMode.OVERWRITE:
            table.delete(wait=True, raise_error=False)

        table = table.ensure_created(
            data,
            schema_mode=schema_mode,
            primary_keys=primary_keys, foreign_keys=foreign_keys
        )
        cast_options = CastOptions.check(options=cast_options).check_target(table.data_field)
        data_df = any_to_spark_dataframe(data, cast_options)
        target = table.delta_spark()

        _spark_options = spark_options if spark_options else {}
        if overwrite_schema:
            _spark_options["overwriteSchema"] = "true"

        def _run() -> None:
            if mode == SaveMode.TRUNCATE:
                cond = _build_match_condition(
                    match_by,
                    left_alias="t",
                    right_alias="s",
                    null_safe=True,
                ) if match_by else None

                if match_by:
                    logger.info(
                        "Spark truncate (match_by=%s): Delta delete matching keys", match_by
                    )
                    distinct_keys = data_df.select(list(match_by)).distinct()
                    (
                        target.alias("t")
                        .merge(distinct_keys.alias("s"), cond)
                        .whenMatchedDelete()
                        .execute()
                    )
                else:
                    logger.info("Spark truncate: Delta delete all rows")
                    target.delete()

                logger.info("Spark write saveAsTable mode=append (after truncate)")
                (
                    data_df.write
                    .format("delta")
                    .mode("append")
                    .options(**_spark_options)
                    .saveAsTable(table.full_name())
                )

            elif match_by and mode != SaveMode.OVERWRITE:
                cond = _build_match_condition(
                    match_by,
                    left_alias="t",
                    right_alias="s",
                    null_safe=True,
                )

                if mode == SaveMode.APPEND:
                    (
                        target.alias("t")
                        .merge(data_df.alias("s"), cond)
                        .whenNotMatchedInsertAll()
                        .execute()
                    )
                else:
                    update_cols_effective = (
                        update_cols
                        if update_cols is not None
                        else [c for c in data_df.columns if c not in match_by]
                    )
                    set_expr = {
                        c: F.expr(f"s.{quote_ident(c)}")
                        for c in update_cols_effective
                    }
                    builder = target.alias("t").merge(data_df.alias("s"), cond)
                    if set_expr:
                        builder = builder.whenMatchedUpdate(set=set_expr)
                    (
                        builder
                        .whenNotMatchedInsertAll()
                        .execute()
                    )
            else:
                logger.info("Spark write saveAsTable mode=append")
                (
                    data_df.write
                    .format("delta")
                    .mode("append")
                    .options(**_spark_options)
                    .saveAsTable(table.full_name())
                )

            if optimize_after_merge and zorder_by:
                logger.info("Delta optimize + zorder (%s)", zorder_by)
                target.optimize().executeZOrderBy(*zorder_by)

            if vacuum_hours is not None:
                logger.info("Delta vacuum retain=%s hours", vacuum_hours)
                target.vacuum(vacuum_hours)

        if wait:
            _run()
        else:
            Job.make(_run).fire_and_forget()

    def drop_table(
        self,
        location: str | None = None,
        *,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        wait: WaitingConfigArg = True,
        raise_error: bool = True,
    ) -> None:
        """
        Drop a table if it exists.

        Args:
            location:
                Fully qualified destination table.
            catalog_name:
                Catalog override when `location` is not fully specified.
            schema_name:
                Schema override when `location` is not fully specified.
            table_name:
                Table name override when `location` is not provided.
            wait:
                Waiting configuration for the drop operation.
            raise_error:
                Whether drop errors should be raised.

        Returns:
            None.
        """
        return self.table(
            location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        ).delete(wait=wait, raise_error=raise_error)

    def create_table(
        self,
        definition: Union[pa.Field, pa.Schema, Any],
        *,
        full_name: str | None = None,
        catalog_name: str | None = None,
        schema_name: str | None = None,
        table_name: str | None = None,
        primary_keys: "list[str] | PrimaryKeySpec | None" = None,
        foreign_keys: "list[ForeignKeySpec] | dict[str, str] | None" = None,
        **kwargs,
    ) -> Table:
        """
        Create a table if it does not already exist.

        Primary and foreign key constraints are applied after the table is
        created via ``ALTER TABLE``, in the order PK → FK.

        Constraint sources (in priority order):

        1. Explicit ``primary_keys`` / ``foreign_keys`` parameters.
        2. Field-level metadata tags ``t:primary_key`` / ``t:foreign_key``
           on the supplied *definition* schema.

        Args:
            definition:
                Table definition — an Arrow field, Arrow schema, or any
                project-supported schema-like object.
            full_name:
                Fully qualified destination table (``catalog.schema.table``).
            catalog_name:
                Catalog override.
            schema_name:
                Schema override.
            table_name:
                Table name override.
            primary_keys:
                Column name(s) to mark as primary key, or a full
                :class:`~yggdrasil.databricks.sql.types.PrimaryKeySpec`.
                Composite keys are expressed as a list of column names:
                ``primary_keys=["trade_date", "instrument_id"]``.
                When ``None``, fields carrying ``t:primary_key`` metadata are
                used automatically.
            foreign_keys:
                A list of
                :class:`~yggdrasil.databricks.sql.types.ForeignKeySpec`
                objects, or a ``{col_name: "cat.sch.tbl.col"}`` dict.
                When ``None``, fields carrying ``t:foreign_key`` metadata are
                used automatically.
            **kwargs:
                Additional arguments forwarded to :meth:`Table.create`.

        Returns:
            The created or existing :class:`Table` handle.
        """
        table = self.table(
            location=full_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
        )

        return table.create(
            definition=definition,
            if_not_exists=True,
            primary_keys=primary_keys,
            foreign_keys=foreign_keys,
            **kwargs,
        )
