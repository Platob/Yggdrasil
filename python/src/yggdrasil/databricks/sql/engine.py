"""
Databricks SQL engine utilities and helpers.

This module provides a thin “do the right thing” layer over:
- Databricks SQL Statement Execution API (warehouse)
- Spark SQL / Delta Lake (when running inside a Spark-enabled context)

It includes helpers to:
- Build fully-qualified table names
- Execute SQL via Spark or Databricks SQL API
- Insert Arrow/Spark/tabular data into Delta tables (append/overwrite/merge)
- Generate DDL from Arrow schemas

Design goals:
- Prefer Spark execution when running inside Databricks with an active SparkSession.
- Otherwise, use the Warehouse Statement Execution API.
- Keep behavior consistent between Spark and SQL paths (especially MERGE semantics).
- Be safe by default: quote identifiers, escape SQL strings, validate column references.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import string
import time
from threading import Thread
from typing import Optional, Union, Any, Dict, Literal, TYPE_CHECKING

import pyarrow as pa
from databricks.sdk.service.sql import Disposition
from yggdrasil.collections.expiring_dict import ExpiringDict
from yggdrasil.enums import SaveMode

from .exceptions import SqlStatementError
from .statement_result import StatementResult
from .types import column_info_to_arrow_field
from .warehouse import SQLWarehouse
from ..workspaces import WorkspaceService, DatabricksPath
from ...enums import FileFormat
from ...pyutils.waiting_config import WaitingConfigArg, WaitingConfig
from ...sql.engine import SQLEngine as BaseSQLEngine
from ...types import is_arrow_type_string_like, is_arrow_type_binary_like, arrow_field_to_schema
from ...types.cast.cast_options import CastOptions
from ...types.cast.registry import convert

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyspark
    import delta
    import pandas
    import polars

    from ...ai.sql_session import SQLAISession, SQLFlavor


__all__ = [
    "SQLEngine",
    "StatementResult",
    "CreateTablePlan",
]


_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _escape_sql_string(s: str) -> str:
    """
    Escape a Python string for safe embedding in a single-quoted SQL literal.

    Args:
        s: Input string.

    Returns:
        SQL-escaped string where single quotes are doubled.
    """
    return s.replace("'", "''")


def _quote_ident(ident: str) -> str:
    """
    Quote a SQL identifier using backticks, escaping embedded backticks.

    Args:
        ident: Identifier to quote (catalog/schema/table/column/etc).

    Returns:
        Backtick-quoted identifier with embedded backticks doubled.
    """
    escaped = ident.replace("`", "``")
    return f"`{escaped}`"


def _needs_column_mapping(col_name: str) -> bool:
    """
    Heuristic: return True if a column name has characters that commonly break without
    Delta column mapping / strict quoting.

    Args:
        col_name: Column name.

    Returns:
        True if column name includes invalid/awkward characters.
    """
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


@dataclasses.dataclass
class CreateTablePlan:
    """
    A plan produced by :meth:`SQLEngine.create_table`.

    Attributes:
        sql:
            The generated CREATE TABLE statement.
        properties:
            Final Delta/Databricks table properties emitted into TBLPROPERTIES.
        warnings:
            Non-fatal issues detected while building the statement (e.g., invalid column names
            while column mapping is disabled).
        arrow_field:
            Arrow field representing the table schema. Convention:
              - If struct: its children become table columns.
              - Otherwise: it represents a single-column table.
        result:
            The execution result when ``execute=True`` was used; otherwise None.
    """

    sql: str
    properties: dict[str, Any]
    warnings: list[str]
    arrow_field: pa.Field
    result: Optional[StatementResult] = None

    @property
    def arrow_schema(self) -> pa.Schema:
        """
        Return the table schema as an Arrow :class:`pyarrow.Schema`.

        Returns:
            Arrow schema derived from :attr:`arrow_field` using project conventions.
        """
        return arrow_field_to_schema(self.arrow_field, None)


@dataclasses.dataclass
class SQLEngine(BaseSQLEngine, WorkspaceService):
    """Execute SQL statements and manage tables via Databricks SQL / Spark."""
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    _warehouse: Optional[SQLWarehouse] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _ai_session: Optional["SQLAISession"] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _cached_queries: Optional[ExpiringDict[str, StatementResult]] = dataclasses.field(default=ExpiringDict, repr=False, hash=False, compare=False)

    # -------------------------------------------------------------------------------------
    # Naming helpers
    # -------------------------------------------------------------------------------------

    def table_full_name(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True,
    ) -> str:
        """
        Build a fully qualified table name (catalog.schema.table).

        Args:
            catalog_name:
                Optional catalog override (defaults to ``self.catalog_name``).
            schema_name:
                Optional schema override (defaults to ``self.schema_name``).
            table_name:
                Table name to qualify.
            safe_chars:
                If True, wraps each identifier in backticks.

        Returns:
            Fully qualified table name string.

        Raises:
            AssertionError: If any of catalog/schema/table is missing after defaults.
        """
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert table_name, "No table name given"

        if safe_chars:
            return f"{_quote_ident(catalog_name)}.{_quote_ident(schema_name)}.{_quote_ident(table_name)}"
        return f"{catalog_name}.{schema_name}.{table_name}"

    def _catalog_schema_table_names(self, full_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Parse a catalog.schema.table string into components.

        Supports partial names:
        - table
        - schema.table
        - catalog.schema.table

        Backticks are stripped from each part.

        Args:
            full_name:
                Fully qualified or partial table name string.

        Returns:
            Tuple (catalog_name, schema_name, table_name), where missing parts may be None
            and caller can fall back to engine defaults.
        """
        parts = [_.strip("`") for _ in full_name.split(".")]

        # Unreachable for .split("."), but harmless and keeps intent explicit.
        if len(parts) == 0:
            return self.catalog_name, self.schema_name, None
        if len(parts) == 1:
            return self.catalog_name, self.schema_name, parts[0]
        if len(parts) == 2:
            return self.catalog_name, parts[0], parts[1]

        catalog_name, schema_name, table_name = parts[-3], parts[-2], parts[-1]
        return catalog_name or self.catalog_name, schema_name or self.schema_name, table_name

    @staticmethod
    def _random_suffix(prefix: str = "") -> str:
        """
        Generate a unique suffix for temporary resources.

        Args:
            prefix: Optional prefix to put before the suffix.

        Returns:
            String of the form "{prefix}{timestamp_ms}_{random8}".
        """
        unique = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        timestamp = int(time.time() * 1000)
        return f"{prefix}{timestamp}_{unique}"

    # -------------------------------------------------------------------------------------
    # Warehouse + AI session
    # -------------------------------------------------------------------------------------

    def warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
    ) -> SQLWarehouse:
        """
        Resolve and cache a Databricks SQL warehouse.

        Behavior:
        - If no warehouse is cached, try to find one (including a starter warehouse).
        - If none found, create or update a warehouse according to workspace policy.
        - Always returns a resolved warehouse (or raises if resolution fails).

        Args:
            warehouse_id:
                Optional explicit warehouse ID to resolve.
            warehouse_name:
                Optional warehouse name to resolve.

        Returns:
            Resolved :class:`SQLWarehouse`.
        """
        if self._warehouse is None:
            wh = SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
            )

            self._warehouse = wh.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                raise_error=False,
                find_starter=True,
            )

            if self._warehouse is None:
                self._warehouse = wh.create_or_update()

        return self._warehouse.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            raise_error=True,
        )

    def ai_session(
        self,
        model: str = "databricks-gemini-2-5-pro",
        flavor: Optional["SQLFlavor"] = None,
    ):
        """
        Create a SQL AI session (LLM-assisted SQL).

        Args:
            model:
                Serving endpoint/model name.
            flavor:
                SQL dialect/flavor. Defaults to Databricks.

        Returns:
            SQLAISession.
        """
        from ...ai.sql_session import SQLAISession, SQLFlavor

        if flavor is None:
            flavor = SQLFlavor.DATABRICKS

        return SQLAISession(
            model=model,
            api_key=self.workspace.current_token(),
            base_url="%s/serving-endpoints" % self.workspace.safe_host,
            flavor=flavor,
        )

    # -------------------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------------------

    def execute(
        self,
        statement: str,
        *,
        row_limit: Optional[int] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = True,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        byte_limit: Optional[int] = None,
        cache_for: Optional[WaitingConfigArg] = None
    ) -> StatementResult:
        """
        Execute a SQL statement using either Spark SQL or the Databricks SQL Statement Execution API.

        Engine selection:
          - If `engine="spark"`, executes using the active SparkSession.
          - If `engine="api"`, executes using the SQL warehouse API.
          - If `engine` is None, defaults to Spark when running inside Databricks with a Spark environment;
            otherwise uses the warehouse API.

        Waiting behavior (API engine only):
          - `wait=True` blocks until the statement reaches a terminal state.
          - `wait=False` returns immediately with a handle; callers can wait later via `StatementResult.wait()`.

        Args:
            statement:
                SQL statement to execute.
            row_limit:
                Optional row limit applied (Spark: `df.limit()`, API: forwarded to API if supported).
            catalog_name:
                Optional catalog override for API engine context.
            schema_name:
                Optional schema override for API engine context.
            wait:
                Whether to wait for completion (API engine only).
            engine:
                Execution engine override: "spark" or "api".
            warehouse_id:
                Warehouse ID override (API engine).
            warehouse_name:
                Warehouse name override (API engine).
            byte_limit:
                Optional maximum bytes returned for results (API engine only).
            cache_for:
                Optional cache time

        Returns:
            StatementResult:
                Unified wrapper over either a Spark DataFrame result or a warehouse API statement result.

        Raises:
            ValueError:
                If `engine="spark"` is selected but there is no active SparkSession.
        """
        # --- Engine auto-detection ---
        if not engine:
            if self.workspace.is_in_databricks_environment():
                engine = "spark"

        statement = statement.strip()

        if cache_for is not None:
            cache_for = WaitingConfig.check_arg(cache_for)

            existing = self._cached_queries.get(statement)

            if existing is not None:
                return existing

        # --- Spark path ---
        if engine == "spark":
            from ...spark.lib import pyspark_sql

            spark_session = pyspark_sql.SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No spark session found to run sql query")

            logger.debug("SPARK SQL executing query:\n%s", statement)

            df: pyspark_sql.DataFrame = spark_session.sql(statement)
            if row_limit:
                df = df.limit(row_limit)

            result = StatementResult(
                workspace_client=self.workspace.sdk(),
                warehouse_id="SparkSQL",
                statement_id="SparkSQL",
                disposition=Disposition.EXTERNAL_LINKS,
            )

            result._spark_df = df
        else:
            # --- Warehouse API path ---
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
            )

        if cache_for is not None:
            self._cached_queries.set(
                key=statement, value=result,
                ttl=cache_for.timeout_total_seconds
            )

        return result

    # -------------------------------------------------------------------------------------
    # Delta table handle (Spark required)
    # -------------------------------------------------------------------------------------

    def spark_table(
        self,
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> "delta.tables.DeltaTable":
        """
        Return a DeltaTable handle for a given table name (Spark context required).

        Args:
            full_name:
                Fully qualified table name (catalog.schema.table). If provided, takes precedence.
            catalog_name:
                Catalog name used when `full_name` is not provided.
            schema_name:
                Schema name used when `full_name` is not provided.
            table_name:
                Table name used when `full_name` is not provided.

        Returns:
            delta.tables.DeltaTable.

        Raises:
            ImportError:
                If delta-spark cannot be imported and runtime install fails.
        """
        from ...spark.lib import pyspark_sql

        try:
            from delta.tables import DeltaTable
        except ImportError:
            from ...pyutils.pyenv import PyEnv

            m = PyEnv.runtime_import_module(module_name="delta.tables", pip_name="delta-spark", install=True)
            DeltaTable = m.DeltaTable

        if not full_name:
            full_name = self.table_full_name(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
            )

        return DeltaTable.forName(
            sparkSession=pyspark_sql.SparkSession.getActiveSession(),
            tableOrViewName=full_name,
        )

    # -------------------------------------------------------------------------------------
    # Insert routing
    # -------------------------------------------------------------------------------------

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
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        wait: Optional[WaitingConfigArg] = True,
        ## Databricks specific
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional["pyspark.sql.SparkSession"] = None,
        spark_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert data into a Delta table.

        Routing strategy:
          - If a SparkSession is available (or `data` is a Spark DataFrame), uses :meth:`spark_insert_into`.
          - Otherwise uses :meth:`arrow_insert_into` which stages Parquet to a temp volume and executes SQL.

        Args:
            data:
                Input data. Supported (path-dependent) inputs include:
                  - Arrow: `pa.Table`, `pa.RecordBatch`, `pa.RecordBatchReader`
                  - Spark: `pyspark.sql.DataFrame`
                  - Tabular Python: dict/list, pandas.DataFrame, polars.DataFrame (convertible)
                  - str: path or special locator (only if your conversion utilities support it)
            mode:
                Save mode/semantics (auto/append/overwrite). Parsed via `SaveMode.from_any`.
            location:
                Fully qualified table name override (catalog.schema.table).
            catalog_name:
                Optional catalog override (used when `location` not provided or partial).
            schema_name:
                Optional schema override (used when `location` not provided or partial).
            table_name:
                Optional table name (used when `location` not provided).
            cast_options:
                Casting rules for aligning input columns to the destination schema.
            overwrite_schema:
                Spark-only: if True, sets write option `overwriteSchema=true`.
            match_by:
                Merge key columns. Enables MERGE semantics (upsert/insert-only/overwrite-by-key).
            wait:
                Waiting completion configuration
            zorder_by:
                Columns for Z-ORDER optimization after write (path-dependent support).
            optimize_after_merge:
                If True, run OPTIMIZE after merge (and ZORDER where supported).
            vacuum_hours:
                If set, run VACUUM with the given retention (hours).
            spark_session:
                Optional SparkSession override. If not provided, uses active session when available.
            spark_options:
                Optional Spark DataFrameWriter options.

        Returns:
            None.
        """
        if spark_session is None:
            try:
                import pyspark.sql as psql

                if isinstance(data, psql.DataFrame):
                    spark_session = data.sparkSession
                else:
                    spark_session = psql.SparkSession.getActiveSession()
            except ImportError:
                pass

        if spark_session is not None:
            return self.spark_insert_into(
                data=data,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                mode=mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                match_by=match_by,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                spark_options=spark_options,
            )
        else:
            return self.arrow_insert_into(
                data=data,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                mode=mode,
                cast_options=cast_options,
                overwrite_schema=overwrite_schema,
                match_by=match_by,
                wait=wait,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
            )

    # -------------------------------------------------------------------------------------
    # Warehouse SQL insert path
    # -------------------------------------------------------------------------------------

    def arrow_insert_into(
        self,
        data: Union[
            pa.Table,
            pa.RecordBatch,
            pa.RecordBatchReader,
            dict,
            list,
            "pandas.DataFrame",
            "polars.DataFrame",
        ],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: SaveMode | str | None = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        wait: Optional[WaitingConfigArg] = True,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        existing_schema: pa.Schema | None = None,
        temp_volume_path: Optional[Union[str, DatabricksPath]] = None,
    ) -> None:
        """
        Insert data using the warehouse SQL path by staging Parquet files to a temporary volume, then executing
        SQL INSERT/MERGE into the target Delta table.

        Notes:
          - If the table does not exist, it is created from the input schema (best-effort) via :meth:`create_table`.
          - If `match_by` is provided, uses MERGE semantics:
              * mode=overwrite: delete matching keys, then insert batch (overwrite-by-key)
              * mode=append: insert-only (no updates)
              * mode=auto/other: full upsert (update matched + insert new)
          - If `match_by` is not provided:
              * mode=overwrite: INSERT OVERWRITE
              * otherwise: INSERT INTO
          - Rows with NULL in any match key are filtered out (mirrors Spark path).

        Args:
            data:
                Input data (Arrow or convertible). Non-Arrow inputs may be converted by project utilities.
            location:
                Fully qualified table name override (catalog.schema.table).
            catalog_name:
                Optional catalog override (used when `location` not provided/partial).
            schema_name:
                Optional schema override (used when `location` not provided/partial).
            table_name:
                Optional table name (used when `location` not provided).
            mode:
                Save mode/semantics (auto/append/overwrite) parsed via `SaveMode.from_any`.
            cast_options:
                Casting rules for aligning staged Parquet columns to destination schema.
            overwrite_schema:
                Reserved for parity with Spark path (unused in SQL staging path).
            match_by:
                Merge key columns enabling MERGE behavior.
            wait:
                Waiting completion configuration
            zorder_by:
                If set, runs `OPTIMIZE ... ZORDER BY (...)` after inserting.
            optimize_after_merge:
                If True and `match_by` is set, runs `OPTIMIZE <table>` after merge (in addition to ZORDER).
            vacuum_hours:
                If set, runs `VACUUM <table> RETAIN <hours> HOURS`.
            existing_schema:
                Optional destination schema (Arrow) to avoid an extra schema fetch.
            temp_volume_path:
                Optional staging path override. If not provided, a temp volume path is allocated.

        Returns:
            None.
        """
        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )

        with self.connect() as connected:
            if existing_schema is None:
                try:
                    existing_schema = connected.get_table_schema(
                        location=location,
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        to_arrow_schema=True,
                    )
                except ValueError as exc:
                    if isinstance(data, (list, dict)):
                        from ...polars.cast import any_to_polars_dataframe

                        data = any_to_polars_dataframe(data, cast_options)

                    logger.warning("%s, creating it from input schema %s", exc, repr(data))

                    plan = connected.create_table(
                        field=data,  # convertible to pa.Field via convert() in create_table
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        if_not_exists=True,
                        execute=True,
                    )

                    try:
                        connected.arrow_insert_into(
                            data=data,
                            location=location,
                            catalog_name=catalog_name,
                            schema_name=schema_name,
                            table_name=table_name,
                            mode="overwrite",
                            cast_options=cast_options,
                            overwrite_schema=overwrite_schema,
                            match_by=match_by,
                            wait=wait,
                            zorder_by=zorder_by,
                            optimize_after_merge=optimize_after_merge,
                            vacuum_hours=vacuum_hours,
                            existing_schema=plan.arrow_schema,
                        )
                        return
                    except Exception:
                        logger.exception(
                            "Arrow insert failed after auto-creating %s; attempting cleanup (DROP TABLE)", location
                        )
                        try:
                            connected.drop_table(location=location, wait=True)
                        except Exception as e:
                            logger.exception("Failed to drop table %s after auto creation error: %s", location, e)
                        raise

            cast_options = CastOptions.check_arg(options=cast_options, target_field=existing_schema)

            logger.debug("Inserting %s into %s", data, location)

            temp_volume_path = (
                self.workspace.tmp_path(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    volume_name="tmp",
                    extension="parquet",
                    max_lifetime=3600,
                )
                if temp_volume_path is None
                else DatabricksPath.parse(obj=temp_volume_path, workspace=connected.workspace)
            )

            temp_volume_path.write_table(
                data,
                file_format=FileFormat.PARQUET,
                cast_options=cast_options,
            )

            mode = SaveMode.from_any(mode, default=SaveMode.AUTO)

            columns = list(existing_schema.names)
            cols_quoted = ", ".join([_quote_ident(c) for c in columns])

            statements: list[str] = []

            if match_by:
                not_null_pred = " AND ".join([f"{_quote_ident(k)} IS NOT NULL" for k in match_by])

                source_sql = f"""SELECT {cols_quoted}
FROM parquet.{_quote_ident(str(temp_volume_path))}
WHERE {not_null_pred}""".strip()

                on_condition = " AND ".join([f"T.{_quote_ident(k)} = S.{_quote_ident(k)}" for k in match_by])

                if mode == SaveMode.OVERWRITE:
                    key_cols = ", ".join([_quote_ident(k) for k in match_by])

                    delete_sql = f"""DELETE FROM {location} AS T
USING (
  SELECT DISTINCT {key_cols}
  FROM parquet.{_quote_ident(str(temp_volume_path))}
  WHERE {not_null_pred}
) AS S
ON {on_condition}""".strip()

                    insert_sql = f"""INSERT INTO {location} ({cols_quoted})
{source_sql}""".strip()

                    # Need consecutive queries
                    wait = True
                    statements.extend([delete_sql, insert_sql])

                elif mode == SaveMode.APPEND:
                    insert_clause = (
                        f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                        f"VALUES ({', '.join([f'S.{_quote_ident(c)}' for c in columns])})"
                    )

                    merge_sql = f"""MERGE INTO {location} AS T
USING (
  {source_sql}
) AS S
ON {on_condition}
{insert_clause}""".strip()

                    statements.append(merge_sql)

                else:
                    update_cols = [c for c in columns if c not in match_by]
                    update_clause = ""
                    if update_cols:
                        update_set = ", ".join([f"T.{_quote_ident(c)} = S.{_quote_ident(c)}" for c in update_cols])
                        update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"

                    insert_clause = (
                        f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                        f"VALUES ({', '.join([f'S.{_quote_ident(c)}' for c in columns])})"
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
                if mode == SaveMode.OVERWRITE:
                    insert_sql = f"""INSERT OVERWRITE {location}
SELECT {cols_quoted}
FROM parquet.{_quote_ident(str(temp_volume_path))}"""
                else:
                    insert_sql = f"""INSERT INTO {location} ({cols_quoted})
SELECT {cols_quoted}
FROM parquet.{_quote_ident(str(temp_volume_path))}"""
                statements.append(insert_sql)

            try:
                for stmt in statements:
                    connected.execute(stmt, wait=wait)
            finally:
                try:
                    Thread(target=temp_volume_path.remove, kwargs={"recursive": True}).start()
                except Exception:
                    logger.exception("Failed cleaning temp volume: %s", temp_volume_path)

            logger.info("Arrow inserted into %s", location)

            if zorder_by:
                zorder_cols = ", ".join([_quote_ident(c) for c in zorder_by])
                connected.execute(f"OPTIMIZE {location} ZORDER BY ({zorder_cols})")

            if optimize_after_merge and match_by:
                connected.execute(f"OPTIMIZE {location}")

            if vacuum_hours is not None:
                connected.execute(f"VACUUM {location} RETAIN {int(vacuum_hours)} HOURS")

        return None

    # -------------------------------------------------------------------------------------
    # Spark insert path
    # -------------------------------------------------------------------------------------

    def spark_insert_into(
        self,
        data: Any,
        *,
        mode: SaveMode | str | None = None,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert data into a Delta table using Spark.

        Accepts a Spark DataFrame or any input convertible to a Spark DataFrame via
        your project's `any_to_spark_dataframe()` utility.

        Behavior:
          - If destination table does not exist: creates it via `saveAsTable(mode="overwrite")`.
          - If `match_by` is provided: uses Delta MERGE semantics.
              * mode=overwrite: overwrite-by-key (delete matching keys, then append)
              * mode=append: insert-only (whenNotMatchedInsertAll)
              * mode=auto/other: upsert (whenMatchedUpdate + whenNotMatchedInsertAll)
          - If no `match_by`: uses DataFrameWriter `saveAsTable()` with append/overwrite.

        Notes:
          - Rows with NULL in any match key are filtered out (mirrors SQL path).
          - `optimize_after_merge` will only run ZORDER when `zorder_by` is provided.
          - If `overwrite_schema` is True, sets Spark option `overwriteSchema=true`.

        Args:
            data:
                Input data convertible to Spark DataFrame.
            mode:
                Save mode/semantics (auto/append/overwrite) parsed via `SaveMode.from_any`.
            location:
                Fully qualified table name override (catalog.schema.table).
            catalog_name:
                Optional catalog override (used when `location` not provided/partial).
            schema_name:
                Optional schema override (used when `location` not provided/partial).
            table_name:
                Optional table name (used when `location` not provided).
            cast_options:
                Casting rules for aligning input columns to destination schema.
            overwrite_schema:
                If True, sets writer option `overwriteSchema=true` (Spark path).
            match_by:
                Merge key columns enabling MERGE behavior.
            zorder_by:
                Z-ORDER columns for optimize after merge (requires `optimize_after_merge=True`).
            optimize_after_merge:
                If True and `zorder_by` is provided, runs Delta optimize ZORDER after merge/write.
            vacuum_hours:
                If set, runs Delta vacuum with given retention hours.
            spark_options:
                Spark DataFrameWriter options dict (e.g., {"mergeSchema": "true"}).

        Returns:
            None.
        """
        from ...spark.cast import any_to_spark_dataframe
        from pyspark.sql import DataFrame
        import pyspark.sql.functions as F

        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )

        logger.info(
            "Spark insert into %s (mode=%s, match_by=%s, overwrite_schema=%s)",
            location,
            mode,
            match_by,
            overwrite_schema,
        )

        spark_options = spark_options if spark_options else {}
        if overwrite_schema:
            spark_options["overwriteSchema"] = "true"

        try:
            existing_schema = self.get_table_schema(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                to_arrow_schema=False,
            )
        except ValueError:
            logger.warning("Destination table missing; creating table %s via overwrite write", location)
            data_df = convert(data, DataFrame)
            data_df.write.mode("overwrite").options(**spark_options).saveAsTable(location)
            return

        cast_options = CastOptions.check_arg(options=cast_options, target_field=existing_schema)
        data_df = any_to_spark_dataframe(data, cast_options)

        if match_by:
            notnull = None
            for k in match_by:
                if k not in data_df.columns:
                    raise ValueError(f"Missing match key '{k}' in DataFrame columns: {data_df.columns}")
                notnull = data_df[k].isNotNull() if notnull is None else notnull & data_df[k].isNotNull()

            data_df = data_df.filter(notnull)
            logger.debug("Filtered null keys for match_by=%s", match_by)

        target = self.spark_table(full_name=location)
        mode = SaveMode.from_any(mode, default=SaveMode.AUTO)

        if match_by:
            cond = " AND ".join([f"t.{_quote_ident(k)} <=> s.{_quote_ident(k)}" for k in match_by])

            if mode == SaveMode.OVERWRITE:
                data_df = data_df.cache()
                distinct_keys = data_df.select([f"{_quote_ident(k)}" for k in match_by]).distinct()

                (
                    target.alias("t")
                    .merge(distinct_keys.alias("s"), cond)
                    .whenMatchedDelete()
                    .execute()
                )

                (
                    data_df.write.format("delta")
                    .mode("append")
                    .options(**spark_options)
                    .saveAsTable(location)
                )

            elif mode == SaveMode.APPEND:
                (
                    target.alias("t")
                    .merge(data_df.alias("s"), cond)
                    .whenNotMatchedInsertAll()
                    .execute()
                )

            else:
                update_cols = [c for c in data_df.columns if c not in match_by]
                set_expr = {c: F.expr(f"s.{_quote_ident(c)}") for c in update_cols}

                (
                    target.alias("t")
                    .merge(data_df.alias("s"), cond)
                    .whenMatchedUpdate(set=set_expr)
                    .whenNotMatchedInsertAll()
                    .execute()
                )

        else:
            spark_mode = "overwrite" if mode == SaveMode.OVERWRITE else "append"
            logger.info("Spark write saveAsTable mode=%s", spark_mode)
            data_df.write.mode(spark_mode).options(**spark_options).saveAsTable(location)

        if optimize_after_merge and zorder_by:
            logger.info("Delta optimize + zorder (%s)", zorder_by)
            target.optimize().executeZOrderBy(*zorder_by)

        if vacuum_hours is not None:
            logger.info("Delta vacuum retain=%s hours", vacuum_hours)
            target.vacuum(vacuum_hours)

        return None

    # -------------------------------------------------------------------------------------
    # Schema + table management
    # -------------------------------------------------------------------------------------

    def get_table_schema(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        to_arrow_schema: bool = True,
    ) -> Union[pa.Field, pa.Schema]:
        """
        Fetch a table schema from Unity Catalog and convert it to Arrow types.

        Args:
            location:
                Optional fully qualified table name (catalog.schema.table).
            catalog_name:
                Optional catalog override (used when `location` not provided/partial).
            schema_name:
                Optional schema override (used when `location` not provided/partial).
            table_name:
                Optional table name (used when `location` not provided).
            to_arrow_schema:
                If True returns a `pa.Schema`.
                If False returns a `pa.Field` with STRUCT type representing the table schema.

        Returns:
            `pa.Schema` or `pa.Field` depending on `to_arrow_schema`.

        Raises:
            ValueError:
                If the table cannot be retrieved from Unity Catalog.
        """
        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=False,
        )

        client = self.workspace.sdk().tables

        try:
            table = client.get(location)
        except Exception as e:
            raise ValueError(f"Table {location} not found, {type(e)} {e}")

        fields = [column_info_to_arrow_field(_) for _ in table.columns]

        metadata = {
            b"engine": b"databricks",
            b"full_name": location.encode("utf-8"),
            b"catalog_name": (catalog_name or "").encode("utf-8"),
            b"schema_name": (schema_name or "").encode("utf-8"),
            b"table_name": (table_name or "").encode("utf-8"),
        }

        if to_arrow_schema:
            return pa.schema(fields, metadata=metadata)

        return pa.field(location, pa.struct(fields), metadata=metadata)

    def drop_table(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = True,
    ) -> None:
        """
        Drop a table if it exists.

        Args:
            location:
                Optional fully qualified table name (catalog.schema.table).
            catalog_name:
                Optional catalog override (used when `location` not provided/partial).
            schema_name:
                Optional schema override (used when `location` not provided/partial).
            table_name:
                Optional table name (used when `location` not provided).
            wait:
                Whether to block until the DROP finishes (API engine only).

        Returns:
            None.
        """
        location, _, _, _ = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )

        logger.debug("Dropping table if exists: %s", location)
        self.execute(f"DROP TABLE IF EXISTS {location}", wait=wait)
        logger.info("Dropped table if exists: %s", location)

    def create_table(
        self,
        field: Union[pa.Field, pa.Schema, Any],
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        storage_location: Optional[str] = None,
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[bool | list[str]] = True,
        comment: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        if_not_exists: bool = True,
        or_replace: bool = False,
        using: str = "DELTA",
        optimize_write: bool = True,
        auto_compact: bool = True,
        enable_cdf: Optional[bool] = None,
        enable_deletion_vectors: Optional[bool] = None,
        target_file_size: Optional[int] = None,
        column_mapping_mode: Optional[str] = None,
        execute: bool = True,
        wait_result: bool = True,
    ) -> CreateTablePlan:
        """
        Generate (and optionally execute) a Databricks/Delta CREATE TABLE statement from an Apache Arrow schema/field,
        with safety checks and performance-oriented defaults.

        Safety/perf behaviors:
          - Quotes identifiers to avoid keyword/name edge cases.
          - Validates `partition_by` / `cluster_by` columns exist in schema.
          - Supports managed or external tables via `storage_location`.
          - Optionally enables Delta Column Mapping (name/id) with required protocol props.
          - Adds workspace default tags into table properties (`tags.<k>`).

        Args:
            field:
                Arrow schema/field describing the table, or any object convertible to `pa.Field` via `convert()`.
                - If `pa.Schema`: all schema fields become columns.
                - If struct `pa.Field`: its children become columns.
                - If non-struct `pa.Field`: single-column table.
            full_name:
                Fully qualified table name "catalog.schema.table". If provided, takes precedence.
            catalog_name:
                Catalog used when `full_name` not provided (or as override for partial `full_name`).
            schema_name:
                Schema used when `full_name` not provided (or as override for partial `full_name`).
            table_name:
                Table used when `full_name` not provided (or as override for partial `full_name`).
            storage_location:
                External storage location path. If set, emits `LOCATION '<path>'` (SQL-escaped).
            partition_by:
                Partition column names. Must exist in schema.
                Note: if set, clustering is not emitted (partition wins).
            cluster_by:
                Controls clustering / liquid clustering.
                - True: emits `CLUSTER BY AUTO`
                - False: emits nothing
                - list[str]: emits `CLUSTER BY (<cols...>)` (must exist in schema)
                Note: only applied when `partition_by` is not set.
            comment:
                Table comment. If None and Arrow metadata contains b"comment", that is used.
            properties:
                Additional/override Delta table properties (caller wins last).
            if_not_exists:
                If True, generates `CREATE TABLE IF NOT EXISTS ...`. Mutually exclusive with `or_replace`.
            or_replace:
                If True, generates `CREATE OR REPLACE TABLE ...`. Mutually exclusive with `if_not_exists`.
            using:
                Storage format keyword (default "DELTA").
            optimize_write:
                Sets `delta.autoOptimize.optimizeWrite`.
            auto_compact:
                Sets `delta.autoOptimize.autoCompact`.
            enable_cdf:
                If set, sets `delta.enableChangeDataFeed`.
            enable_deletion_vectors:
                If set, sets `delta.enableDeletionVectors`.
            target_file_size:
                If set, sets `delta.targetFileSize` (bytes).
            column_mapping_mode:
                Delta column mapping mode:
                - None: auto-detect (enable "name" iff invalid column names exist, else "none")
                - "none": do not enable column mapping
                - "name": enable name-based column mapping
                - "id": enable id-based column mapping
                When enabled (name/id), also sets:
                - `delta.minReaderVersion = 2`
                - `delta.minWriterVersion = 5`
            execute:
                If True, executes the generated SQL via :meth:`execute`.
            wait_result:
                Passed to :meth:`execute` when `execute=True`.

        Returns:
            CreateTablePlan:
                Always returns a plan. If `execute=True`, plan.result is populated.

        Raises:
            ValueError:
                On invalid naming parameters, conflicting flags, invalid column mapping mode, or missing columns.
            SqlStatementError:
                If execution fails and cannot be recovered (e.g., schema creation retry fails).
        """
        if not isinstance(field, pa.Field):
            field = convert(field, pa.Field)

        schema_metadata = field.metadata or {}

        if pa.types.is_struct(field.type):
            arrow_fields = list(field.type)
        else:
            arrow_fields = [field]

        full_name, catalog_name, schema_name, table_name = self._check_location_params(
            location=full_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )

        if comment is None and schema_metadata:
            c = schema_metadata.get(b"comment")
            if isinstance(c, bytes):
                comment = c.decode("utf-8")

        any_invalid = any(_needs_column_mapping(f.name) for f in arrow_fields)
        warnings: list[str] = []

        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        if column_mapping_mode not in ("none", "name", "id"):
            raise ValueError("column_mapping_mode must be one of: None, 'none', 'name', 'id'.")

        col_names = {f.name for f in arrow_fields}

        if partition_by:
            missing = [c for c in partition_by if c not in col_names]
            if missing:
                raise ValueError(f"partition_by contains unknown columns: {missing}")

        if isinstance(cluster_by, list):
            missing = [c for c in cluster_by if c not in col_names]
            if missing:
                raise ValueError(f"cluster_by contains unknown columns: {missing}")

        column_definitions = [self._field_to_ddl(child) for child in arrow_fields]

        if or_replace and if_not_exists:
            raise ValueError("Use either or_replace or if_not_exists, not both.")

        create_kw = "CREATE OR REPLACE TABLE" if or_replace else "CREATE TABLE"
        if if_not_exists and not or_replace:
            create_kw = "CREATE TABLE IF NOT EXISTS"

        sql_parts: list[str] = [
            f"{create_kw} {full_name} (",
            "  " + ",\n  ".join(column_definitions),
            ")",
            f"USING {using}",
        ]

        if partition_by:
            sql_parts.append("PARTITIONED BY (" + ", ".join(_quote_ident(c) for c in partition_by) + ")")
        elif cluster_by:
            if isinstance(cluster_by, bool):
                if cluster_by:
                    sql_parts.append("CLUSTER BY AUTO")
            else:
                sql_parts.append("CLUSTER BY (" + ", ".join(_quote_ident(c) for c in cluster_by) + ")")

        if comment:
            sql_parts.append(f"COMMENT '{_escape_sql_string(comment)}'")

        if storage_location:
            sql_parts.append(f"LOCATION '{_escape_sql_string(storage_location)}'")

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

        if any_invalid and column_mapping_mode == "none":
            warnings.append(
                "Schema has invalid column names but column_mapping_mode='none'. "
                "This will fail unless you rename/escape columns or enable column mapping."
            )

        default_tags = self.workspace.default_tags()
        for k, v in default_tags.items():
            props[f"tags.{k}"] = v

        if props:

            def fmt(_key: str, _value: Any) -> str:
                if isinstance(_value, str):
                    return f"'{_key}' = '{_escape_sql_string(_value)}'"
                if isinstance(_value, bool):
                    return f"'{_key}' = '{'true' if _value else 'false'}'"
                return f"'{_key}' = {_value}"

            sql_parts.append("TBLPROPERTIES (" + ", ".join(fmt(k, v) for k, v in props.items()) + ")")

        statement = "\n".join(sql_parts)
        plan = CreateTablePlan(sql=statement, properties=props, warnings=warnings, arrow_field=field)

        if not execute:
            return plan

        try:
            res = self.execute(statement, wait=wait_result)
        except SqlStatementError as e:
            if "SCHEMA_NOT_FOUND" in e.message:
                self.execute(f"CREATE SCHEMA IF NOT EXISTS {_quote_ident(schema_name)}", wait=True)
                res = self.execute(statement, wait=wait_result)
            else:
                raise

        plan.result = res
        return plan

    def _check_location_params(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """
        Resolve (location OR catalog/schema/table) into a fully-qualified table identifier.

        Args:
            location:
                Fully or partially-qualified table name string. Accepts:
                - table
                - schema.table
                - catalog.schema.table
            catalog_name:
                Optional catalog override, or default when `location` doesn't provide it.
            schema_name:
                Optional schema override, or default when `location` doesn't provide it.
            table_name:
                Optional table override, or default when `location` doesn't provide it.
            safe_chars:
                If True, quotes identifiers using backticks.

        Returns:
            Tuple (location_fqn, catalog_name, schema_name, table_name) where:
            - location_fqn is the resolved full name (quoted if safe_chars=True)
            - the name components are resolved with engine defaults as needed
        """
        if location:
            c, s, t = self._catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = catalog_name or c, schema_name or s, table_name or t

        location = self.table_full_name(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=safe_chars,
        )

        return location, catalog_name or self.catalog_name, schema_name or self.schema_name, table_name

    @staticmethod
    def _field_to_ddl(
        field: pa.Field,
        put_name: bool = True,
        put_not_null: bool = True,
        put_comment: bool = True,
    ) -> str:
        """
        Convert an Arrow field to a Databricks SQL column DDL fragment.

        Supports primitives plus nested types:
        - STRUCT<...>
        - MAP<k,v>
        - ARRAY<elem>

        Args:
            field:
                Arrow field to convert.
            put_name:
                If True, include the column name.
            put_not_null:
                If True, emit NOT NULL when `field.nullable` is False.
            put_comment:
                If True, emit COMMENT when field metadata contains b"comment".

        Returns:
            SQL fragment for a column definition.

        Raises:
            TypeError:
                If a nested Arrow type cannot be represented.
            ValueError:
                If a primitive Arrow type cannot be mapped to a SQL type.
        """
        name = field.name
        nullable_str = " NOT NULL" if put_not_null and not field.nullable else ""
        name_str = f"{_quote_ident(name)} " if put_name else ""

        comment_str = ""
        if put_comment and field.metadata and b"comment" in field.metadata:
            comment = field.metadata[b"comment"].decode("utf-8")
            comment_str = f" COMMENT '{_escape_sql_string(comment)}'"

        if not pa.types.is_nested(field.type):
            sql_type = SQLEngine._arrow_to_sql_type(field.type)
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        if pa.types.is_struct(field.type):
            struct_body = ", ".join([SQLEngine._field_to_ddl(child) for child in field.type])
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        if pa.types.is_map(field.type):
            map_type: pa.MapType = field.type
            key_type = SQLEngine._field_to_ddl(
                map_type.key_field,
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            val_type = SQLEngine._field_to_ddl(
                map_type.item_field,
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            list_type: pa.ListType = field.type
            elem_type = SQLEngine._field_to_ddl(
                list_type.value_field,
                put_name=False,
                put_comment=False,
                put_not_null=False,
            )
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        raise TypeError(f"Cannot make ddl field from {field}")

    @staticmethod
    def _arrow_to_sql_type(arrow_type: Union[pa.DataType, pa.Decimal128Type]) -> str:
        """
        Convert an Arrow data type to a Databricks SQL type string.

        Args:
            arrow_type:
                Arrow type instance to convert.

        Returns:
            Databricks SQL type string.

        Raises:
            ValueError:
                If the Arrow type is unsupported.
        """
        if pa.types.is_boolean(arrow_type):
            return "BOOLEAN"
        if pa.types.is_int8(arrow_type):
            return "TINYINT"
        if pa.types.is_int16(arrow_type):
            return "SMALLINT"
        if pa.types.is_int32(arrow_type):
            return "INT"
        if pa.types.is_int64(arrow_type):
            return "BIGINT"
        if pa.types.is_float32(arrow_type):
            return "FLOAT"
        if pa.types.is_float64(arrow_type):
            return "DOUBLE"
        if is_arrow_type_string_like(arrow_type):
            return "STRING"
        if is_arrow_type_binary_like(arrow_type):
            return "BINARY"
        if pa.types.is_timestamp(arrow_type):
            tz = getattr(arrow_type, "tz", None)
            return "TIMESTAMP" if tz else "TIMESTAMP_NTZ"
        if pa.types.is_date(arrow_type):
            return "DATE"
        if pa.types.is_decimal(arrow_type):
            return f"DECIMAL({arrow_type.precision}, {arrow_type.scale})"
        if pa.types.is_null(arrow_type):
            # Pragmatic fallback: Delta doesn't support a dedicated NULL column type.
            return "STRING"
        raise ValueError(f"Cannot make ddl type for {arrow_type}")
