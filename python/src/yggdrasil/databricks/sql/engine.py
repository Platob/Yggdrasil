"""Databricks SQL engine utilities and helpers.

This module provides a thin “do the right thing” layer over:
- Databricks SQL Statement Execution API (warehouse)
- Spark SQL / Delta Lake (when running inside a Spark-enabled context)

It includes helpers to:
- Build fully-qualified table names
- Execute SQL via Spark or Databricks SQL API
- Insert Arrow/Spark data into Delta tables (append/overwrite/merge)
- Generate DDL from Arrow schemas
"""

import dataclasses
import logging
import random
import string
import time
from threading import Thread
from typing import Optional, Union, Any, Dict, List, Literal

import pyarrow as pa
import pyarrow.dataset as pds

from .statement_result import StatementResult
from .types import column_info_to_arrow_field
from .warehouse import SQLWarehouse
from ..workspaces import WorkspaceService, DatabricksPath
from ...ai.sql_session import SQLAISession, SQLFlavor
from ...libs.databrickslib import databricks_sdk, DatabricksDummyClass
from ...libs.sparklib import SparkSession, SparkDataFrame, pyspark
from ...pyutils.waiting_config import WaitingConfigArg
from ...types import is_arrow_type_string_like, is_arrow_type_binary_like
from ...types.cast.cast_options import CastOptions
from ...types.cast.registry import convert
from ...types.cast.spark_cast import cast_spark_dataframe

try:
    from delta.tables import DeltaTable as SparkDeltaTable
except ImportError:
    class SparkDeltaTable:
        @classmethod
        def forName(cls, *args, **kwargs):
            from delta.tables import DeltaTable
            return DeltaTable.forName(*args, **kwargs)


if databricks_sdk is not None:
    from databricks.sdk.service.sql import (
        Disposition, Format,
        ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem
    )
else:
    Disposition = DatabricksDummyClass
    Format = DatabricksDummyClass
    ExecuteStatementRequestOnWaitTimeout = DatabricksDummyClass
    StatementParameterListItem = DatabricksDummyClass


logger = logging.getLogger(__name__)

if pyspark is not None:
    import pyspark.sql.functions as F


__all__ = [
    "SQLEngine",
    "StatementResult"
]


@dataclasses.dataclass
class CreateTablePlan:
    sql: str
    properties: dict[str, Any]
    warnings: list[str]
    result: Any = None  # StatementResult when executed


_INVALID_COL_CHARS = set(" ,;{}()\n\t=")


def _escape_sql_string(s: str) -> str:
    return s.replace("'", "''")


def _quote_ident(ident: str) -> str:
    # Always quote to be safe; also allows reserved keywords
    escaped = ident.replace("`", "``")
    return f"`{escaped}`"


def _needs_column_mapping(col_name: str) -> bool:
    return any(ch in _INVALID_COL_CHARS for ch in col_name)


@dataclasses.dataclass
class SQLEngine(WorkspaceService):
    """Execute SQL statements and manage tables via Databricks SQL / Spark."""
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    _warehouse: Optional[SQLWarehouse] = dataclasses.field(default=None, repr=False, hash=False, compare=False)
    _ai_session: Optional[SQLAISession] = dataclasses.field(default=None, repr=False, hash=False, compare=False)

    def table_full_name(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True,
    ) -> str:
        """Build a fully qualified table name (catalog.schema.table).

        Args:
            catalog_name: Optional catalog override (defaults to engine.catalog_name).
            schema_name: Optional schema override (defaults to engine.schema_name).
            table_name: Table name to qualify.
            safe_chars: Whether to wrap each identifier in backticks.

        Returns:
            Fully qualified table name string.
        """
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert table_name, "No table name given"

        if safe_chars:
            return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
        return f"{catalog_name}.{schema_name}.{table_name}"

    def _catalog_schema_table_names(self, full_name: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse a catalog.schema.table string into components.

        Supports partial names:
        - table
        - schema.table
        - catalog.schema.table

        Backticks are stripped.

        Args:
            full_name: Fully qualified or partial table name.

        Returns:
            Tuple of (catalog_name, schema_name, table_name).
        """
        parts = [_.strip("`") for _ in full_name.split(".")]

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
        """Generate a unique suffix for temporary resources."""
        unique = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        timestamp = int(time.time() * 1000)
        return f"{prefix}{timestamp}_{unique}"

    def warehouse(
        self,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
    ) -> SQLWarehouse:
        if self._warehouse is None:
            wh = SQLWarehouse(
                workspace=self.workspace,
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name
            )

            self._warehouse = wh.find_warehouse(
                warehouse_id=warehouse_id,
                warehouse_name=warehouse_name,
                raise_error=False
            )

            if self._warehouse is None:
                self._warehouse = wh.create_or_update()

        return self._warehouse.find_warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            raise_error=True
        )

    def ai_session(
        self,
        model: str = "databricks-gemini-2-5-pro",
        flavor: SQLFlavor = SQLFlavor.DATABRICKS
    ):
        return SQLAISession(
            model=model,
            api_key=self.workspace.current_token(),
            base_url="%s/serving-endpoints" % self.workspace.safe_host,
            flavor=flavor
        )

    def execute(
        self,
        statement: Optional[str] = None,
        *,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: Optional[str] = None,
        warehouse_name: Optional[str] = None,
        byte_limit: Optional[int] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None,
        parameters: Optional[List[StatementParameterListItem]] = None,
        row_limit: Optional[int] = None,
        wait_timeout: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = True
    ) -> "StatementResult":
        """Execute a SQL statement via Spark or Databricks SQL Statement Execution API.

        Engine resolution:
        - If `engine` is not provided and a Spark session is active -> uses Spark.
        - Otherwise uses Databricks SQL API (warehouse).

        Waiting behavior (`wait_result`):
        - If True (default): returns a StatementResult in terminal state (SUCCEEDED/FAILED/CANCELED).
        - If False: returns immediately with the initial handle (caller can `.wait()` later).

        Args:
            statement: SQL statement to execute. If None, a `SELECT *` is generated from the table params.
            engine: "spark" or "api".
            warehouse_id: Warehouse override (for API engine).
            warehouse_name: Warehouse name override (for API engine).
            byte_limit: Optional byte limit for results.
            disposition: Result disposition mode (API engine).
            format: Result format (API engine).
            on_wait_timeout: Timeout behavior for waiting (API engine).
            parameters: Optional statement parameters (API engine).
            row_limit: Optional row limit for results (API engine).
            wait_timeout: API wait timeout value.
            catalog_name: Optional catalog override for API engine.
            schema_name: Optional schema override for API engine.
            wait: Whether to block until completion (API engine).

        Returns:
            StatementResult.
        """
        # --- Engine auto-detection ---
        if not engine:
            if pyspark is not None:
                spark_session = SparkSession.getActiveSession()
                if spark_session is not None:
                    engine = "spark"

        # --- Spark path ---
        if engine == "spark":
            spark_session = SparkSession.getActiveSession()
            if spark_session is None:
                raise ValueError("No spark session found to run sql query")

            logger.debug(
                "SPARK SQL executing query:\n%s",
                statement
            )

            df: SparkDataFrame = spark_session.sql(statement)

            if row_limit:
                df = df.limit(row_limit)

            return StatementResult(
                workspace_client=self.workspace.sdk(),
                warehouse_id="SparkSQL",
                statement_id="SparkSQL",
                disposition=Disposition.EXTERNAL_LINKS,
                _spark_df=df,
            )

        wh = self.warehouse(
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
        )

        return wh.execute(
            statement=statement,
            warehouse_id=warehouse_id,
            warehouse_name=warehouse_name,
            byte_limit=byte_limit,
            disposition=disposition,
            format=format,
            on_wait_timeout=on_wait_timeout,
            parameters=parameters,
            wait_timeout=wait_timeout,
            catalog_name=catalog_name,
            schema_name=schema_name,
            wait=wait
        )

    def spark_table(
        self,
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        """Return a DeltaTable handle for a given table name (Spark context required)."""
        if not full_name:
            full_name = self.table_full_name(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
            )
        return SparkDeltaTable.forName(
            sparkSession=SparkSession.getActiveSession(),
            tableOrViewName=full_name,
        )

    def insert_into(
        self,
        data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader, SparkDataFrame],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_session: Optional[SparkSession] = None,
        spark_options: Optional[Dict[str, Any]] = None,
    ):
        """Insert data into a Delta table using Spark when available; otherwise stage Arrow.

        Strategy:
        - If Spark is available and we have an active session (or Spark DF input) -> use `spark_insert_into`.
        - Otherwise -> use `arrow_insert_into` (stages Parquet to a temp volume + runs SQL INSERT/MERGE).

        Args:
            data: Arrow or Spark data to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite").
            cast_options: Optional casting options.
            overwrite_schema: Whether to overwrite schema (Spark path).
            match_by: Merge keys for upserts (MERGE semantics). When set, mode affects behavior.
            zorder_by: Z-ORDER columns (SQL path uses OPTIMIZE ZORDER; Spark path uses Delta optimize API).
            optimize_after_merge: Whether to run OPTIMIZE after a merge (SQL path) / after merge+zorder (Spark path).
            vacuum_hours: Optional VACUUM retention window.
            spark_session: Optional SparkSession override.
            spark_options: Optional Spark write options.

        Returns:
            None (mutates the destination table).
        """

        if pyspark is not None or spark_session is not None:
            spark_session = SparkSession.getActiveSession() if spark_session is None else spark_session

            if spark_session is not None or isinstance(data, SparkDataFrame):
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
            zorder_by=zorder_by,
            optimize_after_merge=optimize_after_merge,
            vacuum_hours=vacuum_hours,
        )

    def arrow_insert_into(
        self,
        data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        existing_schema: pa.Schema | None = None,
        temp_volume_path: Optional[Union[str, DatabricksPath]] = None,
    ):
        """Insert Arrow data by staging Parquet to a temp volume and running Databricks SQL.

        Notes:
        - If the table does not exist, it is created from the input Arrow schema (best-effort).
        - If `match_by` is provided, uses MERGE INTO (upsert).
        - Otherwise uses INSERT INTO / INSERT OVERWRITE depending on mode.

        Args:
            data: Arrow table/batch data to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite"). ("auto" behaves like append here.)
            cast_options: Optional casting options.
            overwrite_schema: Reserved for parity with Spark path (unused here).
            match_by: Merge keys for MERGE INTO upserts.
            zorder_by: Columns for OPTIMIZE ZORDER BY.
            optimize_after_merge: Run OPTIMIZE after MERGE (in addition to ZORDER optimization).
            vacuum_hours: Optional VACUUM retention window in hours.
            existing_schema: Optional pre-fetched destination schema (Arrow).
            temp_volume_path: Optional temp volume path override.

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
                    data_tbl = convert(data, pa.Table)
                    existing_schema = data_tbl.schema
                    logger.warning(
                        "%s, creating it from input schema (columns=%s)",
                        exc,
                        existing_schema.names,
                    )

                    connected.create_table(
                        field=existing_schema,
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        if_not_exists=True,
                    )

                    try:
                        return connected.arrow_insert_into(
                            data=data_tbl,
                            location=location,
                            catalog_name=catalog_name,
                            schema_name=schema_name,
                            table_name=table_name,
                            mode="overwrite",
                            cast_options=cast_options,
                            overwrite_schema=overwrite_schema,
                            match_by=match_by,
                            zorder_by=zorder_by,
                            optimize_after_merge=optimize_after_merge,
                            vacuum_hours=vacuum_hours,
                            existing_schema=existing_schema,
                        )
                    except Exception:
                        logger.exception("Arrow insert failed after auto-creating %s; attempting cleanup (DROP TABLE)", location)
                        try:
                            connected.drop_table(location=location, wait=True)
                        except Exception:
                            logger.exception("Failed to drop table %s after auto creation error", location)
                        raise

            data_tbl = convert(
                data, pa.Table,
                options=cast_options, target_field=existing_schema
            )
            num_rows = data_tbl.num_rows

            logger.debug(
                "Arrow inserting %s rows into %s (mode=%s, match_by=%s, zorder_by=%s)",
                num_rows,
                location,
                mode,
                match_by,
                zorder_by,
            )

            # Write in temp volume
            temp_volume_path = self.workspace.tmp_path(
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_name="tmp",
                extension="parquet"
            ) if temp_volume_path is None else DatabricksPath.parse(obj=temp_volume_path, workspace=connected.workspace)

            logger.debug("Staging Parquet to temp volume: %s", temp_volume_path)
            temp_volume_path.write_arrow_table(data_tbl, file_format=pds.ParquetFileFormat())

            columns = list(existing_schema.names)
            cols_quoted = ", ".join([f"`{c}`" for c in columns])

            statements: list[str] = []

            if match_by:
                on_condition = " AND ".join([f"T.`{k}` = S.`{k}`" for k in match_by])

                update_cols = [c for c in columns if c not in match_by]
                if update_cols:
                    update_set = ", ".join([f"T.`{c}` = S.`{c}`" for c in update_cols])
                    update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"
                else:
                    update_clause = ""

                insert_clause = (
                    f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) "
                    f"VALUES ({', '.join([f'S.`{c}`' for c in columns])})"
                )

                merge_sql = f"""MERGE INTO {location} AS T
USING (
  SELECT {cols_quoted} FROM parquet.`{temp_volume_path}`
) AS S
ON {on_condition}
{update_clause}
{insert_clause}"""
                statements.append(merge_sql)
            else:
                if mode.lower() in ("overwrite",):
                    insert_sql = f"""INSERT OVERWRITE {location}
SELECT {cols_quoted}
FROM parquet.`{temp_volume_path}`"""
                else:
                    insert_sql = f"""INSERT INTO {location} ({cols_quoted})
SELECT {cols_quoted}
FROM parquet.`{temp_volume_path}`"""
                statements.append(insert_sql)

            try:
                for stmt in statements:
                    connected.execute(stmt.strip())
            finally:
                try:
                    Thread(
                        target=temp_volume_path.rmdir,
                        kwargs={
                            "recursive": True
                        }
                    ).start()
                except Exception:
                    logger.exception("Failed cleaning temp volume: %s", temp_volume_path)

            logger.info(
                "Arrow inserted %s rows into %s (mode=%s, match_by=%s, zorder_by=%s)",
                num_rows,
                location,
                mode,
                match_by,
                zorder_by,
            )

            if zorder_by:
                zcols = ", ".join([f"`{c}`" for c in zorder_by])
                optimize_sql = f"OPTIMIZE {location} ZORDER BY ({zcols})"
                logger.info("Running OPTIMIZE ZORDER BY: %s", zorder_by)
                connected.execute(optimize_sql)

            if optimize_after_merge and match_by:
                logger.info("Running OPTIMIZE after MERGE")
                connected.execute(f"OPTIMIZE {location}")

            if vacuum_hours is not None:
                logger.info("Running VACUUM retain=%s hours", vacuum_hours)
                connected.execute(f"VACUUM {location} RETAIN {vacuum_hours} HOURS")

        return None

    def spark_insert_into(
        self,
        data: SparkDataFrame,
        *,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: Optional[list[str]] = None,
        zorder_by: Optional[list[str]] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,
        spark_options: Optional[Dict[str, Any]] = None,
    ):
        """Insert a Spark DataFrame into a Delta table (append/overwrite/merge).

        Behavior:
        - If the table does not exist: creates it via `saveAsTable(location)` (overwrite).
        - If `match_by` is provided: uses Delta MERGE for upserts.
          - If mode == "overwrite": deletes matching keys first, then appends the batch (fast-ish overwrite-by-key).
          - Else: updates matching rows + inserts new ones.
        - Else: uses `DataFrameWriter.saveAsTable` with mode.

        Args:
            data: Spark DataFrame to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite").
            cast_options: Optional casting options (align to destination schema).
            overwrite_schema: Whether to overwrite schema on write (when supported).
            match_by: Merge keys for upserts.
            zorder_by: Z-ORDER columns (used only if `optimize_after_merge` is True).
            optimize_after_merge: Whether to run Delta optimize (and z-order) after merge.
            vacuum_hours: Optional VACUUM retention window in hours.
            spark_options: Optional Spark write options.

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
            data = convert(data, pyspark.sql.DataFrame)
            data.write.mode("overwrite").options(**spark_options).saveAsTable(location)
            return

        if not isinstance(data, pyspark.sql.DataFrame):
            data = convert(data, pyspark.sql.DataFrame, target_field=existing_schema)
        else:
            cast_options = CastOptions.check_arg(options=cast_options, target_field=existing_schema)
            data = cast_spark_dataframe(data, options=cast_options)

        if match_by:
            notnull = None
            for k in match_by:
                if k not in data.columns:
                    raise ValueError(f"Missing match key '{k}' in DataFrame columns: {data.columns}")
                notnull = data[k].isNotNull() if notnull is None else notnull & data[k].isNotNull()

            data = data.filter(notnull)
            logger.debug("Filtered null keys for match_by=%s", match_by)

        target = self.spark_table(full_name=location)

        if match_by:
            cond = " AND ".join([f"t.`{k}` <=> s.`{k}`" for k in match_by])

            if mode.casefold() == "overwrite":
                data = data.cache()
                distinct_keys = data.select([f"`{k}`" for k in match_by]).distinct()

                (
                    target.alias("t")
                    .merge(distinct_keys.alias("s"), cond)
                    .whenMatchedDelete()
                    .execute()
                )

                data.write.format("delta").mode("append").options(**spark_options).saveAsTable(location)
            else:
                update_cols = [c for c in data.columns if c not in match_by]
                set_expr = {c: F.expr(f"s.`{c}`") for c in update_cols}

                (
                    target.alias("t")
                    .merge(data.alias("s"), cond)
                    .whenMatchedUpdate(set=set_expr)
                    .whenNotMatchedInsertAll()
                    .execute()
                )
        else:
            if mode == "auto":
                mode = "append"
            logger.info("Spark write saveAsTable mode=%s", mode)
            data.write.mode(mode).options(**spark_options).saveAsTable(location)

        if optimize_after_merge and zorder_by:
            logger.info("Delta optimize + zorder (%s)", zorder_by)
            target.optimize().executeZOrderBy(*zorder_by)

        if vacuum_hours is not None:
            logger.info("Delta vacuum retain=%s hours", vacuum_hours)
            target.vacuum(vacuum_hours)

    def get_table_schema(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        to_arrow_schema: bool = True,
    ) -> Union[pa.Field, pa.Schema]:
        """Fetch a table schema from Unity Catalog and convert it to Arrow types.

        Args:
            location: Optional Fully qualified location name
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            to_arrow_schema: If True returns pa.Schema; else returns a pa.Field(STRUCT<...>).

        Returns:
            Arrow Schema or a STRUCT Field representing the table.
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
            raise ValueError(f"Table %s not found, {type(e)} {e}" % location)

        fields = [
            column_info_to_arrow_field(_) for _ in table.columns
        ]

        metadata = {
            b"engine": b"databricks",
            b"full_name": location,
            b"catalog_name": catalog_name,
            b"schema_name": schema_name,
            b"table_name": table_name,
        }

        if to_arrow_schema:
            return pa.schema(
                fields,
                metadata=metadata
            )

        return pa.field(
            location,
            pa.struct(fields),
            metadata=metadata
        )

    def drop_table(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        wait: Optional[WaitingConfigArg] = True
    ):
        """Drop a table if it exists."""
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
        field: Union[pa.Field, pa.Schema],
        full_name: Optional[str] = None,            # e.g. catalog.schema.table
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        storage_location: Optional[str] = None,     # external location path
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[bool | list[str]] = True,
        comment: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        if_not_exists: bool = True,
        or_replace: bool = False,
        using: str = "DELTA",
        optimize_write: bool = True,
        auto_compact: bool = True,
        # perf-ish optional knobs (don’t hard-force)
        enable_cdf: Optional[bool] = None,
        enable_deletion_vectors: Optional[bool] = None,
        target_file_size: Optional[int] = None,     # bytes
        # column mapping: None=auto, "none"/"name"/"id" explicit
        column_mapping_mode: Optional[str] = None,
        execute: bool = True,
        wait_result: bool = True,
        return_plan: bool = False,
    ) -> Union[str, CreateTablePlan, "StatementResult"]:
        """
        Generate (and optionally execute) a Databricks/Delta `CREATE TABLE` statement from an Apache Arrow
        schema/field, with integration-friendly safety checks and performance-oriented defaults.

        This helper is meant to be "team-safe":
          - Quotes identifiers (catalog/schema/table/columns) to avoid SQL keyword/name edge cases.
          - Validates `partition_by` / `cluster_by` columns exist in the Arrow schema before generating SQL.
          - Supports managed or external tables via `storage_location`.
          - Optionally enables Delta Column Mapping (name/id) and applies the required protocol upgrades.

        Parameters
        ----------
        field:
            Arrow schema or field describing the table.
            - If `pa.Schema`, all schema fields are used as columns.
            - If `pa.Field` with struct type, its children become columns.
            - If `pa.Field` non-struct, it becomes a single-column table.
        full_name:
            Fully-qualified table name, e.g. `"catalog.schema.table"`.
            If provided, it takes precedence over `catalog_name`/`schema_name`/`table_name`.
            Parts are quoted as needed.
        catalog_name, schema_name, table_name:
            Used to build the table identifier when `table_fqn` is not provided.
            All three must be provided together.
        storage_location:
            If set, emits `LOCATION '<path>'` to create an external Delta table at the given path.
            (Path string is SQL-escaped.)
        partition_by:
            List of partition column names. Must exist in the schema.
            Note: Partitioning is a physical layout choice; only use for low-cardinality columns.
        cluster_by:
            Controls clustering / liquid clustering:
            - True  -> emits `CLUSTER BY AUTO`
            - False -> emits no clustering clause
            - list[str] -> emits `CLUSTER BY (<cols...>)` (all cols must exist in schema)
        comment:
            Optional table comment. If not provided and Arrow metadata contains `b"comment"`, that is used.
        properties:
            Additional/override Delta table properties (final say).
            Example: `{"delta.enableChangeDataFeed": "true"}` or `{"delta.logRetentionDuration": "30 days"}`
        if_not_exists:
            If True, generates `CREATE TABLE IF NOT EXISTS ...`.
            Mutually exclusive with `or_replace`.
        or_replace:
            If True, generates `CREATE OR REPLACE TABLE ...`.
            Mutually exclusive with `if_not_exists`.
        using:
            Storage format keyword. Defaults to `"DELTA"`.
        optimize_write:
            Sets `delta.autoOptimize.optimizeWrite` table property.
        auto_compact:
            Sets `delta.autoOptimize.autoCompact` table property.
        enable_cdf:
            If set, adds `delta.enableChangeDataFeed` property.
            Useful for CDC pipelines; avoid enabling by default if you don't need it.
        enable_deletion_vectors:
            If set, adds `delta.enableDeletionVectors` property.
            Can improve performance for updates/deletes in some workloads (subject to platform support).
        target_file_size:
            If set, adds `delta.targetFileSize` (bytes). Helps guide file sizing and reduce small files.
        column_mapping_mode:
            Delta column mapping mode:
            - None  -> auto-detect: enables `"name"` only if invalid column names are present, else `"none"`
            - "none" -> do not enable column mapping (max compatibility)
            - "name" -> enable name-based column mapping
            - "id"   -> enable id-based column mapping

            When enabled (name/id), this method also sets the required protocol properties:
            `delta.minReaderVersion=2` and `delta.minWriterVersion=5`.
        execute:
            If True, executes the generated SQL via `self.execute(...)`.
            If False, returns the SQL (or plan) without executing.
        wait_result:
            Passed to `self.execute(...)`. If True, blocks until the statement finishes.
        return_plan:
            If True, returns a `CreateTablePlan` containing SQL + applied properties + warnings (+ result if executed).
            If False:
              - returns SQL string when `execute=False`
              - returns `StatementResult` when `execute=True`

        Returns
        -------
        Union[str, CreateTablePlan, StatementResult]
            - If `execute=False` and `return_plan=False`: the SQL string.
            - If `execute=False` and `return_plan=True`: `CreateTablePlan(sql=..., properties=..., warnings=...)`.
            - If `execute=True` and `return_plan=False`: `StatementResult`.
            - If `execute=True` and `return_plan=True`: `CreateTablePlan` with `result` populated.

        Raises
        ------
        ValueError
            If required naming params are missing, if `or_replace` and `if_not_exists` conflict,
            if `column_mapping_mode` is invalid, or if partition/cluster columns are not present.

        Notes
        -----
        - Column mapping is primarily a metadata feature; performance impact is usually negligible vs IO,
          but enabling it affects compatibility with older readers.
        - Partitioning and clustering are workload-dependent: partition for selective pruning on low-cardinality
          columns; cluster for speeding up common filter/join patterns.

        Examples
        --------
        Create a managed Delta table with auto clustering and auto column mapping:
            >>> plan = client.create_table(schema, full_name="main.analytics.events", execute=False, return_plan=True)
            >>> print(plan.sql)

        External table with explicit partitioning and CDF:
            >>> client.create_table(
            ...     schema,
            ...     full_name="main.analytics.events",
            ...     storage_location="abfss://.../events",
            ...     partition_by=["event_date"],
            ...     enable_cdf=True,
            ... )
        """

        if not isinstance(field, (pa.Field, pa.Schema)):
            field = convert(field, pa.Field)

        # ---- Normalize Arrow input ----
        if isinstance(field, pa.Schema):
            arrow_fields = list(field)
            schema_metadata = field.metadata or {}
        else:
            # pa.Field
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
            safe_chars=True
        )

        # ---- Comments ----
        if comment is None and schema_metadata:
            c = schema_metadata.get(b"comment")
            if isinstance(c, bytes):
                comment = c.decode("utf-8")

        # ---- Detect invalid column names -> column mapping auto ----
        any_invalid = any(_needs_column_mapping(f.name) for f in arrow_fields)
        warnings: list[str] = []
        if column_mapping_mode is None:
            column_mapping_mode = "name" if any_invalid else "none"

        if column_mapping_mode not in ("none", "name", "id"):
            raise ValueError("column_mapping_mode must be one of: None, 'none', 'name', 'id'.")

        # ---- Validate partition/cluster columns exist ----
        col_names = {f.name for f in arrow_fields}
        for cols, label in ((partition_by, "partition_by"),):
            if cols:
                missing = [c for c in cols if c not in col_names]
                if missing:
                    raise ValueError(f"{label} contains unknown columns: {missing}")

        if isinstance(cluster_by, list):
            missing = [c for c in cluster_by if c not in col_names]
            if missing:
                raise ValueError(f"cluster_by contains unknown columns: {missing}")

        # ---- Column DDL ----
        # IMPORTANT: your _field_to_ddl should quote names with backticks if needed.
        # I’d recommend it ALWAYS quotes via _quote_ident internally.
        column_definitions = [self._field_to_ddl(child) for child in arrow_fields]

        # ---- Build CREATE TABLE ----
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
                sql_parts.append("CLUSTER BY AUTO")
            else:
                sql_parts.append("CLUSTER BY (" + ", ".join(_quote_ident(c) for c in cluster_by) + ")")

        if comment:
            sql_parts.append(f"COMMENT '{_escape_sql_string(comment)}'")

        if storage_location:
            sql_parts.append(f"LOCATION '{_escape_sql_string(storage_location)}'")

        # ---- Table properties (defaults + overrides) ----
        props: dict[str, Any] = {
            "delta.autoOptimize.optimizeWrite": bool(optimize_write),
            "delta.autoOptimize.autoCompact": bool(auto_compact)
        }

        if enable_cdf is not None:
            props["delta.enableChangeDataFeed"] = bool(enable_cdf)

        if enable_deletion_vectors is not None:
            props["delta.enableDeletionVectors"] = bool(enable_deletion_vectors)

        if target_file_size is not None:
            props["delta.targetFileSize"] = int(target_file_size)

        # Column mapping + required protocol bumps
        if column_mapping_mode != "none":
            props["delta.columnMapping.mode"] = column_mapping_mode
            props["delta.minReaderVersion"] = 2
            props["delta.minWriterVersion"] = 5
        else:
            # only set explicitly if user wants; otherwise leave unset for max compatibility
            pass

        # Let caller override anything (final say)
        if properties:
            props.update(properties)

        if any_invalid and column_mapping_mode == "none":
            warnings.append(
                "Schema has invalid column names but column_mapping_mode='none'. "
                "This will fail unless you rename/escape columns."
            )

        default_tags = self.workspace.default_tags()

        for k, v in default_tags.items():
            props[f"tags.{k}"] = v

        if props:
            def fmt(k: str, v: Any) -> str:
                if isinstance(v, str):
                    return f"'{k}' = '{_escape_sql_string(v)}'"
                if isinstance(v, bool):
                    return f"'{k}' = '{'true' if v else 'false'}'"
                return f"'{k}' = {v}"

            sql_parts.append("TBLPROPERTIES (" + ", ".join(fmt(k, v) for k, v in props.items()) + ")")

        statement = "\n".join(sql_parts)

        plan = CreateTablePlan(sql=statement, properties=props, warnings=warnings)

        if not execute:
            return plan if return_plan else statement

        res = self.execute(statement, wait=wait_result)
        plan.result = res
        return plan if return_plan else res

    def _check_location_params(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True,
    ) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
        """Resolve (location OR catalog/schema/table) into a fully-qualified name."""
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
        """Convert an Arrow Field to a Databricks SQL column DDL fragment."""
        name = field.name
        nullable_str = " NOT NULL" if put_not_null and not field.nullable else ""
        name_str = f"`{name}` " if put_name else ""

        comment_str = ""
        if put_comment and field.metadata and b"comment" in field.metadata:
            comment = field.metadata[b"comment"].decode("utf-8")
            comment_str = f" COMMENT '{comment}'"

        if not pa.types.is_nested(field.type):
            sql_type = SQLEngine._arrow_to_sql_type(field.type)
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        if pa.types.is_struct(field.type):
            child_defs = [SQLEngine._field_to_ddl(child) for child in field.type]
            struct_body = ", ".join(child_defs)
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        if pa.types.is_map(field.type):
            map_type: pa.MapType = field.type
            key_type = SQLEngine._field_to_ddl(map_type.key_field, put_name=False, put_comment=False, put_not_null=False)
            val_type = SQLEngine._field_to_ddl(map_type.item_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            list_type: pa.ListType = field.type
            elem_type = SQLEngine._field_to_ddl(list_type.value_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        raise TypeError(f"Cannot make ddl field from {field}")

    @staticmethod
    def _arrow_to_sql_type(arrow_type: Union[pa.DataType, pa.Decimal128Type]) -> str:
        """Convert an Arrow data type to a Databricks SQL type string."""
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
            return "STRING"
        raise ValueError(f"Cannot make ddl type for {arrow_type}")
