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
from typing import Optional, Union, Any, Dict, List, Literal

import pyarrow as pa

from .statement_result import StatementResult
from .types import column_info_to_arrow_field
from .. import DatabricksPathKind, DatabricksPath
from ..workspaces import WorkspaceService
from ...libs.databrickslib import databricks_sdk
from ...libs.sparklib import SparkSession, SparkDataFrame, pyspark
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
        StatementResponse, Disposition, Format,
        ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem
    )
    StatementResponse = StatementResponse
else:
    class StatementResponse:  # pragma: no cover
        pass


logger = logging.getLogger(__name__)

if pyspark is not None:
    import pyspark.sql.functions as F

__all__ = ["SQLEngine", "StatementResult"]


class SqlExecutionError(RuntimeError):
    """Raised when a SQL statement execution fails."""


@dataclasses.dataclass
class SQLEngine(WorkspaceService):
    """Execute SQL statements and manage tables via Databricks SQL / Spark."""
    warehouse_id: Optional[str] = None
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

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
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name
        return catalog_name, schema_name, table_name

    def _default_warehouse(self, cluster_size: str = "Small"):
        """Pick a default SQL warehouse (best-effort) matching the desired size.

        Args:
            cluster_size: Desired warehouse size (Databricks "cluster_size"), e.g. "Small".
                If empty/None, returns the first warehouse encountered.

        Returns:
            Warehouse object.

        Raises:
            ValueError: If no warehouses exist in the workspace.
        """
        wk = self.workspace.sdk()
        existing = list(wk.warehouses.list())
        first = None

        for warehouse in existing:
            if first is None:
                first = warehouse

            if cluster_size:
                if getattr(warehouse, "cluster_size", None) == cluster_size:
                    logger.debug("Default warehouse match found: id=%s cluster_size=%s", warehouse.id, warehouse.cluster_size)
                    return warehouse
            else:
                logger.debug("Default warehouse selected (first): id=%s", warehouse.id)
                return warehouse

        if first is not None:
            logger.info(
                "No warehouse matched cluster_size=%s; falling back to first warehouse id=%s cluster_size=%s",
                cluster_size,
                getattr(first, "id", None),
                getattr(first, "cluster_size", None),
            )
            return first

        raise ValueError(f"No default warehouse found in {wk.config.host}")

    def _get_or_default_warehouse_id(self, cluster_size: str = "Small") -> str:
        """Return configured warehouse_id or resolve a default one.

        Args:
            cluster_size: Desired warehouse size filter used when resolving defaults.

        Returns:
            Warehouse id string.
        """
        if not self.warehouse_id:
            dft = self._default_warehouse(cluster_size=cluster_size)
            self.warehouse_id = dft.id
            logger.info("Resolved default warehouse_id=%s (cluster_size=%s)", self.warehouse_id, cluster_size)

        return self.warehouse_id

    @staticmethod
    def _random_suffix(prefix: str = "") -> str:
        """Generate a unique suffix for temporary resources."""
        unique = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        timestamp = int(time.time() * 1000)
        return f"{prefix}{timestamp}_{unique}"

    @staticmethod
    def _sql_preview(sql: str, limit: int = 220) -> str:
        """Short, single-line preview for logs (avoids spewing giant SQL)."""
        if not sql:
            return ""
        return sql[:limit] + ("…" if len(sql) > limit else "")

    def execute(
        self,
        statement: Optional[str] = None,
        *,
        engine: Optional[Literal["spark", "api"]] = None,
        warehouse_id: Optional[str] = None,
        byte_limit: Optional[int] = None,
        disposition: Optional["Disposition"] = None,
        format: Optional["Format"] = None,
        on_wait_timeout: Optional["ExecuteStatementRequestOnWaitTimeout"] = None,
        parameters: Optional[List["StatementParameterListItem"]] = None,
        row_limit: Optional[int] = None,
        wait_timeout: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        wait_result: bool = True,
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
            byte_limit: Optional byte limit for results.
            disposition: Result disposition mode (API engine).
            format: Result format (API engine).
            on_wait_timeout: Timeout behavior for waiting (API engine).
            parameters: Optional statement parameters (API engine).
            row_limit: Optional row limit for results (API engine).
            wait_timeout: API wait timeout value.
            catalog_name: Optional catalog override for API engine.
            schema_name: Optional schema override for API engine.
            table_name: Optional table override used when `statement` is None.
            wait_result: Whether to block until completion (API engine).

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

            df: SparkDataFrame = spark_session.sql(statement)

            if row_limit:
                df = df.limit(row_limit)

            logger.info("Spark SQL executed: %s", self._sql_preview(statement))

            # Avoid Disposition dependency if SDK imports are absent
            spark_disp = disposition if disposition is not None else getattr(globals().get("Disposition", object), "EXTERNAL_LINKS", None)

            return StatementResult(
                engine=self,
                statement_id="sparksql",
                disposition=spark_disp,
                _spark_df=df,
            )

        # --- API path defaults ---
        if format is None:
            format = Format.ARROW_STREAM

        if (disposition is None or disposition == Disposition.INLINE) and format in [Format.CSV, Format.ARROW_STREAM]:
            disposition = Disposition.EXTERNAL_LINKS

        if not statement:
            full_name = self.table_full_name(catalog_name=catalog_name, schema_name=schema_name, table_name=table_name)
            statement = f"SELECT * FROM {full_name}"

        if not warehouse_id:
            warehouse_id = self._get_or_default_warehouse_id()

        response = self.workspace.sdk().statement_execution.execute_statement(
            statement=statement,
            warehouse_id=warehouse_id,
            byte_limit=byte_limit,
            disposition=disposition,
            format=format,
            on_wait_timeout=on_wait_timeout,
            parameters=parameters,
            row_limit=row_limit,
            wait_timeout=wait_timeout,
            catalog=catalog_name or self.catalog_name,
            schema=schema_name or self.schema_name,
        )

        execution = StatementResult(
            engine=self,
            statement_id=response.statement_id,
            _response=response,
            _response_refresh_time=time.time(),
            disposition=disposition,
        )

        logger.info(
            "API SQL executed: %s",
            self._sql_preview(statement)
        )

        return execution.wait() if wait_result else execution

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

        if pyspark is not None:
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
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        to_arrow_schema=True,
                    )
                except ValueError as exc:
                    data_tbl = convert(data, pa.Table)
                    existing_schema = data_tbl.schema
                    logger.warning(
                        "Table %s not found (%s). Creating it from input schema (columns=%s)",
                        location,
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
                            connected.drop_table(location=location)
                        except Exception:
                            logger.exception("Failed to drop table %s after auto creation error", location)
                        raise

            transaction_id = self._random_suffix()

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
            temp_volume_path = connected.dbfs_path(
                kind=DatabricksPathKind.VOLUME,
                parts=[catalog_name, schema_name, "tmp", "sql", transaction_id],
            ) if temp_volume_path is None else DatabricksPath.parse(obj=temp_volume_path, workspace=connected.workspace)

            logger.debug("Staging Parquet to temp volume: %s", temp_volume_path)
            temp_volume_path.mkdir()
            temp_volume_path.write_arrow_table(data_tbl)

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
                    temp_volume_path.rmdir(recursive=True)
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

        logger.debug("Incoming Spark columns: %s", data.columns)

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
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        to_arrow_schema: bool = True,
    ) -> Union[pa.Field, pa.Schema]:
        """Fetch a table schema from Unity Catalog and convert it to Arrow types.

        Args:
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            to_arrow_schema: If True returns pa.Schema; else returns a pa.Field(STRUCT<...>).

        Returns:
            Arrow Schema or a STRUCT Field representing the table.
        """
        full_name = self.table_full_name(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=False,
        )

        wk = self.workspace.sdk()

        try:
            table = wk.tables.get(full_name)
        except Exception as e:
            raise ValueError(f"Table %s not found, {type(e)} {e}" % full_name)

        fields = [column_info_to_arrow_field(_) for _ in table.columns]

        if to_arrow_schema:
            return pa.schema(fields, metadata={b"name": table_name})
        return pa.field(table.name, pa.struct(fields))

    def drop_table(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        """Drop a table if it exists."""
        location, _, _, _ = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )
        logger.info("Dropping table if exists: %s", location)
        return self.execute(f"DROP TABLE IF EXISTS {location}")

    def create_table(
        self,
        field: pa.Field,
        location: Optional[str] = None,
        table_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        partition_by: Optional[list[str]] = None,
        cluster_by: Optional[bool | list[str]] = True,
        comment: Optional[str] = None,
        options: Optional[dict] = None,
        if_not_exists: bool = True,
        optimize_write: bool = True,
        auto_compact: bool = True,
        execute: bool = True,
        wait_result: bool = True
    ) -> Union[str, "StatementResult"]:
        """Generate (and optionally execute) CREATE TABLE DDL from an Arrow schema/field.

        Args:
            field: Arrow Field or Schema describing the table. If `field` is a schema, it's converted.
            location: Fully qualified table name override.
            table_name: Table name override (used if location not provided).
            catalog_name: Catalog override.
            schema_name: Schema override.
            partition_by: Optional partition columns.
            cluster_by: If True -> CLUSTER BY AUTO. If list[str] -> CLUSTER BY (..). If False -> no clustering.
            comment: Optional table comment (falls back to field metadata b"comment" when present).
            options: Extra table properties.
            if_not_exists: Add IF NOT EXISTS clause.
            optimize_write: Sets delta.autoOptimize.optimizeWrite table property.
            auto_compact: Sets delta.autoOptimize.autoCompact table property.
            execute: If True, executes DDL and returns StatementResult; otherwise returns SQL string.
            wait_result: Waits execution to complete

        Returns:
            StatementResult if execute=True, else the DDL SQL string.
        """
        if not isinstance(field, pa.Field):
            field = convert(field, pa.Field)

        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True,
        )

        if pa.types.is_struct(field.type):
            children = list(field.type)
        else:
            children = [field]

        column_definitions = [self._field_to_ddl(child) for child in children]

        sql = [
            f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{location} (",
            ",\n  ".join(column_definitions),
            ")",
        ]

        if partition_by:
            sql.append(f"\nPARTITIONED BY ({', '.join(partition_by)})")
        elif cluster_by:
            if isinstance(cluster_by, bool):
                sql.append("\nCLUSTER BY AUTO")
            else:
                sql.append(f"\nCLUSTER BY ({', '.join(cluster_by)})")

        if not comment and field.metadata:
            comment = field.metadata.get(b"comment")

        if isinstance(comment, bytes):
            comment = comment.decode("utf-8")

        if comment:
            sql.append(f"\nCOMMENT '{comment}'")

        options = {} if options is None else options
        options.update({
            "delta.autoOptimize.optimizeWrite": optimize_write,
            "delta.autoOptimize.autoCompact": auto_compact,
        })

        option_strs = []
        for key, value in (options or {}).items():
            if isinstance(value, str):
                option_strs.append(f"'{key}' = '{value}'")
            elif isinstance(value, bool):
                option_strs.append(f"'{key}' = '{'true' if value else 'false'}'")
            else:
                option_strs.append(f"'{key}' = {value}")

        if option_strs:
            sql.append(f"\nTBLPROPERTIES ({', '.join(option_strs)})")

        statement = "\n".join(sql)

        logger.debug(
            "Generated CREATE TABLE DDL for %s:\n%s",
            location, statement
        )

        if execute:
            return self.execute(statement, wait_result=wait_result)
        return statement

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
        name_str = f"{name} " if put_name else ""

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
