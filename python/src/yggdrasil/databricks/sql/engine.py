"""Databricks SQL engine utilities and helpers."""

import dataclasses
import logging
import random
import string
import time
from typing import Optional, Union, Any, Dict, List, Literal

import pyarrow as pa
import pyarrow.parquet as pq

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
    class StatementResponse:
        pass


logger = logging.getLogger(__name__)


if pyspark is not None:
    import pyspark.sql.functions as F

__all__ = [
    "SQLEngine",
    "StatementResult"
]


class SqlExecutionError(RuntimeError):
    """Raised when a SQL statement execution fails."""


@dataclasses.dataclass
class SQLEngine(WorkspaceService):
    """Execute SQL statements and manage tables via Databricks."""
    warehouse_id: Optional[str] = None
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    def table_full_name(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True
    ):
        """Build a fully qualified table name for the current catalog/schema.

        Args:
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Table name to qualify.
            safe_chars: Whether to wrap identifiers in backticks.

        Returns:
            The fully qualified table name.
        """
        catalog_name = catalog_name or self.catalog_name
        schema_name = schema_name or self.schema_name

        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert table_name, "No table name given"

        if safe_chars:
            return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
        return f"{catalog_name}.{schema_name}.{table_name}"

    def _catalog_schema_table_names(
        self,
        full_name: str,
    ):
        """Parse a catalog.schema.table string into components.

        Args:
            full_name: A fully qualified name or partial name.

        Returns:
            A tuple of (catalog_name, schema_name, table_name).
        """
        parts = [
            _.strip("`") for _ in full_name.split(".")
        ]

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

    def _default_warehouse(
        self,
        cluster_size: str = "Small"
    ):
        """Return a default SQL warehouse matching the desired size.

        Args:
            cluster_size: Desired warehouse size filter.

        Returns:
            The matched warehouse object.
        """
        wk = self.workspace.sdk()
        existing = list(wk.warehouses.list())
        first = None

        for warehouse in existing:
            if first is None:
                first = warehouse

            if cluster_size:
                if warehouse.cluster_size == cluster_size:
                    return warehouse
            else:
                return warehouse

        if first is not None:
            return first

        raise ValueError(f"No default warehouse found in {wk.config.host}")

    def _get_or_default_warehouse_id(
        self,
        cluster_size = "Small"
    ):
        """Return the configured warehouse id or a default one.

        Args:
            cluster_size: Desired warehouse size filter.

        Returns:
            The warehouse id string.
        """
        if not self.warehouse_id:
            dft = self._default_warehouse(cluster_size=cluster_size)

            self.warehouse_id = dft.id
        return self.warehouse_id

    @staticmethod
    def _random_suffix(prefix: str = "") -> str:
        """Generate a unique suffix for temporary resources.

        Args:
            prefix: Optional prefix to prepend.

        Returns:
            A unique suffix string.
        """
        unique = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        timestamp = int(time.time() * 1000)
        return f"{prefix}{timestamp}_{unique}"

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
        **kwargs,
    ) -> "StatementResult":
        """
        Execute a SQL statement on a SQL warehouse.

        - If wait=True (default): poll until terminal state.
            - On SUCCEEDED: return final statement object
            - On FAILED / CANCELED: raise SqlExecutionError
        - If wait=False: return initial execution handle without polling.

        Args:
            statement: SQL statement to execute. If omitted, selects from the table.
            engine: Execution engine ("spark" or "api").
            warehouse_id: Optional warehouse id override.
            byte_limit: Optional byte limit for results.
            disposition: Result disposition mode.
            format: Result format for Databricks SQL API.
            on_wait_timeout: Timeout behavior for waiting.
            parameters: Optional statement parameters.
            row_limit: Optional row limit.
            wait_timeout: Optional API wait timeout.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            wait_result: Whether to block until completion.
            **kwargs: Additional API parameters.

        Returns:
            A StatementResult wrapper for the execution.
        """
        if not engine:
            if pyspark is not None:
                spark_session = SparkSession.getActiveSession()

                if spark_session is not None:
                    engine = "spark"

        if engine == "spark":
            spark_session = SparkSession.getActiveSession()

            if spark_session is None:
                raise ValueError("No spark session found to run sql query")

            return StatementResult(
                engine=self,
                statement_id="sparksql",
                disposition=Disposition.EXTERNAL_LINKS,
                _spark_df=spark_session.sql(statement)
            )

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
            **kwargs,
        )

        execution = StatementResult(
            engine=self,
            statement_id=response.statement_id,
            _response=response,
            _response_refresh_time=time.time(),
            disposition=disposition
        )

        return execution.wait() if wait_result else wait_result

    def spark_table(
        self,
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        """Return a DeltaTable handle for a given table name.

        Args:
            full_name: Fully qualified table name.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.

        Returns:
            A Spark DeltaTable handle.
        """
        if not full_name:
            full_name = self.table_full_name(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name
            )

        return SparkDeltaTable.forName(
            sparkSession=SparkSession.getActiveSession(),
            tableOrViewName=full_name
        )

    def insert_into(
        self,
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
            SparkDataFrame
        ],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: list[str] = None,
        zorder_by: list[str] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        spark_session: Optional[SparkSession] = None,
        spark_options: Optional[Dict[str, Any]] = None
    ):
        """Insert data into a table using Spark or Arrow paths.

        Args:
            data: Arrow or Spark data to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite").
            cast_options: Optional casting options.
            overwrite_schema: Whether to overwrite schema (Spark).
            match_by: Optional merge keys for upserts.
            zorder_by: Optional Z-ORDER columns.
            optimize_after_merge: Whether to run OPTIMIZE after merge.
            vacuum_hours: Optional VACUUM retention window.
            spark_session: Optional SparkSession override.
            spark_options: Optional Spark write options.

        Returns:
            None for Arrow inserts, or the Spark insert result.
        """
        # -------- existing logic you provided (kept intact) ----------
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
                    spark_options=spark_options
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
        data: Union[
            pa.Table, pa.RecordBatch, pa.RecordBatchReader,
        ],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        cast_options: Optional[CastOptions] = None,
        overwrite_schema: bool | None = None,
        match_by: list[str] = None,
        zorder_by: list[str] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        existing_schema: pa.Schema | None = None,
        temp_volume_path: Optional[Union[str, DatabricksPath]] = None
    ):
        """Insert Arrow data by staging to a temp volume and running SQL.

        Args:
            data: Arrow table/batch data to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite").
            cast_options: Optional casting options.
            overwrite_schema: Whether to overwrite schema.
            match_by: Optional merge keys for upserts.
            zorder_by: Optional Z-ORDER columns.
            optimize_after_merge: Whether to run OPTIMIZE after merge.
            vacuum_hours: Optional VACUUM retention window.
            existing_schema: Optional pre-fetched schema.
            temp_volume_path: Optional temp volume path override.

        Returns:
            None.
        """
        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True
        )

        with self as connected:
            if existing_schema is None:
                try:
                    existing_schema = connected.get_table_schema(
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        to_arrow_schema=True
                    )
                except ValueError as exc:
                    data = convert(data, pa.Table)
                    existing_schema = data.schema
                    logger.warning(
                        "Table %s not found, %s, creating it based on input data %s",
                        location,
                        exc,
                        existing_schema.names
                    )

                    connected.create_table(
                        field=existing_schema,
                        catalog_name=catalog_name,
                        schema_name=schema_name,
                        table_name=table_name,
                        if_not_exists=True
                    )

                    try:
                        return connected.arrow_insert_into(
                            data=data,
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
                            existing_schema=existing_schema
                        )
                    except:
                        try:
                            connected.drop_table(location=location)
                        except Exception as e:
                            logger.warning("Failed to drop table %s after auto creation on error: %s", location, e)
                        raise

            transaction_id = self._random_suffix()

            data = convert(data, pa.Table, options=cast_options, target_field=existing_schema)

            # Write in temp volume
            temp_volume_path = connected.dbfs_path(
                kind=DatabricksPathKind.VOLUME,
                parts=[catalog_name, schema_name, "tmp", "sql", transaction_id]
            ) if temp_volume_path is None else DatabricksPath.parse(obj=temp_volume_path, workspace=connected.workspace)

            temp_volume_path.mkdir()

            temp_volume_path.write_arrow_table(data)

            # get column list from arrow schema
            columns = [c for c in existing_schema.names]
            cols_quoted = ", ".join([f"`{c}`" for c in columns])

            statements = []

            # Decide how to ingest
            # If merge keys provided -> use MERGE
            if match_by:
                # build ON condition using match_by
                on_clauses = []
                for k in match_by:
                    on_clauses.append(f"T.`{k}` = S.`{k}`")
                on_condition = " AND ".join(on_clauses)

                # build UPDATE set (all columns except match_by)
                update_cols = [c for c in columns if c not in match_by]
                if update_cols:
                    update_set = ", ".join([f"T.`{c}` = S.`{c}`" for c in update_cols])
                    update_clause = f"WHEN MATCHED THEN UPDATE SET {update_set}"
                else:
                    update_clause = ""  # nothing to update

                # build INSERT clause
                insert_clause = f"WHEN NOT MATCHED THEN INSERT ({cols_quoted}) VALUES ({', '.join([f'S.`{c}`' for c in columns])})"

                merge_sql = f"""MERGE INTO {location} AS T
USING (
  SELECT {cols_quoted} FROM parquet.`{temp_volume_path}`
) AS S
ON {on_condition}
{update_clause}
{insert_clause}"""
                statements.append(merge_sql)

            else:
                # No match_by -> plain insert
                if mode.lower() in ("overwrite",):
                    insert_sql = f"""INSERT OVERWRITE {location}
SELECT {cols_quoted}
FROM parquet.`{temp_volume_path}`"""
                else:
                    # default: append
                    insert_sql = f"""INSERT INTO {location} ({cols_quoted})
SELECT {cols_quoted}
FROM parquet.`{temp_volume_path}`"""
                statements.append(insert_sql)

            # Execute statements (use your existing execute helper)
            try:
                for stmt in statements:
                    # trim and run
                    connected.execute(stmt.strip())
            finally:
                try:
                    temp_volume_path.rmdir(recursive=True)
                except Exception as e:
                    logger.warning(e)

            # Optionally run OPTIMIZE / ZORDER / VACUUM if requested (Databricks SQL)
            if zorder_by:
                zcols = ", ".join([f"`{c}`" for c in zorder_by])
                optimize_sql = f"OPTIMIZE {location} ZORDER BY ({zcols})"
                connected.execute(optimize_sql)

            if optimize_after_merge and match_by:
                connected.execute(f"OPTIMIZE {location}")

            if vacuum_hours is not None:
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
        match_by: list[str] = None,
        zorder_by: list[str] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        spark_options: Optional[Dict[str, Any]] = None,
    ):
        """Insert a Spark DataFrame into a Delta table with optional merge semantics.

        Args:
            data: Spark DataFrame to insert.
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            mode: Insert mode ("auto", "append", "overwrite").
            cast_options: Optional casting options.
            overwrite_schema: Whether to overwrite schema.
            match_by: Optional merge keys for upserts.
            zorder_by: Optional Z-ORDER columns.
            optimize_after_merge: Whether to run OPTIMIZE after merge.
            vacuum_hours: Optional VACUUM retention window.
            spark_options: Optional Spark write options.

        Returns:
            None.
        """
        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True
        )

        spark_options = spark_options if spark_options else {}
        if overwrite_schema:
            spark_options["overwriteSchema"] = "true"

        try:
            existing_schema = self.get_table_schema(
                catalog_name=catalog_name, schema_name=schema_name,
                table_name=table_name,
                to_arrow_schema=False
            )
        except ValueError:
            data = convert(data, pyspark.sql.DataFrame)
            data.write.mode("overwrite").options(**spark_options).saveAsTable(location)
            return

        if not isinstance(data, pyspark.sql.DataFrame):
            data = convert(data, pyspark.sql.DataFrame, target_field=existing_schema)
        else:
            cast_options = CastOptions.check_arg(options=cast_options, target_field=existing_schema)
            data = cast_spark_dataframe(data, options=cast_options)

        # --- Sanity checks & pre-cleaning (avoid nulls in keys) ---
        if match_by:
            notnull: pyspark.sql.Column = None

            for k in match_by:
                if k not in data.columns:
                    raise ValueError(f"Missing match key '{k}' in DataFrame columns: {data.columns}")

                notnull = data[k].isNotNull() if notnull is None else notnull & (data[k].isNotNull())

            data = data.filter(notnull)

        # --- Merge (upsert) ---
        target = self.spark_table(full_name=location)

        if match_by:
            # Build merge condition on the composite key
            cond = " AND ".join([f"t.`{k}` <=> s.`{k}`" for k in match_by])

            if mode.casefold() == "overwrite":
                data = data.cache()

                # Step 1: get unique key combos from source
                distinct_keys = data.select([f"`{k}`" for k in match_by]).distinct()

                (
                    target.alias("t")
                    .merge(distinct_keys.alias("s"), cond)
                    .whenMatchedDelete()
                    .execute()
                )

                # Step 3: append the clean batch
                data.write.format("delta").mode("append").saveAsTable(location)
            else:
                update_cols = [c for c in data.columns if c not in match_by]
                set_expr = {
                    c: F.expr(f"s.`{c}`") for c in update_cols
                }

                # Execute MERGE - update matching records first, then insert new ones
                (
                    target.alias("t")
                    .merge(data.alias("s"), cond)
                    .whenMatchedUpdate(set=set_expr)  # update matched rows
                    .whenNotMatchedInsertAll()  # insert new rows
                    .execute()
                )
        else:
            if mode == "auto":
                mode = "append"
            data.write.mode(mode).options(**spark_options).saveAsTable(location)

        # --- Optimize: Z-ORDER for faster lookups by composite key (Databricks) ---
        if optimize_after_merge and zorder_by:
            # pass columns as varargs
            target.optimize().executeZOrderBy(*zorder_by)

        # --- Optional VACUUM ---
        if vacuum_hours is not None:
            # Beware data retention policies; set to a safe value or use default 7 days
            target.vacuum(vacuum_hours)

    def get_table_schema(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        to_arrow_schema: bool = True
    ) -> Union[pa.Field, pa.Schema]:
        """Fetch a table schema from Unity Catalog as Arrow types.

        Args:
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            to_arrow_schema: Whether to return an Arrow schema or field.

        Returns:
            Arrow Schema or Field representing the table.
        """
        full_name = self.table_full_name(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=False
        )

        wk = self.workspace.sdk()

        try:
            table = wk.tables.get(full_name)
        except Exception as e:
            raise ValueError(f"Table %s not found, {type(e)} {e}" % full_name)

        fields = [
            column_info_to_arrow_field(_)
            for _ in table.columns
        ]

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
        """Drop a table if it exists.

        Args:
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.

        Returns:
            The StatementResult from executing the drop statement.
        """
        location, _, _, _ = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True
        )

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
        execute: bool = True
    ) -> str:
        """
        Generate DDL (Data Definition Language) SQL for creating a table from a PyField schema.

        Args:
            field: PyField schema that defines the table structure
            table_name: Name of the table to create (defaults to schema.name)
            catalog_name: Optional catalog name (defaults to "hive_metastore")
            schema_name: Optional schema name (defaults to "default")
            partition_by: Optional list of column names to partition the table by
            comment: Optional table comment
            options: Optional table properties
            if_not_exists: Whether to add IF NOT EXISTS clause

        Returns:
            A SQL string for creating the table
        """
        if not isinstance(field, pa.Field):
            field = convert(field, pa.Field)

        location, catalog_name, schema_name, table_name = self._check_location_params(
            location=location,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=True
        )

        if pa.types.is_struct(field.type):
            children = list(field.type)
        else:
            children = [field]

        # Create the DDL statement
        column_definitions = [
            self._field_to_ddl(child)
            for child in children
        ]

        sql = [
            f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{location} (",
            ",\n  ".join(column_definitions),
            ")"
        ]

        # Add partition by clause if provided
        if partition_by and len(partition_by) > 0:
            sql.append(f"\nPARTITIONED BY ({', '.join(partition_by)})")
        elif cluster_by:
            if isinstance(cluster_by, bool):
                sql.append(f"\nCLUSTER BY AUTO")
            else:
                sql.append(f"\nCLUSTER BY ({', '.join(cluster_by)})")

        # Add comment if provided
        if not comment and field.metadata:
            comment = field.metadata.get(b"comment")

        if isinstance(comment, bytes):
            comment = comment.decode("utf-8")

        if comment:
            sql.append(f"\nCOMMENT '{comment}'")

        # Add options if provided
        options = {} if options is None else options
        options.update({
            "delta.autoOptimize.optimizeWrite": optimize_write,
            "delta.autoOptimize.autoCompact": auto_compact
        })

        option_strs = []

        if options:
            for key, value in options.items():
                if isinstance(value, str):
                    option_strs.append(f"'{key}' = '{value}'")
                elif isinstance(value, bool):
                    b_value = "true" if value else "false"
                    option_strs.append(f"'{key}' = '{b_value}'")
                else:
                    option_strs.append(f"'{key}' = {value}")

        if option_strs:
            sql.append(f"\nTBLPROPERTIES ({', '.join(option_strs)})")

        statement = "\n".join(sql)

        if execute:
            return self.execute(statement)
        return statement

    def _check_location_params(
        self,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True
    ):
        """Resolve location/catalog/schema/table parameters to a full name.

        Args:
            location: Fully qualified table name override.
            catalog_name: Optional catalog override.
            schema_name: Optional schema override.
            table_name: Optional table name override.
            safe_chars: Whether to wrap identifiers in backticks.

        Returns:
            A tuple of (location, catalog_name, schema_name, table_name).
        """
        if location:
            c, s, t = self._catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = catalog_name or c, schema_name or s, table_name or t

        location = self.table_full_name(
            catalog_name=catalog_name, schema_name=schema_name,
            table_name=table_name,
            safe_chars=safe_chars
        )

        return location, catalog_name or self.catalog_name, schema_name or self.schema_name, table_name

    @staticmethod
    def _field_to_ddl(
        field: pa.Field,
        put_name: bool = True,
        put_not_null: bool = True,
        put_comment: bool = True
    ) -> str:
        """
        Convert a PyField to a DDL column definition.

        Args:
            field: The PyField to convert

        Returns:
            A string containing the column definition in DDL format
        """
        name = field.name
        nullable_str = " NOT NULL" if put_not_null and not field.nullable else ""
        name_str = f"{name} " if put_name else ""

        # Get comment if available
        comment_str = ""
        if put_comment and field.metadata and b"comment" in field.metadata:
            comment = field.metadata[b"comment"].decode("utf-8")
            comment_str = f" COMMENT '{comment}'"

        # Handle primitive types
        if not pa.types.is_nested(field.type):
            sql_type = SQLEngine._arrow_to_sql_type(field.type)
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        # Handle struct type
        if pa.types.is_struct(field.type):
            child_defs = [SQLEngine._field_to_ddl(child) for child in field.type]
            struct_body = ", ".join(child_defs)
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        # Handle map type
        if pa.types.is_map(field.type):
            map_type: pa.MapType = field.type
            key_type = SQLEngine._field_to_ddl(map_type.key_field, put_name=False, put_comment=False, put_not_null=False)
            val_type = SQLEngine._field_to_ddl(map_type.item_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        # Handle list type after map
        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            list_type: pa.ListType = field.type
            elem_type = SQLEngine._field_to_ddl(list_type.value_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        # Default fallback to string for unknown types
        raise TypeError(f"Cannot make ddl field from {field}")

    @staticmethod
    def _arrow_to_sql_type(
        arrow_type: Union[pa.DataType, pa.Decimal128Type]
    ) -> str:
        """
        Convert an Arrow data type to SQL data type.

        Args:
            arrow_type: The Arrow data type

        Returns:
            A string containing the SQL data type
        """
        if pa.types.is_boolean(arrow_type):
            return "BOOLEAN"
        elif pa.types.is_int8(arrow_type):
            return "TINYINT"
        elif pa.types.is_int16(arrow_type):
            return "SMALLINT"
        elif pa.types.is_int32(arrow_type):
            return "INT"
        elif pa.types.is_int64(arrow_type):
            return "BIGINT"
        elif pa.types.is_float32(arrow_type):
            return "FLOAT"
        elif pa.types.is_float64(arrow_type):
            return "DOUBLE"
        elif is_arrow_type_string_like(arrow_type):
            return "STRING"
        elif is_arrow_type_binary_like(arrow_type):
            return "BINARY"
        elif pa.types.is_timestamp(arrow_type):
            tz = getattr(arrow_type, "tz", None)

            if tz:
                return "TIMESTAMP"
            return "TIMESTAMP_NTZ"
        elif pa.types.is_date(arrow_type):
            return "DATE"
        elif pa.types.is_decimal(arrow_type):
            precision = arrow_type.precision
            scale = arrow_type.scale
            return f"DECIMAL({precision}, {scale})"
        elif pa.types.is_null(arrow_type):
            return "STRING"
        else:
            raise ValueError(f"Cannot make ddl type for {arrow_type}")
