import dataclasses
import io
import logging
import random
import string
import time
from functools import cached_property
from typing import Optional, Union, Generator, Any, Dict, List

import pyarrow as pa
import pyarrow.ipc as pipc
import pyarrow.parquet as pq

from .types import column_info_to_arrow_field
from ..workspaces import WorkspaceService
from ...libs.databrickslib import databricks_sdk
from ...libs.pandaslib import pandas
from ...libs.polarslib import polars
from ...libs.sparklib import SparkSession, SparkDataFrame, pyspark
from ...requests.session import YGGSession
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
    from databricks.sdk.service.sql import StatementState, StatementResponse, Disposition, Format, \
    ExecuteStatementRequestOnWaitTimeout, StatementParameterListItem

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
    pass


@dataclasses.dataclass
class SQLEngine(WorkspaceService):
    warehouse_id: Optional[str] = None

    _http_path: str = dataclasses.field(init=False, default=None)
    _was_connected: bool = dataclasses.field(init=False, default=False)

    @staticmethod
    def _table_full_name(
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True
    ):
        assert catalog_name, "No catalog name given"
        assert schema_name, "No schema name given"
        assert table_name, "No table name given"

        if safe_chars:
            return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
        return f"{catalog_name}.{schema_name}.{table_name}"

    @staticmethod
    def _catalog_schema_table_names(
        full_name: str,
    ):
        parts = [
            _.strip("`") for _ in full_name.split(".")
        ]

        if len(parts) == 0:
            return None, None, None
        if len(parts) == 1:
            return None, None, parts[0]
        if len(parts) == 2:
            return None, parts[0], parts[1]

        return parts[-3], parts[-2], parts[-1]

    def _default_warehouse(
        self,
        cluster_size: str = "Small"
    ):
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
        if not self.warehouse_id:
            dft = self._default_warehouse(cluster_size=cluster_size)

            self.warehouse_id = dft.id
        return self.warehouse_id

    def _get_temp_volume(
        self,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
    ):
        pass

    @staticmethod
    def _random_suffix(prefix: str = "") -> str:
        unique = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        timestamp = int(time.time() * 1000)
        return f"{prefix}{timestamp}_{unique}"

    def execute(
        self,
        statement: Optional[str] = None,
        *,
        byte_limit: Optional[int] = None,
        disposition: Optional[Disposition] = None,
        format: Optional[Format] = None,
        on_wait_timeout: Optional[ExecuteStatementRequestOnWaitTimeout] = None,
        parameters: Optional[List[StatementParameterListItem]] = None,
        row_limit: Optional[int] = None,
        wait_timeout: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        wait: bool = True,
        timeout: Optional[float] = 600.0,
        poll_interval: float = 1.0,
        **kwargs,
    ):
        """
        Execute a SQL statement on a SQL warehouse.

        - If wait=True (default): poll until terminal state.
            - On SUCCEEDED: return final statement object
            - On FAILED / CANCELED: raise SqlExecutionError
        - If wait=False: return initial execution handle without polling.
        """
        if format is None:
            format = Format.ARROW_STREAM

        if (disposition is None or disposition == Disposition.INLINE) and format in [Format.CSV, Format.ARROW_STREAM]:
            disposition = Disposition.EXTERNAL_LINKS

        if not statement:
            full_name = self._table_full_name(catalog_name=catalog_name, schema_name=schema_name, table_name=table_name)
            statement = f"SELECT * FROM {full_name}"

        with self as connected:
            wk = connected.workspace.sdk()

            response = wk.statement_execution.execute_statement(
                statement=statement,
                warehouse_id=self._get_or_default_warehouse_id(),
                byte_limit=byte_limit,
                disposition=disposition,
                format=format,
                on_wait_timeout=on_wait_timeout,
                parameters=parameters,
                row_limit=row_limit,
                wait_timeout=wait_timeout,
                catalog=catalog_name,
                schema=schema_name,
                **kwargs,
            )
            execution = StatementResult(
                engine=connected,
                response=response
            )

            if not wait:
                # Caller handles polling / status themselves
                return execution
            return execution.wait(timeout=timeout, poll_interval=poll_interval)

    def get_statement(self, statement_id: str):
        return StatementResult(
            engine=self,
            response=self.sdk().statement_execution.get_statement(statement_id),
        )

    def spark_table(
        self,
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        if not full_name:
            full_name = self._table_full_name(
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
        # -------- existing logic you provided (kept intact) ----------
        if pyspark is not None:
            spark_session = SparkSession.getActiveSession()

            if spark_session or isinstance(data, SparkDataFrame):
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
    ):
        if location:
            c, s, t = self._catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = catalog_name or c, schema_name or s, table_name or t

        transaction_id = self._random_suffix()

        existing_schema = None

        with self as connected:
            try:
                existing_schema = connected.get_table_schema(
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    table_name=table_name,
                    to_arrow_schema=True
                )
                # normalize arrow tabular input
                data = convert(data, pa.Table, options=cast_options, target_field=existing_schema)
            except ValueError as exc:
                logger.warning(
                    "Falling back to create-or-overwrite for %s.%s.%s after cast failure: %s",
                    catalog_name,
                    schema_name,
                    table_name,
                    exc,
                )
                logger.debug(
                    "Existing schema fields: %s",
                    [f"{f.name}:{f.type}" for f in existing_schema] if existing_schema else None,
                )
                data = convert(data, pa.Table)
                existing_schema = data.schema
                statement = connected.create_table_ddl(
                    field=existing_schema,
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    table_name=table_name,
                    if_not_exists=False
                )
                connected.execute(statement)
                mode = "overwrite"

            # Write in temp volume
            databricks_tmp_folder = connected.temp_volume_folder(
                suffix=transaction_id,
                catalog_name=catalog_name,
                schema_name=schema_name,
                volume_name="tmp"
            )
            databricks_tmp_path = databricks_tmp_folder + "/data.parquet"

            buffer = io.BytesIO()

            # remove pandas index if present in the schema helper
            pq.write_table(data, buffer, compression="snappy")
            buffer.seek(0)

            connected.upload_content_file(
                target_path=databricks_tmp_path,
                content=buffer,
            )

            # build fully-qualified table name
            full_table = connected._table_full_name(
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name
            )

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

                merge_sql = f"""MERGE INTO {full_table} AS T
USING (
  SELECT {cols_quoted} FROM parquet.`{databricks_tmp_folder}`
) AS S
ON {on_condition}
{update_clause}
{insert_clause}"""
                statements.append(merge_sql)

            else:
                # No match_by -> plain insert
                if mode.lower() in ("overwrite",):
                    insert_sql = f"""INSERT OVERWRITE {full_table}
SELECT {cols_quoted}
FROM parquet.`{databricks_tmp_folder}`"""
                else:
                    # default: append
                    insert_sql = f"""INSERT INTO {full_table} ({cols_quoted})
SELECT {cols_quoted}
FROM parquet.`{databricks_tmp_folder}`"""
                statements.append(insert_sql)

            # Execute statements (use your existing execute helper)
            try:
                for stmt in statements:
                    # trim and run
                    connected.execute(stmt.strip())
            finally:
                connected.delete_path(databricks_tmp_folder, recursive=True)

            # Optionally run OPTIMIZE / ZORDER / VACUUM if requested (Databricks SQL)
            if zorder_by:
                zcols = ", ".join([f"`{c}`" for c in zorder_by])
                optimize_sql = f"OPTIMIZE {full_table} ZORDER BY ({zcols})"
                connected.execute(optimize_sql)

            if optimize_after_merge and match_by:
                connected.execute(f"OPTIMIZE {full_table}")

            if vacuum_hours is not None:
                connected.execute(f"VACUUM {full_table} RETAIN {vacuum_hours} HOURS")

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
        if location:
            c, s, t = self._catalog_schema_table_names(location)
            catalog_name, schema_name, table_name = catalog_name or c, schema_name or s, table_name or t
        else:
            location = self._table_full_name(
                catalog_name=catalog_name, schema_name=schema_name,
                table_name=table_name
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
            try:
                data = convert(data, pyspark.sql.DataFrame, target_field=existing_schema)
            except Exception as exc:
                logger.error(
                    "Failed to cast data to Spark DataFrame for %s with schema %s: %s",
                    location,
                    [f"{f.name}:{f.dataType}" for f in existing_schema],
                    exc,
                )
                raise
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

            if mode.casefold() == "auto":
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
        full_name = self._table_full_name(
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

    @classmethod
    def create_table_ddl(
        cls,
        field: pa.Field,
        table_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        partition_by: Optional[list[str]] = None,
        comment: Optional[str] = None,
        options: Optional[dict] = None,
        if_not_exists: bool = True
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

        table_name = table_name or field.name
        catalog_name = catalog_name or "hive_metastore"
        schema_name = schema_name or "default"
        full_table_name = cls._table_full_name(
            catalog_name, schema_name, table_name,
            safe_chars=True
        )

        # Create the DDL statement
        sql = [f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{full_table_name} ("]

        # Generate column definitions
        column_defs = []

        if pa.types.is_struct(field.type):
            children = list(field.type)
        else:
            children = [field]

        for child in children:
            column_def = cls._field_to_ddl(child)
            column_defs.append(column_def)

        sql.append(",\n  ".join(column_defs))
        sql.append(")")

        # Add partition by clause if provided
        if partition_by and len(partition_by) > 0:
            sql.append(f"\nPARTITIONED BY ({', '.join(partition_by)})")
        else:
            sql.append(f"\nCLUSTER BY AUTO")

        # Add comment if provided
        if not comment and field.metadata:
            comment = field.metadata.get(b"comment")

        if isinstance(comment, bytes):
            comment = comment.decode("utf-8")

        if comment:
            sql.append(f"\nCOMMENT '{comment}'")

        # Add options if provided
        option_strs = []

        if options:
            for key, value in options.items():
                if isinstance(value, str):
                    option_strs.append(f"'{key}' = '{value}'")
                else:
                    option_strs.append(f"'{key}' = {value}")

        for dft in (
            "'delta.autoOptimize.optimizeWrite' = 'true'",
            "'delta.autoOptimize.autoCompact' = 'true'"
        ):
            option_strs.append(dft)

        if option_strs:
            sql.append(f"\nTBLPROPERTIES ({', '.join(option_strs)})")

        return "\n".join(sql)

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
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type) or pa.types.is_string_view(arrow_type):
            return "STRING"
        elif pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type) or pa.types.is_binary_view(arrow_type) or pa.types.is_fixed_size_binary(arrow_type):
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


@dataclasses.dataclass
class StatementResult:
    engine: SQLEngine
    response: StatementResponse

    def __iter__(self):
        return self.arrow_batches()

    @property
    def workspace(self):
        return self.engine.workspace

    @property
    def status(self):
        return self.response.status

    @property
    def state(self):
        return self.status.state

    @property
    def statement_id(self):
        return self.response.statement_id

    @property
    def manifest(self):
        return self.response.manifest

    @property
    def result(self):
        return self.response.result

    @property
    def external_links(self):
        return self.response.result.external_links

    def _fetch_chunk(self, chunk_index: int):
        if not self.workspace:
            raise ValueError("Workspace is required to fetch additional result chunks")

        sdk = self.workspace.sdk()
        return sdk.statement_execution.get_statement_result_chunk_n(
            statement_id=self.statement_id,
            chunk_index=chunk_index,
        )

    @property
    def done(self):
        return self.state in [StatementState.CANCELED, StatementState.CLOSED, StatementState.FAILED, StatementState.SUCCEEDED]

    @property
    def failed(self):
        return self.state in [StatementState.CANCELED, StatementState.FAILED]

    def raise_for_status(self):
        if self.failed:
            # grab error info if present
            err = self.status.error
            message = err.message or "Unknown SQL error"
            error_code = err.error_code
            sql_state = getattr(err, "sql_state", None)

            parts = [message]
            if error_code:
                parts.append(f"error_code={error_code}")
            if sql_state:
                parts.append(f"sql_state={sql_state}")

            raise SqlExecutionError(
                f"Statement {self.statement_id} {self.state}: " + " | ".join(parts)
            )

    def wait(
        self,
        timeout: Optional[int] = None,
        poll_interval: Optional[float] = None
    ):
        start = time.time()
        poll_interval = poll_interval or 1
        current = self

        while True:
            current = self.engine.get_statement(current.statement_id)
            current.raise_for_status()

            if current.done:
                break

            # still running / queued / pending
            if timeout is not None and (time.time() - start) > timeout:
                raise TimeoutError(
                    f"Statement {current.statement_id} did not finish within {timeout} seconds "
                    f"(last state={current})"
                )

            poll_interval = max(10, poll_interval * 1.2)
            time.sleep(poll_interval)

        return current

    @cached_property
    def arrow_schema(self):
        fields = [
            column_info_to_arrow_field(_) for _ in self.manifest.schema.columns
        ]
        return pa.schema(fields)

    def arrow_table(self, max_workers: int | None = None) -> pa.Table:
        batches = list(self.arrow_batches(max_workers=max_workers))

        if not batches:
            # empty table with no columns
            return pa.Table.from_batches([], schema=self.arrow_schema)

        return pa.Table.from_batches(batches)

    def arrow_batches(
        self, max_workers: int | None = None
    ) -> Generator[pa.RecordBatch, None, None]:
        if self.manifest and self.manifest.format != Format.ARROW_STREAM:
            raise ValueError("Cannot convert to arrow batches, run execute(..., format=Format.ARROW_STREAM)")

        result_data = self.result

        if result_data.external_links is not None:
            session = YGGSession()
            link = None

            while True:
                for link in result_data.external_links:
                    resp = session.get(link.external_link, verify=False, timeout=10)
                    resp.raise_for_status()

                    buf = pa.BufferReader(resp.content)

                    # If it's an IPC *stream*:
                    reader = pipc.open_stream(buf)

                    # If it’s an IPC *file*, use:
                    # reader = pipc.open_file(buf)

                    # reader yields RecordBatch objects
                    for batch in reader:
                        yield batch

                    if not link.next_chunk_internal_link:
                        break

                    # /api/2.0/sql/statements/01f0d056-0596-194e-b011-aef9049504bf/result/chunks/1
                    try:
                        chunk_index = int(link.next_chunk_internal_link.split("/")[-1])
                        result_data = self.workspace.sdk().statement_execution.get_statement_result_chunk_n(
                            statement_id=self.statement_id,
                            chunk_index=chunk_index
                        )
                    except Exception as e:
                        raise SqlExecutionError(
                            f"Cannot retrieve data batch from {link.next_chunk_internal_link!r}: {e}")

                break
        else:
            raise ValueError("Cannot convert to arrow batches, run execute(..., format=Format.ARROW_STREAM)")

    def to_pandas(
        self,
        max_workers: int | None = None
    ) -> "pandas.DataFrame":
        return self.arrow_table(max_workers=max_workers).to_pandas()

    def to_polars(
        self,
        max_workers: int | None = None
    ) -> "polars.DataFrame":
        return polars.DataFrame(self.arrow_table(max_workers=max_workers))

