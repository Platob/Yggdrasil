import base64
import dataclasses
import io
import itertools
import json
import os
import random
import re
import string
import time
from typing import Optional, Union, Generator, Any, Dict, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .exception import DBXTableNotFound
from ..workspaces import DBXWorkspaceObject, DBXWorkspace
from ...libs import SparkSession, SparkDataFrame, pyspark, require_pyspark

try:
    from delta.tables import DeltaTable as SparkDeltaTable
except ImportError:
    SparkDeltaTable = None


try:
    from databricks.sdk.service.sql import State
except ImportError:
    pass


__all__ = [
    "DBXSQL",
]


STRING_TYPE_MAP = {
    # boolean
    "BOOL": pa.bool_(),
    "BOOLEAN": pa.bool_(),

    # string / text
    "CHAR": pa.string(),
    "NCHAR": pa.string(),
    "VARCHAR": pa.string(),
    "NVARCHAR": pa.string(),
    "STRING": pa.string(),
    "TEXT": pa.large_string(),
    "LONGTEXT": pa.large_string(),

    # integers
    "TINYINT": pa.int8(),
    "SMALLINT": pa.int16(),
    "INT2": pa.int16(),

    "INT": pa.int32(),
    "INTEGER": pa.int32(),
    "INT4": pa.int32(),

    "BIGINT": pa.int64(),
    "INT8": pa.int64(),

    # unsigned → widen (Arrow has no unsigned for many)
    "UNSIGNED TINYINT": pa.int16(),
    "UNSIGNED SMALLINT": pa.int32(),
    "UNSIGNED INT": pa.int64(),
    "UNSIGNED BIGINT": pa.uint64() if hasattr(pa, "uint64") else pa.int64(),

    # floats
    "FLOAT": pa.float32(),
    "REAL": pa.float32(),
    "DOUBLE": pa.float64(),
    "DOUBLE PRECISION": pa.float64(),

    # numeric/decimal — regex later for DECIMAL(p,s)
    "NUMERIC": pa.decimal128(38, 18),
    "DECIMAL": pa.decimal128(38, 18),

    # date/time/timestamp
    "DATE": pa.date32(),
    "TIME": pa.time64("ns"),
    "TIMESTAMP": pa.timestamp("us", "UTC"),
    "TIMESTAMP_NTZ": pa.timestamp("us"),
    "DATETIME": pa.timestamp("us", "UTC"),

    # binary
    "BINARY": pa.binary(),
    "VARBINARY": pa.binary(),
    "BLOB": pa.binary(),

    # json-like
    "JSON": pa.string(),
    "JSONB": pa.string(),

    # other structured text
    "UUID": pa.string(),
    "XML": pa.string(),

    # explicit arrow large types
    "LARGE_STRING": pa.large_string(),
    "LARGE_BINARY": pa.large_binary(),
}

_decimal_re = re.compile(r"^DECIMAL\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$", re.IGNORECASE)
_array_re = re.compile(r"^ARRAY\s*<\s*(.+)\s*>$", re.IGNORECASE)
_map_re = re.compile(r"^MAP\s*<\s*(.+?)\s*,\s*(.+)\s*>$", re.IGNORECASE)
_struct_re = re.compile(r"^STRUCT\s*<\s*(.+)\s*>$", re.IGNORECASE)

def _split_top_level_commas(s: str):
    parts, cur, depth = [], [], 0
    for ch in s:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts

def parse_sql_type_to_pa(type_str: str) -> pa.DataType:
    """
    Adapted parser that:
      - looks up base types in STRING_TYPE_MAP (expects uppercase keys)
      - supports DECIMAL(p,s), ARRAY<...>, MAP<k,v>, STRUCT<...> recursively
      - raises ValueError if it cannot map the provided type string
    """
    if not type_str:
        raise ValueError("Empty type string")

    raw = str(type_str).strip()

    # DECIMAL(p,s)
    m = _decimal_re.match(raw)
    if m:
        precision = int(m.group(1)); scale = int(m.group(2))
        return pa.decimal128(precision, scale)

    # ARRAY<...>
    m = _array_re.match(raw)
    if m:
        inner = m.group(1).strip()
        return pa.list_(parse_sql_type_to_pa(inner))

    # MAP<k,v>
    m = _map_re.match(raw)
    if m:
        key_raw = m.group(1).strip()
        val_raw = m.group(2).strip()
        key_type = parse_sql_type_to_pa(key_raw)
        val_type = parse_sql_type_to_pa(val_raw)
        return pa.map_(key_type, val_type)

    # STRUCT<...>
    m = _struct_re.match(raw)
    if m:
        inner = m.group(1).strip()
        parts = _split_top_level_commas(inner)
        fields = []
        for p in parts:
            if ':' not in p:
                # defensive fallback
                fname = p
                ftype = pa.string()
            else:
                fname, ftype_raw = p.split(':', 1)
                fname = fname.strip()
                ftype = parse_sql_type_to_pa(ftype_raw.strip())
            fields.append(pa.field(fname, ftype, nullable=True))
        return pa.struct(fields)

    # normalize and strip size/precision suffixes: e.g. VARCHAR(255) -> VARCHAR
    base = re.sub(r"\(.*\)\s*$", "", raw).strip().upper()

    # direct lookup in provided map
    if base in STRING_TYPE_MAP:
        return STRING_TYPE_MAP[base]

    # nothing matched — raise so caller knows it's unknown
    raise ValueError(f"Cannot convert string data type '{type_str}' to arrow")


# create a safe local staging area for ingestion ops
try:
    DEFAULT_STAGING_ALLOWED_LOCAL_PATH = os.path.join(
        os.path.expanduser("~"),
        ".ygg", "databricks", "sql", "staging"
    )
except OSError:
    DEFAULT_STAGING_ALLOWED_LOCAL_PATH = None


@dataclasses.dataclass
class DBXSQL(DBXWorkspaceObject):
    warehouse_id: Optional[str] = None

    _http_path: str = dataclasses.field(init=False, default=None)
    _was_connected: bool = dataclasses.field(init=False, default=False)

    @classmethod
    def find_in_env(
        cls,
        env: Mapping = None,
        prefix: Optional[str] = None,
        workspace: Optional[DBXWorkspace] = None
    ):
        env = env or os.environ
        prefix = prefix or "DATABRICKS_SQL_"
        workspace = workspace or DBXWorkspace.find_in_env(env=env)

        options = {
            k: env.get(prefix + k.upper())
            for k in (
                "warehouse_id"
            )
            if env.get(prefix + k.upper())
        }

        return cls(
            workspace=workspace,
            **options
        )


    @staticmethod
    def table_full_name(
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        safe_chars: bool = True
    ):
        assert catalog_name, "No catalog name give"
        assert schema_name, "No schema name give"
        assert table_name, "No table name give"

        if safe_chars:
            return f"`{catalog_name}`.`{schema_name}`.`{table_name}`"
        return f"{catalog_name}.{schema_name}.{table_name}"

    @staticmethod
    def catalog_schema_table_names(
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


    @staticmethod
    def volume_full_path(
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        volume_name: Optional[str] = None,
        shared: bool = False
    ):
        assert catalog_name, "No catalog name give"
        assert schema_name, "No schema name give"
        assert volume_name, "No table name give"
        prefix = "Shared" if shared else "Volumes"

        return f"/{prefix}/{catalog_name}/{schema_name}/{volume_name}"

    def default_warehouse(
        self,
        find_alive: bool = True
    ):
        wk = self.workspace.sdk()
        existing = wk.warehouses.list(page_size=5)
        first = None

        for warehouse in existing:
            if first is None:
                first = warehouse

            if find_alive:
                if warehouse.state == State.RUNNING:
                    return warehouse
            else:
                return warehouse

        if first is not None:
            return first

        raise ValueError(f"No default warehouse found in {wk.config.host}")

    def get_or_default_warehouse_id(
        self,
        find_alive: bool = True
    ):
        if not self.warehouse_id:
            dft = self.default_warehouse(find_alive=find_alive)

            self.warehouse_id = dft.id
        return self.warehouse_id

    def get_temp_volume(
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
        statement: str,
        **kwargs
    ):
        wk = self.workspace.sdk()

        execution = wk.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.get_or_default_warehouse_id(),
            **kwargs
        )

        return execution

    def read_arrow_batches(
        self,
        statement: str,
        *,
        batch_size: int | None = None,
    ) -> pa.RecordBatchReader:
        """Stream query results from a temporary workspace export as Arrow batches."""
        query = statement.strip().rstrip(";")
        if not query:
            raise ValueError("Statement must not be empty")

        temp_folder = self.temp_volume_folder(suffix=self._random_suffix("read_"))
        fs = self.fs()

        try:
            fs.rm(temp_folder, recurse=True)
        except Exception:
            pass

        copy_sql = f"""
        COPY INTO '{temp_folder}'
        FROM ({query})
        FILEFORMAT PARQUET
        """

        self.execute(copy_sql)

        file_infos = [
            info
            for info in fs.ls(temp_folder)
            if not str(getattr(info, "path", "")).endswith("/")
        ]

        def batch_iterator():
            try:
                for info in file_infos:
                    with fs.open(info.path, "rb") as handle:
                        parquet_file = pq.ParquetFile(handle)
                        yield from parquet_file.iter_batches(batch_size=batch_size)
            finally:
                try:
                    fs.rm(temp_folder, recurse=True)
                except Exception:
                    pass

        iterator = batch_iterator()
        first_batch = next(iterator, None)

        if first_batch is None:
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])

        return pa.RecordBatchReader.from_batches(
            first_batch.schema,
            itertools.chain([first_batch], iterator)
        )

    def spark_table(
        self,
        full_name: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
    ):
        if not full_name:
            full_name = self.table_full_name(
                catalog_name=catalog_name, schema_name=schema_name, table_name=table_name
            )

        return SparkDeltaTable.forName(
            sparkSession=SparkSession.getActiveSession(),
            tableOrViewName=full_name
        )

    def insert_into(
        self,
        data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader, str],
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        safe: Optional[bool] = None,
        fill_columns: Optional[bool] = None,
        overwrite_schema: bool | None = None,
        match_by: list[str] = None,
        zorder_by: list[str] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        spark_session: Optional[SparkSession] = None,
        spark_options: Optional[Dict[str, Any]] = None
    ):
        if SparkSession is not None:
            spark_session = SparkSession.getActiveSession()

        # -------- existing logic you provided (kept intact) ----------
        if spark_session or isinstance(data, SparkDataFrame):
            return self.spark_insert_into(
                data=data,
                location=location,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                mode=mode,
                safe=safe,
                fill_columns=fill_columns,
                overwrite_schema=overwrite_schema,
                match_by=match_by,
                zorder_by=zorder_by,
                optimize_after_merge=optimize_after_merge,
                vacuum_hours=vacuum_hours,
                spark_session=spark_session,
                spark_options=spark_options
            )

        transaction_id = self._random_suffix()

        try:
            schema = self.get_table_schema(
                catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
                spark_session=spark_session,
                to_arrow=False
            )
        except DBXTableNotFound:
            data = safe_arrow_tabular(data)
            schema = PyField.from_any(data)
            statement = self.create_table_ddl(
                schema=schema,
                catalog_name=catalog_name,
                schema_name=schema_name,
                table_name=table_name,
                if_not_exists=False
            )
            self.execute(statement)
            mode = "overwrite"

        # normalize arrow tabular input
        data: Union[pa.Table, pa.RecordBatch, pa.RecordBatchReader] = schema.cast_arrow_tabular(
            data,
            safe=safe,
            fill_columns=fill_columns,
        )

        # Write in temp volume
        databricks_tmp_folder = self.temp_volume_folder(suffix=transaction_id)
        databricks_tmp_path = databricks_tmp_folder + "/data.parquet"

        # Use pyarrow to write parquet to local tmp path (robust across pyarrow versions)
        # If `data` is a RecordBatchReader, convert to table
        if isinstance(data, pa.RecordBatchReader):
            table: pa.Table = pa.Table.from_batches(list(data))
        elif isinstance(data, pa.RecordBatch):
            table: pa.Table = pa.Table.from_batches([data])
        elif isinstance(data, pa.Table):
            table: pa.Table = data
        else:
            raise TypeError("Unsupported data type for parquet write")

        buffer = io.BytesIO()

        # remove pandas index if present in the schema helper
        pq.write_table(table, buffer, compression="snappy")

        buffer = base64.b64encode(buffer.getvalue()).decode("utf8")

        self.fs().put(
            file=databricks_tmp_path,
            contents=buffer,
            overwrite=True
        )

        # build fully-qualified table name
        full_table = self.table_full_name(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name
        )

        # get column list from arrow schema
        columns = [c for c in table.schema.names]
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

            merge_sql = f"""
            MERGE INTO {full_table} AS T
            USING (
              SELECT {cols_quoted}
              FROM parquet.`{databricks_tmp_folder}`
            ) AS S
            ON {on_condition}
            {update_clause}
            {insert_clause}
            """
            statements.append(merge_sql)

        else:
            # No match_by -> plain insert
            if mode.lower() in ("overwrite",):
                insert_sql = f"""
                INSERT OVERWRITE {full_table}
                SELECT {cols_quoted}
                FROM parquet.`{databricks_tmp_folder}`
                """
            else:
                # default: append
                insert_sql = f"""
                INSERT INTO {full_table} ({cols_quoted})
                SELECT {cols_quoted}
                FROM parquet.`{databricks_tmp_folder}`
                """
            statements.append(insert_sql)

        # Execute statements (use your existing execute helper)
        try:
            for stmt in statements:
                # trim and run
                self.execute(stmt.strip())
        finally:
            self.fs().rm(databricks_tmp_folder, recurse=True)

        # Optionally run OPTIMIZE / ZORDER / VACUUM if requested (Databricks SQL)
        if zorder_by:
            zcols = ", ".join([f"`{c}`" for c in zorder_by])
            optimize_sql = f"OPTIMIZE {full_table} ZORDER BY ({zcols})"
            self.execute(optimize_sql)

        if optimize_after_merge and match_by:
            self.execute(f"OPTIMIZE {full_table}")

        if vacuum_hours is not None:
            self.execute(f"VACUUM {full_table} RETAIN {vacuum_hours} HOURS")

        return None

    @require_pyspark
    def spark_insert_into(
        self,
        data: SparkDataFrame,
        *,
        location: Optional[str] = None,
        catalog_name: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: Optional[str] = None,
        mode: str = "auto",
        safe: Optional[bool] = None,
        fill_columns: Optional[bool] = None,
        overwrite_schema: bool | None = None,
        match_by: list[str] = None,
        zorder_by: list[str] = None,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        spark_session: Optional[SparkSession] = None,
        spark_options: Optional[Dict[str, Any]] = None,
        existing_schema: Optional[pa.Schema] = None
    ):
        location = location or self.table_full_name(catalog_name=catalog_name, schema_name=schema_name, table_name=table_name)
        spark_options = spark_options if spark_options else {}
        if overwrite_schema:
            spark_options["overwriteSchema"] = "true"

        try:
            existing_schema = existing_schema or self.get_table_schema(
                catalog_name=catalog_name, schema_name=schema_name, table_name=table_name,
                spark_session=spark_session,
                to_arrow=False
            )
            data = safe_spark_dataframe(data, field=existing_schema)
        except:
            data = safe_spark_dataframe(data)
            data.write.mode("overwrite").options(**spark_options).saveAsTable(location)
            return

        if not isinstance(data, SparkDataFrame):
            data = existing_schema.cast_arrow_tabular(
                data,
                safe=safe,
                fill_columns=fill_columns
            )
            spark_session = spark_session or SparkSession.getActiveSession()
            if not spark_session:
                raise ValueError("Need initialized spark session")
            data = spark_session.createDataFrame(data)
        else:
            data = existing_schema.cast_spark_dataframe(
                data, safe=safe, fill_columns=fill_columns
            )

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
                    c: pyspark.sql.functions.expr(f"s.`{c}`") for c in update_cols
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
        spark_session: Optional[SparkSession] = None,
        to_arrow: bool = False
    ):
        full_name = self.table_full_name(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=table_name,
            safe_chars=False)

        if SparkDeltaTable is not None:
            spark_session = spark_session or SparkSession.getActiveSession()

            if spark_session:
                schema = PyField.from_spark(SparkDeltaTable.forName(
                    sparkSession=SparkSession.getActiveSession(),
                    tableOrViewName=full_name
                ).toDF())

                return schema

        wk = self.workspace.sdk()

        try:
            columns =  wk.tables.get(full_name).columns
        except Exception as e:
            raise DBXTableNotFound(f"Table %s not found, {type(e)} {e}" % full_name)

        fields = [
            self._column_info_to_arrow_field(_)
            for _ in columns
        ]

        arrow_field = pa.field(
            name=full_name,
            type=pa.struct(fields),
            nullable=False
        )

        return arrow_field if to_arrow else PyStructField.from_arrow_field(arrow_field)

    @staticmethod
    def _column_info_to_arrow_field(col):
        from databricks.sdk.service.catalog import ColumnInfo
        col: ColumnInfo = col

        parsed = json.loads(col.type_json)
        arrow_type = parse_sql_type_to_pa(col.type_text)

        return pa.field(
            col.name,
            arrow_type,
            nullable=col.nullable,
            metadata=parsed.get("metadata", {}) or {}
        )

    @staticmethod
    def arrow_to_insert_statements(
        data: Union[pa.Table, pa.RecordBatch],
        table_name: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        batch_size: int = 100
    ) -> Generator[str, None, None]:
        """
        Convert an Arrow Table or RecordBatch to SQL INSERT statements.

        Args:
            data: PyArrow Table or RecordBatch to convert
            table_name: The name of the table to insert into
            catalog: Optional catalog name (defaults to the connection's default_catalog)
            schema: Optional schema name (defaults to the connection's default_schema)
            batch_size: Number of rows per INSERT statement

        Returns:
            Generator of INSERT SQL statements
        """
        # Ensure data is a Table
        if isinstance(data, pa.RecordBatch):
            data = pa.Table.from_batches([data])

        # Build fully qualified table name
        qualified_table_name = table_name
        if schema:
            qualified_table_name = f"{schema}.{qualified_table_name}"
        if catalog:
            qualified_table_name = f"{catalog}.{qualified_table_name}"

        # Get column names
        column_names = data.column_names
        columns_clause = ", ".join(f"`{col}`" for col in column_names)

        def format_value(v):
            """Helper function to format values for SQL, handling nested structures"""
            if v is None:
                return "NULL"
            elif isinstance(v, (int, float)):
                return str(v)
            elif isinstance(v, bool):
                return "TRUE" if v else "FALSE"
            elif isinstance(v, (list, tuple)):
                # Format array values
                formatted_items = [format_value(item) for item in v]
                array_str = ", ".join(formatted_items)
                return f"ARRAY[{array_str}]"
            elif isinstance(v, dict):
                # Format struct/map values
                formatted_items = [
                    f"{format_value(k)} => {format_value(v)}"
                    for k, v in v.items()
                ]
                map_str = ", ".join(formatted_items)
                return f"MAP({map_str})"
            else:
                # Escape single quotes and wrap in quotes for strings
                val_str = str(v).replace("'", "''")
                return f"'{val_str}'"

        # Process in batches
        num_rows = data.num_rows
        for i in range(0, num_rows, batch_size):
            batch = data.slice(i, min(batch_size, num_rows - i))

            # Start building the INSERT statement
            insert_stmt = f"INSERT INTO {qualified_table_name} ({columns_clause}) VALUES "

            # Add value tuples
            values = []
            for row_idx in range(batch.num_rows):
                row_values = []
                for col_idx, col_name in enumerate(column_names):
                    val = batch.column(col_idx)[row_idx].as_py()
                    formatted_val = format_value(val)
                    row_values.append(formatted_val)

                values.append(f"({', '.join(row_values)})")

            insert_stmt += ", ".join(values)
            yield insert_stmt

    @classmethod
    def create_table_ddl(
        cls, schema: PyField,
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
            schema: PyField schema that defines the table structure
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
        table_name = table_name or schema.name
        catalog_name = catalog_name or "hive_metastore"
        schema_name = schema_name or "default"
        full_table_name = f"{catalog_name}.{schema_name}.{table_name}"

        # Create the DDL statement
        sql = [f"CREATE TABLE {'IF NOT EXISTS ' if if_not_exists else ''}{full_table_name} ("]

        # Generate column definitions
        column_defs = []
        children = schema.children if isinstance(schema, PyStructField) else [schema]

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
        comment = comment or schema.metadata_str.get("comment")
        if comment:
            sql.append(f"\nCOMMENT '{comment}'")

        # Add options if provided
        if options:
            option_strs = []
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
        field: PyField,
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
        if not pa.types.is_nested(field.arrow_type):
            sql_type = DBXSQL._arrow_to_sql_type(field.arrow_type)
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        # Handle struct type
        if isinstance(field, PyStructField):
            child_defs = [DBXSQL._field_to_ddl(child) for child in field.children]
            struct_body = ", ".join(child_defs)
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        # Handle map type
        if isinstance(field, PyMapField):
            key_type = DBXSQL._field_to_ddl(field.key_field, put_name=False, put_comment=False, put_not_null=False)
            val_type = DBXSQL._field_to_ddl(field.value_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        # Handle list type after map
        if isinstance(field, PyListField):
            elem_type = DBXSQL._field_to_ddl(field.value_field, put_name=False, put_comment=False, put_not_null=False)
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        # Default fallback to string for unknown types
        raise TypeError(f"Cannot make ddl field from {field}")

    @staticmethod
    def _arrow_to_sql_type(arrow_type: pa.DataType) -> str:
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
        elif PyField.is_arrow_string_like(arrow_type):
            return "STRING"
        elif PyField.is_arrow_binary_like(arrow_type):
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
        else:
            raise ValueError(f"Cannot make ddl type for {arrow_type}")
