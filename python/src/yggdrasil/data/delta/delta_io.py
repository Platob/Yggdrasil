from typing import Optional

from ..data_io import DataIO, SaveMode
from ..table_location import TableLocation
from ...types import DataField
from ...utils.spark_utils import spark_sql

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None


__all__ = [
    "DeltaIO"
]


class DeltaIO(DataIO):

    def __init__(
        self,
        spark: spark_sql.SparkSession
    ):
        self.spark = spark

    def get_delta_table(self, location: TableLocation):
        return DeltaTable.forName(self.spark, location.delta_table_name())

    def get_schema(self, location: TableLocation) -> DataField:
        df = self.get_delta_table(location).toDF()

        return DataField.from_spark_type(
            name=location.table_name, spark_type=df.schema,
            nullable=False,
            metadata=None
        )

    def delete_table(self, location: TableLocation) -> DataField:
        full_name = location.sql_full_name(decorator="`", separator=".")

        self.spark.sql(f"DELETE TABLE IF EXISTS {full_name}")

    def read_spark(
        self,
        location: TableLocation,
        schema: Optional[DataField] = None,
        **kwargs
    ) -> spark_sql.DataFrame:
        """
        Read a Delta table (supports Unity Catalog qualified names).
        Accepts optional kwargs that are passed as reader options (e.g., versionAsOf, timestampAsOf).
        If `schema` is supplied it will be used to cast the returned DataFrame via DataField.cast_spark_dataframe.
        """
        full_name = location.delta_table_name()

        # Use Spark's table reader for catalog-qualified Delta tables.
        # Accept reader options via kwargs (e.g., versionAsOf=123, timestampAsOf="2023-01-01")
        reader = self.spark.read.format("delta")
        for k, v in kwargs.items():
            # spark expects option values as str for many options; leave flexibility to caller
            reader = reader.option(k, v)

        try:
            df = reader.table(full_name)
        except Exception:
            # Fallback: try to read by path if TableLocation provides one via sql_full_name
            # This is a best-effort fallback; if it fails we'll re-raise the original error.
            try:
                path = location.full_path  # may or may not exist on TableLocation
            except Exception:
                raise
            df = reader.load(path)

        if schema is not None:
            df = schema.cast_spark_dataframe(df)

        return df

    def _write_spark(
        self,
        location: TableLocation,
        df: spark_sql.DataFrame,
        mode: SaveMode = SaveMode.Overwrite,
        **kwargs
    ) -> None:
        """
        Write DataFrame to Delta using saveAsTable for catalog-qualified names.
        kwargs are forwarded to the writer as options (e.g., mergeSchema="true", path="...").
        """
        full_name = location.delta_table_name()

        # Map SaveMode enum -> spark mode string. If SaveMode provides .name, rely on that.
        try:
            spark_mode = mode.name.lower()
        except Exception:
            # Fallback to string conversion
            spark_mode = str(mode).lower()

        writer = df.write.format("delta").mode(spark_mode)

        # forward writer options
        for k, v in kwargs.items():
            writer = writer.option(k, v)

        # Use saveAsTable for Unity Catalog / managed/external table names
        writer.saveAsTable(full_name)

    def unity_catalog_write(
        self,
        df, *,
        location: TableLocation,
        match_keys: list[str] = None,
        zorder_cols: list[str] = None,
        create_catalog_if_missing: bool = False,
        create_schema_if_missing: bool = True,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
    ) -> None:
        """
        Idempotent upsert (MERGE) of df into a Delta table in Unity Catalog.
        - MATCH on `match_keys`.
        - UPDATE non-key columns on match.
        - INSERT on not matched.
        - CREATE TABLE IF NOT EXISTS (with schema from df).
        - Optionally OPTIMIZE ZORDER and VACUUM.
        """
        # --- Sanity checks & pre-cleaning (avoid nulls in keys) ---
        if match_keys:
            for k in match_keys:
                if k not in df.columns:
                    raise ValueError(f"Missing match key '{k}' in DataFrame columns: {df.columns}")

            df = df.dropna(subset=list(match_keys))  # enforce keys not null

        spark = df.sparkSession
        schema = self.get_schema(location)
        df = schema.cast_spark_dataframe(df)
        full_name = location.delta_table_name()

        # --- Ensure catalog/schema exist (Unity Catalog) ---
        if location.catalog_name and create_catalog_if_missing:
            spark.sql(f"CREATE CATALOG IF NOT EXISTS `{location.catalog_name}`")
        if location.schema_name and create_schema_if_missing:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{location.catalog_name}`.`{location.schema_name}`")

        # --- Merge (upsert) ---
        target = None
        try:
            target = self.get_delta_table(location)
        except Exception:
            # target will remain None if table doesn't exist
            target = None

        if match_keys and target is not None:
            # Build merge condition on the composite key
            cond = " AND ".join([f"t.`{k}` <=> s.`{k}`" for k in match_keys])

            # Execute MERGE - delete matching records first, then insert new ones
            (
                target.alias("t")
                .merge(df.alias("s"), cond)
                .whenMatchedDelete()  # Remove existing records that match
                .whenNotMatchedInsertAll()  # Insert all new records
                .execute()
            )
        else:
            # No match keys provided or target does not exist -> simple write/create behavior.
            if target is None:
                # Table doesn't exist: create it with df schema
                df.write.format("delta").mode("overwrite").saveAsTable(full_name)
            else:
                # Table exists but no match key specified: append incoming rows.
                # This cannot be idempotent; caller should provide match_keys for upserts.
                df.write.format("delta").mode("append").saveAsTable(full_name)

        # --- Optimize: Z-ORDER for faster lookups by composite key (Databricks) ---
        if optimize_after_merge and zorder_cols:
            cols = ", ".join([f"`{c}`" for c in zorder_cols])
            spark.sql(f"OPTIMIZE {full_name} ZORDER BY ({cols})")

        # --- Optional VACUUM ---
        if vacuum_hours is not None:
            # Beware data retention policies; set to a safe value or use default 7 days
            spark.sql(f"VACUUM {full_name} RETAIN {vacuum_hours} HOURS")
