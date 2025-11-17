from dataclasses import dataclass
from typing import Any

from ..data_io import DataTableIO
from ..table_location import TableLocation
from ...types import DataField
from ...libutils.py_utils import safe_dict, safe_str
from ...libutils.spark_utils import spark_sql, safe_spark_dataframe

try:
    from delta.tables import DeltaTable
except ImportError:
    DeltaTable = None


__all__ = [
    "DeltaTableIO"
]


@dataclass
class DeltaTableIO(DataTableIO):
    spark: spark_sql.SparkSession | None

    @classmethod
    def parse_any(
        cls,
        path: Any,
        schema: DataField | None = None,
        spark_session: spark_sql.SparkSession | None = None
    ):
        location = TableLocation.parse_any(path)

        if not schema:
            try:
                schema = cls.load_schema(
                    location=location,
                    spark_session=spark_session
                )
            except Exception:
                pass

        return DeltaTableIO(
            location=location,
            schema=schema,
            spark=cls.get_spark(raise_error=False)
        )

    @classmethod
    def get_spark_delta_table(
        cls,
        location: TableLocation,
        spark_session: spark_sql.SparkSession
    ):
        location = TableLocation.parse_any(location)

        if location.entity:
            return DeltaTable.forName(
                spark_session,
                location.entity.delta_table_full_name()
            )
        return DeltaTable.forPath(
            spark_session,
            location.fs_path
        )

    @classmethod
    def load_schema(
        cls,
        location: TableLocation,
        spark_session: spark_sql.SparkSession | None = None
    ) -> DataField:
        df = cls.get_spark_delta_table(
            location,
            spark_session=spark_session or cls.get_spark()
        ).toDF()

        return DataField.from_spark_type(
            name=location.entity.table_name,
            spark_type=df.schema,
            nullable=False,
            metadata=None
        )

    def unity_catalog_write(
        self,
        df, *,
        mode: str,
        overwrite_schema: bool | None = None,
        match_keys: list[str] = None,
        zorder_cols: list[str] = None,
        create_catalog_if_missing: bool = False,
        create_schema_if_missing: bool = True,
        optimize_after_merge: bool = False,
        vacuum_hours: int | None = None,  # e.g., 168 for 7 days
        spark_options: dict[str, str] | None = None
    ) -> None:
        """
        Idempotent upsert (MERGE) of df into a Delta table in Unity Catalog.
        - MATCH on `match_keys`.
        - UPDATE non-key columns on match.
        - INSERT on not matched.
        - CREATE TABLE IF NOT EXISTS (with schema from df).
        - Optionally OPTIMIZE ZORDER and VACUUM.
        """
        schema = self.load_schema(self.location)

        if schema and not isinstance(df, spark_sql.DataFrame):
            df = schema.cast_arrow_tabular(df)

        spark_session = self.spark
        df = safe_spark_dataframe(df, spark_session=spark_session)

        # --- Sanity checks & pre-cleaning (avoid nulls in keys) ---
        if match_keys:
            for k in match_keys:
                if k not in df.columns:
                    raise ValueError(f"Missing match key '{k}' in DataFrame columns: {df.columns}")

            df = df.dropna(subset=list(match_keys))  # enforce keys not null

        spark_options = safe_dict(spark_options, default={}, check_key=safe_str)
        if overwrite_schema:
            spark_options["overwriteSchema"] = "true"

        entity_object = self.location.entity
        save_path = self.location.entity.delta_table_full_name() if self.location.entity else self.location.fs_path

        if entity_object:
            # --- Ensure catalog/schema exist (Unity Catalog) ---
            if entity_object.catalog_name and create_catalog_if_missing:
                spark_session.sql(f"CREATE CATALOG IF NOT EXISTS `{entity_object.catalog_name}`")

            if entity_object.schema_name and create_schema_if_missing:
                spark_session.sql(f"CREATE SCHEMA IF NOT EXISTS `{entity_object.catalog_name}`.`{entity_object.schema_name}`")

        # --- Merge (upsert) ---
        try:
            target = self.get_spark_delta_table(
                location=self.location,
                spark_session=spark_session
            )
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
                (
                    df.write.format("delta")
                    .mode("overwrite")
                    .options(**spark_options)
                    .saveAsTable(save_path)
                )
            else:
                # Table exists but no match key specified: append incoming rows.
                # This cannot be idempotent; caller should provide match_keys for upserts.
                (
                    df
                    .write.format("delta").mode(mode)
                    .options(**spark_options)
                    .saveAsTable(save_path)
                )

        # --- Optimize: Z-ORDER for faster lookups by composite key (Databricks) ---
        if target:
            if optimize_after_merge and zorder_cols:
                # pass columns as varargs
                target.optimize().executeZOrderBy(*zorder_cols)

            # --- Optional VACUUM ---
            if vacuum_hours is not None:
                # Beware data retention policies; set to a safe value or use default 7 days
                target.vacuum(vacuum_hours)
