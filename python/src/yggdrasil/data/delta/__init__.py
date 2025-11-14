"""
Delta Lake data reader and writer implementation.

This module provides classes for reading from and writing to Delta Lake tables,
supporting both path-based access and Unity Catalog table names.

Examples:
    Path-based access:

    ```python
    from yggdrasil.data.delta import DeltaDataReader, DeltaDataWriter
    from pyspark.sql import SparkSession

    # Initialize Spark session
    spark = SparkSession.builder.appName("DeltaExample").getOrCreate()

    # Create a reader for path-based Delta table
    reader = DeltaDataReader(
        location="/path/to/delta/table",
        spark_session=spark
    )

    # Read into different formats
    spark_df = reader.read_spark()
    polars_df = reader.read_polars()

    # Create a writer for path-based Delta table
    writer = DeltaDataWriter(
        location="/path/to/output/table",
        spark_session=spark,
        save_mode="overwrite"
    )

    # Write from different formats
    writer.write_spark(spark_df)
    writer.write_polars(polars_df)
    ```

    Unity Catalog access:

    ```python
    from yggdrasil.data.delta import DeltaDataReader, DeltaDataWriter
    from pyspark.sql import SparkSession

    # Initialize Spark session
    spark = SparkSession.builder.appName("DeltaExample").getOrCreate()

    # Create a reader for Unity Catalog table
    reader = DeltaDataReader(
        location="trading.ba_3mv_polaris__p__volcano_ref_input.asset_forecasts",
        spark_session=spark,
        use_catalog=True
    )

    # Read into different formats
    spark_df = reader.read_spark()
    polars_df = reader.read_polars()

    # Access the underlying DeltaTable object if needed
    delta_table = reader.get_delta_table()

    # Create a writer for Unity Catalog table
    writer = DeltaDataWriter(
        location="trading.output_schema.result_table",
        spark_session=spark,
        save_mode="append",
        use_catalog=True
    )

    # Write from different formats
    writer.write_spark(spark_df)
    writer.write_polars(polars_df)
    ```

    Creating tables from DataField schema:

    ```python
    from yggdrasil.data.delta import DeltaDataWriter
    from yggdrasil.types.field import DataField
    from pyspark.sql import SparkSession

    # Initialize Spark session
    spark = SparkSession.builder.appName("DeltaSchemaExample").getOrCreate()

    # Create a writer for Unity Catalog
    writer = DeltaDataWriter(
        location="trading.ba_3mv_polaris__p__volcano_ref_input.asset_forecasts",
        spark_session=spark,
        use_catalog=True
    )

    # Define schema using DataField with column comments
    schema = DataField(
        name="asset_forecasts",
        field_type="struct",
        children=[
            DataField(name="asset_class", field_type="string", nullable=True,
                     metadata={"comment": "Asset classification, e.g. Equity, FX, Commodity"}),
            DataField(name="asset_code", field_type="string", nullable=True,
                     metadata={"comment": "Ticker / instrument code (unique per asset class)"}),
            DataField(name="source", field_type="string", nullable=True,
                     metadata={"comment": "Data source identifier (feed name or provider)"}),
            DataField(name="as_of_date", field_type="date", nullable=True,
                     metadata={"comment": "Snapshot date (UTC date), used for partition pruning"}),
            DataField(name="as_of_timestamp", field_type="timestamp", nullable=True,
                     metadata={"comment": "Ingestion/snapshot timestamp in UTC"}),
            DataField(name="timestamp", field_type="timestamp", nullable=True,
                     metadata={"comment": "Event timestamp (UTC) for the price observation"}),
            DataField(name="start_timestamp", field_type="timestamp", nullable=True,
                     metadata={"comment": "Start of the sampling interval (== timestamp)"}),
            DataField(name="end_timestamp", field_type="timestamp", nullable=True,
                     metadata={"comment": "End of the sampling interval (== timestamp + 1 hour)"}),
            DataField(name="sampling", field_type="int", nullable=True,
                     metadata={"comment": "Sampling interval in seconds (should be 3600)"}),
            DataField(name="price", field_type="decimal", nullable=True,
                     metadata={"comment": "Price value (DECIMAL for financial exactness)"}),
            DataField(name="currency", field_type="string", nullable=True,
                     metadata={"comment": "ISO currency code of the price (e.g. USD)"}),
            DataField(name="quantity_unit", field_type="string", nullable=True,
                     metadata={"comment": "Unit of quantity (e.g. barrels, tonnes, shares)"})
        ]
    )

    # Create the table with partitioning, properties, and comments
    writer.create_table_from_schema(
        schema=schema,
        partition_by="as_of_date",
        table_comment="Hourly pricing snapshot table: one row per asset timestamp. as_of_timestamp in UTC.",
        table_properties={
            "delta.autoOptimize.optimizeWrite": "true",
            "delta.autoOptimize.autoCompact": "true",
            "delta.minReaderVersion": "2",
            "delta.minWriterVersion": "7"
        },
        analyze_columns=["asset_code", "asset_class", "as_of_date"]
    )
    ```

    This will create a Delta table equivalent to the following SQL:

    ```sql
    CREATE TABLE IF NOT EXISTS trading.ba_3mv_polaris__p__volcano_ref_input.asset_forecasts
    (
      asset_class       STRING  COMMENT 'Asset classification, e.g. Equity, FX, Commodity',
      asset_code        STRING  COMMENT 'Ticker / instrument code (unique per asset class)',
      source            STRING  COMMENT 'Data source identifier (feed name or provider)',
      as_of_date        DATE    COMMENT 'Snapshot date (UTC date), used for partition pruning',
      as_of_timestamp   TIMESTAMP COMMENT 'Ingestion/snapshot timestamp in UTC',
      timestamp         TIMESTAMP COMMENT 'Event timestamp (UTC) for the price observation',
      start_timestamp   TIMESTAMP COMMENT 'Start of the sampling interval (== timestamp)',
      end_timestamp     TIMESTAMP COMMENT 'End of the sampling interval (== timestamp + 1 hour)',
      sampling          INT     COMMENT 'Sampling interval in seconds (should be 3600)',
      price             DECIMAL(38,12) COMMENT 'Price value (DECIMAL for financial exactness)',
      currency          STRING  COMMENT 'ISO currency code of the price (e.g. USD)',
      quantity_unit     STRING  COMMENT 'Unit of quantity (e.g. barrels, tonnes, shares)'
    )
    USING DELTA
    PARTITIONED BY (as_of_date)
    COMMENT 'Hourly pricing snapshot table: one row per asset timestamp. as_of_timestamp in UTC.'
    TBLPROPERTIES (
      'delta.autoOptimize.optimizeWrite' = 'true',
      'delta.autoOptimize.autoCompact' = 'true',
      'delta.minReaderVersion' = '2',
      'delta.minWriterVersion' = '7'
    );
    ```

    And will also run:

    ```sql
    ANALYZE TABLE trading.ba_3mv_polaris__p__volcano_ref_input.asset_forecasts
    COMPUTE STATISTICS FOR COLUMNS asset_code, asset_class, as_of_date;
    ```
"""

from .delta_reader import DeltaDataReader
from .delta_writer import DeltaDataWriter

__all__ = ["DeltaDataReader", "DeltaDataWriter"]