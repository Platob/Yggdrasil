# yggdrasil.databricks.sql

Unified SQL execution for Databricks with typed result wrappers, Arrow-first conversions, and convenience helpers for warehouses, catalogs, schemas, tables, and columns.

## Recommended one-liner

```python
from yggdrasil.databricks import DatabricksClient

print(DatabricksClient().sql.execute("SELECT 1 AS value").to_polars())
```

## SQL execution features

```python
from yggdrasil.databricks import DatabricksClient

sql = DatabricksClient(host="https://<workspace>", token="<token>").sql
```

- Execute ad-hoc SQL: `stmt = sql.execute("SELECT current_timestamp() AS ts")`
- Wait + error handling: `stmt.wait().raise_for_status()`
- Arrow-first consumption: `table = stmt.to_arrow_table()`
- pandas/Polars conversion: `df = stmt.to_pandas(); lf = stmt.to_polars()`
- Spark conversion: `spark_df = stmt.to_spark()`
- Use explicit context and fully-qualified names like `main.default.table`

## End-to-end write + read-back examples (pyarrow, pandas, polars, pyspark, unstructured)

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.arrow.lib import pyarrow as pa
from yggdrasil.pandas.lib import pandas as pd
from yggdrasil.polars.lib import polars as pl

client = DatabricksClient(host="https://<workspace>", token="<token>")
sql = client.sql
spark = client.spark_connect()  # Spark Connect session

table_name = "main.default.demo_ingest_all_formats"

# 1) Create a target Delta table (all examples write into this same table)
sql.execute(f"""
CREATE TABLE IF NOT EXISTS {table_name} (
  id BIGINT,
  source STRING,
  payload STRING
) USING DELTA
""")

# Optional clean slate
sql.execute(f"DELETE FROM {table_name}")

# 2) Write from pyarrow
arrow_table = pa.table({
    "id": [1],
    "source": ["pyarrow"],
    "payload": ['{"k":"v-arrow"}'],
})
sql.arrow_insert_into(table_name, arrow_table)

# 3) Write from pandas
pandas_df = pd.DataFrame(
    [{"id": 2, "source": "pandas", "payload": '{"k":"v-pandas"}'}]
)
sql.insert_into(table_name, pandas_df)

# 4) Write from polars
polars_df = pl.DataFrame(
    {"id": [3], "source": ["polars"], "payload": ['{"k":"v-polars"}']}
)
sql.insert_into(table_name, polars_df)

# 5) Write from pyspark DataFrame
spark_df = spark.createDataFrame(
    [{"id": 4, "source": "pyspark", "payload": '{"k":"v-spark"}'}]
)
sql.spark_insert_into(table_name, spark_df)

# 6) Write unstructured records (plain Python dict/list payload)
unstructured_rows = [
    {"id": 5, "source": "unstructured", "payload": "raw note: hello world"},
    {"id": 6, "source": "unstructured", "payload": '{"freeform": [1,2,3]}'},
]
sql.insert_into(table_name, unstructured_rows)

# 7) Read back once, then project to each format
stmt = sql.execute(f"SELECT * FROM {table_name} ORDER BY id")

as_arrow = stmt.to_arrow_table()
as_pandas = stmt.to_pandas()
as_polars_lazy = stmt.to_polars()         # LazyFrame
as_polars_df = stmt.to_polars(stream=False)  # DataFrame
as_spark = stmt.to_spark(spark=spark)
as_pylist = stmt.to_pylist()              # unstructured Python list[dict]

print(as_arrow)
print(as_pandas)
print(as_polars_df)
as_spark.show(truncate=False)
print(as_pylist)
```

### Why this pattern works

- You can ingest different producer formats into one Delta target table.
- You can fan out one SQL result into Arrow/pandas/Polars/Spark/Python-native outputs.
- The unstructured path (`list[dict]` and freeform string payloads) is useful for logs, notes, and semi-structured ingestion before normalization.

## Table DDL/DML shortcuts

- Build a typed table handle: `orders = sql.table("main.sales.orders")`
- Create table from schema: `sql.create_table("main.sales.orders", schema=arrow_schema)`
- Insert rows (generic): `sql.insert_into("main.sales.orders", [{"id": 1, "amount": 10.5}])`
- Insert Arrow table: `sql.arrow_insert_into("main.sales.orders", arrow_table)`
- Insert Spark DataFrame: `sql.spark_insert_into("main.sales.orders", spark_df)`
- Drop table safely: `sql.drop_table("main.sales.orders", if_exists=True)`

## Warehouse management

```python
from yggdrasil.databricks import DatabricksClient

wh = DatabricksClient().warehouses.find_default()
```

- Check state: `wh.state`
- Start / stop: `wh.start().wait_for_status("RUNNING")`; `wh.stop()`
- Execute with warehouse binding: `wh.execute("SELECT current_catalog()")`
- Find named warehouse: `DatabricksClient().warehouses.find_warehouse("analytics")`
- Create/update warehouse: `DatabricksClient().warehouses.create_or_update(name="analytics", cluster_size="Small")`

## Catalog, schema, and table navigation

```python
from yggdrasil.databricks import DatabricksClient

catalogs = DatabricksClient().catalogs
```

- Catalog lookup: `main = catalogs.catalog("main")`
- Schema lookup: `sales = catalogs.schema("main.sales")`
- Table lookup: `orders = catalogs.table("main.sales.orders")`
- Dict-style pathing: `orders = catalogs["main"]["sales"]["orders"]`
- Remote discovery: `found = DatabricksClient().tables.find_table("main.sales.orders")`
- List tables: `list(DatabricksClient().tables.list_tables("main.sales"))`
