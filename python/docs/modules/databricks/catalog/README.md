# yggdrasil.databricks.catalog / schema / table / column

Unity Catalog resource singletons. `Catalog`, `Schema` (UCSchema), `Table`, and `Column` wrap the corresponding UC entities — singleton-cached per identity, TTL-aware metadata, DDL helpers, and dict-style navigation.

All four are reachable from `DatabricksClient` without importing the sub-packages directly.

## One-liners

```python
from yggdrasil.databricks import DatabricksClient

catalog = DatabricksClient().catalogs["main"]
schema  = DatabricksClient().catalogs["main"]["default"]
table   = DatabricksClient().catalogs["main"]["default"]["orders"]
```

## Catalog

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
catalogs = client.catalogs

# Lookup
main = catalogs.catalog("main")        # or catalogs["main"]

# Metadata
print(main.full_name())                # "main"
print(main.owner)                      # "admin@example.com"
print(main.comment)                    # optional description
print(main.explore_url)                # Databricks UI URL

# Iterate schemas
for schema in main.schemas():
    print(schema.full_name())

# Create / ensure
from yggdrasil.databricks.catalog import Catalog
cat = Catalog(client=client, name="analytics")
cat.create(if_not_exists=True, comment="Analytics catalog")

# Check existence
print(cat.exists)

# Delete
cat.delete(if_exists=True)
```

## Schema

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
schema = client.catalogs.schema("main.default")   # or client.catalogs["main"]["default"]

# Metadata
print(schema.full_name())         # "main.default"
print(schema.catalog_name)        # "main"
print(schema.owner)

# Iterate tables
for table in schema.tables():
    print(table.full_name())

# Create / ensure
schema.create(if_not_exists=True, comment="Raw landing zone")
print(schema.exists)
schema.delete(if_exists=True)
```

## Table

```python
from yggdrasil.databricks import DatabricksClient
import pyarrow as pa

client = DatabricksClient()

# Lookup
orders = client.catalogs["main"]["sales"]["orders"]
# or
orders = client.tables.find_table("main.sales.orders")

# Metadata
print(orders.full_name())              # "main.sales.orders"
print(orders.catalog_name)            # "main"
print(orders.schema_name)             # "sales"
print(orders.owner)
print(orders.storage_location)        # Delta path
print(orders.data_source_format)      # "DELTA"
print(orders.exists)

# Create from Arrow schema
schema = pa.schema([
    pa.field("id",       pa.int64(),   nullable=False),
    pa.field("amount",   pa.float64()),
    pa.field("placed_at", pa.timestamp("us", tz="UTC")),
])
orders.create(schema=schema, if_not_exists=True, comment="Sales orders")

# Truncate / delete
orders.truncate()
orders.delete(if_exists=True)

# Read Arrow batches (uses SQL under the hood)
for batch in orders.read_arrow_batches():
    print(batch.num_rows)

# Write Arrow table (appends by default)
arrow_table = pa.table({"id": [1], "amount": [99.50], "placed_at": [...]})
orders.write_arrow_table(arrow_table)
```

## Column

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
columns = client.catalogs["main"]["sales"]["orders"].columns()

for col in columns:
    print(col.name, col.type_name, col.nullable)

# Lookup a single column
col = client.columns.find_column("main.sales.orders", "amount")
print(col.comment)
```

## Dict-style navigation

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()

# Full path in one expression
table = client.catalogs["main"]["sales"]["orders"]

# List all tables in a schema
for tbl in client.catalogs["main"]["sales"]:
    print(tbl.full_name())

# Tables service
tables_svc = client.tables
print(list(tables_svc.list_tables("main.sales")))
```

## Async / staged insert (high-volume)

For high-volume writes, `Table` exposes an async staged-insert path that uploads Parquet to a Volume, then issues a `COPY INTO`:

```python
from yggdrasil.databricks import DatabricksClient
import pyarrow as pa

client = DatabricksClient()
table  = client.catalogs["main"]["raw"]["events"]

# Stage + commit in one call (Arrow table)
table.async_insert(arrow_table, wait=True)
```

## Constraints

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
constraints = client.constraints          # TableConstraints service

# Add a primary-key constraint
constraints.add_primary_key("main.sales.orders", columns=["id"])

# Add a foreign-key constraint
constraints.add_foreign_key(
    "main.sales.order_lines",
    columns=["order_id"],
    references="main.sales.orders(id)",
)

# Drop a named constraint
constraints.drop_constraint("main.sales.order_lines", name="fk_order")
```
