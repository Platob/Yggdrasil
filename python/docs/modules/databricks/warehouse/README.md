# yggdrasil.databricks.warehouse

SQL Warehouse lifecycle management — find, start, stop, create, and execute SQL through a warehouse handle.

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

stmt = DatabricksClient().warehouses.find_default().execute("SELECT current_user()")
print(stmt.to_polars())
```

## Find a warehouse

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
wh_svc = client.warehouses

# Default warehouse (first RUNNING or first available)
wh = wh_svc.find_default()

# By name (exact)
wh = wh_svc.find_warehouse("analytics")

# Iterate all warehouses
for wh in wh_svc.list_warehouses():
    print(wh.name, wh.state, wh.id)
```

## Inspect state

```python
wh = DatabricksClient().warehouses.find_default()

print(wh.name)
print(wh.id)
print(wh.state)            # "RUNNING" / "STOPPED" / "STARTING" / ...
print(wh.is_running)
print(wh.is_serverless)
print(wh.details)          # full EndpointInfo object
print(wh.explore_url)      # Databricks UI link
```

## Start / stop

```python
wh = DatabricksClient().warehouses.find_default()

# Start and wait until RUNNING
wh.start().wait_for_status()

# Stop
wh.stop()
```

## Execute SQL

```python
from yggdrasil.databricks import DatabricksClient

wh = DatabricksClient().warehouses.find_default()

# One-shot: execute + consume
stmt = wh.execute("SELECT id, amount FROM main.sales.orders LIMIT 100")
df   = stmt.to_polars()

# Wait and raise on error before consuming
stmt = wh.execute("SELECT 1 AS x")
stmt.wait().raise_for_status()
tbl  = stmt.to_arrow_table()
```

## Create or update a warehouse

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()

wh = client.warehouses.create_or_update(
    name="analytics",
    cluster_size="Small",         # "2X-Small" … "4X-Large"
    min_num_clusters=1,
    max_num_clusters=3,
    auto_stop_mins=30,
    enable_serverless_compute=True,
)
print(wh.id, wh.name)
```

## Delete

```python
wh = DatabricksClient().warehouses.find_warehouse("analytics")
wh.delete()
```

## Warehouse as SQL executor

The `SQLWarehouse` object is callable — pass it as the backing executor to any service that accepts one:

```python
from yggdrasil.databricks import DatabricksClient

client  = DatabricksClient()
wh      = client.warehouses.find_default()

# Run multiple statements through the same warehouse
for sql in [
    "OPTIMIZE main.sales.orders",
    "ANALYZE TABLE main.sales.orders COMPUTE STATISTICS",
]:
    wh.execute(sql).wait().raise_for_status()
```
