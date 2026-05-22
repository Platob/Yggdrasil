# yggdrasil.postgres

Arrow-native PostgreSQL backend. Tables implement `Tabular` — Arrow / Polars / pandas conversions and the cast registry all work out-of-the-box. Uses ADBC for the Arrow fast path and psycopg3 for the row fallback.

**Optional dependencies:** `psycopg` (psycopg3), `adbc-driver-postgresql`.

## One-liner

```python
from yggdrasil.postgres import PostgresEngine

engine = PostgresEngine(uri="postgresql://user:pass@localhost:5432/mydb")
tbl    = engine.catalog("mydb").schema("public").table("orders")
print(tbl.read_arrow_table())
```

## Connect

```python
from yggdrasil.postgres import PostgresEngine, PostgresConnection, normalize_postgres_uri

# Full URI
engine = PostgresEngine(uri="postgresql://user:pass@host:5432/mydb")

# Connection object
conn = PostgresConnection(uri="postgresql://user:pass@host:5432/mydb")

# Normalize / validate a URI string
uri = normalize_postgres_uri("postgres://localhost/mydb")
```

## Database / schema / table hierarchy

```python
from yggdrasil.postgres import PostgresEngine

engine = PostgresEngine(uri="postgresql://localhost/mydb")

# Navigate the hierarchy
catalog = engine.catalog("mydb")        # PostgresCatalog
schema  = catalog.schema("public")      # PostgresSchema
table   = schema.table("orders")        # PostgresTable

# List
for s in catalog.schemas():
    print(s.name)

for t in schema.tables():
    print(t.full_name())

# Full qualified name
print(table.full_name())   # "mydb.public.orders"
```

## Create a table

```python
import pyarrow as pa
from yggdrasil.postgres import PostgresEngine

engine = PostgresEngine(uri="postgresql://localhost/mydb")
table  = engine.catalog("mydb").schema("public").table("orders")

schema = pa.schema([
    pa.field("id",        pa.int64(),   nullable=False),
    pa.field("customer",  pa.string()),
    pa.field("amount",    pa.float64()),
    pa.field("placed_at", pa.timestamp("us", tz="UTC")),
])
table.create(schema=schema, missing_ok=True)
```

## Read

```python
table = engine.catalog("mydb").schema("public").table("orders")

# Arrow (ADBC fast path)
arrow_tbl = table.read_arrow_table()

# Polars
import polars as pl
df = table.read_polars_frame()

# pandas
import pandas as pd
df = table.read_pandas_frame()

# With a SQL filter pushed down
from yggdrasil.io.tabular import TabularReadOptions
options = TabularReadOptions(filter="amount > 100 AND placed_at > '2026-01-01'")
tbl = table.read_arrow_table(options=options)

# Selected columns only
options = TabularReadOptions(columns=["id", "amount"])
tbl = table.read_arrow_table(options=options)
```

## Write

```python
import pyarrow as pa
from yggdrasil.data.enums import Mode

arrow_tbl = pa.table({
    "id":        [1, 2, 3],
    "customer":  ["Alice", "Bob", "Carol"],
    "amount":    [99.50, 149.00, 79.99],
})

# Append
table.write_arrow_table(arrow_tbl, mode=Mode.APPEND)

# Overwrite (TRUNCATE + insert)
table.write_arrow_table(arrow_tbl, mode=Mode.OVERWRITE)

# Upsert on primary key
table.write_arrow_table(arrow_tbl, mode=Mode.UPSERT, merge_keys=["id"])

# From Polars
import polars as pl
table.write_polars_frame(pl.from_arrow(arrow_tbl))

# From pandas
table.write_pandas_frame(arrow_tbl.to_pandas())
```

## Table lifecycle

```python
table = engine.catalog("mydb").schema("public").table("orders")

print(table.exists)
print(table.columns())     # list[Column]

# Truncate (keep structure, delete rows)
table.truncate()

# Rename
table.rename("orders_archive")

# Set a comment
table.set_comment("Customer orders from the sales system")

# Drop
table.delete(if_exists=True)
```

## Column introspection

```python
for col in table.columns():
    print(col.name, col.type_name, col.nullable, col.default)
```

## SQL utilities

```python
from yggdrasil.postgres import quote_ident, quote_qualified_ident, sql_literal

print(quote_ident("my table"))                    # '"my table"'
print(quote_qualified_ident("public", "orders"))  # '"public"."orders"'
print(sql_literal("O'Brien"))                     # "'O''Brien'"
```

## Type mapping

```python
from yggdrasil.postgres import arrow_to_postgres_type, postgres_to_arrow_type
import pyarrow as pa

pg_type    = arrow_to_postgres_type(pa.float64())  # "double precision"
arrow_type = postgres_to_arrow_type("float8")       # pa.float64()
```

## Full pipeline

```python
from yggdrasil.postgres import PostgresEngine
from yggdrasil.data.enums import Mode
import pyarrow as pa

engine = PostgresEngine(uri="postgresql://localhost/mydb")
table  = engine.catalog("mydb").schema("raw").table("events")

schema = pa.schema([
    pa.field("event_id", pa.string(), nullable=False),
    pa.field("user_id",  pa.int64()),
    pa.field("ts",       pa.timestamp("us", tz="UTC")),
    pa.field("payload",  pa.string()),
])
table.ensure_created(schema=schema)

for batch in fetch_batches():
    table.write_arrow_table(batch, mode=Mode.UPSERT, merge_keys=["event_id"])

print(table.read_polars_frame().shape)
```
