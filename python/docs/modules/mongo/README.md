# yggdrasil.mongo

Arrow-native MongoDB backend. Surfaces a collection as a `Tabular`, so Arrow / Polars / pandas / Spark conversions and the cast registry all light up out-of-the-box.

**Optional dependencies:** `pymongo` (required), `pymongoarrow` (optional, Arrow fast path).

## One-liner

```python
from yggdrasil.mongo import MongoEngine

coll = MongoEngine().database("mydb").collection("events")
print(coll.read_arrow_table())
```

## Connect

```python
from yggdrasil.mongo import MongoEngine, MongoConnection

# Default: localhost:27017
engine = MongoEngine()

# Custom URI
engine = MongoEngine(uri="mongodb://user:pass@host:27017/mydb?authSource=admin")

# Explicit connection
conn = MongoConnection(uri="mongodb://localhost:27017")
db   = conn.database("mydb")
coll = db.collection("events")
```

## Database and collection hierarchy

```python
from yggdrasil.mongo import MongoEngine

engine = MongoEngine()

# Navigate
db    = engine.database("mydb")
coll  = db.collection("events")

# List databases / collections
for db_name in engine.list_databases():
    print(db_name)

for coll_name in db.list_collections():
    print(coll_name)

# Full name
print(coll.full_name)   # "mydb.events"
```

## Read

```python
from yggdrasil.mongo import MongoEngine

coll = MongoEngine().database("mydb").collection("events")

# Arrow (fastest — uses pymongoarrow if installed)
arrow_table = coll.read_arrow_table()

# Polars
df_polars = coll.read_polars_frame()

# pandas
df_pandas = coll.read_pandas_frame()

# With a filter
tbl = coll.read_arrow_table(filter={"status": "active"})

# Python list of dicts (fallback path)
docs = coll.read_pylist()
```

## Write

```python
import pyarrow as pa
from yggdrasil.mongo import MongoEngine

coll  = MongoEngine().database("mydb").collection("events")
table = pa.table({
    "user_id": [1, 2, 3],
    "action":  ["click", "view", "purchase"],
    "ts":      pa.array([...], type=pa.timestamp("us", tz="UTC")),
})

# Append (insert)
coll.write_arrow_table(table)

# Overwrite (drop + insert)
from yggdrasil.data.enums import Mode
coll.write_arrow_table(table, mode=Mode.OVERWRITE)

# Upsert on a key field
coll.write_arrow_table(table, mode=Mode.UPSERT, merge_keys=["user_id"])

# Write from Polars
import polars as pl
coll.write_polars_frame(pl.DataFrame({"id": [1, 2]}))

# Write from pandas
import pandas as pd
coll.write_pandas_frame(pd.DataFrame({"id": [1, 2]}))
```

## Collection lifecycle

```python
from yggdrasil.mongo import MongoEngine

coll = MongoEngine().database("mydb").collection("events")

# Create with options
coll.create(validator={"$jsonSchema": {"bsonType": "object"}})
coll.ensure_created()    # no-op if already exists

# Count
print(coll.count())
print(coll.count(filter={"status": "active"}))

# Indexes
coll.create_index([("user_id", 1)], unique=True)
coll.create_index([("ts", -1)])
print(coll.indexes())

# Truncate (delete all documents, keep collection + indexes)
coll.truncate()

# Rename
coll.rename("events_archive")

# Drop
coll.delete(if_exists=True)
```

## Infer Arrow schema from documents

```python
from yggdrasil.mongo import MongoEngine, infer_arrow_schema_from_documents

docs   = [{"id": 1, "v": 2.5, "ts": "2026-01-01"}, ...]
schema = infer_arrow_schema_from_documents(docs)
print(schema)
```

## BSON ↔ Arrow type conversion

```python
from yggdrasil.mongo import bson_to_arrow_type, arrow_to_bson_type_name
import pyarrow as pa

arrow_type = bson_to_arrow_type("string")          # pa.string()
bson_name  = arrow_to_bson_type_name(pa.float64()) # "double"
```

## Execute a raw MongoDB command

```python
from yggdrasil.mongo import MongoEngine, MongoCommand

engine = MongoEngine()
cmd    = MongoCommand(database="mydb", command={"ping": 1})
result = engine.execute(cmd)
print(result.to_pylist())
```

## Full ingest pipeline

```python
from yggdrasil.mongo import MongoEngine
from yggdrasil.data.enums import Mode
import pyarrow as pa

engine = MongoEngine(uri="mongodb://localhost:27017")
coll   = engine.database("analytics").collection("page_views")

# Create collection with index
coll.ensure_created()
coll.create_index([("session_id", 1), ("ts", -1)])

# Stream in Arrow batches
for batch in fetch_batches_from_api():
    coll.write_arrow_table(batch, mode=Mode.APPEND)

# Read back and convert
arrow_table = coll.read_arrow_table()
df          = coll.read_polars_frame()
print(df.shape)
```
