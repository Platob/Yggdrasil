# yggdrasil.databricks.volume

Unity Catalog Volume resource — singleton-cached per `(catalog, schema, volume)`, TTL-aware metadata, credential vending, and path resolution into S3 / ADLS / GCS storage.

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

vol = DatabricksClient().volumes["main.raw.landing"]
print(vol.storage_location())
```

## Find a volume

```python
from yggdrasil.databricks import DatabricksClient

client  = DatabricksClient()
volumes = client.volumes

# By three-part name
vol = volumes["main.raw.landing"]
# or equivalently
vol = volumes.volume("main", "raw", "landing")

# Iterate all volumes in a schema
for v in volumes.list("main.raw"):
    print(v.full_name(), v.volume_type)
```

## Metadata

```python
vol = DatabricksClient().volumes["main.raw.landing"]

print(vol.full_name())           # "main.raw.landing"
print(vol.name)                  # "landing"
print(vol.catalog_name)          # "main"
print(vol.schema_name)           # "raw"
print(vol.volume_type)           # "MANAGED" or "EXTERNAL"
print(vol.owner)
print(vol.comment)
print(vol.storage_location())    # s3://bucket/path or abfss://...
print(vol.volume_id)
print(vol.exists)
print(vol.explore_url)           # Databricks UI link
```

## Create / ensure

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
vol    = client.volumes["main.raw.landing"]

# Create if it doesn't exist
vol.create(missing_ok=True, comment="Raw landing area")

# Ensure with a single call (idempotent)
vol.ensure_created(comment="Raw landing area")

print(vol.exists)
```

## Delete

```python
vol.delete(if_exists=True)
```

## Path access (files on the volume)

`Volume.path(...)` returns a `VolumePath` — the same path type used throughout `yggdrasil.databricks.fs`:

```python
vol  = DatabricksClient().volumes["main.raw.landing"]
path = vol.path("events/2026/05/21/batch.parquet")

# Read / write via the path interface
path.write_bytes(b"hello")
content = path.read_bytes()

# List
for p in path.parent.iterdir():
    print(p)

# Upload a local file
import pathlib
path.upload(pathlib.Path("/tmp/local_batch.parquet"))
```

## Arrow filesystem

For bulk Arrow I/O (Parquet, Feather, CSV) the volume exposes an Arrow filesystem:

```python
import pyarrow.parquet as pq
import pyarrow as pa

vol = DatabricksClient().volumes["main.raw.landing"]
fs  = vol.arrow_filesystem()

# Write Parquet to the volume
table = pa.table({"id": [1, 2, 3], "v": [4.0, 5.0, 6.0]})
pq.write_table(table, "events/batch.parquet", filesystem=fs)

# Read back
tbl = pq.read_table("events/batch.parquet", filesystem=fs)
```

## AWS S3 credentials (credential vending)

For external volumes backed by S3, the volume vends temporary credentials that expire and auto-refresh:

```python
from yggdrasil.data.enums import Mode

vol   = DatabricksClient().volumes["main.external.s3data"]
creds = vol.temporary_credentials(mode=Mode.READ)

print(creds.access_key_id)
print(creds.secret_access_key)
print(creds.session_token)

# Or use the auto-refreshing AWS filesystem directly
fs = vol.aws(mode=Mode.READ)
# Then pass `fs` to any Arrow/pyarrow I/O call
```

## High-volume Parquet ingest pattern

```python
from yggdrasil.databricks import DatabricksClient
import pyarrow as pa
import pyarrow.parquet as pq

client = DatabricksClient()
vol    = client.volumes["main.raw.landing"]
table  = client.catalogs["main"]["raw"]["events"]

# 1. Write Parquet shards to the volume
fs = vol.arrow_filesystem()
for i, batch in enumerate(produce_batches()):
    pq.write_table(batch, f"events/shard_{i:04d}.parquet", filesystem=fs)

# 2. COPY INTO from the volume
client.sql.execute(f"""
COPY INTO main.raw.events
FROM '{vol.storage_location()}/events/'
FILEFORMAT = PARQUET
""").wait().raise_for_status()
```
