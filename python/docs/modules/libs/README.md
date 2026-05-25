# Optional dependency guards — `yggdrasil.lazy_imports`

`yggdrasil.lazy_imports` is the single guard module for every optional dependency in the library. All optional packages are imported through here — not via bare `import polars` calls — so a base install that lacks the engine still gets a clean, actionable `ImportError` instead of a silent failure or a cryptic `ModuleNotFoundError` three frames deep.

## Import patterns

Two equivalent shapes:

```python
# 1. Attribute access — identical to `import polars` but deferred
from yggdrasil.lazy_imports import polars
df = polars.DataFrame({"a": [1, 2, 3]})

# 2. Function call — same result, reads more explicitly in long import graphs
from yggdrasil.lazy_imports import polars_module
polars = polars_module()
```

Both resolve lazily on first touch and cache the result. Multiple calls return the same module object.

## Available lazy attributes (third-party packages)

| Attribute | Install extra | Notes |
| --- | --- | --- |
| `polars` | core dep | Always available in standard installs |
| `pandas` | `pip install pandas` | Also via `ygg[bigdata]` |
| `pyarrow` | core dep | Hard runtime dependency — always present |
| `fastapi` | `pip install "ygg[api]"` | FastAPI service |
| `requests` | `pip install requests` | Stdlib fallback available |
| `xxhash` | core dep | Required for URL/payload hashing |
| `confluent_kafka` | `pip install "ygg[kafka]"` | Kafka producer/consumer |
| `databricks_sdk` | `pip install "ygg[databricks]"` | Databricks SDK |
| `AccountClient` | `ygg[databricks]` | Account-scoped Databricks client class |
| `WorkspaceClient` | `ygg[databricks]` | Workspace-scoped Databricks client class |
| `Config` | `ygg[databricks]` | Databricks SDK `Config` dataclass |
| `DatabricksError` | `ygg[databricks]` | Root Databricks SDK exception |
| `psycopg` | `pip install "ygg[postgres]"` | psycopg3 (PostgreSQL) |
| `adbc_dbapi` | `pip install "ygg[postgres]"` | ADBC Arrow-native Postgres path |
| `pymongo` | `pip install "ygg[mongo]"` | pymongo |
| `bson` | `pip install "ygg[mongo]"` | bson (ships with pymongo) |
| `pymongoarrow` | `pip install "ygg[mongo]"` | Arrow-native MongoDB path |
| `sqlglot` | `pip install "ygg[sql]"` | SQL parser |

```python
from yggdrasil.lazy_imports import polars, pandas, confluent_kafka, sqlglot
```

## Probe helpers (non-raising)

Use these when you need conditional logic without a try/except:

```python
from yggdrasil.lazy_imports import (
    has_polars,
    has_pymongo,
    has_pymongoarrow,
    has_psycopg,
    has_adbc,
    has_sqlglot,
)

if has_polars():
    from yggdrasil.lazy_imports import polars
    df = polars.DataFrame({"x": [1, 2, 3]})

if has_pymongo():
    from yggdrasil.lazy_imports import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017")
```

## Install on first touch

The `install=True` flag tells the guard to `pip install` the package if it is not found. Use only in scripts or automation — never in library code:

```python
from yggdrasil.lazy_imports import polars_module, pandas_module

# Script / notebook: auto-install if missing
polars = polars_module(install=True)
pandas = pandas_module(install=True)
```

## Internal class loaders (always available)

These loaders pull internal yggdrasil classes without triggering circular imports. They are used by the cast registry and other internal layers — you rarely need them directly, but they exist and are cached:

```python
from yggdrasil.lazy_imports import (
    bytes_io_class,        # → BytesIO
    io_class,              # → IO (base buffer class)
    field_class,           # → Field
    schema_class,          # → Schema
    tabular_io_class,      # → Tabular
    databricks_path_class, # → DatabricksPath
    databricks_client_class, # → DatabricksClient
    aws_client_class,      # → AWSClient
    aws_s3_path_class,     # → S3Path
    path_class,            # → yggdrasil.io.path.Path
    local_path_class,      # → LocalPath
    struct_type_class,     # → StructType
    media_type_class,      # → MediaType
    media_types_class,     # → MediaTypes
    mime_type_class,       # → MimeType
    mime_types_class,      # → MimeTypes
)

BytesIO = bytes_io_class()   # live class
```

## Spark-specific helpers

PySpark 3.5 and earlier expose two parallel `DataFrame` / `Column` class hierarchies (classic and Spark Connect). These helpers return the right tuple for `isinstance` checks regardless of version:

```python
from yggdrasil.lazy_imports import spark_dataframe_classes, spark_column_classes

def is_spark_df(obj) -> bool:
    return isinstance(obj, spark_dataframe_classes())

def is_spark_col(obj) -> bool:
    return isinstance(obj, spark_column_classes())
```

## pyarrow submodule loaders

```python
from yggdrasil.lazy_imports import pyarrow_compute_module, pyarrow_dataset_module

pc = pyarrow_compute_module()   # pyarrow.compute
ds = pyarrow_dataset_module()   # pyarrow.dataset

arr = pc.cast(pc.array([1, 2, 3]), pc.int64())
```

## PATH_SCHEME_FACTORY — path class dispatch

`PATH_SCHEME_FACTORY` maps URL schemes to the right path class loader. Used internally by `URL.resolve()` and path-factory helpers:

```python
from yggdrasil.lazy_imports import PATH_SCHEME_FACTORY

# {"file": local_path_class, "s3": aws_s3_path_class, "dbfs": databricks_path_class, ...}
LocalPath = PATH_SCHEME_FACTORY["file"]()
S3Path    = PATH_SCHEME_FACTORY["s3"]()
```

## Pattern: library code that handles multiple engines

```python
from yggdrasil.lazy_imports import has_polars, has_pymongo, polars_module, pymongo_module

def read_source(uri: str):
    if uri.startswith("mongodb://"):
        if not has_pymongo():
            raise ImportError("Install pymongo: pip install 'ygg[mongo]'")
        client = pymongo_module().MongoClient(uri)
        return client

    if has_polars():
        return polars_module().read_csv(uri)

    raise ValueError(f"No engine available to read {uri!r}")
```
