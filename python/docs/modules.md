# Module index

This index reflects the current package layout under [`python/src/yggdrasil`](https://github.com/Platob/Yggdrasil/tree/main/python/src/yggdrasil).

## Core data and schema

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.data` | Cast registry, `CastOptions`, `DataType`, `Field`/`Schema`, `DataTable`, normalized enums | [types](modules/types/README.md) |
| `yggdrasil.data.cast` | Converter registry, dispatch, options | [types](modules/types/README.md) |
| `yggdrasil.arrow` | Arrow type inference and casting helpers | [arrow](modules/arrow/README.md) |
| `yggdrasil.dataclasses` | Dataclass → Arrow field, `ExpiringDict`, `WaitingConfig`, `Singleton` | [dataclasses](modules/dataclasses/README.md) |
| `yggdrasil.exceptions` | `YGGException` hierarchy — `CastError`, `HTTPError`, and all HTTP status types | [exceptions](modules/exceptions/README.md) |

## Dataframe engines

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.polars` | Polars bridge: cast helpers, lib guard, `PolarsTestCase` | [engines](modules/engines/README.md) |
| `yggdrasil.pandas` | pandas bridge: cast helpers, lib guard, `PandasTestCase` | [engines](modules/engines/README.md) |
| `yggdrasil.spark` | Spark bridge: cast helpers, `SparkTestCase`, Arrow optimizations | [engines](modules/engines/README.md) |

Engines register their converters **on import** — pull them in once at startup if you want a specific cast to fire.

## IO and transport

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.io` | `URL`, `SendConfig`/`SendManyConfig`, codecs, media types | [io](modules/io/README.md) |
| `yggdrasil.http_` | `HTTPSession` (preferred HTTP client) | [http_](modules/http_/README.md) |

## Serialization

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.pickle` | Wire format: `dumps`/`loads`/`dump`/`load`/`serialize`, `Serialized`, `Tags`, codecs | [pickle](modules/pickle/README.md) |
| `yggdrasil.pickle.json` | orjson-backed JSON with datetime/UUID/dataclass/Enum/Decimal/set/Path support | [pickle](modules/pickle/README.md) |

## Databricks

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.databricks` | `DatabricksClient` entry point | [databricks](modules/databricks/README.md) |
| `yggdrasil.databricks.sql` | SQL execution engine + helpers | [databricks/sql](modules/databricks/sql/README.md) |
| `yggdrasil.databricks.catalog` | Unity Catalog `Catalog` resource + `Catalogs` service | [catalog/schema/table/column](modules/databricks/catalog/README.md) |
| `yggdrasil.databricks.schema` | Unity Catalog `Schema` resource + `Schemas` service | [catalog/schema/table/column](modules/databricks/catalog/README.md) |
| `yggdrasil.databricks.table` | Unity Catalog `Table` resource + `Tables` service | [catalog/schema/table/column](modules/databricks/catalog/README.md) |
| `yggdrasil.databricks.column` | Unity Catalog `Column` resource + `Columns` service | [catalog/schema/table/column](modules/databricks/catalog/README.md) |
| `yggdrasil.databricks.volume` | Unity Catalog `Volume` — credentials, storage path, Arrow filesystem | [volume](modules/databricks/volume/README.md) |
| `yggdrasil.databricks.warehouse` | `SQLWarehouse` lifecycle + `Warehouses` service | [warehouse](modules/databricks/warehouse/README.md) |
| `yggdrasil.databricks.compute` | Cluster lifecycle + remote execution | [compute](modules/databricks/compute/README.md), [remote](modules/databricks/compute/remote/README.md) |
| `yggdrasil.databricks.cluster` | `Cluster` + `ServerlessCluster`, execute Python remotely | [cluster](modules/databricks/cluster/README.md) |
| `yggdrasil.databricks.ai` | Vector Search endpoints, indexes, similarity search | [ai](modules/databricks/ai/README.md) |
| `yggdrasil.databricks.workspaces` | Workspace + path helpers | [workspaces](modules/databricks/workspaces/README.md) |
| `yggdrasil.databricks.fs` | DBFS / Volume / Workspace files | [fs](modules/databricks/fs/README.md) |
| `yggdrasil.databricks.secrets` | Scope/secret helpers | [secrets](modules/databricks/secrets/README.md) |
| `yggdrasil.databricks.iam` | Users + groups (workspace/account) | [iam](modules/databricks/iam/README.md) |
| `yggdrasil.databricks.account` | Account-level service | [account](modules/databricks/account/README.md) |
| `yggdrasil.databricks.constraints` | Table constraints (PK, FK) | [catalog/schema/table/column](modules/databricks/catalog/README.md) |

## Storage and databases

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.aws` | `AWSClient` singleton, S3 filesystem, credential management | [aws](modules/aws/README.md) |
| `yggdrasil.delta` | Delta Lake log reader (back-compat shim → `yggdrasil.io.nested.delta`) | [delta](modules/delta/README.md) |

## Domain utilities

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.jwt` | Parsing-only JWT primitives (`JWTToken`, `JWTParseError`) | [jwt](modules/jwt/README.md) |

## Platform and utilities

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.pyutils` | `retry`, `parallelize` | [pyutils](modules/pyutils/README.md) |
| `yggdrasil.concurrent` | `Job`, `AsyncJob`, `ThreadJob`, `JobPoolExecutor` | [concurrent](modules/concurrent/README.md) |
| `yggdrasil.environ` | Runtime import / install logic | [optional libs](modules/libs/README.md) |

## See also

- [Module pages index](modules/README.md) — all curated module pages.
- [API Reference](api/index.md) — auto-generated from the source tree.
