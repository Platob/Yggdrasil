# Module index

This index reflects the current package layout under [`python/src/yggdrasil`](https://github.com/Platob/Yggdrasil/tree/main/python/src/yggdrasil).

## Core data and schema

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.data` | Cast registry, `CastOptions`, `DataType`, `Field`/`Schema`, `DataTable`, normalized enums | [types](modules/types/README.md) |
| `yggdrasil.data.cast` | Converter registry, dispatch, options | [types](modules/types/README.md) |
| `yggdrasil.arrow` | Arrow type inference and casting helpers | [arrow](modules/arrow/README.md) |
| `yggdrasil.dataclasses` | Dataclass → Arrow field, waiting/expiring utilities | [dataclasses](modules/dataclasses/README.md) |

## Dataframe engines

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.polars` | Polars bridge: `cast.py`, `lib.py`, `tests.py` | [engine cast helpers](modules/types/cast/README.md) |
| `yggdrasil.pandas` | pandas bridge | [engine cast helpers](modules/types/cast/README.md) |
| `yggdrasil.spark`  | Spark bridge | [engine cast helpers](modules/types/cast/README.md) |

Engines register their converters **on import** — pull them in once at startup if you want a specific cast to fire.

## IO and transport

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.io` | `BytesIO`, `URL`, `SendConfig`/`SendManyConfig`, codecs, media types | [io](modules/io/README.md) |
| `yggdrasil.io.http_` | `HTTPSession` (preferred client) | [http_](modules/io/http_/README.md) |

## Databricks

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.databricks` | `DatabricksClient` entry point | [databricks](modules/databricks/README.md) |
| `yggdrasil.databricks.sql` | SQL execution + Unity Catalog | [databricks/sql](modules/databricks/sql/README.md) |
| `yggdrasil.databricks.compute` | Cluster lifecycle + remote execution | [databricks/compute](modules/databricks/compute/README.md), [remote](modules/databricks/compute/remote/README.md) |
| `yggdrasil.databricks.workspaces` | Workspace + path helpers | [databricks/workspaces](modules/databricks/workspaces/README.md) |
| `yggdrasil.databricks.fs` | DBFS / Volume / Workspace files | [databricks/fs](modules/databricks/fs/README.md) |
| `yggdrasil.databricks.secrets` | Scope/secret helpers | [databricks/secrets](modules/databricks/secrets/README.md) |
| `yggdrasil.databricks.iam` | Users + groups (workspace/account) | [databricks/iam](modules/databricks/iam/README.md) |
| `yggdrasil.databricks.jobs` | Typed `NotebookConfig` widgets | [databricks/jobs](modules/databricks/jobs/README.md) |
| `yggdrasil.databricks.account` | Account-level service | [databricks/account](modules/databricks/account/README.md) |
| `yggdrasil.databricks.genie` | Conversational analytics | [databricks/genie](modules/databricks/genie/README.md) |

## Platform and utilities

| Module | Purpose | Page |
|---|---|---|
| `yggdrasil.pyutils` | `retry`, `parallelize` | [pyutils](modules/pyutils/README.md) |
| `yggdrasil.concurrent` | `Job`, `JobPoolExecutor` | [concurrent](modules/concurrent/README.md) |
| `yggdrasil.environ` | Runtime import / install logic | [optional libs](modules/libs/README.md) |
| `yggdrasil.fastapi` | FastAPI service powering the Power Query connector | [API Reference](api/index.md) |
| `yggdrasil.pickle` | Custom serialization (cloudpickle/dill/zstandard) | — |
| `yggdrasil.blake3` / `yggdrasil.xxhash` | Optional hashing | — |
| `yggdrasil.mongo` / `yggdrasil.mongoengine` | Mongo helpers | — |
| `yggdrasil.fxrates` | FX-rate helpers | — |

## See also

- [Module pages index](modules/README.md) — all curated module pages.
- [API Reference](api/index.md) — auto-generated from the source tree.
