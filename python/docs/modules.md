# Yggdrasil Python — module reference

## Core

| Module | What it does |
|---|---|
| [`yggdrasil.arrow`](modules/arrow/README.md) | Infer Arrow fields/schemas from Python type hints |
| [`yggdrasil.data.cast`](modules/types/README.md) | `CastOptions`, `convert`, `register_converter` |
| [`yggdrasil.arrow.cast`](modules/types/cast/README.md) | Table-level casting: Arrow, pandas, Polars, Spark |
| [`yggdrasil.dataclasses`](modules/dataclasses/README.md) | `dataclass_to_arrow_field` |
| [`yggdrasil.pyutils`](modules/pyutils/README.md) | `retry`, `parallelize` |
| [`yggdrasil.concurrent`](modules/concurrent/README.md) | `JobPoolExecutor`, `Job` |
| [`yggdrasil.requests`](modules/requests/README.md) | `YGGSession` — retry-enabled HTTP |
| [`yggdrasil.io`](modules/io/README.md) | `BytesIO`, `Codec`, `MediaType` |
| [`yggdrasil.deltalake`](modules/deltalake/README.md) | `DeltaTable` — no Spark required |

## Databricks

| Module | What it does |
|---|---|
| [`yggdrasil.databricks`](modules/databricks/README.md) | Overview and patterns |
| [`yggdrasil.databricks.workspaces`](modules/databricks/workspaces/README.md) | Paths + files across DBFS / Workspace / Volumes |
| [`yggdrasil.databricks.sql`](modules/databricks/sql/README.md) | `SQLEngine`, `StatementResult` |
| [`yggdrasil.databricks.compute`](modules/databricks/compute/README.md) | `Cluster`, `ExecutionContext` |
| [`yggdrasil.databricks.compute.remote`](modules/databricks/compute/remote/README.md) | `@databricks_remote_compute` |
| [`yggdrasil.databricks.jobs`](modules/databricks/jobs/README.md) | `NotebookConfig` — typed widget params |

## Onboarding path

1. **`data.cast`** → understand `CastOptions` and `convert`
2. **`arrow`** / **`arrow.cast`** → schema inference and table casting
3. **`pyutils`** → wrap IO/network calls with `@retry`, fan-out with `@parallelize`
4. **`databricks.workspaces`** + **`databricks.sql`** → Databricks pipelines
5. **`databricks.jobs`** → typed notebook contracts
6. **`concurrent`** + **`io`** → advanced streaming and buffering
