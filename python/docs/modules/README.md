# Module index

## Core

- [arrow](arrow/README.md) — `arrow_field_from_hint`, Python→Arrow type map
- [data.cast](types/README.md) — `CastOptions`, `convert`, `register_converter`
- [arrow.cast](types/cast/README.md) — `cast_arrow_tabular`, per-engine casting
- [dataclasses](dataclasses/README.md) — `dataclass_to_arrow_field`
- [pyutils](pyutils/README.md) — `retry`, `parallelize`
- [concurrent](concurrent/README.md) — `JobPoolExecutor`, `Job`
- [requests](requests/README.md) — `YGGSession`
- [io](io/README.md) — `BytesIO`, `Codec`, `MediaType`
- [deltalake](deltalake/README.md) — `DeltaTable`

## Databricks

- [databricks](databricks/README.md) — overview
- [databricks.workspaces](databricks/workspaces/README.md) — paths and file IO
- [databricks.sql](databricks/sql/README.md) — SQL execution and results
- [databricks.compute](databricks/compute/README.md) — cluster lifecycle
- [databricks.compute.remote](databricks/compute/remote/README.md) — remote decorator
- [databricks.jobs](databricks/jobs/README.md) — typed notebook config

## Reading order

1. `data.cast` → understand the casting model
2. `arrow` → schema inference from type hints
3. `arrow.cast` → apply schemas to real tables
4. `pyutils` → retries and parallelism
5. `databricks.workspaces` + `databricks.sql` → Databricks integration
6. `databricks.jobs` → typed job parameters
7. `concurrent` / `io` → advanced use cases
