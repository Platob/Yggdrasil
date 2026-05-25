# Module pages

Hand-written, copy-paste-friendly pages for the most-used surfaces. The auto-generated pages live under [API Reference](../api/index.md).

## Core

- [arrow](arrow/README.md) — Arrow type inference and casting helpers.
- [data.cast](types/README.md) — `convert`, `register_converter`, `CastOptions`.
- [engine cast helpers](types/cast/README.md) — `cast_arrow_tabular`, `cast_polars_dataframe`, `cast_pandas_dataframe`, `cast_spark_dataframe`.
- [dataclasses](dataclasses/README.md) — `dataclass_to_arrow_field`, `WaitingConfig`, `ExpiringDict`.

## IO and HTTP

- [io](io/README.md) — `BytesIO`, `URL`, `Memory`, `MemoryStream`, `SendConfig`/`SendManyConfig`.
- [http_](http_/README.md) — `HTTPSession`, prepared requests, batch dispatch.

## Databricks

- [databricks](databricks/README.md) — `DatabricksClient` overview.
- [databricks/sql](databricks/sql/README.md) — SQL execution + Unity Catalog.
- [databricks/compute](databricks/compute/README.md) and [compute/remote](databricks/compute/remote/README.md).
- [databricks/workspaces](databricks/workspaces/README.md), [databricks/fs](databricks/fs/README.md).
- [databricks/secrets](databricks/secrets/README.md), [databricks/iam](databricks/iam/README.md).
- [databricks/jobs](databricks/jobs/README.md) — `NotebookConfig`, `Job`, `JobTask`, `TaskParameters`.
- [databricks/account](databricks/account/README.md), [databricks/genie](databricks/genie/README.md).
- [databricks/workflow](databricks/workflow/README.md) — `@flow` / `@task` declarative pipelines.
- [databricks/ai](databricks/ai/README.md) — Vector Search endpoints and indexes.
- [databricks/catalog](databricks/catalog/README.md) — Unity Catalog hierarchy + table resources.
- [databricks/volume](databricks/volume/README.md), [databricks/warehouse](databricks/warehouse/README.md), [databricks/cluster](databricks/cluster/README.md).

## Serialization and Encoding

- [pickle + json](pickle/README.md) — binary pickle wire format, orjson-backed JSON.

## Engines

- [polars / pandas / spark](engines/README.md) — engine bridges and cast helpers.

## Storage and Databases

- [mongo](mongo/README.md) — Arrow-native MongoDB.
- [postgres](postgres/README.md) — Arrow-native PostgreSQL (ADBC fast path).
- [aws](aws/README.md) — `AWSClient`, `S3Path`, Arrow filesystem.
- [delta](delta/README.md) — Delta Lake log reader (no Spark required).

## Streaming

- [kafka](kafka/README.md) — Kafka topic as `Tabular`.

## Domain

- [fxrate](fxrate/README.md) — FX rate fetching with multi-source fallback.
- [jwt](jwt/README.md) — JWT parsing primitives (RFC 7519).
- [exceptions](exceptions/README.md) — `YGGException` hierarchy.

## Utilities

- [pyutils](pyutils/README.md) — `retry`, `parallelize`.
- [concurrent](concurrent/README.md) — `Job`, `JobPoolExecutor` with backpressure.
- [environ](environ/README.md) — `PyEnv`, `runtime_import_module`, `SystemParameters`.
- [optional libs](libs/README.md) and [extension helpers](libs/extensions/README.md) — `yggdrasil.lazy_imports`.

## Service

- [fastapi](fastapi/README.md) — FastAPI service backing the Power Query / Excel connector.

> The `deltalake` page is retained only as a [status note](deltalake/README.md) — `yggdrasil.deltalake` is not part of the current package tree.
