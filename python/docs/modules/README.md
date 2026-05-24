# Module pages

Hand-written, copy-paste-friendly pages for the most-used surfaces. Progress from one-liners to complex use cases in each page. The auto-generated API reference lives under [API Reference](../api/index.md).

## Core data and schema

- [arrow](arrow/README.md) — Arrow type inference, casting, `pyarrow.compute` patterns, streaming IPC, cross-engine tests.
- [data.cast](types/README.md) — `DataType`, `Field`, `Schema`, `convert`, `register_converter`, `CastOptions`, geo types, schema set-ops.
- [engine cast helpers](types/cast/README.md) — `cast_arrow_tabular`, `cast_polars_dataframe`, `cast_pandas_dataframe`, `cast_spark_dataframe`.
- [dataclasses](dataclasses/README.md) — `ExpiringDict`, `WaitingDict`, Arrow-aware `@yggdataclass`, `WaitingConfig`, `Singleton`.

## IO and HTTP

- [io](io/README.md) — `URL` (rich API), `BytesIO`, `SendConfig`/`SendManyConfig`/`CacheConfig`, primitive formats, Delta log reader.
- [http_](http_/README.md) — `HTTPSession`, auth subclassing, retry policy, batch dispatch, streaming, Arrow conversion.

## Serialization

- [pickle + json](pickle/README.md) — `dumps`/`loads`, `yggdrasil.pickle.json`, codec selection, cloudpickle closures.

## Engine bridges

- [Polars / pandas / Spark](engines/README.md) — converter registration, cast helpers, vectorised patterns, `Dataset` API.

## Databricks

- [Overview](databricks/README.md) — `DatabricksClient`, auth, full workflows, schema-driven DDL, sharded inserts.
- [sql](databricks/sql/README.md) — SQL execution, Unity Catalog CRUD.
- [compute](databricks/compute/README.md) — cluster lifecycle, `ExecutionContext`, `@databricks_remote_compute`, fan-out.
- [compute.remote](databricks/compute/remote/README.md) — remote function decorator.
- [workspaces](databricks/workspaces/README.md) — client setup, path helpers, multi-workspace routing.
- [fs](databricks/fs/README.md) — DBFS / Volume / Workspace paths, Arrow FS integration.
- [secrets](databricks/secrets/README.md) — scope/secret management, rotation, bootstrap patterns.
- [iam](databricks/iam/README.md) — user/group management, account-level governance, onboarding.
- [jobs](databricks/jobs/README.md) — `NotebookConfig`, widget init, dependency sniffing.
- [workflow](databricks/workflow/README.md) — `@flow` / `@task`, scheduled pipelines, DAG composition.
- [account](databricks/account/README.md) — account-level IAM, metastores, multi-workspace provisioning.
- [genie](databricks/genie/README.md) — conversational analytics, conversation lifecycle, eval runs.
- [catalog / schema / table / column](databricks/catalog/README.md) — Unity Catalog resource tree.
- [volume](databricks/volume/README.md) — Unity Catalog Volume lifecycle and file operations.
- [warehouse](databricks/warehouse/README.md) — SQL Warehouse lifecycle.
- [cluster](databricks/cluster/README.md) — all-purpose cluster management.
- [ai](databricks/ai/README.md) — Vector Search indexes and similarity search.

## Storage and databases

- [mongo](mongo/README.md) — `MongoEngine`, `MongoCollection`, statements, Arrow conversion.
- [postgres](postgres/README.md) — `PostgresEngine`, Unity Catalog-style resource hierarchy.
- [aws](aws/README.md) — `AWSClient`, S3 paths, Databricks credential refresh.
- [delta](delta/README.md) — Delta Lake log reader (no Spark), snapshots, file manifests.

## Streaming

- [kafka](kafka/README.md) — Kafka producer/consumer patterns.

## Domain

- [fxrate](fxrate/README.md) — FX rate fetching, geo enrichment, scheduled Databricks ingestion.
- [jwt](jwt/README.md) — JWT token creation and verification.
- [exceptions](exceptions/README.md) — `YGGException` hierarchy, HTTP status errors, retry patterns.

## Utilities

- [pyutils](pyutils/README.md) — `@retry` (backoff, jitter, timeout, async), `@parallelize`.
- [concurrent](concurrent/README.md) — `Job`, `JobPoolExecutor`, backpressure, ordered/completion modes.
- [optional libs](libs/README.md) — `lib.py` guard pattern, extras map, converter registration.
- [extension helpers](libs/extensions/README.md) — per-engine Arrow bridging helpers.

> The `deltalake` page is retained as a [status note](deltalake/README.md) — `yggdrasil.io.nested.delta` is the current import path.
