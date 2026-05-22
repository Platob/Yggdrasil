# Module pages

Hand-written, copy-paste-friendly pages for the most-used surfaces. The auto-generated pages live under [API Reference](../api/index.md).

## Core

- [arrow](arrow/README.md) — Arrow type inference and casting helpers.
- [data.cast](types/README.md) — `convert`, `register_converter`, `CastOptions`.
- [engine cast helpers](types/cast/README.md) — `cast_arrow_tabular`, `cast_polars_dataframe`, `cast_pandas_dataframe`, `cast_spark_dataframe`.
- [dataclasses](dataclasses/README.md) — `dataclass_to_arrow_field`, `WaitingConfig`, `ExpiringDict`.

## IO and HTTP

- [io](io/README.md) — `BytesIO`, `URL`, `Memory`, `IOStats` root primitives.
- [io.path](io/path/README.md) — `Path`, `LocalPath`, `RemotePath` — filesystem abstraction.
- [io.tabular](io/tabular/README.md) — `ArrowTabular`, `LazyTabular`, `UnionTabular` — in-memory Tabular.
- [io.primitive](io/primitive/README.md) — `ParquetFile`, `CSVFile`, `NDJSONFile`, `ArrowIPCFile`, `XLSXFile`.
- [io.nested](io/nested/README.md) — `FolderPath`, `ZipFile`, `DeltaFolder`.
- [io.http_](io/http_/README.md) — `HTTPSession`, prepared requests, batch dispatch.

## Databricks

- [databricks](databricks/README.md) — `DatabricksClient` overview.
- [databricks/sql](databricks/sql/README.md) — SQL execution + Unity Catalog.
- [databricks/compute](databricks/compute/README.md) and [compute/remote](databricks/compute/remote/README.md).
- [databricks/workspaces](databricks/workspaces/README.md), [databricks/fs](databricks/fs/README.md).
- [databricks/secrets](databricks/secrets/README.md), [databricks/iam](databricks/iam/README.md).
- [databricks/jobs](databricks/jobs/README.md), [databricks/workflow](databricks/workflow/README.md).
- [databricks/account](databricks/account/README.md), [databricks/genie](databricks/genie/README.md).
- [catalog / schema / table / column](databricks/catalog/README.md), [volume](databricks/volume/README.md).
- [warehouse](databricks/warehouse/README.md), [cluster](databricks/cluster/README.md).
- [ai / Vector Search](databricks/ai/README.md).

## Storage & Databases

- [mongo](mongo/README.md) — Arrow-native MongoDB.
- [mongoengine](mongoengine/README.md) — MongoEngine integration + `with_mongo_connection`.
- [postgres](postgres/README.md) — Arrow-native PostgreSQL.
- [aws](aws/README.md) — `AWSClient`, `S3Path`.
- [delta](delta/README.md) — Delta Lake log reader (no Spark).

## Streaming

- [kafka](kafka/README.md) — `KafkaIO` — Kafka topic as `Tabular`.

## Domain

- [fxrate](fxrate/README.md) — FX rate lookup with multi-source fallback.
- [jwt](jwt/README.md) — `JWTToken` parsing.
- [exceptions](exceptions/README.md) — `YGGException` hierarchy.

## Serialization

- [pickle + json](pickle/README.md) — Binary wire format + orjson-backed JSON layer.

## Engines

- [polars / pandas / spark](engines/README.md) — Engine bridges and TestCase bases.

## Utilities

- [pyutils](pyutils/README.md) — `retry`, `parallelize`.
- [concurrent](concurrent/README.md) — `Job`, `JobPoolExecutor`.
- [environ](environ/README.md) — `PyEnv`, `UserInfo`, `SystemParameters`, `runtime_import_module`.
- [optional libs](libs/README.md) — `lib.py` guard pattern + extra matrix.
- [extension helpers](libs/extensions/README.md).
- [fastapi](fastapi/README.md) — FastAPI service entry point.

> The `deltalake` page is retained only as a [status note](deltalake/README.md) — `yggdrasil.deltalake` is not part of the current package tree.
