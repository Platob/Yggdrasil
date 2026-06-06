# Module pages

Hand-written, copy-paste-friendly pages for the most-used surfaces. The auto-generated pages live under [API Reference](../api/index.md).

## Core

- [arrow](arrow/README.md) — Arrow type inference and casting helpers.
- [data.cast](types/README.md) — `convert`, `register_converter`, `CastOptions`.
- [engine cast helpers](types/cast/README.md) — `cast_arrow_tabular`, `cast_polars_dataframe`, `cast_pandas_dataframe`, `cast_spark_dataframe`.
- [dataclasses](dataclasses/README.md) — `dataclass_to_arrow_field`, `WaitingConfig`, `Expiring`.

## IO and HTTP

- [io](io/README.md) — `URL`, Tabular IO holders, codecs.
- [http_](http_/README.md) — `HTTPSession`, prepared requests, batch dispatch.

## Databricks

- [databricks](databricks/README.md) — `DatabricksClient` overview.
- [databricks/sql](databricks/sql/README.md) — SQL execution + Unity Catalog.
- [databricks/compute](databricks/compute/README.md) and [compute/remote](databricks/compute/remote/README.md).
- [databricks/workspaces](databricks/workspaces/README.md), [databricks/fs](databricks/fs/README.md).
- [databricks/secrets](databricks/secrets/README.md), [databricks/iam](databricks/iam/README.md).
- [databricks/account](databricks/account/README.md).

## Utilities

- [pyutils](pyutils/README.md) — `retry`, `parallelize`.
- [concurrent](concurrent/README.md) — `Job`, `JobPoolExecutor`.
- [optional libs](libs/README.md) and [extension helpers](libs/extensions/README.md).

> The `deltalake` page is retained only as a [status note](deltalake/README.md) — `yggdrasil.deltalake` is not part of the current package tree.
