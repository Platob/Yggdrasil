# Module reference overview

Use this page as a map of the Python submodules shipped with Yggdrasil. Each section highlights when to use the component and points to the detailed page in `docs/modules/`.

## `yggdrasil.dataclasses`
Enhanced dataclasses with safe initialization, default handling, and Arrow schema generation. Ideal for defining typed payloads that must travel across engines.
- [Detailed doc](modules/dataclasses/README.md)

## `yggdrasil.types`
Central casting registry, Arrow inference from type hints, and sensible defaults. Use when normalizing data across Python/Arrow/Polars/pandas/Spark.
- [Detailed doc](modules/types/README.md)

## `yggdrasil.libs`
Optional bridges to external dataframe libraries plus dependency guards. Helpful for converting Arrow schemas to Polars/pandas/Spark dtypes or gating optional imports.
- [Detailed doc](modules/libs/README.md)

## `yggdrasil.requests`
HTTP utilities with retries and Azure MSAL authentication. Use for resilient API calls and service-to-service auth.
- [Detailed doc](modules/requests/README.md)

## `yggdrasil.databricks`
Wrappers around the Databricks SDK for workspaces, SQL, jobs, and compute management. Use for table ingestion, query execution, and cluster automation.
- [Detailed doc](modules/databricks/README.md)

## `yggdrasil.pyutils`
Concurrency and retry decorators that underpin the package utilities. Use to parallelize work or add backoff to unreliable operations.
- [Detailed doc](modules/pyutils/README.md)

## `yggdrasil.ser`
Serialization helpers and dependency inspection utilities. Use when you need to extract function source or validate optional dependencies.
- [Detailed doc](modules/ser/README.md)
