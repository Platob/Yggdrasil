# Module reference overview

Use this page as a map of the Python submodules shipped with Yggdrasil. Each section highlights when to use the component and points to the detailed page in `docs/modules/`.

## `yggdrasil.dataclasses`
Enhanced dataclasses with safe initialization, default handling, and Arrow schema generation. Ideal for defining typed payloads that must travel across engines.
- [Detailed doc](modules/dataclasses/README.md)

## `yggdrasil.types`
Central casting registry, Arrow inference from type hints, and sensible defaults. Use when normalizing data across Python/Arrow/Polars/Pandas/Spark.
- [Detailed doc](modules/types/README.md)

## `yggdrasil.libs`
Optional dependency guards plus Spark/Polars helper functions. Helpful for environment checks and Arrow/Spark type conversions.
- [Detailed doc](modules/libs/README.md)

## `yggdrasil.requests`
MSAL-backed request sessions and retry-enabled request utilities.
- [Detailed doc](modules/requests/README.md)

## `yggdrasil.databricks`
Wrappers around the Databricks SDK for workspace, SQL, jobs, and compute management.
- [Detailed doc](modules/databricks/README.md)

## `yggdrasil.pyutils`
Concurrency, retry, environment, and serialization utilities.
- [Detailed doc](modules/pyutils/README.md)
