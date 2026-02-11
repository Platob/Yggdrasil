# Python modules documentation index

This folder documents the Python modules shipped with Yggdrasil.

If you are building data products on Databricks, start with the Databricks section below. If you are integrating with mixed local + cloud runtimes, combine `types`, `pyutils`, and the Databricks modules.

## Core modules

- [yggdrasil.types](types/README.md)
  Type defaults, Arrow inference, and casting interoperability.
- [yggdrasil.pyutils](pyutils/README.md)
  Retry, waiting, parallel operations, module loading, serialization helpers.
- [yggdrasil.requests](requests/README.md)
  Session wrappers and auth-ready request workflows.

## Databricks modules

- [yggdrasil.databricks](databricks/README.md)
  Databricks integration entrypoint.
- [yggdrasil.databricks.workspaces](databricks/workspaces/README.md)
  Path and file operations across DBFS / Workspace / Volumes.
- [yggdrasil.databricks.sql](databricks/sql/README.md)
  Statement execution, Spark fallback, and structured result handling.
- [yggdrasil.databricks.compute](databricks/compute/README.md)
  Cluster management and command execution contexts.
- [yggdrasil.databricks.compute.remote](databricks/compute/remote/README.md)
  Decorator-based remote execution for local function code.
- [yggdrasil.databricks.jobs](databricks/jobs/README.md)
  Typed config ingestion via widgets and job parameters.

## Optional ecosystem helpers

- [yggdrasil.libs](libs/README.md)
- [yggdrasil.libs.extensions](libs/extensions/README.md)
- [yggdrasil.types.cast](types/cast/README.md)

## Suggested reading order for new users

1. `types`
2. `pyutils`
3. `databricks.workspaces`
4. `databricks.sql`
5. `databricks.jobs`
6. `databricks.compute` / `compute.remote`
