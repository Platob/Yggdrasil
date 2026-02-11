# Yggdrasil Python module reference

This guide is the landing page for the Python package documentation in `yggdrasil`.

Use it when you want to quickly answer:
- **What module should I start with?**
- **Which module is Databricks-first?**
- **Where do I find examples for production integration?**

---

## Quick module map

### Core building blocks
- [`yggdrasil.types`](modules/types/README.md)
  Type normalization and schema inference for Arrow/Pandas/Polars/Spark.
- [`yggdrasil.pyutils`](modules/pyutils/README.md)
  Retry, parallelism, import safety, and runtime utilities.
- [`yggdrasil.requests`](modules/requests/README.md)
  Request sessions and authentication helpers.

### Databricks platform modules
- [`yggdrasil.databricks`](modules/databricks/README.md)
  Main entrypoint for Databricks-oriented orchestration.
- [`yggdrasil.databricks.workspaces`](modules/databricks/workspaces/README.md)
  Unified path/filesystem operations for DBFS, Workspace files, and Volumes.
- [`yggdrasil.databricks.sql`](modules/databricks/sql/README.md)
  SQL execution via Spark or Databricks SQL statement API.
- [`yggdrasil.databricks.compute`](modules/databricks/compute/README.md)
  Cluster lifecycle + command execution.
- [`yggdrasil.databricks.jobs`](modules/databricks/jobs/README.md)
  Typed notebook/job parameter ingestion using widgets.

### Optional helpers
- [`yggdrasil.libs`](modules/libs/README.md)
  Optional dependency guards + extension helpers.
- [`yggdrasil.libs.extensions`](modules/libs/extensions/README.md)
  Extra helpers for dataframe workflows.
- [`yggdrasil.types.cast`](modules/types/cast/README.md)
  Engine-specific casting layer for strict schema enforcement.

---

## Recommended onboarding path

1. Start with `types` to define your expected schemas.
2. Add `pyutils` retry/parallel controls in jobs and services.
3. For Databricks pipelines, integrate `databricks.workspaces` and `databricks.sql` first.
4. Introduce `databricks.compute` only when remote execution from local code is required.
5. Use `databricks.jobs` for reproducible notebook parameter contracts.

---

## Documentation style promise

Each module page includes:
- **What it solves** (when to use / when not to use).
- **Bootstrap snippets** you can copy into notebooks, jobs, CI scripts, and services.
- **Integration patterns** for local + Databricks runtime parity.
