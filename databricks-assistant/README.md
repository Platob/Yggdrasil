# Databricks Assistant config for `ygg[data,databricks]`

Instruction files and Skills that teach the Databricks Assistant (Genie Code)
how to write idiomatic code against [Yggdrasil](https://github.com/Platob/Yggdrasil)
(`pip install "ygg[data,databricks]"`).

## What's here

| File | Where it goes in the Assistant settings panel |
| --- | --- |
| [`.assistant_workspace_instructions.md`](.assistant_workspace_instructions.md) | **Workspace instructions** — admin-managed, loaded for every user in the workspace. |
| [`user_instructions.md`](user_instructions.md) | **User instructions** — personal file an individual user can paste into "Add instructions file". |
| [`skills/`](skills/) | **Skills folder** — markdown skills the Assistant can route to per task. Each file is a self-contained skill. |

## Installing into a Databricks workspace

The Assistant configuration UI exposes three slots (see the workspace's
**Assistant settings** sidebar):

1. **Workspace instructions** — copy `.assistant_workspace_instructions.md`
   into the workspace root (a workspace admin can do this from the settings
   panel via "Edit" on the `.assistant_workspace_instructions.md` slot).
2. **User instructions** — each user clicks **Add instructions file** and
   pastes the contents of `user_instructions.md`.
3. **Skills folder** — click **Create skills folder**, then upload the
   files from [`skills/`](skills/) into it. Workspace admins can use the
   **Create workspace skills folder** slot to publish them to every user.

The Assistant picks skills by matching the task against each skill's
"When to use" section, so the filename and that section are what matter
most for routing.

## Skills inventory

| Skill | Covers |
| --- | --- |
| [`ygg-install`](skills/ygg-install.md) | `%pip install "ygg[data,databricks]"`, version pinning, env vars |
| [`ygg-databricks-client`](skills/ygg-databricks-client.md) | `DatabricksClient`, auth, services, resource singletons |
| [`ygg-databricks-sql`](skills/ygg-databricks-sql.md) | `dbc.sql.execute(...)`, parameter binding, warehouse vs cluster routing |
| [`ygg-databricks-tables`](skills/ygg-databricks-tables.md) | `Table.create / insert / async_insert / merge / delete_where` |
| [`ygg-databricks-files`](skills/ygg-databricks-files.md) | `DatabricksPath`, DBFS / Volume / Workspace IO |
| [`ygg-databricks-jobs`](skills/ygg-databricks-jobs.md) | Run / wait on jobs, secrets, clusters, warehouses, `WaitingConfig` |
| [`ygg-databricks-job-workflows`](skills/ygg-databricks-job-workflows.md) | `dbc.jobs.create_or_update`, `JobTask.from_callable`, cron / file-arrival schedules, multi-task DAGs |
| [`ygg-databricks-genie`](skills/ygg-databricks-genie.md) | `dbc.genie.ask`, `GenieSpace`, `GenieConversation` |
| [`ygg-ingestion-pipeline`](skills/ygg-ingestion-pipeline.md) | End-to-end recipe: HTTP / API / S3 → discover → cast → Unity Catalog → schedule |
| [`ygg-schema-discovery`](skills/ygg-schema-discovery.md) | Sample an unknown endpoint, infer + tighten a `Schema`, validate against fresh data |
| [`ygg-data-modeling`](skills/ygg-data-modeling.md) | Schema-per-source, `raw_<entity>` + provenance, PK / FK / partition via `Field` metadata, cross-source joins via shared ISO dims |
| [`ygg-curated-views`](skills/ygg-curated-views.md) | UTC timestamps, decimal money, ISO currency / country / language / timezone, naming, table vs view |
| [`ygg-trading-commodity`](skills/ygg-trading-commodity.md) | Trading / commodity / energy market data: MIC codes, ENTSO-E EIC, OHLCV bars, contracts, FX, idempotency for corrections / settlements |
| [`ygg-mlops`](skills/ygg-mlops.md) | MLflow on Databricks + UC registry, `ml_<task>_features` / `_predictions` / `_runs` tables, autoML candidate discovery, drift detection |
| [`ygg-cast`](skills/ygg-cast.md) | `convert(value, target)`, `CastOptions`, registry extension |
| [`ygg-schema-fields`](skills/ygg-schema-fields.md) | `Field` / `Schema` / `DataType`, schema intent |
| [`ygg-statement-result`](skills/ygg-statement-result.md) | `StatementResult` / `Tabular` / `DataTable` consumption, streaming |
| [`ygg-enums`](skills/ygg-enums.md) | `ByteUnit`, `Currency`, `MimeType`, `TimeZone`, … |
| [`ygg-json-pickle`](skills/ygg-json-pickle.md) | `yggdrasil.pickle.json`, `serde`, singleton-by-config pickling |
| [`ygg-http`](skills/ygg-http.md) | `HTTPSession`, `HTTPRequest`, `HTTPResponse`, `URL`, retries / caching |
| [`ygg-benchmarks`](skills/ygg-benchmarks.md) | `python/benchmarks/`, before/after workflow, `run_all.py`, picking the right metric |
| [`ygg-logging`](skills/ygg-logging.md) | `<Verb> <ResourceNoun> %r (...)`, `%r` lazy logging, anti-patterns |
| [`ygg-pitfalls`](skills/ygg-pitfalls.md) | Post-generation checklist — row loops, bare imports, pre-checks, etc. |

## Autonomous ingestion workflow

The skills are organised so a prompt like *"ingest this API into
`main.sales.orders` every hour"* (plus a docs URL or sample payload)
can be answered end-to-end without further questions:

1. [`ygg-schema-discovery`](skills/ygg-schema-discovery.md) — probe
   the source, infer a `Schema`, tighten it, commit the literal.
2. [`ygg-data-modeling`](skills/ygg-data-modeling.md) — pick the
   layout (one schema per source, `raw_<entity>` + provenance,
   PK / FK / partition via `Field` metadata).
3. [`ygg-databricks-tables`](skills/ygg-databricks-tables.md) —
   reconcile catalog / schema / table, `ensure_created(schema=...)`.
4. [`ygg-http`](skills/ygg-http.md) +
   [`ygg-cast`](skills/ygg-cast.md) — pull pages with
   `HTTPSession` (or `SchemaSession` when responses *are* the
   bronze cache), cast through the schema, write via
   `Table.insert / merge / async_insert`.
5. [`ygg-databricks-job-workflows`](skills/ygg-databricks-job-workflows.md)
   — stage the callable via `Job.pytask`, attach a
   `CronSchedule` / `FileArrivalTriggerConfiguration`.
6. [`ygg-curated-views`](skills/ygg-curated-views.md) — standardise
   UTC timestamps, decimal money, ISO codes; expose the curated
   layer that BI / ML read.
7. [`ygg-benchmarks`](skills/ygg-benchmarks.md) — add a bench for the
   hot transform path before merging.

[`ygg-ingestion-pipeline`](skills/ygg-ingestion-pipeline.md) is the
master recipe that chains the seven.

## Keeping these files in sync with the library

Skills reference public surface (`yggdrasil.data.cast.convert`,
`DatabricksClient`, `Volume`, `Table`, `SQLEngine`, …). When that surface
changes in `python/src/yggdrasil/`, update the skill that mentions it. The
canonical style/voice rules live in [`../AGENTS.md`](../AGENTS.md); the
skills here are condensed call-site guidance for end users, not a
replacement for `AGENTS.md`.
