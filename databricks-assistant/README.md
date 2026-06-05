# Databricks Assistant config for `ygg[databricks]`

Instruction files and Skills that teach the Databricks Assistant to write
idiomatic code against [Yggdrasil](https://github.com/Platob/Yggdrasil)
(`pip install "ygg[databricks]"`, import `yggdrasil`).

Every example here is verified against the real public API in
`python/src/yggdrasil/databricks/` — not aspirational.

## What's here

| File | Where it goes |
| --- | --- |
| [`.assistant_workspace_instructions.md`](.assistant_workspace_instructions.md) | **Workspace instructions** — loaded for every user. |
| [`user_instructions.md`](user_instructions.md) | **User instructions** — personal preferences. |
| [`skills/`](skills/) | **Skills folder** — markdown skills the Assistant routes to per task. |

## Installing into a Databricks workspace

1. **Workspace instructions** — paste `.assistant_workspace_instructions.md`
   into the workspace instructions slot (admin panel).
2. **User instructions** — each user pastes `user_instructions.md` into
   "Add instructions file".
3. **Skills** — upload the files from `skills/` into the workspace or user
   skills folder.

## Skills inventory

| Skill | Covers |
| --- | --- |
| [`ygg-install`](skills/ygg-install.md) | `%pip install "ygg[databricks]"`, extras, version pinning |
| [`ygg-databricks-client`](skills/ygg-databricks-client.md) | `DatabricksClient`, auth, the `dbc.<service>` map, secrets |
| [`ygg-databricks-sql`](skills/ygg-databricks-sql.md) | `dbc.sql.execute` → `StatementResult`, `Table` create/insert/upsert |
| [`ygg-databricks-files`](skills/ygg-databricks-files.md) | `DatabricksPath`, Volumes, DBFS, Workspace files |
| [`ygg-spark-tabular`](skills/ygg-spark-tabular.md) | `Dataset` / `SparkDataset` — map, apply, filter, parallelize, to_table |
| [`ygg-databricks-jobs`](skills/ygg-databricks-jobs.md) | Jobs & runs, `@task` / `@flow`, schedules, secrets |

> Genie is intentionally **not** covered — `dbc.genie` does not exist in
> this library. Don't add a Genie skill until the API ships.

## Keeping in sync

Skills mirror the public surface of `yggdrasil.databricks`. When that
surface changes in `python/src/yggdrasil/databricks/`, update the matching
skill in the same change. Before documenting a method, confirm it exists
(grep the source) — the cardinal failure mode for an assistant is
confidently suggesting an API that was never implemented.

Quick anchors for the current surface:

- Services: `python/src/yggdrasil/databricks/client.py` (the `dbc.<service>` properties).
- SQL: `databricks/sql/engine.py`; results: `data/statement.py` + `io/tabular/base.py` (the `to_*` aliases).
- Tables: `databricks/table/table.py` (`ensure_created`, `insert`, `insert_into`).
- Files: `databricks/path.py` + `databricks/fs/`.
- Jobs / `@task` / `@flow`: `databricks/job/` (`service.py`, `job.py`, `run.py`, `skeleton.py`).
- Spark Dataset: `spark/tabular.py`.
