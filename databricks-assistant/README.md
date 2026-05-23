# Databricks Assistant config for `ygg[databricks]`

Instruction files and Skills that teach the Databricks Assistant how
to write idiomatic code against
[Yggdrasil](https://github.com/Platob/Yggdrasil)
(`pip install "ygg[databricks]"`).

## What's here

| File | Where it goes |
| --- | --- |
| [`.assistant_workspace_instructions.md`](.assistant_workspace_instructions.md) | **Workspace instructions** — loaded for every user. |
| [`user_instructions.md`](user_instructions.md) | **User instructions** — personal preferences. |
| [`skills/`](skills/) | **Skills folder** — markdown skills the Assistant routes to per task. |

## Installing into a Databricks workspace

1. **Workspace instructions** — copy
   `.assistant_workspace_instructions.md` into the workspace
   instructions slot (admin panel).
2. **User instructions** — each user pastes `user_instructions.md`
   into "Add instructions file".
3. **Skills folder** — upload the files from `skills/` into the
   workspace or user skills folder.

## Skills inventory

| Skill | Covers |
| --- | --- |
| [`ygg-install`](skills/ygg-install.md) | `%pip install "ygg[databricks]"`, version pinning |
| [`ygg-databricks-client`](skills/ygg-databricks-client.md) | `DatabricksClient`, auth, services, resource singletons |
| [`ygg-spark-tabular`](skills/ygg-spark-tabular.md) | `Dataset` / `SparkTabular` — map, apply, filter, parallelize, to_table |
| [`ygg-databricks-sql`](skills/ygg-databricks-sql.md) | SQL execution, `StatementResult`, table create/insert/merge |
| [`ygg-databricks-files`](skills/ygg-databricks-files.md) | `DatabricksPath`, DBFS, Volumes, Workspace files |
| [`ygg-databricks-jobs`](skills/ygg-databricks-jobs.md) | Jobs, `@task` / `@flow` workflows, schedules, secrets |
| [`ygg-databricks-genie`](skills/ygg-databricks-genie.md) | Genie Q&A, spaces, conversations |

## Keeping in sync

Skills reference the public surface of `yggdrasil`. When that surface
changes in `python/src/yggdrasil/`, update the relevant skill.
