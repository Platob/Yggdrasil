# Databricks Assistant config for `ygg[databricks]`

Instruction files and Skills that teach the Databricks Assistant (the
in-product **Genie** that writes and runs code) to drive
[Yggdrasil](https://github.com/Platob/Yggdrasil) (`pip install
"ygg[databricks]"`, import `yggdrasil`) **in Python on serverless — never
through a terminal/CLI**.

## Where the content lives

The skills and guidance are packaged **inside ygg** so the CLI can deploy
them — they are the single source of truth:

```
python/src/yggdrasil/databricks/assistant/
  workspace_instructions.md     # assistant guidance (workspace-wide)
  user_instructions.md          # per-user preferences
  skills/                       # per-task Skills the Assistant routes to
```

Read them programmatically:

```python
from yggdrasil.databricks import assistant

assistant.workspace_instructions()   # str
assistant.user_instructions()        # str
assistant.skills()                   # {"<skill>.md": markdown, ...}
```

## Deploy into a workspace

```bash
ygg databricks seed            # provisions the workspace AND deploys the bundle
ygg databricks seed --check    # read-only: report what's present
```

`seed`'s **assistant** step uploads the bundle to the workspace and makes a
best-effort attempt at any live Assistant-settings API:

| Artifact | Lands at |
| --- | --- |
| Assistant guidance (workspace) | `/Workspace/Shared/.ygg/assistant/workspace_instructions.md` |
| Workspace skills | `/Workspace/Shared/.ygg/assistant/skills/*.md` |
| User guidance | `/Workspace/Users/<me>/.ygg/assistant/user_instructions.md` |
| User skills | `/Workspace/Users/<me>/.ygg/assistant/skills/*.md` |

Then point the Databricks Assistant at those folders / paste the guidance
into the workspace + user instruction slots (Settings → Assistant).

## Skills inventory

| Skill | Covers |
| --- | --- |
| `ygg-serverless-runtime` | The baseline: **no CLI/`%sh`** on serverless, use the **pre-built ygg image** + default environment, CLI→Python cheatsheet |
| `ygg-install` | `import yggdrasil` first; `%pip install "ygg[databricks]"` into the default env only on `ModuleNotFoundError` |
| `ygg-databricks-client` | `DatabricksClient`, auth, the `dbc.<service>` map, secrets |
| `ygg-databricks-sql` | `dbc.sql.execute` → `StatementResult`, `Table` create/insert/upsert |
| `ygg-databricks-files` | `DatabricksPath`, Volumes, DBFS, Workspace files |
| `ygg-spark-tabular` | `Dataset` / `SparkDataset` — map, apply, filter, parallelize, to_table |
| `ygg-databricks-jobs` | Jobs & runs, `@task` / `@flow`, schedules, secrets |

## Keeping in sync

Skills mirror the public surface of `yggdrasil.databricks`. When that
surface changes in `python/src/yggdrasil/databricks/`, update the matching
skill in the same change. Before documenting a method, confirm it exists
(grep the source) — the cardinal failure mode for an assistant is
confidently suggesting an API that was never implemented.
