# User instructions — `ygg[databricks]`

I work in Python notebooks and jobs on Databricks with
[Yggdrasil](https://github.com/Platob/Yggdrasil) (`pip install "ygg[databricks]"`).

## Preferences

- Install `ygg[databricks]` first; restart Python after `%pip install`.
  Don't use `databricks-sdk` directly — go through `DatabricksClient` and
  its `dbc.<service>` accessors.
- Read data with `dbc.sql.execute(q)` (→ `result.to_polars()` /
  `to_arrow_table()` / `to_pandas()`) or `dbc.dataset(q)` for Spark.
- Write tables with `tbl.insert(data)`; upsert with
  `tbl.insert(data, match_by=["id"])`; create with
  `tbl.ensure_created(schema)`. No `merge` / `async_insert` /
  `delete_where` — those don't exist.
- Files: `dbc.path(uri)` or `DatabricksPath.from_(uri)` — auto-dispatches
  to `VolumePath` / `DBFSPath` / `WorkspacePath`. Use `read_bytes`/
  `write_bytes`, `iterdir`/`ls`, `remove`/`unlink` (no `glob`/`rename`).
- Distributed work: `Dataset` (`map`, `apply`, `filter`, `to_table`) and
  `dbc.parallelize(inputs, fn, schema=...)` (inputs first).
- Schedule with `@task` / `@flow` (from `yggdrasil.databricks.job`) or
  `dbc.jobs.create_or_update(name, tasks=[...])`.
- No row-by-row Python loops over data — vectorise with `pyarrow.compute`,
  Polars, or Spark.
- Use `yggdrasil.pickle.json` instead of stdlib `json`.

## Style

- Short comments only where the WHY is non-obvious.
- Direct and concise — show the snippet, skip the preamble.
- Act autonomously; only ask when genuinely ambiguous.
- Verify method names against the real API before suggesting them. If
  unsure whether something exists, say so rather than inventing it.
