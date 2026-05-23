# User instructions — `ygg[databricks]`

I work in Python notebooks on Databricks with
[Yggdrasil](https://github.com/Platob/Yggdrasil) (`pip install "ygg[databricks]"`).

## Preferences

- Always install `ygg[databricks]` first. Don't use `databricks-sdk`
  directly — go through `DatabricksClient` and its services.
- Use `Dataset` / `SparkTabular` for distributed transforms — `map`,
  `apply`, `filter`, `parallelize`, `to_table`.
- Read data via `dbc.sql.execute(q)` or `dbc.dataset(q)`. Write via
  `tbl.insert()` / `tbl.merge()` / `ds.to_table()`.
- Use `DatabricksPath.from_(path)` for filesystem — auto-dispatches
  to `VolumePath` / `DBFSPath` / `WorkspacePath`.
- Schedule with `@task` / `@flow` or `dbc.jobs.create_or_update()`.
- No row-by-row Python loops. Vectorise with `pyarrow.compute`,
  Polars expressions, or Spark operations.
- Use `yggdrasil.pickle.json` instead of stdlib `json`.

## Style

- Short comments only where the WHY is non-obvious.
- Direct and concise responses. Show the snippet, not the preamble.
- Act autonomously — only ask when genuinely ambiguous.
