# User instructions — Yggdrasil (`ygg[data,databricks]`)

I primarily work in Python notebooks on Databricks against
[Yggdrasil](https://github.com/Platob/Yggdrasil) (PyPI: `ygg`, import:
`yggdrasil`). Tailor suggestions to that stack.

## Preferences

- **Idiomatic stack:** Arrow-first frames via `yggdrasil.data`, then
  Polars/pandas/Spark via the engine bridges. Reach past
  `yggdrasil.data` only when the abstraction genuinely can't cover the
  case.
- **Imports:** `from yggdrasil.databricks import DatabricksClient`,
  `from yggdrasil.data.cast import convert`,
  `from yggdrasil.data import DataField, Schema, DataType`. Use the
  `lib.py` guards (`from yggdrasil.polars.lib import polars`) for
  optional engines.
- **Casting:** Prefer `convert(value, target)` and
  `cast_arrow_tabular(t, CastOptions(target_field=schema))` over
  per-column `.cast()` chains.
- **No row-loops over data.** No `for row in df.iterrows()`, no
  `array.to_pylist()` followed by a comprehension. Vectorise via
  `pyarrow.compute`, Polars expressions, or numpy ufuncs.
- **JSON:** Use `yggdrasil.pickle.json` (orjson-backed) instead of
  stdlib `json` — handles datetime / UUID / Path / Enum / dataclass.
- **Lifecycle ops:** Call the resource singleton's own method
  (`volume.create(...)`, `schema.delete(...)`, `table.read_info()`),
  not `ws.volumes.create(...)` directly — the singleton method wraps
  retries, cache warm-up, and `if_not_exists` / `missing_ok` ergonomics.

## Style

- Short, blunt comments only where the WHY is non-obvious (engine
  edge case, schema invariant, workaround). Skip "loop through fields"
  prose.
- Log lines follow `<Verb> <ResourceNoun> %r (key=value, …)` — use
  `%r` lazy interpolation, not f-strings, in `LOGGER.debug` /
  `LOGGER.info`.
- Type hints match runtime, including `| None` on nullable returns.
- Keyword-only arguments for ambiguous options.

## Autonomy on ingestion tasks

When I paste API docs / a Swagger URL / an S3 bucket / a vendor
sample and say "ingest this", I expect a working pipeline, not
fragments. The skills are wired for that — chain them without
asking permission at each step:

1. **Probe + discover schema** (sample 100–500 rows, infer via
   `Field.from_arrow_schema` / `Field.from_polars_schema`, tighten
   for nullability / decimal / timezone) → commit the `Schema(...)`
   literal to source.
2. **Reconcile target** via `dbc.catalog(...).ensure_created()` →
   `dbc.schema(...).ensure_created()` → `dbc.table(...).ensure_created(schema=...)`.
3. **Write the fetch-and-load callable** using `HTTPSession` for
   HTTP, `DatabricksPath`/`Path` for S3/Volume, and `Table.insert`
   / `merge` / `async_insert` for the write (pick from the size
   table in [`ygg-ingestion-pipeline`](skills/ygg-ingestion-pipeline.md)).
4. **Schedule** via `dbc.jobs.create_or_update(name=..., schedule=CronSchedule(...))`
   + `job.pytask(callable, ..., task_key=...).create()` — auto-deps
   resolve via the AST walker, splat `dbc.jobs.userinfo_defaults()`
   for git source / notifications / tags.
5. **Benchmark** the hot transform path before merging — see
   [`ygg-benchmarks`](skills/ygg-benchmarks.md). Quote before /
   after numbers in the commit body.

If something is genuinely ambiguous (idempotency strategy, whether
to overwrite vs append, secret-scope name), ask once with the
options. Otherwise pick a defensible default and document it in a
short comment, don't stall.

## Tone in responses

Direct and concise. State the change, show the snippet, point me to
the relevant module path. Skip "Great question!" preambles.
