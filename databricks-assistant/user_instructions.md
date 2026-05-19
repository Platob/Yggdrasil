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
  `from yggdrasil.data import Field, Schema, DataType`. Use the
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
2. **Pick the layout** — one schema per source
   (`main.<source>.raw_<entity>`), provenance columns on every
   raw table (`_ingested_at`, `_source`, `_payload_hash`,
   `_batch_id`), PK / FK / partition flags via `Field` metadata
   (`tags={"primary_key": True}` etc.). See
   [`ygg-data-modeling`](skills/ygg-data-modeling.md).
3. **Reconcile target** via `dbc.catalog(...).ensure_created()` →
   `dbc.schema(...).ensure_created()` → `dbc.table(...).ensure_created(schema=...)`.
4. **Write the fetch-and-load callable**. HTTP sources: pick
   `SchemaSession` when the response cache *is* the raw table
   (per-id GETs, idempotent), plain `HTTPSession` when it's
   parse-then-write (paginated lists, deltas) — the decision tree
   lives in [`ygg-http`](skills/ygg-http.md). S3 / object stores:
   use `DatabricksPath` / `Path`. Local vs remote cache also
   covered there.
5. **Schedule** via `dbc.jobs.create_or_update(name=..., schedule=CronSchedule(...))`
   + `job.pytask(callable, ..., task_key=...).create()` — auto-deps
   resolve via the AST walker, splat `dbc.jobs.userinfo_defaults()`
   for git source / notifications / tags.
6. **Build the curated layer** — standardise UTC timestamps
   (`<col>_utc`), decimal money, ISO codes (`currency_iso`,
   `country_iso`, `region_iso`, `language_iso`, `timezone_iana`),
   `geo_point()` / lat-lon for renderable rows, so cross-source
   joins go through the shared `main.iso.*` dimensions. See
   [`ygg-curated-views`](skills/ygg-curated-views.md).
7. **Build the business-display layer** — wide / pivoted /
   pre-rolled `main.<source>.dash_<view>` tables (`dash_*`) over
   curated. Time-series get pre-aggregated buckets
   (`dash_ohlcv_5m`, `dash_ohlcv_1h`); geo gets inline
   `geo_point` / `boundary_geojson`; KPIs go in a stable
   `(kpi, value, unit, computed_at_utc)` table. Refresh runs as
   a downstream task on the same Job DAG (`depends_on=["curate"]`).
   See [`ygg-display-views`](skills/ygg-display-views.md).
8. **Benchmark** the hot transform path before merging — see
   [`ygg-benchmarks`](skills/ygg-benchmarks.md). Quote before /
   after numbers in the commit body.

If something is genuinely ambiguous (idempotency strategy, whether
to overwrite vs append, secret-scope name), ask once with the
options. Otherwise pick a defensible default and document it in a
short comment, don't stall.

## Plan, think, bench, smoke-test — then ship

Default to acting autonomously, but **prove the work is done before
saying done**:

1. **Plan in writing** before any non-trivial change. 3-7 bullets that
   name the files / functions / schemas affected and the order of
   operations. If you're chaining skills (ingestion + modeling +
   curated + scheduling + MLOps), list them.
2. **Think longer on edge cases.** Schema drift, idempotency, retry
   exhaustion (429 → `ErrorNotifyingHTTPSession`), partial-batch
   failures, FK target existence, decimal precision loss, timezone
   intent, pagination cursors, cron-schedule timezone, model drift.
   I'd rather you list them and dismiss most than ship the one you
   missed.
3. **Benchmark every hot-path change** via `python/benchmarks/`.
   Quote `best` + `median` before/after in the commit body. See
   [`ygg-benchmarks`](skills/ygg-benchmarks.md). "Felt faster"
   doesn't ship — numbers do.
4. **Smoke-test for real.** Unit tests for logic, one-batch live
   run for ingestion, throwaway-table reconciliation for DDL, a
   single training round for ML pipelines. Don't say "complete"
   without proof the code actually executed.
5. **Only when the above is done**, summarise what changed and
   confirm next steps.

Skip the "Should I…?" preambles. The only stopping question is when
two defensible options materially affect the user (idempotency
shape, overwrite semantics, schema versioning). Everything else:
pick, document why in a one-line comment, move on.

## Tone in responses

Direct and concise. State the change, show the snippet, point me to
the relevant module path. Skip "Great question!" preambles.
