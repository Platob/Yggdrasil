# Skill: trigger and wait on Databricks jobs / secrets / compute

## When to use

The user asks to "run a job", "trigger a workflow", "wait for a job
run", "list job runs", "read a secret", "start / restart / terminate
a cluster", or "list warehouses".

## Jobs

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
job = dbc.job(123456789)          # Job singleton by id
job.run_now(notebook_params={"date": "2026-05-19"})
job.wait()                         # blocks until terminal
job.last_run.result_state          # SUCCESS / FAILED / …
```

`dbc.jobs.list(...)`, `dbc.jobs.find(name="…")` for discovery. Use
the `Job` singleton's own `.run_now(...)` / `.wait(...)` /
`.cancel(...)` rather than `ws.jobs.run_now(...)` — the singleton
wraps the SDK call with project defaults, the `_store_infos` cache,
and the standard retry / waiting policies (`WaitingConfig`).

## Secrets

```python
val = dbc.secrets.get("my-scope", "db-password")
```

Don't print secrets in logs. The library masks them in `__repr__`,
but downstream code is your responsibility.

## Compute / clusters

```python
cluster = dbc.cluster("0123-456789-abcdef")
cluster.start()
cluster.wait_for_state("RUNNING")
cluster.terminate()
```

Use the `Cluster.ensure_running()` helper when "start if stopped,
wait, then proceed" is the intent — it folds the state check + start
+ wait into one call with the right retries.

## Warehouses

```python
wh = dbc.warehouse("my-sql-warehouse")
wh.ensure_running()
wh.execute("SELECT 1")           # convenience pass-through to dbc.sql
```

Warehouse metadata is cached in an `ExpiringDict`; let the cache do
its job and don't hand-roll a second one.

## Waiting / retries

Most lifecycle methods accept a `WaitingConfigArg` keyword
(`timeout=`, `interval=`, etc.) that builds a `WaitingConfig`. Pass
seconds / `timedelta` directly:

```python
job.wait(timeout=600)            # seconds
cluster.wait_for_state("RUNNING", timeout=timedelta(minutes=10))
```

## Don'ts

- Don't sleep-poll a job state in a Python loop; the singleton's
  `.wait(...)` already does exponential backoff with the right
  budget.
- Don't fork a parallel pickled-credentials path to run tasks across
  Spark — `DatabricksClient` is already picklable + singleton-by-
  config across workers.
- Don't shell out to `databricks` CLI from a notebook when a service
  method (`dbc.jobs.run_now`, `dbc.secrets.get`, …) does the same in
  one round trip.
