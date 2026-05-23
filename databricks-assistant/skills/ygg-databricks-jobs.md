# Skill: create, schedule, and run Databricks jobs

## When to use

The user asks to create a job, schedule a pipeline, run a function as
a Databricks task, set up a cron schedule, or use `@task` / `@flow`.

## Run existing jobs

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
job = dbc.jobs.get(name="my-pipeline")
job.run_now()
job.wait()
job.last_run.result_state  # SUCCESS / FAILED
```

## Create a job with `@task` / `@flow`

```python
from yggdrasil.databricks.workflow import flow, task, secret

@task
def ingest(date: str) -> str:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    (dbc.dataset(f"SELECT * FROM vendor.raw WHERE date = '{date}'")
     .map(clean)
     .to_table("main.curated.events"))
    return "main.curated.events"

@task(retries=2)
def load(path: str, api_key: str = secret("vendor", "api-key")) -> None:
    ...

@flow(name="daily-ingest", schedule="0 2 * * *")
def daily(date: str = "2025-01-01"):
    p = ingest(date)
    load(p)

daily.deploy()                  # upsert Databricks Job
daily.run(date="2026-05-23")    # trigger it
```

- `@task` runs unchanged locally (for testing) and as a Databricks
  task inside a `@flow` trace.
- `@flow.deploy()` traces the body, stages tasks, upserts the Job.
- `secret("scope", "key")` resolves at runtime — cleartext never
  lives on disk.

## Stage a callable directly (no decorator)

```python
job = dbc.jobs.create_or_update(
    name="orders_etl",
    tasks=[],
    **dbc.jobs.userinfo_defaults(),   # git source, notifications, tags
)

# Single task
job.pytask(ingest_fn, "2026-01-01", task_key="ingest").create()

# Multi-task DAG
extract = job.pytask(extract_fn, task_key="extract").create()
transform = job.pytask(
    transform_fn,
    task_key="transform",
    depends_on=["extract"],
).create()
load = job.pytask(load_fn, task_key="load", depends_on=["transform"]).create()
```

`pytask` extracts the function source, stages a `.py` runner in
Workspace, and AST-walks imports to resolve pip dependencies
automatically.

## Schedules

```python
from databricks.sdk.service.jobs import CronSchedule

dbc.jobs.create_or_update(
    name="hourly_ingest",
    tasks=[...],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 * * * ?",  # every hour
        timezone_id="UTC",
    ),
)
```

| Want | Quartz cron |
| --- | --- |
| Every hour | `0 0 * * * ?` |
| Every 15 min | `0 */15 * * * ?` |
| Daily at 02:30 UTC | `0 30 2 * * ?` |
| Weekdays at 09:00 | `0 0 9 ? * MON-FRI` |

## Compute

Default is **serverless** (fast cold start, internal pipelines).
Use a **classic cluster** for ingestion that hits the public internet:

```python
job.pytask(
    ingest_fn,
    task_key="ingest",
    existing_cluster_id="0123-...",  # cluster with internet egress
).create()
```

## Secrets

```python
val = dbc.secrets.get("my-scope", "db-password")
dbc.secrets["my-scope/db-password"]  # dict-like access
```

## Don'ts

- Don't sleep-poll a job — `job.wait()` does exponential backoff.
- Don't `ws.jobs.create(...)` directly — use
  `dbc.jobs.create_or_update()`.
- Don't hand-write the `.py` runner — `pytask` stages it for you.
- Don't list pip deps manually — `auto_dependencies` walks the AST.
