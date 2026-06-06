# Skill: create, schedule, and run Databricks jobs

## When to use

The user asks to create a job, schedule a pipeline, run a function as a
Databricks task, set up a cron schedule, or use `@task` / `@flow`.

## Run an existing job

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
job = dbc.jobs.get(name="my-pipeline")    # or dbc.jobs.get(job_id=123)

run = job.run()                # → JobRun (awaitable). Or: job.run_and_wait()
run.wait()
run.result_state               # SUCCESS / FAILED / CANCELED / ...
run.duration_seconds
```

Pass parameters when triggering:

```python
job.run(parameters={"date": "2026-05-23"})          # job-level params
job.run(notebook_params={"date": "2026-05-23"})     # notebook widgets
job.run(python_params=["--date", "2026-05-23"])     # wheel/script argv
```

## Define a job with `@task` / `@flow`

Import the decorators from `yggdrasil.databricks.job` (Prefect-style):

```python
from yggdrasil.databricks.job import task, flow

@task
def ingest(date: str) -> str:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    dbc.dataset(f"SELECT * FROM vendor.raw WHERE date = '{date}'") \
       .map(clean).to_table("main.curated.events")
    return "main.curated.events"

@task(retries=2)
def publish(table: str) -> None:
    ...

@flow(name="daily-ingest")
def daily(date: str = "2025-01-01"):
    table = ingest(date)
    publish(table)

daily.deploy(dbc)              # trace the body, stage tasks, upsert the Job
daily(date="2026-05-23")       # run it
```

- A `@task` / `@flow` **runs in-process when called inside Databricks**
  (for testing) and is dispatched as a real Databricks run otherwise.
  `daily.local(...)` forces in-process; `daily.submit(...)` runs it in the
  background and returns a `Future`.
- `@task` options: `name`, `key`, `retries`, `retry_delay_seconds`,
  `depends_on=(...)` (task keys for an explicit DAG edge).
- Flows/tasks default to **serverless** compute and ship your live code as
  a wheel — no manual cluster wiring.

## Define a job with explicit tasks

When you'd rather build the task list yourself:

```python
from databricks.sdk.service.jobs import Task, NotebookTask

dbc.jobs.create_or_update(
    name="orders_etl",
    tasks=[
        Task(task_key="ingest", notebook_task=NotebookTask(notebook_path="/…/ingest")),
        Task(task_key="load",   depends_on=[{"task_key": "ingest"}],
             notebook_task=NotebookTask(notebook_path="/…/load")),
    ],
)
```

`create_or_update` upserts by name (creates if absent, updates if present).

## Schedules

Schedules use the SDK's `CronSchedule` (quartz), passed to
`create_or_update(schedule=...)` or to `@flow(trigger=...)`:

```python
from databricks.sdk.service.jobs import CronSchedule

dbc.jobs.create_or_update(
    name="hourly_ingest",
    tasks=[...],
    schedule=CronSchedule(quartz_cron_expression="0 0 * * * ?", timezone_id="UTC"),
)
```

| Want | Quartz cron |
| --- | --- |
| Every hour | `0 0 * * * ?` |
| Every 15 min | `0 */15 * * * ?` |
| Daily at 02:30 UTC | `0 30 2 * * ?` |
| Weekdays at 09:00 | `0 0 9 ? * MON-FRI` |

## Inspect runs

```python
run = dbc.job_runs.get(run_id=987654321)
run.state; run.result_state; run.state_message
run.debug()                    # human-readable dump: state + DAG + task logs/stderr
run.logs(task_key="ingest")    # one task's output
run.cancel()
run.repair(rerun_tasks=["load"], wait=True)   # rerun failed tasks

for r in job.list_runs(active_only=True):
    print(r.run_id, r.state)
```

## Secrets

```python
dbc.secrets.create_secret("db-password", "<value>", scope="vendor")

dbc.secrets["vendor/db-password"].svalue()     # read the string value
```

## Don'ts

- **No `yggdrasil.databricks.workflow` module** — import `task` / `flow`
  from `yggdrasil.databricks.job`.
- **No `secret("scope", "key")` helper** — read via `dbc.secrets`.
- **No `job.pytask(...)`** — use `@task` / `@flow`, or
  `dbc.jobs.create_or_update(name, tasks=[...])` with SDK `Task` objects.
- It's `job.run()` (not `run_now()`), and `@flow(trigger=CronSchedule(...))`
  (not a `schedule="0 2 * * *"` string).
- Don't sleep-poll a run — `run.wait()` does the backoff.
- Don't `dbc.workspace_client().jobs.create(...)` — use
  `dbc.jobs.create_or_update()`.
