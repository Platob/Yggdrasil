# yggdrasil.databricks.job

Databricks Jobs lifecycle — create, trigger, poll, cancel, and repair job runs with an awaitable interface.

## One-liner

```python
from yggdrasil.databricks import DatabricksClient

run = DatabricksClient().jobs["my-etl-job"].run_and_wait()
print(run.state, run.duration_seconds)
```

## Find a job

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
jobs   = client.jobs

# By name
job = jobs.get("my-etl-job")

# By numeric ID (int or string)
job = jobs.get(12345)
job = jobs.get("12345")

# Dict-style
job = jobs["my-etl-job"]

# List all
for j in jobs:
    print(j.name, j.job_id, j.explore_url)
```

## Inspect a job

```python
job = DatabricksClient().jobs.get("my-etl-job")

print(job.name)
print(job.job_id)
print(job.explore_url)     # Databricks UI link
print(job.tags)
print(job.tasks)           # list[Task] from settings
```

## Trigger a run

```python
job = DatabricksClient().jobs.get("my-etl-job")

# Fire-and-forget (default)
run = job.run()

# Block until done
run = job.run_and_wait()

# With parameters
run = job.run(parameters={"env": "prod", "date": "2026-01-15"}, wait=True)

# With notebook overrides
run = job.run(notebook_params={"input_path": "/data/raw"}, wait=60)
```

## Wait / cancel / repair a run

`JobRun` implements `Awaitable` — same backoff + timeout contract as every other async surface in yggdrasil.

```python
run = job.run()

# Poll until terminal
run.wait()

# Check state
print(run.state)           # State.SUCCEEDED / State.FAILED / ...
print(run.is_done)
print(run.is_succeeded)
print(run.duration_seconds)

# Cancel a running job
run.cancel()

# Repair (rerun failed tasks)
run.repair(wait=True)
```

## Inspect run tasks

```python
run = job.run_and_wait()

for task in run.tasks:
    print(task.task_key, task.state, task.duration_seconds)
    if task.is_failed:
        print(f"  FAILED: {task.state_message}")
```

## List runs

```python
job = DatabricksClient().jobs.get("my-etl-job")

# Recent runs
for run in job.list_runs(limit=5):
    print(run.run_id, run.state, run.duration_seconds)

# Active runs only
for run in job.list_runs(active_only=True):
    print(run.run_id, run.state)

# Latest run
latest = job.latest_run()

# All runs across all jobs
for run in DatabricksClient().job_runs.list(limit=10):
    print(run.job_id, run.run_id, run.state)
```

## Create a job

```python
from databricks.sdk.service.jobs import Task, NotebookTask

client = DatabricksClient()

job = client.jobs.create(
    "daily-etl",
    tasks=[
        Task(
            task_key="extract",
            notebook_task=NotebookTask(notebook_path="/Workspace/etl/extract"),
        ),
        Task(
            task_key="transform",
            notebook_task=NotebookTask(notebook_path="/Workspace/etl/transform"),
            depends_on=[{"task_key": "extract"}],
        ),
    ],
    cluster="0601-123456-abc12345",
    permissions=["users", "data-team@example.com"],
    timeout_seconds=7200,
)
```

## Update / delete

```python
job = client.jobs.get("daily-etl")

# Update settings
job.update(timeout_seconds=3600, max_concurrent_runs=2)

# Upsert (create if missing, update if exists)
job = client.jobs.create_or_update("daily-etl", timeout_seconds=3600)

# Delete
job.delete()

# Cancel all active runs
job.cancel_all_runs()
```
