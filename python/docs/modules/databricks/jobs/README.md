# yggdrasil.databricks.jobs

Databricks Jobs — lifecycle management, typed notebook parameter parsing, task definition helpers, and programmatic run control.

For the higher-level **`@flow` / `@task` declarative API**, see the [workflow module](../workflow/README.md).

---

## One-liner

```python
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class MyConfig(NotebookConfig):
    catalog: str = "main"
    table:   str = "events"

cfg = MyConfig.from_environment()
```

---

## 1) `NotebookConfig` — typed job/widget parameters

`NotebookConfig` is the recommended way to pass typed parameters into Databricks notebooks and Jobs.

### Declare and read parameters

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog:  str  = "main"
    schema:   str  = "raw"
    table:    str  = "events"
    max_rows: int  = 100_000
    dry_run:  bool = False

# In a Databricks job: reads from job task parameters
cfg = IngestConfig.from_environment()

# In an interactive notebook: creates dbutils widgets
cfg = IngestConfig.init_widgets()

# In a job task context: reads from job task invocation params
cfg = IngestConfig.init_job()

print(cfg.catalog, cfg.schema, cfg.table, cfg.dry_run)
```

### Access dbutils

```python
from yggdrasil.databricks.jobs import NotebookConfig
from dataclasses import dataclass

@dataclass
class Config(NotebookConfig):
    path: str = "/Volumes/main/landing/"

cfg = Config.from_environment()
dbutils = cfg.get_dbutils()  # None when not in a Databricks runtime

if dbutils:
    files = dbutils.fs.ls(cfg.path)
```

---

## 2) `Job` — job resource lifecycle

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
jobs   = client.jobs

# Find a job by name
job = jobs.find("my-daily-etl")
print(job.job_id, job.name, job.url)

# Get all jobs
for j in jobs.list():
    print(j.job_id, j.name)

# Create or update a job
job = jobs.create_or_update(
    name="my-daily-etl",
    tasks=[...],   # list[JobTask]
    schedule="0 2 * * *",
)

# Trigger a run
run = job.run(wait=True, job_parameters={"date": "2026-05-23"})
print(run.is_successful, run.run_id, run.url)

# Get the latest run
latest = job.latest_run()
print(latest.state, latest.run_duration)

# List all runs
for run in job.runs(limit=10):
    print(run.run_id, run.state, run.start_time)

# Delete a job
job.delete()
```

---

## 3) `JobRun` — run lifecycle and repair

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
job    = client.jobs.find("my-daily-etl")

# Trigger a run and wait
run = job.run(wait=True)

if run.is_successful:
    print("Done:", run.run_id)
elif run.is_failed:
    # Rerun only failed tasks
    repaired = run.repair(rerun_all_failed_tasks=True, wait=True)
    print("Repair result:", repaired.state)

# Cancel a running run
run2 = job.run(wait=False)
run2.cancel()
```

---

## 4) `JobTask` — task definitions

```python
from yggdrasil.databricks.jobs import JobTask

# Serverless Python task from a callable
def my_etl():
    from yggdrasil.databricks import DatabricksClient
    DatabricksClient.current().sql.execute("INSERT INTO main.raw.events ...")

task = JobTask.from_callable(my_etl, task_key="etl")

# Notebook task
notebook_task = JobTask.notebook(
    task_key="run-notebook",
    notebook_path="/Workspace/Shared/notebooks/daily_ingest",
    base_parameters={"date": "{{job.parameters.date}}"},
)

# Python wheel task
wheel_task = JobTask.wheel(
    task_key="run-wheel",
    package_name="my_etl_pkg",
    entry_point="daily_ingest",
    parameters=["--date", "{{job.parameters.date}}"],
)

# Dependency chain
task2 = JobTask.from_callable(
    my_downstream,
    task_key="downstream",
    depends_on=["etl"],
)
```

---

## 5) `TaskParameters` — parse job parameters programmatically

```python
from yggdrasil.databricks.jobs import read_job_parameters, read_widgets

# In a job context: read job parameters as a dict
params = read_job_parameters()
print(params.get("date"), params.get("catalog"))

# In a notebook context: read widget values
widgets = read_widgets()
print(widgets.get("table"))
```

---

## 6) Dependency sniffing for serverless tasks

`JobTask.from_callable` automatically sniffs the function's imports and translates them into pip specs for the job environment:

```python
from yggdrasil.databricks.jobs import JobTask, sniff_imports, dependencies_to_pip_specs

def my_task():
    import polars as pl
    import pyarrow as pa
    from yggdrasil.data.cast.registry import convert
    ...

# List what would be installed
imports = sniff_imports(my_task)
pip_specs = dependencies_to_pip_specs(imports)
print(pip_specs)   # ["polars", "pyarrow>=20", "ygg[data]"]

# Build the task (installs deps automatically)
task = JobTask.from_callable(my_task, task_key="my-task")
```

---

## 7) End-to-end: create a multi-task Job

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.jobs import JobTask

client = DatabricksClient()

def extract():
    from yggdrasil.databricks import DatabricksClient
    c = DatabricksClient.current()
    c.sql.execute("INSERT OVERWRITE main.staging.events SELECT * FROM raw.events")

def transform():
    from yggdrasil.databricks import DatabricksClient
    c = DatabricksClient.current()
    c.sql.execute("INSERT OVERWRITE main.curated.events SELECT id, ts_utc FROM main.staging.events")

def load():
    from yggdrasil.databricks import DatabricksClient
    c = DatabricksClient.current()
    c.sql.execute("REFRESH TABLE main.dash_events")

t_extract   = JobTask.from_callable(extract,   task_key="extract")
t_transform = JobTask.from_callable(transform, task_key="transform", depends_on=["extract"])
t_load      = JobTask.from_callable(load,      task_key="load",      depends_on=["transform"])

job = client.jobs.create_or_update(
    name="daily-etl-pipeline",
    tasks=[t_extract, t_transform, t_load],
    schedule="0 3 * * *",         # daily at 03:00 UTC
    tags={"team": "data", "env": "prod"},
)

print(f"Job created: {job.url}")

# Trigger manually
run = job.run(wait=True)
print("Run outcome:", run.state)
```

---

## 8) `WorkspacePyPI` — workspace-hosted PyPI registry

When a workspace hosts its own PyPI server for internal packages:

```python
from yggdrasil.databricks.jobs import WorkspacePyPI, JobTask

registry = WorkspacePyPI(
    url="https://<workspace>/api/2.0/libraries/pypi",
    token="<token>",
)

task = JobTask.from_callable(
    my_func,
    task_key="run",
    extra_pip_sources=[registry],
)
```

---

## Cross-references

- [workflow module](../workflow/README.md) — `@flow` / `@task` declarative API built on top of Jobs
- [Databricks overview](../README.md) — client bootstrap and auth
