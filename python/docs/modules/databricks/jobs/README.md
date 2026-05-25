# yggdrasil.databricks.jobs

Typed parameter parsing for Databricks notebooks and jobs, plus `Job` / `JobTask` / `JobRun` resource classes for the Jobs API. `NotebookConfig` is an alias for `SystemParameters` (see [environ](../../environ/README.md)).

## Surface map

| Symbol | Purpose |
| --- | --- |
| `NotebookConfig` | Alias for `SystemParameters` — typed parameter bag for notebooks and job tasks |
| `TaskParameters` | Raw parameter dict from all sources (argv, env, widgets, job params) |
| `Job` | Databricks Job resource singleton — run, list, cancel, repair |
| `JobRun` | A single job run — wait, cancel, inspect |
| `JobTask` | A single task within a job — create, update, delete, `from_callable` |
| `Jobs` | Service layer accessed via `DatabricksClient().jobs` |
| `get_dbutils` | Return `dbutils` when available, else `None` |
| `read_widgets` | Read current notebook widget values |
| `read_job_parameters` | Read current job parameter values |
| `sniff_imports` | Introspect a callable's imports for dependency resolution |

---

## 1) NotebookConfig — typed notebook / job parameters

`NotebookConfig` reads parameters from the first available source: Databricks widget → job parameter → environment variable → CLI argument → default.

### Minimal example

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class IngestConfig(NotebookConfig):
    catalog: str
    schema: str
    table: str
    dry_run: bool = False

cfg = IngestConfig.from_environment()
print(cfg.catalog, cfg.schema, cfg.table, cfg.dry_run)
```

### With defaults and type casting

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class TrainConfig(NotebookConfig):
    catalog: str = "main"
    schema: str = "ml"
    model_name: str = "churn_v1"
    max_rows: int = 50_000
    dry_run: bool = True

# Read from widgets in a notebook
cfg = TrainConfig.init_widgets()

# Read from all available sources in a job task
cfg = TrainConfig.from_environment()
print(cfg)
```

### Initialize notebook widgets

```python
from dataclasses import dataclass
from yggdrasil.databricks.jobs import NotebookConfig

@dataclass
class Config(NotebookConfig):
    catalog: str = "main"
    schema: str = "staging"
    start_date: str = "2026-01-01"

# Register widgets in the notebook (no-op if already registered)
Config.init_widgets()
cfg = Config.from_environment()
```

### Access `dbutils`

```python
from yggdrasil.databricks.jobs import get_dbutils, NotebookConfig

dbutils = get_dbutils()   # None when not in Databricks

@dataclass
class Config(NotebookConfig):
    catalog: str = "main"

cfg = Config.from_environment()
dbutils = cfg._dbutils   # same handle
```

---

## 2) TaskParameters — raw parameter dict

Read the raw string-valued parameters from all sources without type casting:

```python
from yggdrasil.databricks.jobs import task_parameters, read_widgets, read_job_parameters

# All sources merged
params = task_parameters()
print(params)   # {"catalog": "main", "schema": "default", ...}

# Widget values only (notebook context)
widgets = read_widgets()

# Job parameter values only
job_params = read_job_parameters()
```

---

## 3) Job — Databricks Job resource

Access via the client's `jobs` service or directly:

```python
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()
jobs   = client.jobs   # Jobs service
```

### List jobs

```python
for job in client.jobs.list(limit=20):
    print(job.job_id, job.settings.name)
```

### Get a specific job

```python
job = client.jobs.get(job_id=123456)
print(job.settings.name, job.settings.schedule)
```

### Trigger a run

```python
job = client.jobs.get(job_id=123456)

run = job.run(
    job_parameters={"catalog": "main", "schema": "staging"},
    wait=True,    # block until the run completes
)
print(run.state.result_state)   # "SUCCESS"
```

### List runs

```python
job = client.jobs.get(job_id=123456)

for run in job.runs(limit=10):
    print(run.run_id, run.state.life_cycle_state)
```

### Cancel all active runs

```python
job.cancel_all_runs()
```

---

## 4) JobTask — task management

`JobTask` wraps a single task within a Databricks Job. The most powerful feature is `from_callable`, which serializes a Python function to a notebook or script, walks its imports for dependency inference, and registers it as a task on the job.

### Create a task from a Python callable

```python
from yggdrasil.databricks.jobs import JobTask
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()

def my_etl(catalog: str, schema: str) -> None:
    from yggdrasil.databricks import DatabricksClient
    dbc = DatabricksClient()
    dbc.sql.execute(f"OPTIMIZE {catalog}.{schema}.orders")

task = JobTask.from_callable(
    func=my_etl,
    task_key="etl",
    job=client.jobs.get(job_id=123456),
    libraries=["ygg[databricks]"],
)
task.create()
```

### Update a task

```python
task.update(timeout_seconds=3600, max_retries=2)
```

### Delete a task

```python
task.delete()
```

---

## 5) Dependency introspection

Before deploying a callable as a job task, inspect its imports to infer the required pip packages:

```python
from yggdrasil.databricks.jobs import sniff_imports, dependencies_to_pip_specs

def my_fn():
    import polars as pl
    import yggdrasil.databricks
    return pl.DataFrame({"x": [1, 2]})

deps = sniff_imports(my_fn)
specs = dependencies_to_pip_specs(deps)
print(specs)   # ["polars", "ygg[databricks]"]
```

---

## 6) WorkspacePyPI — install from a workspace-local PyPI

For air-gapped environments, packages can be stored in a DBFS/Volume path and installed from there:

```python
from yggdrasil.databricks.jobs import WorkspacePyPI
from yggdrasil.databricks import DatabricksClient

client = DatabricksClient()

pypi = WorkspacePyPI(
    client=client,
    root="dbfs:/FileStore/packages",
)

# Upload a wheel
pypi.upload("/local/path/my_package-1.0.0-py3-none-any.whl")

# Install in job task libraries
specs = pypi.library_spec("my_package")
```

---

## 7) userinfo helpers — tag and notify from git context

When running inside a CI/CD pipeline with git context, attach git metadata and email notifications to job settings:

```python
from yggdrasil.databricks.jobs import (
    userinfo_tags,
    userinfo_git_source,
    userinfo_email_notifications,
    userinfo_job_settings,
)

tags   = userinfo_tags()             # {"git_commit": "abc123", "git_branch": "main", ...}
source = userinfo_git_source()       # GitSource with branch, commit, provider
notifs = userinfo_email_notifications()  # EmailNotifications from git config user.email
```
