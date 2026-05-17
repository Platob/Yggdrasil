# yggdrasil.databricks.workflow

Prefect-style declarative workflows for Databricks Jobs.

Two decorators (`@flow`, `@task`) plus a `secret(...)` placeholder
turn a plain Python function into a deployed Databricks Job —
multi-task DAGs included.

## Quick start

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.workflow import flow, task, secret


@task
def extract(date: str) -> str:
    return f"/Volumes/raw/{date}"


@task(retries=2, environment_key="ygg-default")
def load(path: str, api_key: str = secret("vendor", "api-key")) -> str:
    import requests
    return requests.post(URL, json={"path": path}, auth=api_key).text


@flow(name="daily-etl", schedule="0 2 * * *")
def daily_etl(date: str = "2025-01-01"):
    p = extract(date)
    load(p)


client = DatabricksClient(host="https://<workspace>", token="<token>")
with client:
    daily_etl.deploy()                              # upsert the Databricks Job
    run = daily_etl.run(date="2025-01-15", wait=True)
```

The flow body is normal Python. `daily_etl(date="2025-01-15")` runs every
task locally — the same DatabricksClient is used to resolve `SecretRef`
defaults, so the function body sees a real string. With Databricks Connect
configured, SQL / Spark calls inside task bodies route to the live cluster
without any workflow-layer changes.

## Three execution modes

| Mode | Trigger | What runs where |
| ---- | ------- | ---------------- |
| **Local** | `daily_etl(date="…")` | Every `@task` body executes in-process. `SecretRef` defaults resolve via `DatabricksClient.current().secrets[...]`. No Job is created. |
| **Databricks Connect** | Same as local, with `DATABRICKS_*` env vars set or a `DatabricksClient` in scope | Same as local, but `pyspark` / SQL calls in task bodies route to the remote cluster. Still no Job — convenient for iteration. |
| **Deployed** | `daily_etl.deploy()` then `.run()` | The DAG is staged as a multi-task Databricks Job. Each task runs on serverless (default) or on the compute you pinned via `existing_cluster_id` / `job_cluster_key` / `new_cluster`. Return values flow through `dbutils.jobs.taskValues`. |

## How DAG capture works

`@flow` runs the wrapped function once in *trace mode* (a
`contextvars.ContextVar` switches every `@task` into "record" mode).
A `@task` call inside the trace returns a `TaskNode` future instead of
running the body. The flow body composes those futures the same way it
would compose normal values:

```python
@flow
def pipeline(date):
    a = step_a(date)        # TaskNode
    b = step_b(a)           # TaskNode, depends_on=[step_a]
    step_c(a, b)            # TaskNode, depends_on=[step_a, step_b]
```

`Flow.deploy()` walks the captured nodes, calls
`stage_python_callable` (or `stage_python_notebook_callable` when
`task_type="notebook"`) for each, wires the staged `Task` objects with
their `depends_on` edges, and calls `Jobs.create_or_update`. The same
return-value passthrough mechanism `Job.from_callable` uses is in
play: each staged invocation is wrapped in
`_ygg_runtime.publish_return(...)` so downstream tasks read upstream
results via `dbutils.jobs.taskValues.get`.

## Secrets

`secret("scope", "key")` returns a `SecretRef` placeholder. The
cleartext is *never* in the staged source on disk — the rendered
invocation calls `_ygg_runtime.secret('scope', 'key')` at task-execution
time, which fetches from the Databricks Secrets API via the current
client. Locally (or under Databricks Connect), `WorkflowTask.__call__`
materialises the same `SecretRef` defaults the moment the function is
called, so the function body always sees a real string regardless of
where it runs.

```python
@task
def call_vendor(payload: dict, api_key: str = secret("vendor", "api-key")):
    requests.post(VENDOR_URL, json=payload, headers={"Authorization": api_key})

# Local:  api_key materialised from the bound DatabricksClient
call_vendor({"q": "ping"})

# Deployed: api_key materialised on the cluster from the workspace secrets API
my_flow.deploy()
```

## Cluster / serverless binding

Every `@task` accepts the same compute kwargs the underlying SDK
`Task` supports:

```python
@task(environment_key="ygg-default")     # serverless (default)
@task(existing_cluster_id="0123-…-abcd") # reuse a named cluster
@task(job_cluster_key="prod-2x")         # cluster declared on the parent job
@task(new_cluster=ClusterSpec(...))      # per-task ephemeral cluster
```

Setting any cluster binding clears the default serverless `environment_key`
so the task lands on classic compute.

## Reference

- `flow(func=None, *, name=None, schedule=None, timezone="UTC", parameters=None, tags=None, permissions=None, **job_settings)` — see `Flow`.
- `task(func=None, *, task_key=None, task_type="spark", retries=None, environment_key=..., existing_cluster_id=None, job_cluster_key=None, new_cluster=None, **task_fields)` — see `WorkflowTask`.
- `secret(scope, key=None)` — returns a `SecretRef`. Accepts `"scope/key"` shorthand.
- `runtime.secret`, `runtime.task_value`, `runtime.publish_return` — runtime helpers the staged scripts call. Importable directly if you want to use them inside a task body without the `secret(...)` default magic.
