# Skill: create Databricks Jobs and schedule them with `dbc.jobs.create*`

## When to use

The user asks to "create a job", "schedule a pipeline", "wire this
Python function as a Databricks task", "add a cron schedule", "make
this run hourly / nightly", "stage code to Workspace and trigger
it from a job", or pastes a function and asks "make this a
scheduled job". Pair with [`ygg-databricks-jobs`](ygg-databricks-jobs.md)
(triggering / waiting on existing runs) and
[`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md) (full
ingestion recipe that ends in a scheduled job).

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
dbc.jobs                         # Jobs service
dbc.jobs.create(...)             # create new
dbc.jobs.create_or_update(...)   # upsert by name (idempotent)
dbc.jobs.create_for_user(...)    # sugar: pulls UserInfo defaults
```

All three return a `Job` singleton bound to the workspace, cached by
`job_id`. Re-running with the same name on `create_or_update` updates
the existing definition in place — no duplicate-job sprawl.

## Stage a Python callable as a task

The most common shape: "I have `def ingest(since): …`, run it
nightly". `JobTask.from_callable` (used internally by `Job.pytask`)
extracts the function's source via `inspect.getsource`, writes a
`.py` runner under `/Workspace/Shared/.ygg/jobs/<task_key>/`, and
configures the task to invoke it with literal args — **no pickling**,
the script Databricks runs is the exact source of the function.

```python
def ingest(since_iso: str) -> int:
    ...

job = dbc.jobs.create_or_update(
    name="ingest_orders",
    tasks=[],
    **dbc.jobs.userinfo_defaults(),
)

job.pytask(
    ingest,
    "2026-05-19T00:00:00Z",
    task_key="ingest",
).create()
```

`auto_dependencies=True` (default) walks the rendered script's AST,
resolves every top-level import via `sniff_imports`, and feeds the
result through `dependencies_to_pip_specs` to derive pinned pip
requirements. They land on the task's serverless `JobEnvironment`,
so the runner has every transitive package the function uses without
the caller spelling them out.

Pass `extra_dependencies=("vendor-sdk==1.2.3",)` to add a wheel that
isn't directly imported (e.g. an entry-point plugin), or
`exclude_modules=("polars",)` to drop a noisy import that's already
provisioned on the cluster.

## Schedules and triggers

```python
from databricks.sdk.service.jobs import (
    CronSchedule, Continuous, FileArrivalTriggerConfiguration, TriggerSettings,
)

# Cron — hourly at minute 0, in UTC
dbc.jobs.create_or_update(
    name="ingest_orders",
    tasks=[...],
    schedule=CronSchedule(
        quartz_cron_expression="0 0 * * * ?",
        timezone_id="UTC",
        pause_status="UNPAUSED",
    ),
)

# Continuous — keep one run alive
dbc.jobs.create_or_update(
    name="kafka_bridge",
    tasks=[...],
    continuous=Continuous(pause_status="UNPAUSED"),
)

# File-arrival trigger — fire on new file in a Volume / S3 path
dbc.jobs.create_or_update(
    name="land_files",
    tasks=[...],
    trigger=TriggerSettings(
        file_arrival=FileArrivalTriggerConfiguration(
            url="s3://vendor-feeds/orders/",
            min_time_between_triggers_seconds=60,
        ),
        pause_status="UNPAUSED",
    ),
)
```

Quartz cron quick reference (Databricks dialect — six fields with a
leading seconds slot, ``?`` in either day field):

| Want | Expression |
| --- | --- |
| Every hour, top of hour | `0 0 * * * ?` |
| Every 15 min | `0 */15 * * * ?` |
| Daily at 02:30 UTC | `0 30 2 * * ?` |
| Weekdays at 09:00 | `0 0 9 ? * MON-FRI` |
| Monthly on day 1 at 00:00 | `0 0 0 1 * ?` |

## UserInfo defaults (recommended)

`dbc.jobs.userinfo_defaults()` pulls `git_source`,
`email_notifications`, and `tags` from the running process's
`UserInfo` (Databricks notebook context, git remote, current user).
Splat it into any `create` / `create_or_update` call — caller-supplied
kwargs win on collision:

```python
dbc.jobs.create_or_update(
    name="ingest_orders",
    tasks=[...],
    schedule=CronSchedule(quartz_cron_expression="0 0 * * * ?", timezone_id="UTC"),
    **dbc.jobs.userinfo_defaults(),                    # git, notifications, tags
    tags={"Env": "prod", "Owner": "data-platform"},    # overrides the UserInfo tags
)
```

`create_for_user(...)` is the sugar shape — same as the splat but
configurable per call (`include_git_source=`, `notification_events=`,
…).

## Multi-task jobs

```python
job = dbc.jobs.create_or_update(name="orders_etl", tasks=[])

extract = job.pytask(extract_fn, "2026-05-19", task_key="extract").create()
transform = job.pytask(
    transform_fn,
    task_key="transform",
    depends_on=["extract"],
).create()
load = job.pytask(
    load_fn,
    task_key="load",
    depends_on=["transform"],
).create()
```

`depends_on=[...]` accepts task keys (strings) or `JobTask` objects.
Tasks in the same job share the cached `JobEnvironment` when they
declare the same `environment_key` — set it once, all three tasks
re-use the wheel set.

## Parameters and bound arguments

Two ways to parametrise a callable task:

1. **Bind literals at staging time** — args become Python literals in
   the rendered script (`func(*args, **kwargs)`):

   ```python
   job.pytask(ingest, "2026-05-19T00:00:00Z", region="EU", task_key="ingest").create()
   ```

2. **Read job parameters at runtime** — when the schedule should
   inject `{{ start_time }}` / `{{ job.parameters.region }}`:

   ```python
   def ingest():
       import os
       since = os.environ["JOB_PARAMETER_SINCE"]
       ...

   dbc.jobs.create_or_update(
       name="ingest_orders",
       tasks=[...],
       parameters=[{"name": "since", "default": "{{job.start_time.iso_date}}"}],
   )
   ```

Mix-and-match by capturing job parameters via task `notebook_params` /
`python_params` in your `JobTask` config.

## Cluster vs serverless — pick by workload

Default `JobTask.from_callable` runs on **serverless**. That's the
right answer for **internal pipelines** (curated rebuilds, `dash_*`
refreshes, dim joins, anything that talks only to Unity Catalog
warehouses + Delta tables) — fast cold start, auto-scale, `ygg` is
already provisioned via the AST-walked dependency resolution.

It's the **wrong** answer for **ingestion pipelines that hit the
public internet** (HTTP APIs, S3, vendor SFTP, third-party
webhooks). Databricks serverless compute has no outbound internet
access by default — egress is gated by the workspace network policy
and most workspaces ship without one configured. Use a **custom
all-purpose cluster** for ingestion, with `ygg[data,databricks,http]`
preinstalled.

| Task | Compute | Why |
| --- | --- | --- |
| Ingestion (HTTP API, S3 / GCS / Azure, FTP, webhooks) | **Multi-node all-purpose cluster** with outbound internet | Serverless has no public-internet egress on most workspaces; multi-node spreads outbound IPs across workers so per-IP API rate limits hit N× later. Use single-node only for low-volume / per-account-quota sources. |
| Curated rebuild (raw_ → curated) | **Serverless** (`environment_key=DEFAULT_ENVIRONMENT_KEY`) | Internal to UC; fast cold start; ygg provisioned via AST deps. |
| `dash_*` display rebuilds | **Serverless** | Same as curated — internal joins / windows. |
| ML training (small data) | **Serverless** | UC-only; MLflow tracking is internal. |
| ML training (heavy / GPU) | **Job cluster** (single-use, sized for the workload) | Serverless GPU is restricted; one-shot job cluster avoids leaving paid GPUs idle. |

### Provisioning an ingestion cluster with ygg pre-installed

**Default to multi-node, not single-node, for external API
ingestion.** Each worker has its own outbound IP behind the
workspace NAT; spreading the requests across N workers means an
API's per-IP rate limit hits N× before it throttles you, and a
single noisy neighbour doesn't poison the whole feed. Single-node
ingestion is the right shape only for low-volume sources or for
APIs with per-account (not per-IP) limits.

```python
INGESTION_CLUSTER_SPEC = {
    "spark_version": "16.4.x-scala2.12",
    "node_type_id": "Standard_DS3_v2",
    "num_workers": 4,                                # multi-node by default;
                                                     # spreads requests across
                                                     # 4 worker IPs to reduce
                                                     # per-IP rate-limit risk.
    "autoscale": {"min_workers": 2, "max_workers": 8},  # let it grow under
                                                        # heavy fan-out.
    "data_security_mode": "SINGLE_USER",
    # NETWORK ACCESS — required for ingestion (default workspaces lock
    # serverless egress; classic clusters honour the workspace's NAT /
    # public-IP policy).
    "custom_tags": {"egress": "internet"},
    # Pre-install ygg so the staged Python task starts from the right
    # baseline. PyPi index libraries land before the script runs.
    "library_specifications": [
        {"pypi": {"package": "ygg[data,databricks,http]"}},
    ],
}

cluster = dbc.clusters.create(name="ygg-ingest-default", **INGESTION_CLUSTER_SPEC)
cluster.wait_for_state("RUNNING")
```

When you genuinely need single-node (very-low-volume source,
per-account quota, debug runs), set `num_workers=0` plus the
`singleNode` Spark conf:

```python
SINGLE_NODE_OVERRIDES = {
    "num_workers": 0,
    "spark_conf": {
        "spark.databricks.cluster.profile": "singleNode",
        "spark.master": "local[*, 4]",
    },
}
```

To take real advantage of the worker IPs, drive the fetch in
parallel (one task per shard / symbol / date / region) — Spark
`mapInPandas`, `foreachPartition`, `yggdrasil.concurrent` helpers,
or one `JobTask` per shard. A multi-node cluster sitting behind a
single-threaded ingestion script wastes the worker IPs.

Then pin the ingestion task to it:

```python
job.pytask(
    ingest_orders_since,
    "2026-05-19T00:00:00Z",
    task_key="ingest",
    existing_cluster_id=cluster.cluster_id,
).create()
```

Curated / display / ML tasks on the **same job** keep the
serverless default — the cluster pin is per-task. A multi-task DAG
ends up with one classic ingestion task and N serverless downstream
tasks, all in one Job.

## Pinning compute explicitly (when you need to override)

To pin a serverless task to a specific environment (Python version,
pinned wheels), or to pin classic compute on a per-task basis:

```python
job.pytask(
    ingest,
    task_key="ingest",
    new_cluster={
        "spark_version": "15.4.x-scala2.12",
        "node_type_id": "Standard_DS3_v2",
        "num_workers": 0,            # single-node
    },
).create()

# …or share an existing all-purpose cluster:
job.pytask(ingest, task_key="ingest", existing_cluster_id="0123-…").create()
```

For internal pipelines the serverless default beats a tiny
single-node cluster on cold-start. Pin classic compute when you
need: a specific Spark version, a JVM lib, an init script, **or
internet egress for ingestion** (see the workload table above —
serverless has no public outbound on most workspaces).

## Triggering + waiting (already covered)

See [`ygg-databricks-jobs`](ygg-databricks-jobs.md) for `.run_now`,
`.wait`, `.last_run`, secrets, clusters, warehouses, and
`WaitingConfig`.

## Don'ts

- Don't `ws.jobs.create(...)` raw — go through `dbc.jobs.create /
  create_or_update / create_for_user`. They wrap retries, the
  name → id cache, auto-volume-creation on `InvalidParameterValue`,
  and `userinfo_defaults` (git source, notifications, tags).
- Don't define a job twice — `create_or_update(name=...)` upserts by
  name. `create(...)` raises on duplicates.
- Don't hand-write the Workspace `.py` runner; `JobTask.from_callable`
  stages it for you, AST-walked dependencies and all.
- Don't pickle the callable into the task — the staging path is the
  exact source, not a pickled closure. Closures captured from the
  enclosing scope **won't** survive — pass values as arguments.
- Don't list pip deps manually unless you actually need a wheel
  `auto_dependencies` can't see (an entry-point plugin). `[data]` /
  `[databricks]` / `[http]` extras land via the AST walker.
- Don't set `pause_status="PAUSED"` and forget — the job ships
  scheduled but won't run; reviewers won't catch it.
- Don't sleep-poll a run; `job.wait()` does exponential backoff with
  the right budget.
