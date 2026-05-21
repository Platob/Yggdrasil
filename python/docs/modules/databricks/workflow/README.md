# yggdrasil.databricks.workflow

> **Prefect-style declarative workflows for Databricks Jobs.**
> Two decorators, one runtime helper, three execution modes.

Take a plain Python function whose body calls other Python functions,
decorate them with `@flow` / `@task`, and you have a multi-task
Databricks Job you can deploy, schedule, run, and observe. The same
function still works as ordinary Python — call it directly for unit
tests, or with Databricks Connect to drive Spark against a live cluster
from your laptop. The only difference between "local" and "cluster" is
which `DatabricksClient` is in scope.

```python
from yggdrasil.databricks.workflow import flow, task, secret

@task
def extract(date: str) -> str:
    return f"/Volumes/raw/{date}"

@task(retries=2)
def load(path: str, api_key: str = secret("vendor", "api-key")) -> str:
    return f"loaded {path} with {api_key[:4]}…"

@flow(name="daily-etl", schedule="0 2 * * *")
def daily_etl(date: str = "2025-01-01"):
    p = extract(date)
    load(p)

daily_etl.deploy()                              # → Databricks Job
run = daily_etl.run(date="2025-01-15", wait=True)
```

---

## Table of contents

1. [Concepts](#1-concepts)
2. [Quick start](#2-quick-start)
3. [The three execution modes](#3-the-three-execution-modes)
4. [Use cases](#4-use-cases)
    - [4.1 Linear pipeline](#41-linear-pipeline)
    - [4.2 Fan-out + fan-in DAG](#42-fan-out-fan-in-dag)
    - [4.3 Same task called multiple times](#43-same-task-called-multiple-times)
    - [4.4 Secrets as parameter defaults](#44-secrets-as-parameter-defaults)
    - [4.5 Secrets at runtime via `ygg.secret`](#45-secrets-at-runtime-via-yggsecret)
    - [4.6 Reading upstream return values with `ygg.task_value`](#46-reading-upstream-return-values-with-yggtask_value)
    - [4.7 Retries, timeouts, conditional execution](#47-retries-timeouts-conditional-execution)
    - [4.8 Notebook-style tasks (per-cell logs)](#48-notebook-style-tasks-per-cell-logs)
    - [4.9 Cluster placement: serverless, existing cluster, job cluster, new cluster](#49-cluster-placement)
    - [4.10 Side-effect-only ordering with `.after(...)`](#410-side-effect-only-ordering-with-after)
    - [4.11 Scheduling, time zones, and pause status](#411-scheduling-time-zones-and-pause-status)
    - [4.12 Tags, permissions, UserInfo defaults](#412-tags-permissions-userinfo-defaults)
    - [4.13 Re-deploying on every commit (CI/CD)](#413-re-deploying-on-every-commit-cicd)
    - [4.14 Reaching SQL and Volumes from inside a task](#414-reaching-sql-and-volumes-from-inside-a-task)
    - [4.15 Running with Databricks Connect](#415-running-with-databricks-connect)
    - [4.16 Unit-testing flows (mocking secrets)](#416-unit-testing-flows-mocking-secrets)
    - [4.17 Repairing a failed run](#417-repairing-a-failed-run)
    - [4.18 Introspecting a captured DAG without deploying](#418-introspecting-a-captured-dag-without-deploying)
    - [4.19 Pinning the target workspace at decoration time](#419-pinning-the-target-workspace-at-decoration-time)
    - [4.20 Cross-workspace secrets](#420-cross-workspace-secrets)
    - [4.21 The auto-derived `[YGG][project/version]` prefix](#421-the-auto-derived-yggprojectversion-prefix)
    - [4.22 Source-attribution metadata (tags + descriptions)](#422-source-attribution-metadata-tags-descriptions)
    - [4.23 Parallel fan-out with `.map(...)`](#423-parallel-fan-out-with-map)
    - [4.24 File-arrival triggers (`@flow(file_trigger=...)`)](#424-file-arrival-triggers-flowfile_trigger)
5. [The `ygg` runtime module](#5-the-ygg-runtime-module)
6. [API reference](#6-api-reference)
7. [Limitations](#7-limitations)
8. [Internals (how it works)](#8-internals-how-it-works)

---

## 1) Concepts

| Concept | What it is |
| ------- | ---------- |
| **`@task`** | A Python callable wrapped with workflow metadata (task key, retries, environment, cluster). Outside a flow trace it runs as plain Python; inside it records a `TaskNode` future. |
| **`@flow`** | The function whose body composes `@task` calls. Calling `flow.deploy()` traces the body and creates a Databricks Job; calling `flow.run()` triggers a run. The function itself is still a normal callable. |
| **`secret(scope, key)`** | A `SecretRef` placeholder used as a parameter default. Cleartext is never on disk — at runtime the staged invocation calls `ygg.secret('scope', 'key')` to fetch the live value. |
| **`TaskNode`** | A future captured during the trace. Passing one into a downstream task implies a `depends_on` edge plus a `ygg.task_value(...)` read for value pass-through. |
| **`FlowParam`** | Sentinel for a flow-level parameter. Becomes a `JobParameterDefinition` on the deployed Job; per-run overrides land via `Job.run(job_parameters=...)`. |
| **`ygg`** | The runtime helper module imported by every staged script: `ygg.secret`, `ygg.task_value`, `ygg.publish_return`. Importable in your own task bodies too. |
| **Trace mode** | A `contextvars.ContextVar` set by `Flow.trace()` / `Flow.deploy()`. While active, every `@task` call returns a `TaskNode` instead of running the body. |

---

## 2) Quick start

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.databricks.workflow import flow, task

@task
def extract():
    return [1, 2, 3]

@task
def total(values: list) -> int:
    return sum(values)

@flow(name="hello-flow")
def hello_flow():
    v = extract()
    total(v)

with DatabricksClient(host="https://<workspace>", token="<token>"):
    hello_flow.deploy()                          # upserts the Job
    run = hello_flow.run(wait=True)              # triggers + blocks
    print(run.is_successful, run.explore_url)
```

The flow is also a plain function — `hello_flow()` just runs in-process, no workspace contacted.

---

## 3) The three execution modes

A single flow definition runs in three modes. **The function body never changes.**

| Mode | Call shape | Where does each task body execute? | Use it for |
| ---- | ---------- | ----------------------------------- | ---------- |
| **Local** | `my_flow(date="…")` | In-process. `SecretRef` defaults resolve via `DatabricksClient.current()`. | Unit tests, fast iteration, debugging on a laptop. |
| **Databricks Connect** | Same as local, with `DATABRICKS_*` env vars set (and `databricks-connect` installed) | Same as local. SQL / Spark calls inside task bodies route to the remote cluster transparently. | Driving real cluster Spark from a notebook / IDE without deploying a Job. |
| **Deployed** | `my_flow.deploy()` then `.run()` | One Databricks Task per node in the trace, scheduled by the Jobs service. Return values flow through `dbutils.jobs.taskValues`. | Production. |

> **Why three modes?** Authoring vs. testing vs. deployment shouldn't be three different shapes of code. The flow body is the single source of truth — the workflow layer just changes what happens *inside* each `@task` call depending on the `DatabricksClient` and trace context in scope.

---

## 4) Use cases

Every example is self-contained — copy, paste, decorate, deploy.

### 4.1 Linear pipeline

```python
from yggdrasil.databricks.workflow import flow, task

@task
def fetch(url: str) -> str:
    import urllib.request
    return urllib.request.urlopen(url).read().decode()

@task
def parse(body: str) -> int:
    return sum(1 for line in body.splitlines() if line.strip())

@task
def store(count: int, table: str) -> None:
    from yggdrasil.databricks.client import DatabricksClient
    DatabricksClient.current().sql.execute(
        f"INSERT INTO {table} VALUES ({count})",
    )

@flow(name="line-counter")
def line_counter(url: str = "https://example.com/data.txt", table: str = "main.demo.counts"):
    body = fetch(url)
    n = parse(body)
    store(n, table)

line_counter.deploy()
line_counter.run(url="https://…", table="main.demo.counts")
```

Three sequential tasks. `parse` depends on `fetch` (its `body` argument is a `TaskNode`), `store` depends on `parse`.

### 4.2 Fan-out + fan-in DAG

```python
from yggdrasil.databricks.workflow import flow, task

@task
def split(seed: int) -> list[int]:
    return [seed, seed + 1, seed + 2]

@task
def square(values: list[int]) -> int:
    return sum(v * v for v in values)

@task
def cube(values: list[int]) -> int:
    return sum(v * v * v for v in values)

@task
def summarize(sq: int, cb: int) -> str:
    return f"sq={sq} cb={cb}"

@flow(name="fanout-fanin")
def fanout_fanin(seed: int = 1):
    values = split(seed)
    sq = square(values)         # depends_on=[split]
    cb = cube(values)           # depends_on=[split]
    summarize(sq, cb)           # depends_on=[square, cube]

fanout_fanin.deploy()
```

The Databricks Job runs `square` and `cube` in parallel after `split` completes; `summarize` waits for both.

### 4.3 Same task called multiple times

Trace collisions are auto-suffixed `_2`, `_3`, … so the staged task keys stay unique:

```python
@task
def ingest(date: str) -> str: ...

@flow(name="three-days")
def three_days():
    ingest("2025-01-01")    # task_key = "ingest"
    ingest("2025-01-02")    # task_key = "ingest_2"
    ingest("2025-01-03")    # task_key = "ingest_3"

three_days.deploy()
```

If you want explicit keys, pin them with `task(task_key="ingest_2025_01_01")`. The suffixing is just the default.

### 4.4 Secrets as parameter defaults

The most ergonomic form — declare the secret right next to the parameter that needs it:

```python
from yggdrasil.databricks.workflow import flow, task, secret

@task
def call_vendor(payload: dict, api_key: str = secret("vendor", "api-key")) -> dict:
    import requests
    r = requests.post("https://vendor.example/ingest", json=payload, headers={
        "Authorization": f"Bearer {api_key}",
    })
    r.raise_for_status()
    return r.json()

@flow(name="vendor-sync")
def vendor_sync():
    call_vendor({"event": "ping"})

vendor_sync.deploy()
```

What happens:
- `secret("vendor", "api-key")` returns a `SecretRef("vendor", "api-key")` at decoration time.
- The cleartext **never** lives in the staged `.py` source.
- The staged invocation reads `call_vendor(payload=…, api_key=ygg.secret('vendor', 'api-key'))`. At call time the cluster fetches it from the Databricks Secrets API.
- When run locally, `WorkflowTask.__call__` materialises the same `SecretRef` via the bound `DatabricksClient`, so the function body sees a real string identically.

`secret("scope/key")` and `secret("scope:key")` shortcuts both work.

### 4.5 Secrets at runtime via `ygg.secret`

When you want a secret *inside* the body (not bound as a parameter), use the runtime module directly. It works the same locally, under Databricks Connect, and on the cluster:

```python
from yggdrasil.databricks.workflow import flow, task, ygg

@task
def query_vendor() -> str:
    # ygg is the runtime helper module — staged tasks import it
    # automatically; you can also import it explicitly here for
    # inline use.
    api_key = ygg.secret("vendor", "api-key")
    import requests
    return requests.get("https://vendor.example/ping", auth=("bearer", api_key)).text

@flow(name="ad-hoc")
def ad_hoc():
    query_vendor()
```

### 4.6 Reading upstream return values with `ygg.task_value`

A `TaskNode` passed as an argument to a downstream task gets wired through `dbutils.jobs.taskValues` automatically. You can read other published values explicitly too:

```python
from yggdrasil.databricks.workflow import flow, task, ygg

@task
def write_marker(path: str) -> str:
    open(path, "w").write("ok")
    return path

@task
def verify():
    marker_path = ygg.task_value("write_marker", default="/tmp/missing")
    # On Databricks: reads the value write_marker published.
    # Locally / no dbutils: returns the default.
    assert open(marker_path).read() == "ok"

@flow(name="marker-flow")
def marker_flow():
    p = write_marker("/Volumes/main/demo/marker.txt")
    verify()                              # implicit ordering — see below
```

If `verify` needs to wait for `write_marker` but doesn't take its return as an argument, attach an explicit edge — see [4.10](#410-side-effect-only-ordering-with-after).

### 4.7 Retries, timeouts, conditional execution

Every kwarg accepted by `databricks.sdk.service.jobs.Task` flows through unchanged:

```python
from databricks.sdk.service.jobs import RunIf

@task(
    retries=3,
    timeout_seconds=600,
    run_if=RunIf.ALL_DONE,                  # run even if upstreams failed
    description="Flush staging table",
)
def flush(table: str) -> None: ...
```

`retries` maps to `Task.max_retries`. `description`, `timeout_seconds`, `run_if`, `email_notifications`, `webhook_notifications`, `health`, etc. all pass through as **task_fields**.

### 4.8 Notebook-style tasks (per-cell logs)

For tasks where you want stdout / `LOGGER` lines to surface under their own cell in the Databricks UI, switch the rendering flavour to `notebook`:

```python
@task(task_type="notebook")
def long_running_etl(date: str):
    import logging
    LOGGER = logging.getLogger("etl")
    LOGGER.info("Phase 1: loading …")
    # …
    LOGGER.info("Phase 2: aggregating …")
    # …
```

Same function, just rendered as a Databricks-format `.py` notebook (one cell for the import + metadata, one for captured locals, one for the function body, one for the invocation). The Databricks UI surfaces each `LOGGER` line under the cell that produced it, which is much easier to debug than the single-stream output of a `SparkPythonTask`.

### 4.9 Cluster placement

`@task` accepts the three cluster bindings the SDK supports — *one* per task. Setting any of them clears the default serverless `environment_key`.

```python
# Serverless (default) — ygg-default environment with ygg[data,databricks].
@task
def serverless_step(): ...

# Reuse a named cluster.
@task(existing_cluster_id="0123-456789-abcd1234")
def warm_cluster_step(): ...

# A "job cluster" declared on the parent Job — share across tasks.
@task(job_cluster_key="prod-2x")
def shared_cluster_step(): ...

# Per-task ephemeral cluster (spec passed verbatim to Task.new_cluster).
from databricks.sdk.service.compute import ClusterSpec
@task(new_cluster=ClusterSpec(spark_version="15.4.x-scala2.12", num_workers=4))
def own_cluster_step(): ...
```

To declare a job cluster the whole flow can reuse, pass `job_clusters=[…]` through the `@flow` decorator's `**job_settings`:

```python
from databricks.sdk.service.jobs import JobCluster
from databricks.sdk.service.compute import ClusterSpec

@flow(
    name="shared-cluster-flow",
    job_clusters=[
        JobCluster(
            job_cluster_key="prod-2x",
            new_cluster=ClusterSpec(spark_version="15.4.x-scala2.12", num_workers=2),
        ),
    ],
)
def shared_cluster_flow():
    shared_cluster_step()
```

### 4.10 Side-effect-only ordering with `.after(...)`

When task B must run after task A, but doesn't take its output as an argument:

```python
@task
def drain_staging(path: str): ...

@task
def reload(): ...

@flow
def reload_flow():
    d = drain_staging("/Volumes/main/staging/")
    reload.after(d)(reload)()
    # ↑ wraps `reload`, returns a callable; calling it inside the
    #   trace registers a node with depends_on=[d].
```

Pattern: `<task>.after(*upstreams)(<task>)(...args)`. The double-call is intentional — `after(...)` returns a decorator-shaped wrapper so the same pattern works at top-level (`some_task = task.after(other)(task)`) and inside flow bodies.

### 4.11 Scheduling, time zones, and pause status

```python
@flow(
    name="hourly",
    schedule="0 0 * * * ?",                  # Quartz cron
    timezone="Europe/Paris",
    pause_status="PAUSED",                   # deploy paused, unpause via UI
)
def hourly(): ...
```

`schedule` accepts:
- a Quartz cron string (coerced into `CronSchedule` using `timezone`),
- a pre-built `databricks.sdk.service.jobs.CronSchedule`,
- `None` to deploy without a schedule.

For file-arrival or table-update triggers, pass a fully-built `TriggerSettings` through `**job_settings`:

```python
from databricks.sdk.service.jobs import (
    TriggerSettings, FileArrivalTriggerConfiguration,
)

@flow(
    name="on-file",
    trigger=TriggerSettings(
        file_arrival=FileArrivalTriggerConfiguration(
            url="/Volumes/main/landing/inbox/",
            min_time_between_triggers_seconds=60,
        ),
    ),
)
def on_file(): ...
```

### 4.12 Tags, permissions, UserInfo defaults

```python
@flow(
    name="reporting",
    tags={"Team": "Data", "Env": "Prod"},
    permissions=["data-team", "alice@example.com"],
)
def reporting(): ...

reporting.deploy(userinfo_defaults=True)        # pre-fills git_source +
                                                # email_notifications
                                                # from the running user
```

`permissions` accepts the same shapes `Jobs.create` does — bare strings (email → user, other → group), already-built `JobAccessControlRequest` objects. `userinfo_defaults=True` runs the same `UserInfo` discovery as `Jobs.create_for_user`.

### 4.13 Re-deploying on every commit (CI/CD)

`deploy()` is idempotent — calling it again **updates** the Job in place. Wire it into your CI:

```python
# scripts/deploy_flows.py
from yggdrasil.databricks import DatabricksClient
from my_pipelines import reporting, daily_etl, vendor_sync

with DatabricksClient():                         # picks up DATABRICKS_HOST + DATABRICKS_TOKEN
    for flow in (reporting, daily_etl, vendor_sync):
        flow.deploy()
        print("deployed", flow.name)
```

A staged task whose body hasn't changed lands on the same `main-<digest>.py` path (the digest is content-stable), so re-deploys don't accumulate near-duplicate files. Pip deps re-sniff on every deploy and merge into the matching `JobEnvironment`.

### 4.14 Reaching SQL and Volumes from inside a task

The whole `yggdrasil.databricks` surface is available — same as in any Yggdrasil-using code:

```python
@task
def daily_dim_load(target: str = "main.dim.customer"):
    from yggdrasil.databricks.client import DatabricksClient

    client = DatabricksClient.current()

    # Pull from SQL warehouse
    df = client.sql.execute(
        "SELECT * FROM raw.customer WHERE updated_at >= current_date()",
    ).to_polars()

    # Write to a volume (parquet)
    out = client.volumes["main.staging.daily"].as_path() / "customer.parquet"
    out.write_bytes(df.to_arrow().serialize().to_pybytes())

    # Apply into dim table via Spark or warehouse
    client.sql.execute(
        f"INSERT OVERWRITE {target} SELECT * FROM raw.customer",
    )

@flow(name="dim-loader")
def dim_loader():
    daily_dim_load()
```

### 4.15 Running with Databricks Connect

Set up your environment with the usual Databricks Connect variables (or a `~/.databrickscfg` profile):

```bash
export DATABRICKS_HOST=https://my-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=dapi…
export DATABRICKS_CLUSTER_ID=0123-456789-abcd1234   # for compute
```

Now run the flow function from your laptop:

```python
from my_pipelines import dim_loader

dim_loader()                # tasks execute locally, Spark routes to the cluster
```

No code change. The `DatabricksClient.current()` picks up the env vars; Spark sessions opened inside the task body talk to the remote cluster via `databricks-connect`; secrets resolve from the live workspace.

> **Tip:** Use this for iterating on transformation logic against real data without paying the Job-deployment round trip. Switch to `dim_loader.deploy()` + `.run()` when you want the Databricks scheduler involved.

### 4.16 Unit-testing flows (mocking secrets)

Two patterns. **Pattern A** — pass concrete values explicitly:

```python
def test_call_vendor_explicit():
    result = call_vendor(payload={"x": 1}, api_key="test-key")
    assert "loaded" in result
```

The caller's `api_key` wins over the `SecretRef` default — no workspace touched.

**Pattern B** — mock the workspace client's secrets service:

```python
from unittest.mock import MagicMock, patch

def test_flow_locally():
    with patch("yggdrasil.databricks.client.DatabricksClient.current") as cur:
        client = MagicMock()
        client.secrets.__getitem__.return_value.svalue.return_value = "FAKE"
        cur.return_value = client
        my_flow()                                # SecretRef defaults → "FAKE"
```

Or use the `DatabricksTestCase` base, which already mocks the SDK boundary.

### 4.17 Repairing a failed run

```python
run = my_flow.run(wait=True)
if not run.is_successful:
    # Re-run only the failed task(s)
    run.repair(rerun_all_failed_tasks=True, wait=True)
```

`JobRun.repair` mirrors the SDK shape — see the [jobs module docs](../jobs/README.md) for the full surface.

### 4.18 Introspecting a captured DAG without deploying

`Flow.trace(**overrides)` returns the captured `TaskNode` list. Useful for testing the DAG shape, generating diagrams, or asserting invariants:

```python
nodes = daily_etl.trace()
for n in nodes:
    upstreams = [d.task_key for d in n.depends_on]
    print(f"{n.task_key} depends_on={upstreams}")

# Assert a flow contains the expected tasks
assert {n.task_key for n in nodes} == {"extract", "load_to_warehouse"}
```

`trace()` is also what `deploy()` calls under the hood; the same trace context machinery powers both.

### 4.19 Pinning the target workspace at decoration time

By default `Flow.deploy()` / `Flow.run()` use whichever `DatabricksClient` is current — picked up from `DATABRICKS_*` env vars, a `with DatabricksClient(...)` block, or `DatabricksClient.set_current(...)`. Pin a specific target at decoration time when the flow always targets one workspace:

```python
from yggdrasil.databricks.client import DatabricksClient
from yggdrasil.databricks.workflow import flow, task

prod_eu = DatabricksClient(
    host="https://prod-eu.cloud.databricks.com",
    token=os.environ["PROD_EU_TOKEN"],
)

@flow(name="reporting", client=prod_eu)
def reporting():
    daily_dim_load()

# Both deploy and run go to prod_eu — no matter the caller's env vars.
reporting.deploy()
reporting.run(wait=True)
```

Per-call overrides still win:

```python
reporting.deploy(client=staging_client)        # one-off staging deploy
reporting.run(client=staging_client)
```

A `@task(client=...)` override pins the staging workspace for one task while the rest of the flow lives in the parent flow's workspace — useful when a task's staged source must live alongside resources in a different workspace. The cluster running the task must have access to the staging path, so this is a corner case; most users only need `@flow(client=...)`.

### 4.20 Cross-workspace secrets

A `secret(...)` placeholder can target a workspace other than the one running the task. Pass a host URL or a live `DatabricksClient`:

```python
from yggdrasil.databricks.workflow import flow, task, secret

prod_eu = DatabricksClient(host="https://prod-eu.cloud.databricks.com", token=…)

@task
def cross_region_sync(
    api_key: str = secret("vendor", "eu-key", client=prod_eu),
    backup_key: str = secret("vendor", "us-key"),  # current workspace
):
    requests.post(…, auth=api_key)
    requests.post(BACKUP_URL, auth=backup_key)

@flow(name="dual-region", prefix=False)
def dual_region():
    cross_region_sync()
```

The staged invocation reads:

```python
cross_region_sync(
    api_key=ygg.secret('vendor', 'eu-key', host='https://prod-eu.cloud.databricks.com'),
    backup_key=ygg.secret('vendor', 'us-key'),
)
```

`ygg.secret(host=...)` builds a fresh `DatabricksClient` for that host on the spot (inheriting auth from the cluster's environment), fetches the secret, and returns the cleartext.

### 4.21 The auto-derived `[YGG][project/version]` prefix

By default every deployed Job is renamed to `[YGG][<project>/<version>] <flow.name>`:

```python
@flow(name="daily-etl")
def daily_etl(): ...

daily_etl.deploy()      # registered as "[YGG][my-pipelines/1.2.3] daily-etl"
```

The prefix comes from `UserInfo.current()`:

| Field | Resolution order |
| ----- | ---------------- |
| `<project>` | `UserInfo.product` (the `[project] name` in the nearest `pyproject.toml`) → `UserInfo.hostname` → `"ygg"` |
| `<version>` | `UserInfo.product_version` → flow source-file git commit short hash → `"ygg-<yggdrasil version>"` |

This makes deployed jobs **scannable in the Databricks UI** — filter by `[YGG][` to see every yggdrasil-shipped job, and the project segment tells you which repo / version drove the latest deploy.

Toggle policies:

```python
@flow(name="daily-etl", prefix=False)            # raw name, no prefix
@flow(name="daily-etl")                          # auto-prefix (default)
@flow(name="daily-etl", prefix="[ETL-PROD] ")    # literal prefix string
```

`Flow.deployed_name` returns the final name (with prefix applied) — handy for tests and for asserting against the workspace job list.

### 4.22 Source-attribution metadata (tags + descriptions)

Every deploy auto-derives provenance from the wrapped Python functions and surfaces it in three places:

1. **Job tags**, keyed under `ygg.*` — searchable in the Databricks UI ("Jobs › Filter by tag"):

   | Tag key | Source |
   | ------- | ------ |
   | `ygg.flow` | The flow's `name` argument. |
   | `ygg.module` | `flow_func.__module__` |
   | `ygg.qualname` | `flow_func.__qualname__` |
   | `ygg.source_file` | Absolute path returned by `inspect.getsourcefile`. |
   | `ygg.source_url` | HTTPS link to the line on GitHub / GitLab / Bitbucket / Azure DevOps (when the file lives in a git checkout with a recognised remote). |
   | `ygg.git_commit` | Current commit short hash. |
   | `ygg.git_branch` | Current branch (omitted in detached-HEAD state). |
   | `ygg.version` | Yggdrasil package version that built the staged source. |

2. **Job description** — the flow's first docstring line followed by a `Flow source:` footer that lists the source URL, commit, and yggdrasil version. Caller-supplied descriptions still win on collision.

3. **Task description** — the same footer scoped to the task's own function, appended after the auto-derived signature description from `stage_python_callable`.

Example footer:

```
Flow source: my_pipelines.daily_etl.daily_etl
  https://github.com/acme/data-platform/blob/abc123…/pipelines/daily_etl.py#L42
  git: abc123def012 (main)
  yggdrasil=0.7.93
```

Need to add custom tags? `metadata.collect_source_metadata(func, extra={"team": "data"})` plumbs through to the deploy-time tag bag (used internally by the workflow layer — pass the same shape via the `@flow(tags={...})` kwarg).

The git probe is best-effort and cached per source-file directory — N tasks in one flow trigger at most one `git` subprocess invocation per file's repo root. Missing git, non-git source trees, detached HEADs, and unfamiliar remote URLs all degrade gracefully.

### 4.23 Parallel fan-out with `.map(...)`

Tasks expose a `.map(iterable, *constants, **kw_constants)` method that fans the body out over an iterable in parallel. The same call shape adapts to both execution modes:

| Mode | What happens | Returns |
| ---- | ------------ | ------- |
| **Local** (no active trace) | The body runs across a `concurrent.futures.ThreadPoolExecutor` (default) or `ProcessPoolExecutor` (`pool="process"`), preserving input order. `SecretRef` defaults resolve once up-front so per-item submissions don't pay a Secrets round-trip each. | `list` of results. |
| **Trace** (inside `@flow`) | One `TaskNode` is registered per element — the Databricks scheduler runs the resulting tasks in parallel, subject to the job's `max_concurrent_runs` and the cluster's slot capacity. Auto-suffix gives every node a unique `task_key` (`step`, `step_2`, …). | `list[TaskNode]`. |

Local fan-out — same shape as a list comprehension but parallelised through a thread pool:

```python
from yggdrasil.databricks.workflow import task

@task(pool="thread", max_workers=8)
def fetch(url: str) -> bytes:
    import urllib.request
    return urllib.request.urlopen(url).read()

bodies = fetch.map([
    "https://example.com/a",
    "https://example.com/b",
    "https://example.com/c",
])
# bodies == [b"<page a>", b"<page b>", b"<page c>"]
```

Inside a flow — one Databricks task per element, all running in parallel after the upstream completes:

```python
from yggdrasil.databricks.workflow import flow, task

@task
def list_partitions(date: str) -> list[str]:
    return [f"/Volumes/raw/{date}/p{i}.parquet" for i in range(8)]

@task
def process(partition: str, target: str) -> int:
    # ... heavy per-partition work ...
    return 42

@task
def total(counts: list[int]) -> int:
    return sum(counts)

@flow(name="partitioned-load")
def partitioned_load(date: str = "2025-01-01", target: str = "main.dim.events"):
    parts = list_partitions(date)         # 1 task
    # NOTE: ``.map`` must iterate at trace time — pass a literal /
    # flow-parameter-typed iterable, not a TaskNode. For dynamic
    # fan-out from an upstream task's value, use the Databricks
    # ``ForEachTask`` primitive via the ``**job_settings`` passthrough.
    counts = process.map(
        ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"],
        target=target,
    )                                     # 8 tasks, parallel
    total(counts)                         # 1 reducer, depends_on=all 8
```

The downstream reducer sees the list of `TaskNode` futures as a single argument; `TaskNode._resolve_deps` walks `list` / `tuple` / `dict` containers so each mapped node becomes a `depends_on` edge automatically — no manual wiring.

Pool flavours:

| `pool` | Executor | When to reach for it |
| ------ | -------- | -------------------- |
| `"thread"` (default) | `ThreadPoolExecutor` | I/O-bound bodies — HTTP, Databricks SDK calls, SQL warehouse round trips, `requests.get`, blob storage reads/writes. Most task workloads. |
| `"process"` | `ProcessPoolExecutor` | CPU-bound bodies — heavy numpy / pyarrow / regex work. The task function and arguments must be picklable; with the `@task` decorator shadowing the function's module-level name, this typically requires the `ygg[pickle]` extra (installs `dill`, which patches the multiprocessing pickler at import time). |

Per-call overrides:

```python
fetch.map(urls, pool="thread", max_workers=16)             # one-off bump
fetch.map(urls, executor=existing_thread_pool)             # reuse a pool
```

When an external executor is supplied, `.map(...)` does **not** shut it down on return — caller owns the lifetime.

### 4.24 File-arrival triggers (`@flow(file_trigger=...)`)

A `@flow` runs on whatever signals you wire onto its `JobSettings` — `schedule=` for cron, plus a `file_trigger=` for the Databricks file-arrival trigger that fires whenever new files land at a workspace path or Volumes URL. Both can coexist (the job runs on whichever fires first).

Simple string form — just point at the directory:

```python
from yggdrasil.databricks.workflow import flow, task

@task
def ingest(): ...

@flow(name="on-file-arrival", file_trigger="/Volumes/main/landing/inbox/")
def on_file_arrival():
    ingest()
```

Dict form — opt into the debounce knobs:

```python
@flow(
    name="on-file-arrival",
    file_trigger={
        "url": "/Volumes/main/landing/inbox/",
        "min_time_between_triggers_seconds": 60,    # don't refire within 60s
        "wait_after_last_change_seconds": 30,       # wait for batch to settle
    },
)
def on_file_arrival():
    ingest()
```

Pre-built SDK objects pass through verbatim:

```python
from databricks.sdk.service.jobs import (
    FileArrivalTriggerConfiguration, TriggerSettings, PauseStatus,
)

@flow(
    name="on-file-arrival",
    file_trigger=FileArrivalTriggerConfiguration(
        url="/Volumes/main/landing/inbox/",
        wait_after_last_change_seconds=30,
    ),
)
def on_file_arrival(): ...

# Full TriggerSettings — wire model / periodic / table_update triggers
# alongside the file-arrival shorthand:
@flow(
    name="multi-trigger",
    file_trigger=TriggerSettings(
        file_arrival=FileArrivalTriggerConfiguration(url="/Volumes/x/"),
        pause_status=PauseStatus.PAUSED,
    ),
)
def multi_trigger(): ...
```

The flow's `pause_status` (set via `@flow(pause_status="PAUSED")`) flows through into the coerced `TriggerSettings` for the string / dict / `FileArrivalTriggerConfiguration` shapes — paused-on-deploy means the trigger doesn't fire until an operator unpauses it. The pre-built `TriggerSettings` form is pass-through; we don't second-guess the caller's choice.

Cron + file-arrival together — the job runs on either signal:

```python
@flow(
    name="dual-signal",
    schedule="0 0 6 * * ?",                              # daily 06:00
    file_trigger="/Volumes/main/landing/urgent/",        # plus on file arrival
)
def dual_signal():
    ingest()
```

Conflict guard: pass either `file_trigger=...` (the shorthand) **or** `trigger=...` via `**job_settings` (the raw `TriggerSettings`) — both target the same `JobSettings.trigger` slot and the decorator raises rather than silently picking a side.

> **Task-level note.** Databricks file-arrival triggers are job-level (one trigger per job), so the kwarg lives on `@flow`, not on `@task`. A flow whose body declares a single task is the canonical pattern for "this task fires on file arrival." For task-level fan-out triggered by an upstream task's listing, use `.map(...)` ([4.23](#423-parallel-fan-out-with-map)) or wire a Databricks `ForEachTask` through the `**job_settings` passthrough.

---

## 5) The `ygg` runtime module

Every staged task imports it as `ygg`. You can do the same from your own task bodies:

```python
from yggdrasil.databricks.workflow import ygg
```

| Function | Purpose |
| -------- | ------- |
| `ygg.secret(scope, key)` | Resolve a Databricks secret against `DatabricksClient.current()`. Returns the cleartext string. |
| `ygg.task_value(task_key, value_key="__ygg_return__", *, default=None)` | Read a value an upstream task published via `dbutils.jobs.taskValues`. Returns `default` outside a Databricks runtime. |
| `ygg.publish_return(value, *, value_key="__ygg_return__")` | Mirror *value* onto the current run's task-values map (no-op outside Databricks). Returns *value* unchanged so it composes naturally inside `return` statements. |

The constant `ygg.RETURN_VALUE_KEY` (`"__ygg_return__"`) is the key under which `publish_return` stores function returns — overridable if you need a custom convention.

> **Note:** The legacy alias `runtime` (i.e. `from yggdrasil.databricks.workflow import runtime`) still works; `ygg` is the canonical name and what every staged script uses.

---

## 6) API reference

### `flow(func=None, *, name=None, schedule=None, file_trigger=None, timezone="UTC", pause_status=None, parameters=None, tags=None, permissions=None, prefix=True, client=None, **job_settings)`

Decorator factory. Returns a `Flow`. Usable bare (`@flow`) or parametrised (`@flow(name=…)`).

| Argument | Effect |
| -------- | ------ |
| `name` | Flow / Job name. Defaults to `func.__name__`. |
| `schedule` | Quartz cron string or pre-built `CronSchedule`. |
| `file_trigger` | File-arrival trigger config. Accepts a workspace path / Volumes URL string, a dict of `FileArrivalTriggerConfiguration` kwargs (`url` required; `min_time_between_triggers_seconds`, `wait_after_last_change_seconds` optional), a pre-built `FileArrivalTriggerConfiguration`, or a full `TriggerSettings`. Mutually exclusive with passing `trigger=…` through `**job_settings`. See [4.24](#424-file-arrival-triggers-flowfile_trigger). |
| `timezone` | Time zone for cron coercion. Default `"UTC"`. |
| `pause_status` | `"PAUSED"` / `"UNPAUSED"` / `PauseStatus`. Applied to both the cron schedule and the coerced `file_trigger`. |
| `parameters` | Extra `JobParameterDefinition` entries beyond the auto-derived ones. |
| `tags` | Job tags. Auto-derived `ygg.*` metadata tags are merged under these (caller wins). |
| `permissions` | ACL entries; same shape as `Jobs.create`. |
| `prefix` | `True` (default) → auto `[YGG][project/version] ` prefix; `False` → bare name; `str` → literal prefix. |
| `client` | Pin a target `DatabricksClient`. Per-call `deploy(client=…)` / `run(client=…)` still wins. |
| `**job_settings` | Forwarded verbatim to `JobSettings` — `timeout_seconds`, `max_concurrent_runs`, `email_notifications`, `webhook_notifications`, `notification_settings`, `health`, `git_source`, `job_clusters`, `trigger` (only when `file_trigger` is unset), `queue`, `run_as`, `format`, … |

### `Flow.deploy(*, service=None, client=None, userinfo_defaults=False, trace_overrides=None, **extra_job_settings) -> Job`

Trace the body, stage every task, and upsert via `Jobs.create_or_update`. Returns the `Job` handle. Extra `**extra_job_settings` merge with `@flow`-level settings.

### `Flow.run(*, service=None, client=None, wait=False, deploy_if_missing=True, **job_parameters) -> JobRun`

Trigger a run of the deployed Job. Auto-deploys if the Job isn't in the workspace yet (turn off with `deploy_if_missing=False`).

### `Flow.trace(**overrides) -> list[TaskNode]`

Capture the DAG without deploying. Useful for tests and diagram generation.

### `Flow.__call__(*args, **kwargs)`

Run the flow body locally (no trace, no workspace round-trip). The body's `@task` calls run in-process and `SecretRef` defaults resolve via `DatabricksClient.current()`.

---

### `task(func=None, *, task_key=None, task_type="spark", retries=None, environment_key=..., existing_cluster_id=None, job_cluster_key=None, new_cluster=None, pool=None, max_workers=None, client=None, **task_fields)`

Decorator factory. Returns a `WorkflowTask`.

| Argument | Maps to |
| -------- | ------- |
| `task_key` | `Task.task_key` (default `func.__name__`). Auto-suffixed `_2`, `_3`, … on collision inside a flow. |
| `task_type` | `"spark"` → `SparkPythonTask` (flat `.py`); `"notebook"` → `NotebookTask` (cell-split). |
| `retries` | `Task.max_retries`. |
| `environment_key` | `Task.environment_key`. Default `"ygg-default"`. Set to a different key (and declare the matching `JobEnvironment` on the flow), or `None` to leave it off. |
| `existing_cluster_id` / `job_cluster_key` / `new_cluster` | The three classic-compute bindings on `Task`. Setting any clears `environment_key`. |
| `pool` | Default executor flavour for `WorkflowTask.map(...)` in local mode — `"thread"` (default) for I/O-bound bodies, `"process"` for CPU-bound bodies. Ignored in trace mode (Databricks' scheduler owns parallelism there). See [4.23](#423-parallel-fan-out-with-map). |
| `max_workers` | Default worker count for `.map(...)`'s local-mode pool. `None` defers to the executor's own default. Per-call `.map(max_workers=…)` overrides this. |
| `client` | Override the staging workspace (where the `.py` is written) for this one task. Rare — the deployed Job still runs in the flow's workspace. |
| `**task_fields` | `Task.description`, `Task.timeout_seconds`, `Task.run_if`, `Task.email_notifications`, `Task.webhook_notifications`, `Task.health`, `Task.disabled`, `Task.libraries`, … |

### `WorkflowTask.__call__(*args, **kwargs)`

Outside a trace: resolve `SecretRef` args/defaults and call the wrapped function. Inside a trace: register a `TaskNode` and return it.

### `WorkflowTask.map(iterable, *constants, pool=None, max_workers=None, executor=None, **kw_constants) -> list`

Prefect-style parallel fan-out. Outside a trace, runs the body across a `ThreadPoolExecutor` (or `ProcessPoolExecutor` with `pool="process"`), returning a `list` of results in input order. Inside a trace, registers one `TaskNode` per element of *iterable* — the Databricks scheduler runs those tasks in parallel. *constants* / *kw_constants* pass through on every call. See [4.23](#423-parallel-fan-out-with-map).

### `WorkflowTask.after(*upstreams)`

Returns a decorator that, when applied to a task and called inside a trace, adds explicit `depends_on` edges to the resulting node. See [4.10](#410-side-effect-only-ordering-with-after).

### `WorkflowTask.stage(client, node) -> Task`

Render *node* as a Databricks `Task`. Called by `Flow.deploy`; usually you don't need to call it directly.

---

### `secret(scope, key=None, /, *, client=None) -> SecretRef`

Build a `SecretRef`. Accepts two-arg form (`secret("vendor", "api-key")`) or single-arg shortcut (`secret("vendor/api-key")` / `secret("vendor:api-key")`).

`client` (keyword-only) pins a workspace target for resolution. Accepts a live `DatabricksClient` (the base URL is extracted) or a bare host string. When pinned, the staged repr becomes `"ygg.secret('<scope>', '<key>', host='<url>')"` and the cluster builds a fresh `DatabricksClient` for that host at call time.

### `SecretRef.__repr__() -> str`

Returns `"ygg.secret('<scope>', '<key>')"` (or the `host=`-suffixed form when pinned). The staged-script renderer reads this verbatim — any change is an ABI break for the staged code.

---

### `ygg` module surface

| Symbol | Signature |
| ------ | --------- |
| `ygg.secret(scope, key, *, host=None)` | Fetch cleartext from the current `DatabricksClient`, or build a fresh client for the named host. |
| `ygg.task_value(task_key, value_key="__ygg_return__", *, default=None)` | Read from `dbutils.jobs.taskValues`. |
| `ygg.publish_return(value, *, value_key="__ygg_return__")` | Mirror *value* onto `dbutils.jobs.taskValues` and return it. |
| `ygg.RETURN_VALUE_KEY` | `"__ygg_return__"` — the default key. |

---

## 7) Limitations

- **Function source must come from an importable file.** `inspect.getsource` is the staging contract — lambdas, REPL-defined functions, and dynamically generated code won't stage. (Same constraint as `Job.from_callable`.)
- **Closures are limited.** Same-module callables and literal constants the body references *are* inlined into the staged source. Names imported from other modules ride through the auto-dep path (pip install + import) — they're not inlined. Closure cells holding non-literal values are skipped.
- **Trace mode is single-threaded.** A flow body runs sequentially in trace mode; conditional branches based on values that are `TaskNode`s won't work (you'd be branching on a future, not a value). Build branches statically — call `@task` decorators inside `if` blocks that depend on **deploy-time inputs**, not on task results. For fan-out of N parallel tasks over a known iterable, use [`WorkflowTask.map(...)`](#423-parallel-fan-out-with-map) — the iterable must be materialisable at trace time (a literal list, a flow parameter, etc.), not a `TaskNode`.
- **Return-value passthrough uses `dbutils.jobs.taskValues`.** Values must be JSON-serialisable (Databricks restriction); large payloads should be passed via paths instead.
- **One `Job` per flow.** Multi-job orchestration (one flow triggering another job's run) is doable by calling `Jobs` directly inside a task body, but isn't a first-class workflow primitive yet.

---

## 8) Internals (how it works)

### Trace-on-deploy DAG capture

When `Flow.deploy()` runs, it opens a `TraceContext` (a `contextvars.ContextVar`) and calls the flow function once. Every `@task`-decorated callable consults the context var on every call:

```python
def __call__(self, *args, **kwargs):
    trace = current_trace()
    if trace is None:
        # Plain Python — run the body.
        return self.func(*resolved_args, **resolved_kwargs)
    # Recording — return a future.
    node = TaskNode(spec=self, task_key=self.task_key, args=args, kwargs=kwargs)
    return trace.register(node)
```

Dependencies fall out naturally: a `TaskNode` returned by an upstream task and passed into a downstream task gets picked up in the downstream node's `__post_init__` (`_resolve_deps`), which scans `args` + `kwargs.values()` for `TaskNode` instances.

### Staging — function source → `.py` on workspace

`Flow._stage_nodes` walks the captured trace and calls `WorkflowTask.stage(client, node)` on each. That in turn calls into the existing `yggdrasil.databricks.jobs.task.stage_python_callable` (or `stage_python_notebook_callable` for `task_type="notebook"`) which:

1. Extracts the function source via `inspect.getsource`.
2. Inlines same-module helper callables and literal constants the body references.
3. Wraps the body with `@checkargs` for input coercion.
4. Renders the invocation as `ygg.publish_return(func(arg1=..., arg2=...))`.
5. Writes the result to `/Workspace/Shared/.ygg/jobs/<task_key>/main-<digest>.py` (the digest is content-stable — re-staging the same body lands on the same file).
6. Sniffs imports from the rendered AST and feeds the result through `dependencies_to_pip_specs` to populate the matching `JobEnvironment`.

### Why `ygg.secret(...)` and not `{{secrets/scope/key}}`?

Databricks supports `{{secrets/<scope>/<key>}}` substitution in `spark_env_vars` and a couple other slots, but **not** in `SparkPythonTask.parameters`. To get uniform behaviour across both task types (Spark + notebook) and to keep the same code path for local + Databricks Connect runs, we resolve secrets at runtime via the workspace SDK from inside the staged invocation. The cluster pays one extra `GetSecret` call per use — cheap, and the same call shape regardless of where the task runs.

### Why `TaskNode.__repr__` returns a `ygg.task_value(...)` call

The existing `stage_python_callable` renders bound arguments by calling `repr(value)`. By making `TaskNode.__repr__` return a syntactically valid Python expression that calls back into the runtime, we avoid having to special-case `TaskNode` in the renderer — `repr` produces the right thing for both literals and our custom sentinels. Same trick for `SecretRef.__repr__`.

### Singleton-by-config caching

`Flow` and `WorkflowTask` are not singleton-cached themselves — they're decoration-time wrappers. The deployed `Job` *is* a singleton (per `yggdrasil.databricks.jobs.Job`), so re-running `flow.deploy()` returns the same instance with refreshed details. Run handles (`JobRun`) are also singletons (per workspace + run id).

---

## Cross-references

- [`yggdrasil.databricks.jobs`](../jobs/README.md) — the underlying Jobs primitives.
- [`yggdrasil.databricks.secrets`](../secrets/README.md) — what `ygg.secret` resolves against.
- [`yggdrasil.databricks.client`](../README.md#2-authentication-patterns) — `DatabricksClient` auth patterns.
- [`yggdrasil.databricks.fs`](../fs/README.md) — `DatabricksPath`, `VolumePath`, `WorkspacePath`.
