# Databricks CLI (`ygg databricks`)

**YGGDBKS** is a Databricks management CLI powered by the same
`yggdrasil.databricks` service layer documented in the
[Databricks guide](databricks.md). Where that guide is about the Python
API, this page is about the terminal: discovering and driving clusters,
SQL warehouses, jobs, the Workspace/Volumes/DBFS filesystem, and shipping
your code to serverless — all without writing a script.

```bash
pip install "ygg[databricks]"
ygg databricks --help
```

Everything lives under the unified `ygg` entry point:

```bash
ygg databricks <group> <action> [flags]
```

| Group | What it manages |
|---|---|
| [`clusters`](#clusters) | All-purpose compute clusters — list/get/create/delete/start/stop |
| [`warehouses`](#warehouses) | SQL warehouses — list/get/create/delete/start/stop |
| [`job`](#jobs) | Jobs & runs — list/get/run/runs/logs/cancel/repair/delete |
| [`fs`](#filesystem-fs) | Files across Workspace / Volumes / DBFS — ls/cat/write/put/get/mkdir/rm/stat/cp/mv |
| [`wheel`](#wheels-wheel) | Wheel registry — build/upload/deploy/list in the workspace PyPI-like index |
| [`deploy`](#deploy) | Build + upload wheels and assemble serverless environments |
| [`seed`](#seed) | One-shot readiness — check (and provision) wheels, environments, a default warehouse, and config |

---

## Authentication

Every invocation builds a `DatabricksClient` before running the command.
You authenticate exactly the same way as the Python client — via flags,
environment variables, or a config profile. **Flags win; anything you
omit falls back to the `DATABRICKS_*` environment and `~/.databrickscfg`.**

```bash
# Explicit host + PAT
ygg databricks --host https://my-workspace.cloud.databricks.com \
               --token dapi... \
               clusters list

# A named profile from ~/.databrickscfg
ygg databricks --profile prod warehouses list

# Nothing at all — read DATABRICKS_HOST / DATABRICKS_TOKEN / etc. from the env
export DATABRICKS_HOST=https://my-workspace.cloud.databricks.com
export DATABRICKS_TOKEN=dapi...
ygg databricks job list
```

### Client flags

These are accepted **before** the command group and shared by every
sub-command:

| Flag | Env fallback | Purpose |
|---|---|---|
| `--host` | `DATABRICKS_HOST` | Workspace URL or hostname |
| `--token` | `DATABRICKS_TOKEN` | Personal access token |
| `--profile` | `DATABRICKS_CONFIG_PROFILE` | Profile in `~/.databrickscfg` |
| `--debug` | — | Set the `yggdrasil` logger to `DEBUG` |

For OAuth (service principal), Azure, or Google auth, set the standard
`DATABRICKS_*` / cloud-provider environment variables — the underlying
`DatabricksClient` picks them up automatically. (The richer flag surface
— `--client-id`, `--auth-type`, `--azure-tenant-id`, … — is exposed by
the extensible sub-service base described in
[Extending the CLI](#extending-the-cli).)

!!! tip "Where does output go?"
    Machine-readable result rows (ids, names, JSON) are written to
    **stdout** so you can pipe them. Status lines, spinners, and errors go
    to **stderr**. That means `ygg databricks clusters list | awk '{print $1}'`
    gives you clean cluster ids.

The CLI paints its coral **YGGDBKS** banner and colored status glyphs even
when stdout is not a TTY (so it renders inside a Databricks job or notebook
panel). Set `NO_COLOR=1` to opt out.

---

## Clusters

Manage all-purpose compute clusters via `client.compute.clusters`.

```bash
ygg databricks clusters <action> [flags]
```

### `clusters list`

List clusters, optionally filtered by name. Prints
`cluster_id<TAB>cluster_name<TAB>state` per row.

```bash
ygg databricks clusters list
ygg databricks clusters list --name etl
```

### `clusters get`

Show a cluster's runtime, node type, and worker count. Resolve by id or
by name.

```bash
ygg databricks clusters get --id 0712-200234-abcd1234
ygg databricks clusters get --name shared-etl
```

### `clusters create`

Create a cluster. With no `--file`, the spec is assembled from the
individual flags; the cluster is created without blocking
(`wait=False`). Prints the new `cluster_id<TAB>name`.

| Flag | Maps to |
|---|---|
| `--name` (required) | `cluster_name` |
| `--node-type` | `node_type_id` |
| `--num-workers` | `num_workers` |
| `--spark-version` | `spark_version` |
| `--autotermination-minutes` | `autotermination_minutes` |
| `--single-user` | `single_user_name` |
| `-f`, `--file` | Cluster config YAML (overrides the other flags) |

```bash
ygg databricks clusters create \
  --name etl \
  --node-type i3.xlarge \
  --num-workers 2 \
  --spark-version 15.4.x-scala2.12 \
  --autotermination-minutes 30
```

### `clusters delete`

```bash
ygg databricks clusters delete --id 0712-200234-abcd1234
```

### `clusters start` / `clusters stop`

Start a stopped cluster or stop a running one (non-blocking). Resolve by
id or name.

```bash
ygg databricks clusters start --name shared-etl
ygg databricks clusters stop  --id 0712-200234-abcd1234
```

---

## Warehouses

Manage SQL warehouses via `client.warehouses`.

```bash
ygg databricks warehouses <action> [flags]
```

### `warehouses list`

Prints `warehouse_id<TAB>warehouse_name` per row.

```bash
ygg databricks warehouses list
```

### `warehouses get`

```bash
ygg databricks warehouses get --id 0abc123def456789
ygg databricks warehouses get --name "Serverless Starter"
```

### `warehouses create`

Create a SQL warehouse (non-blocking). Prints the new
`warehouse_id<TAB>name`.

| Flag | Maps to |
|---|---|
| `--name` (required) | warehouse name |
| `--cluster-size` | `2X-Small`, `X-Small`, `Small`, `Medium`, `Large`, … |
| `--type` | `PRO` or `CLASSIC` |
| `--serverless` | enable serverless compute |
| `--auto-stop-mins` | idle minutes before auto-stop |

```bash
ygg databricks warehouses create \
  --name analytics \
  --cluster-size Small \
  --type PRO \
  --serverless \
  --auto-stop-mins 10
```

### `warehouses delete`

```bash
ygg databricks warehouses delete --id 0abc123def456789
```

### `warehouses start` / `warehouses stop`

```bash
ygg databricks warehouses start --name analytics
ygg databricks warehouses stop  --id 0abc123def456789
```

---

## Jobs

Manage Databricks jobs and their runs via `client.jobs` /
`client.job_runs`. The job **target** is always either a numeric job id or
a job name — the CLI resolves whichever you pass.

```bash
ygg databricks job <action> [flags]
```

### `job list`

Prints `job_id<TAB>name` per row.

```bash
ygg databricks job list
ygg databricks job list --name nightly-etl --limit 10
```

### `job get`

Show a job and its task DAG — task keys and the edges between them, plus
the workspace explore URL when available.

```bash
ygg databricks job get 123456789
ygg databricks job get nightly-etl
```

### `job run`

Trigger a run, optionally passing parameters and blocking until it
finishes.

| Flag | Purpose |
|---|---|
| `--param k=v` | Job-level parameter (repeatable) |
| `--notebook-param k=v` | Notebook widget value (repeatable) |
| `--python-param v` | Python wheel/script argument (repeatable) |
| `--wait` | Block until the run completes |
| `--timeout` | Seconds to wait when `--wait` (default `1800`) |

```bash
# Fire and forget
ygg databricks job run nightly-etl

# Pass params and wait, exiting non-zero on failure
ygg databricks job run nightly-etl \
  --param env=prod --param window=2026-06-04 \
  --wait --timeout 3600
```

When `--wait` is set, a success prints the duration; a failure prints the
state, the state message, and any captured stderr, and the command exits
`1`.

### `job runs`

List recent runs of a job. Prints `run_id<TAB>state<TAB>duration`.

```bash
ygg databricks job runs nightly-etl --limit 20
ygg databricks job runs nightly-etl --active
```

### `job logs`

Print a run's output. Without `--task`, prints the full debug dump across
tasks; with `--task`, restricts to that one task key.

```bash
ygg databricks job logs 987654321
ygg databricks job logs 987654321 --task transform
```

### `job cancel`

```bash
ygg databricks job cancel 987654321
```

### `job repair`

Re-run failed tasks of a run. With no `--task` it repairs all failed
tasks; repeat `--task` to target specific keys. `--wait` blocks for the
repair to complete.

```bash
ygg databricks job repair 987654321
ygg databricks job repair 987654321 --task transform --task load --wait
```

### `job delete`

```bash
ygg databricks job delete nightly-etl
ygg databricks job delete 123456789
```

---

## Filesystem (`fs`)

A single filesystem CLI over the `DatabricksPath` abstraction — **uniform
across Workspace, Unity Catalog Volumes, and DBFS**. Every `<uri>` is
resolved with `DatabricksPath.from_(uri, client=...)`, so the same verbs
work everywhere, and `cp`/`mv` move bytes *across surfaces* (e.g. a
Workspace file into a Volume) through one read/write contract.

```bash
ygg databricks fs <action> <uri> [flags]
```

URI prefixes:

| Surface | Example URI |
|---|---|
| DBFS | `dbfs:/tmp/data.parquet` |
| Volumes | `/Volumes/main/default/raw/data.parquet` |
| Workspace | `/Workspace/Users/me@co.com/notebook` |

### `fs ls`

List a directory. `-l` shows kind + human size; `-r` recurses.

```bash
ygg databricks fs ls /Volumes/main/default/raw
ygg databricks fs ls -l /Volumes/main/default/raw
ygg databricks fs ls -r dbfs:/tmp
```

### `fs cat`

Stream a file's raw bytes to stdout.

```bash
ygg databricks fs cat /Volumes/main/default/raw/config.json
```

### `fs write`

Write to a path, creating parent directories. Source is `--data` (literal
text), `--file` (local file bytes), or — if neither is given — **stdin**.

```bash
ygg databricks fs write dbfs:/tmp/hello.txt --data "hello"
ygg databricks fs write dbfs:/tmp/blob.bin --file ./local.bin
echo "piped" | ygg databricks fs write dbfs:/tmp/piped.txt
```

### `fs put` / `fs get`

Upload a local file to a remote path, or download a remote file locally.

```bash
ygg databricks fs put ./report.parquet /Volumes/main/default/out/report.parquet
ygg databricks fs get /Volumes/main/default/out/report.parquet ./report.parquet
```

### `fs mkdir`

Create a directory (parents allowed).

```bash
ygg databricks fs mkdir /Volumes/main/default/staging
```

### `fs rm`

Remove a file, or a directory with `-r`.

```bash
ygg databricks fs rm dbfs:/tmp/hello.txt
ygg databricks fs rm -r /Volumes/main/default/staging
```

### `fs stat`

Show path, kind, size, and mtime.

```bash
ygg databricks fs stat /Volumes/main/default/raw/data.parquet
```

### `fs cp` / `fs mv`

Copy or move bytes — including **across surfaces**. `mv` is a copy
followed by a delete of the source.

```bash
# DBFS → Volume
ygg databricks fs cp dbfs:/tmp/data.parquet /Volumes/main/default/raw/data.parquet

# Workspace → Volume, removing the original
ygg databricks fs mv /Workspace/Shared/old.csv /Volumes/main/default/archive/old.csv
```

---

## Wheels (`wheel`)

The wheel registry lifecycle on its own — building, uploading, and
browsing the workspace's PyPI-like wheel index
(`/Workspace/Shared/pypi/<dist>/...` by default). Where [`deploy`](#deploy)
ships the whole ygg image in one shot (wheel **plus** the serverless
`JobEnvironment`), `wheel` lets you drive each step.

```bash
ygg databricks wheel <action> [flags]
```

Like `deploy`, the wheel is built from the **live package on disk**: the
machinery synthesizes a buildable project from the *installed* package's
own files + metadata, so the artifact runs exactly the code you have now —
dev checkout or pip-installed.

### `wheel build`

Build wheel(s) locally from a package — **no workspace upload, no
credentials needed** (it runs `uv`/`pip` locally). Prints the produced
`.whl` paths to stdout.

| Flag | Purpose |
|---|---|
| `--out-dir` | Directory to write the `.whl`(s) into (default: a temp dir) |
| `--extra` | Optional-dependency extra to fold in (repeatable) |
| `-r`, `--requirement` | Extra requirement to bundle alongside (repeatable) |
| `--no-deps` | Pure-python project wheel only; deps resolve at install time |
| `--all-versions` | A wheel for every supported Python (3.10–3.13) |

```bash
ygg databricks wheel build yggdrasil --out-dir ./dist
ygg databricks wheel build mypkg --no-deps --out-dir ./dist
ygg databricks wheel build yggdrasil --extra databricks --all-versions
```

### `wheel upload`

Upload one or more **prebuilt** `.whl` files into the registry. Prints
each resulting workspace path.

```bash
ygg databricks wheel upload ./dist/mypkg-1.0-py3-none-any.whl
ygg databricks wheel upload ./dist/*.whl --workspace-dir /Workspace/Shared/pypi
```

This pairs with `wheel build` for an air-gapped flow: build locally on a
machine with the package, then upload the artifacts from one that has
workspace access.

### `wheel deploy`

Build the live package **and** upload its wheel(s) in one step — the wheel
half of `deploy` without the environment JSON. Prints the workspace
path(s). Same build flags as `wheel build`.

```bash
ygg databricks wheel deploy yggdrasil --extra databricks
ygg databricks wheel deploy mypkg -r "pandas>=2" --no-deps
```

### `wheel list`

Browse the registry. With no package it lists the distribution folders;
with a package (import or distribution name) it lists that distribution's
deployed `.whl` files.

```bash
ygg databricks wheel list              # ygg/  mypkg/  ...
ygg databricks wheel list ygg          # /Workspace/Shared/pypi/ygg/ygg-0.8.45-py3-none-any.whl
ygg databricks wheel list --workspace-dir /Workspace/Users/me@co.com/pypi
```

---

## Deploy

Ship your Python code to Databricks serverless. The machinery in
`yggdrasil.databricks.job.wheel` builds a wheel from the **live package on
disk**, uploads it into the workspace's PyPI-like registry
(`/Workspace/Shared/pypi/<dist>/` by default), and assembles the
serverless `JobEnvironment` that installs it.

```bash
ygg databricks deploy [subcommand] [flags]
```

Shared flags (accepted on `deploy` and most subcommands):

| Flag | Purpose |
|---|---|
| `--workspace-dir` | PyPI-like registry root (default `/Workspace/Shared/pypi`) |
| `--rebuild` | Force a fresh build even if the version is already deployed |
| `--all-versions` | One wheel + environment per supported Python (3.10–3.13) |

### Bare `deploy`

Ships the **ygg image**: builds/uploads the wheel(s), then prints the
serverless `JobEnvironment` JSON assembled off that fresh build (no double
build).

```bash
ygg databricks deploy
ygg databricks deploy --rebuild
ygg databricks deploy --all-versions
```

### `deploy ygg`

Build + upload only the versioned ygg image wheel(s); prints the deployed
workspace path(s).

```bash
ygg databricks deploy ygg
ygg databricks deploy ygg --all-versions
```

### `deploy wheel <package>`

Build + upload **any** package's wheel(s) by import or distribution name.

| Flag | Purpose |
|---|---|
| `--extra` | Optional-dependency extra to fold in (repeatable) |
| `-r`, `--requirement` | Extra requirement to bundle alongside (repeatable) |
| `--no-deps` | Project wheel only; deps resolve on the cluster |
| `--all-versions` | A wheel for every supported Python |

```bash
ygg databricks deploy wheel yggdrasil --extra databricks
ygg databricks deploy wheel mypkg -r "pandas>=2" --no-deps
```

### `deploy environment`

Build the ygg wheel(s) and print the serverless `JobEnvironment`(s) as
JSON — ready to paste into a job's `environments` block. Aliased as
`deploy env`.

| Flag | Purpose |
|---|---|
| `--key` | `environment_key` for the config (default `default`) |
| `--env-version` | Serverless environment version (default: matched to local Python) |
| `--rebuild` | Force a fresh wheel build first |
| `--all-versions` | One `JobEnvironment` per supported Python plus a default |

```bash
ygg databricks deploy environment --key default
ygg databricks deploy env --all-versions > environments.json
```

---

## Seed

One command to answer *"is this workspace ready to run ygg, and if not,
make it ready"*. `seed` walks four prerequisites and, by default,
provisions anything missing:

| Area | Checks | Provisions (default mode) |
|---|---|---|
| **config** | connectivity, host, current user, default catalog/schema, workspace id | — (read-only) |
| **wheels** | the versioned ygg image wheel in the registry | builds + uploads it ([`deploy ygg`](#deploy)) |
| **environments** | the serverless env version + runtime deps | assembles the `JobEnvironment` off the fresh wheel |
| **warehouses** | a SQL warehouse to execute against | ensures a default (creates a serverless one if none) |

```bash
ygg databricks seed
```

| Flag | Purpose |
|---|---|
| `--check` | **Read-only**: report readiness, create/upload nothing. Exits `1` if anything is missing — use it as a CI gate. |
| `--rebuild` | Force a fresh wheel build even if the version is already deployed. |
| `--all-versions` | Seed a wheel + environment for every supported Python (3.10–3.13). |
| `--workspace-dir` | PyPI-like registry root (default `/Workspace/Shared/pypi`). |

The report is sectioned with status glyphs — `✓` ready, `▲` missing,
`✗` errored:

```text
  ygg databricks seed  (check mode)
  ●  config
    host      https://ws.example.com
    user      me@example.com
    catalog   main
  ✓  config reachable
  ●  wheels
  ▲  ygg 0.8.52 wheel not deployed under /Workspace/Shared/pypi/ygg
  ●  environments
  ✓  environment config resolvable
  ●  warehouses
  ✓  1 warehouse(s) available
  ▲  prerequisites missing — run `ygg databricks seed` to provision
```

Connectivity is the gate: if the workspace can't be reached (bad host /
token), `seed` reports the auth error and exits non-zero without touching
anything else. Use it right after configuring a workspace, or in CI:

```bash
# fail the pipeline unless the workspace is fully provisioned
ygg databricks --profile prod seed --check
```

---

## Extending the CLI

There are two surfaces in the tree:

- **`yggdrasil.databricks.cli`** — the working `ygg databricks` CLI
  documented above. Each group is a small command class
  (`ClustersCommand`, `WarehousesCommand`, `JobsCommand`, `FSCommand`,
  `WheelCommand`, `DeployCommand`, `SeedCommand`) that registers an argparse sub-parser and dispatches
  straight into the service layer. To add a group, write a class with a
  `register(subparsers)` classmethod and add it to
  `cli/services/__init__.py` + the registration block in `cli/__init__.py`.

- **`yggdrasil.cli.databricks`** — an abstract base
  (`DatabricksCLI`) for **standalone** `ygg-<service>` console scripts that
  need the full `DatabricksClient` flag surface (`--host`, `--token`,
  `--profile`, `--auth-type`, `--client-id`, all the Azure/Google flags,
  …). Subclass it, override `add_service_arguments(parser)` for your
  service's flags, implement `run()`, and expose a one-line
  `main(argv)` that calls `cls.parse_and_run(argv)`. The base owns the
  shared client flag group and the client-construction handshake (exit
  code `2` on construction failure), so a new sub-service CLI is rarely
  more than a ~30-line subclass.

---

## Troubleshooting

- **`401` / `403`** — verify the host + token pair and whether you need
  workspace vs. account scope. Re-run with `--debug` for the full SDK
  traceback.
- **`no job matching ...` / `Warehouse not found`** — the name didn't
  resolve. Use `job list` / `warehouses list` to confirm the exact name,
  or pass the numeric id.
- **`fs` path not found** — pick the right prefix: `dbfs:/...` for DBFS,
  `/Volumes/...` for Volumes, `/Workspace/...` for Workspace files.
- **Job run exits `1` under `--wait`** — that's by design: a failed run
  surfaces its state message + stderr and returns non-zero so scripts and
  CI can branch on it.
- **`deploy` build is slow / re-builds every time** — drop `--rebuild` to
  reuse an already-deployed version; the wheel is keyed by version.
- **Interrupted** — `Ctrl+C` exits cleanly with code `130`.

## See also

- [Databricks (Python API)](databricks.md) — the service layer this CLI drives.
- [`ygg` node CLI](node.md) — node lifecycle and execution.
- [databricks/job](../modules/databricks/job/README.md), [databricks/fs](../modules/databricks/fs/README.md), [databricks/compute](../modules/databricks/compute/README.md), [databricks/warehouse](../modules/databricks/warehouse/README.md).
