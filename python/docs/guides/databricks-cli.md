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
| [`configure`](#configure) | Write a `~/.databrickscfg` profile and remember it as the current session |
| [`clusters`](#clusters) | All-purpose compute clusters — list/get/create/delete/start/stop |
| [`warehouses`](#warehouses) | SQL warehouses — list/get/create/delete/start/stop |
| [`sql`](#sql) | Run SQL and export results — query/export (incl. `export --statement-id`) |
| [`job`](#jobs) | Jobs & runs — list/get/run/runs/logs/cancel/repair/delete |
| [`fs`](#filesystem-fs) | Files across Workspace / Volumes / DBFS — ls/cat/write/put/get/mkdir/rm/stat/cp/mv |
| [`wheel`](#wheels-wheel) | Wheel registry CRUD — create/find/update/delete/list in the workspace index |
| [`deploy`](#deploy) | Deploy the current project — environment + default warehouse + cluster |

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

## Configure

Set up authentication once, the way `databricks configure` does — write a
profile into `~/.databrickscfg` — then have ygg **remember it as the
current session** so later tooling can default to "the workspace I last
configured".

```bash
ygg databricks configure <action> [flags]
```

### Bare `configure`

Write (or update) a profile, verify the credentials, and remember the
session. Host and token are taken from flags, or **prompted for
interactively** (the token hidden) when omitted — exactly like
`databricks configure --token`. Existing profiles in the file are
preserved; only the named section is rewritten, and the file is chmod'd to
`600`.

```bash
# Interactive — prompts for host, then a hidden token
ygg databricks configure

# Non-interactive — write the DEFAULT profile
ygg databricks configure --host https://my-ws.cloud.databricks.com --token dapi...

# A named profile
ygg databricks configure --profile prod --host https://prod... --token dapi...

# OAuth service principal instead of a PAT
ygg databricks configure --profile sp --host https://ws \
  --client-id <id> --client-secret <secret>

# SSO — interactive browser login; no secret on disk, the session token
# is captured into the remembered session
ygg databricks configure --profile browser --host https://ws --sso
```

| Flag | Purpose |
|---|---|
| `--profile` | Profile name to write (default `DEFAULT`) |
| `--host` | Workspace URL (prompted if omitted; `https://` is prepended when missing) |
| `--token` | Personal access token (prompted hidden if omitted) |
| `--client-id` / `--client-secret` | OAuth service-principal credentials — written instead of a token |
| `--sso` | Authenticate via SSO (interactive browser). No static secret is written; the resolved session token is dumped into the session |
| `--auth-type` | Explicit auth type (`external-browser`, `azure-cli`, `databricks-cli`, …); implies `--sso` when no token/secret is given |
| `--account-id` | Account id for account-level profiles |
| `--config-file` | Config file path (default `$DATABRICKS_CONFIG_FILE` or `~/.databrickscfg`) |
| `--no-verify` | Skip the credential check (don't call the workspace) |
| `--no-session` | Write the profile but don't remember it as the current session |

After writing, the freshly-saved profile is loaded into a
`DatabricksClient`, the current user is resolved to confirm the
credentials work, that client becomes the process **current** client, and
a non-sensitive snapshot of the session — profile, host, user,
workspace/account ids, timestamp — is dumped into the session folder
`~/.config/databricks-sdk-py/sessions/` as `<hostname>.json` (the
per-machine default). **No secrets** (token / client secret) are written
into the session file. A failed verification still keeps the profile on
disk (it just warns).

For an **SSO** login (`--sso` / `--auth-type`) the credential isn't on disk
— it's an ephemeral bearer minted by the interactive flow — so the
resolved session token *is* captured into the snapshot (`access_token`),
letting later tooling replay the session without re-prompting the browser.
The session file is then locked to owner-only (`600`).

### `configure list`

List the profiles in `~/.databrickscfg` (`profile<TAB>host`), marking the
one remembered as the current session.

```bash
ygg databricks configure list
```

### `configure session`

Print the remembered latest-session metadata (the JSON snapshot).

```bash
ygg databricks configure session
```

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

## SQL

Run SQL against the workspace's SQL warehouse and **export results to a
file** — locally, to a Volume/DBFS/Workspace path, or to `s3://`.

```bash
ygg databricks sql <action> [flags]
```

Both actions accept warehouse routing and bind params:

| Flag | Purpose |
|---|---|
| `--warehouse-id` / `--warehouse-name` | Run on a specific warehouse (default: the workspace default) |
| `--param k=v` | Bind a `:name` parameter (repeatable) — never f-string user values |
| `--format` | Override the export format (`csv`/`parquet`/`arrow`/`ndjson`/`json`) |

### `sql query`

Execute a statement and either **preview** it (default: a 50-row
preview, printed as clean rows so it stays pipeable) or write the full
result to `--target`. It echoes the **`statement_id`** to stderr so you can
re-export the same result later without re-running.

```bash
ygg databricks sql query "SELECT * FROM main.default.orders LIMIT 50"
ygg databricks sql query "SELECT * FROM main.default.orders" --target orders.parquet
ygg databricks sql query "SELECT * FROM t WHERE id = :id" --param id=42 --limit 1000
```

Aliased as `sql exec` / `sql run`.

### `sql export`

Write a result to `--target`, sourced either from an **already-executed
statement** (`--statement-id`) or from a query run on the spot
(`--query`). The Databricks Statement Execution API keeps a finished
result available for a window, so re-fetching by id costs no re-run.

```bash
# Re-fetch a prior statement's result and write it out
ygg databricks sql export --statement-id 01ef0a2b-… --target /Volumes/main/default/stg/out.csv

# Run-and-export in one step
ygg databricks sql export --query "SELECT * FROM main.default.orders" --target out.parquet

# Force a format when the target has no usable extension
ygg databricks sql export --statement-id 01ef… --target ./result --format parquet
```

The export **format** is taken from the target's extension (`.csv`,
`.parquet`, `.arrow`, `.ndjson`, `.json`) unless `--format` overrides it.
A target shaped like `dbfs:/…`, `/Volumes/…`, or `/Workspace/…` is written
into the workspace; anything else (local path, `s3://…`) goes through the
generic path layer.

!!! tip "query → export workflow"
    `sql query` prints the `statement_id`; capture it and hand it to
    `sql export --statement-id` to materialise the same result in any
    format/destination without paying for the query twice.

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

### `fs create-notebook`

Create a notebook at a **`/Workspace`** path (imports via the Workspace
`SOURCE` format so the object lands as a real notebook, not a plain file).
`--language` picks `PYTHON` (default), `SQL`, `SCALA`, or `R`; source comes
from `--data` (literal) or `--file` (local file), or is empty when neither
is given. Parents are created automatically; `--overwrite` replaces an
existing notebook.

```bash
# empty Python notebook
ygg databricks fs create-notebook /Workspace/Users/me@co.com/scratch

# SQL notebook from a literal body
ygg databricks fs create-notebook /Workspace/Shared/report \
  --language SQL --data "SELECT 1"

# import a local source file, replacing any existing notebook
ygg databricks fs create-notebook /Workspace/Shared/etl \
  --file ./etl.py --overwrite
```

### `fs run-notebook`

Submit a **`/Workspace`** notebook as a one-time job run (a `run_id` with
no persisted job). `--param k=v` (repeatable) passes notebook parameters —
they land on the run's widget bindings, so inside the notebook they're
caught by `SystemParameters` (the union of `dbutils.widgets` and
`{{job.parameters.*}}`) or `dbutils.widgets.get(<name>)`.

Compute is defaulted for you: `--cluster <id>` pins existing compute,
otherwise the run goes **serverless**. The serverless environment is
resolved automatically — `--environment <name>` selects a deployed base
environment (or a `.yml` path) from the shared environment path, and the
default picks up the **running client project's** environment when deployed,
else the deployed **ygg** base environment
(`/Workspace/Shared/environment/ygg/ygg-<version>-py3XX.yml`), falling back to
the workspace default serverless compute when none is deployed. Blocks until the
run finishes (`--timeout` seconds) unless `--no-wait` is given.

```bash
# run on serverless (auto ygg env), waiting for the result
ygg databricks fs run-notebook /Workspace/Shared/etl \
  --param date=2024-01-01 --param region=eu

# run on a named, deployed serverless environment
ygg databricks fs run-notebook /Workspace/Shared/Meteologica/databricks/espark_category.py \
  --environment meteologica --param category=wind

# fire-and-forget on a specific cluster
ygg databricks fs run-notebook /Workspace/Shared/etl \
  --cluster 0123-456789-abcde --no-wait
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

Uniform **CRUD** over the workspace's PyPI-like wheel registry
(`/Workspace/Shared/pypi/<dist>/<version>/...` — distribution **and** version are
folder levels). A wheel is keyed by `(project, version)`.
`create`/`find` **fetch** it — a local path (with a `pyproject.toml`) is built
from source (`uv build`), anything else is a **PyPI** project downloaded by
name (`pip download`) — and upload it. Fetches use a local on-disk cache so
repeated builds/downloads/uploads are cheap. Every project, `ygg` included, is
handled the same way. Versions parse + compare via
[`yggdrasil.version.VersionInfo`](../).

```bash
ygg databricks wheel <action> <project> [version] [flags]
```

| Action | Purpose |
|---|---|
| `create <project> [version]` | Fetch (build/download) + upload the wheel(s) |
| `find <project> [version]` | Get the wheel, **building it on a miss** (`--no-install` to skip) |
| `get <project> [version]` | Get a deployed wheel, **never builds** |
| `update <project> [version]` | Re-fetch + overwrite |
| `delete <project> [version]` | Remove (a version, or all) |
| `list [project]` | Browse: distributions, or a project's wheels |

Common flags: `--python 3.11` (target a Python), `--extra <name>` (fold in an
extra, repeatable), `--deps` (also upload the whole dependency closure),
`--rebuild` (force a fresh fetch), `--workspace-dir`.

```bash
ygg databricks wheel create ./my-app                 # build the local project, upload
ygg databricks wheel create polars 1.2.0 --deps      # mirror a PyPI release + closure
ygg databricks wheel find ygg                         # get-or-build the ygg wheel
ygg databricks wheel get mypkg 1.0                    # deployed lookup, no build
ygg databricks wheel update ygg                       # re-fetch + overwrite
ygg databricks wheel delete mypkg 0.9                 # remove one version
ygg databricks wheel list                             # ygg/  mypkg/  ...
ygg databricks wheel list ygg                         # the ygg distribution's wheels
```

---

## Environments (`environment`)

Uniform **CRUD** over reusable base environments under
`/Workspace/Shared/environment/<proj>/<version>/`. An environment is keyed by
`(project, version)`. `create`/`find` fetch the project + its **whole dependency
closure as wheels** (zero-PyPI — the runtime never resolves from a live index)
and write the serverless `<stem>.yml` + cluster `<stem>.requirements.txt`. Same
local-path-or-PyPI rule and the same actions as `wheel`:

```bash
ygg databricks environment create <project> [version] [--python 3.11] [--extra <name>]
ygg databricks environment find <project> [version]    # build on a miss
ygg databricks environment get  <project> [version]    # never builds
ygg databricks environment update <project> [version]  # re-build + overwrite
ygg databricks environment delete <project> [version]
ygg databricks environment list
```

## Deploy

Take **your own project** to Databricks in one command. `ygg databricks deploy
[path]` builds the project's base environment (via
[`environment create`](#environments-environment) — wheel closure, zero-PyPI),
then provisions its **default serverless SQL warehouse** and **default
single-user cluster** — both named for the project's capitalized display name —
wired to that env config. The project is discovered from `path` (a dir or
`pyproject.toml`) or the cwd. The client is bound to the deployed project +
version first (so the warehouse/cluster resolve to *this* project), and the
warehouse + cluster are created **fire-and-forget** — the command doesn't block
on them reaching a running state.

```bash
ygg databricks deploy [path] [flags]
```

| Flag | Purpose |
|---|---|
| `--extra` | optional-dependency extra to fold into the environment (repeatable) |
| `--python` | Python version to build the environment for, e.g. `3.11` (repeatable; default: the interpreter running the CLI) |
| `--rebuild` | Force a fresh build of the wheel closure + env config |
| `--no-cluster` | Don't provision the default single-user cluster |
| `--no-warehouse` | Don't provision the default serverless SQL warehouse |
| `--single-user` | Single-user owner for the cluster (default: the current user) |
| `--workspace-dir` | Environment root (default `/Workspace/Shared/environment`) |

```bash
ygg databricks deploy                          # discover from the cwd
ygg databricks deploy ./my-app --extra databricks
ygg databricks deploy --python 3.11 --python 3.12   # one environment per Python
ygg databricks deploy --rebuild                # force a fresh build of everything
ygg databricks deploy --no-cluster             # env + warehouse only
```

With several `--python` flags, one environment is built per version (the
cluster, which runs a single Python, installs the first; serverless picks the
matching environment per job at runtime).

> The lower-level wheel/environment CRUD lives under the dedicated
> [`ygg databricks wheel`](#wheels-wheel) and
> [`ygg databricks environment`](#environments-environment) commands.

---

## Extending the CLI

There are two surfaces in the tree:

- **`yggdrasil.databricks.cli`** — the working `ygg databricks` CLI
  documented above. Each group is a small command class
  (`ConfigureCommand`, `ClustersCommand`, `WarehousesCommand`, `SQLCommand`, `JobsCommand`, `FSCommand`,
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
- [Loki agent](loki.md) — the global yggdrasil agent.
- [databricks/job](../modules/databricks/job/README.md), [databricks/fs](../modules/databricks/fs/README.md), [databricks/compute](../modules/databricks/compute/README.md), [databricks/warehouse](../modules/databricks/warehouse/README.md).
