# yggdrasil.environ

Runtime environment management — Python interpreter discovery, virtual-env
lifecycle, package installation, import-or-install, and user-identity
resolution.

## Surface map

| Symbol | Use for |
|---|---|
| `PyEnv` | Manage a Python interpreter + its venv — install, run, import |
| `runtime_import_module` | Import a module, auto-installing it if missing |
| `UserInfo` | Resolve the current user's identity (email, name, hostname, compute context) |
| `SystemParameters` | Read typed parameters from env vars / job widgets |
| `cached_from_import` | Cache-on-import guard — import once, freeze the result |

---

## 1) One-liners

```python
from yggdrasil.environ import PyEnv, runtime_import_module, UserInfo

# Current Python interpreter (the one running this process)
env = PyEnv.current()

# Import (and auto-install if missing)
orjson = runtime_import_module("orjson", pip_package="orjson")

# Current user
me = UserInfo.current()
print(me.email, me.hostname)
```

---

## 2) `PyEnv` — interpreter and venv management

### Locate interpreters

```python
from yggdrasil.environ import PyEnv

# The running interpreter
env = PyEnv.current()
print(env.executable)   # /usr/bin/python3.12
print(env.version)      # (3, 12, 0)

# A specific version (finds the first matching python on PATH)
env312 = PyEnv.from_version("3.12")
env310 = PyEnv.from_version("3.10")

# A concrete path
env = PyEnv.from_path("/opt/homebrew/bin/python3")

# A venv directory
env = PyEnv.from_venv("/project/.venv")
```

### Create / reuse a virtual environment

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

env = PyEnv.current()

# Create a new venv (uses `uv venv` when available, falls back to venv)
venv = env.create_venv(Path("/tmp/my-venv"))

# Create only if it doesn't already exist
venv = env.ensure_venv(Path("/tmp/my-venv"))

# Delete
venv.delete()
```

### Install / update / uninstall packages

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()

# Install into the current env
env.install("httpx", "rich>=12")

# Install from a requirements file
env.install_requirements("/project/requirements.txt")

# Install extras of a package
env.install("ygg[databricks,data]")

# Uninstall
env.uninstall("httpx")
```

### Run Python code or a module

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()

# Run a script file
result = env.run_file("/scripts/ingest.py", args=["--date", "2026-01-01"])
print(result.stdout)

# Run a module
result = env.run_module("pytest", args=["tests/"])
print(result.returncode)

# Execute a Python expression
result = env.run_code("import sys; print(sys.version)")
```

### Import-or-install at runtime

```python
from yggdrasil.environ import runtime_import_module

# Auto-installs if missing; raises on failure
httpx = runtime_import_module("httpx")
resp  = httpx.get("https://example.com")

# Explicit pip name when it differs from the import name
yaml = runtime_import_module("yaml", pip_package="PyYAML")

# Pin a version
pydantic = runtime_import_module("pydantic", pip_package="pydantic>=2")
```

---

## 3) `UserInfo` — identity resolution

`UserInfo` auto-detects: email from git config, Databricks notebook context,
Spark jobs, or the system user. It's a singleton — every call returns the
same instance.

```python
from yggdrasil.environ import UserInfo

me = UserInfo.current()

print(me.email)         # "alice@example.com" or None
print(me.first_name)    # "Alice"
print(me.last_name)     # "Smith"
print(me.hostname)      # "ws-12345.azuredatabricks.net" or local hostname
print(me.key)           # stable identity key (sha256 of email or hostname+user)
print(me.hash)          # int64 hash (usable as dict key / Arrow column)
```

### Databricks-context link

```python
from yggdrasil.environ import UserInfo

me = UserInfo.current()

# Resolves a URL to the current Databricks job run, notebook, or workspace
link = me.databricks_link(kind="auto")  # "auto" | "job_run" | "notebook_id"
print(link)   # https://adb-123.azuredatabricks.net/...
```

### Arrow struct schema

`UserInfo` ships a canonical Arrow schema so you can embed it verbatim in
table schemas for provenance tracking:

```python
from yggdrasil.environ.userinfo import USERINFO_SCHEMA
import pyarrow as pa

schema = pa.schema([
    pa.field("event_id", pa.string()),
    pa.field("user",     USERINFO_SCHEMA.field("hash")),  # int64
    pa.field("ts",       pa.timestamp("us", tz="UTC")),
])
```

---

## 4) `SystemParameters` — typed env-var reading

```python
from yggdrasil.environ import SystemParameters
import os

params = SystemParameters.from_env()

# Read a string parameter (env var)
catalog = params.get("CATALOG", default="main")

# Read with type coercion
max_rows = params.get("MAX_ROWS", default=1000, dtype=int)
dry_run  = params.get("DRY_RUN",  default=False, dtype=bool)
```

---

## 5) `cached_from_import`

Guarantees an expensive import runs exactly once per process, then freezes
the result in a module-level name. Used internally by `lib.py` guards:

```python
from yggdrasil.environ import cached_from_import

# Equivalent to: `import polars as pl` — but cached
polars = cached_from_import("polars")
print(polars.__version__)

# With a helpful install message on failure
blake3 = cached_from_import(
    "blake3",
    install_hint="pip install ygg[pickle]",
)
```

---

## 6) Advanced: full venv bootstrap flow

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

# Build a fresh isolated venv for a Databricks job's dependencies
job_venv = Path("/Volumes/main/raw/landing/.venv")
env      = PyEnv.current()
venv     = env.ensure_venv(job_venv)
venv.install("ygg[databricks,data,http]", "httpx", "orjson")

# Run the job entrypoint inside the venv
result = venv.run_module(
    "mypackage.ingest",
    args=["--catalog", "main", "--date", "2026-01-01"],
)
if result.returncode != 0:
    raise RuntimeError(result.stderr)
```
