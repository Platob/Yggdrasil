# yggdrasil.environ

Python environment management — virtual-env lifecycle, package management, runtime imports, and typed parameter resolution for notebooks and scripts.

## What this module gives you

| Symbol | Purpose |
| --- | --- |
| `PyEnv` | Manages a Python interpreter: create/reuse venvs, install packages, run subprocesses, resolve Spark sessions |
| `runtime_import_module` | Import-or-install a module against the current interpreter |
| `SystemParameters` | Typed parameter bag that reads from CLI args, env vars, Databricks widgets, and job params |
| `UserInfo` | Current user and host introspection |

---

## PyEnv — Python interpreter wrapper

`PyEnv` wraps a single Python interpreter and exposes venv lifecycle, package management (uv-first, pip fallback), and subprocess execution.

### Get the current interpreter

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()   # wraps sys.executable
print(env.python_path)
print(env.version_info)   # VersionInfo(major=3, minor=12, ...)
```

### Resolve by path or version selector

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

# Explicit path
env = PyEnv.instance(python_path=Path("/usr/bin/python3.12"))

# Version string — finds a compatible interpreter
env = PyEnv.instance(python_path="3.12")
```

### Create a virtual environment

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

env = PyEnv.current()

# Create a new venv at the given directory
venv = env.create(venv_dir=Path("/tmp/my-venv"), seed=True)   # seed installs pip/wheel
print(venv.python_path)
print(venv.root_path)    # /tmp/my-venv
print(venv.bin_path)     # /tmp/my-venv/bin (or Scripts on Windows)
```

### Get or reuse an existing venv

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

env = PyEnv.current()

# Creates if missing, returns existing if present
venv = env.venv(venv_dir=Path("/tmp/shared-venv"))
```

### Install packages

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()

# Single package
env.install("polars")
env.install("ygg[databricks]")

# Multiple packages
env.install(["polars", "pyarrow>=20", "duckdb"])

# Pin version
env.install("polars==1.20.0")
```

### Update packages

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()
env.update("polars")   # upgrade to latest
```

### Uninstall packages

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()
env.uninstall("polars")
```

### Run Python code in a subprocess

```python
from yggdrasil.environ import PyEnv

env = PyEnv.current()

stdout = env.run_python_code("import sys; print(sys.version)")
print(stdout)
```

### Environment detection

```python
from yggdrasil.environ import PyEnv

print(PyEnv.in_databricks())          # True if running on a Databricks cluster
print(PyEnv.in_databricks_notebook()) # True if running in a Databricks notebook cell
print(PyEnv.in_aws_lambda())          # True if running in AWS Lambda
print(PyEnv.in_aws_batch())           # True if running in AWS Batch
print(PyEnv.in_aws())                 # True on any AWS-managed surface
print(PyEnv.should_use_databricks_connect())  # True outside Databricks with DATABRICKS_HOST set
```

### Spark session resolution

```python
from yggdrasil.environ import PyEnv

# Resolve a SparkSession (creates one if none exists)
spark = PyEnv.current().spark_session()

# With Databricks Connect (outside cluster)
spark = PyEnv.current().spark_session(connect=True)
```

---

## runtime_import_module

Module-level convenience for the most common use case: import a module, installing it first if missing.

```python
from yggdrasil.environ import runtime_import_module

# Import (raises ImportError if not installed)
polars = runtime_import_module("polars")

# Import or install if missing
polars = runtime_import_module("polars", install=True)

# Install with a different pip name
jwt = runtime_import_module("jwt", pip_name="PyJWT", install=True)
```

---

## SystemParameters — typed notebook / job config

`SystemParameters` is a typed mapping that reads parameter values from multiple sources in priority order: Databricks widget → job parameter → environment variable → CLI argument → default. It auto-casts string values to the field's declared type.

### Minimal example

```python
from dataclasses import dataclass
from yggdrasil.environ import SystemParameters

@dataclass
class Config(SystemParameters):
    catalog: str
    schema: str
    table: str
    dry_run: bool = False
    max_rows: int = 10_000

# Read from all available sources (widgets, job params, env, CLI)
cfg = Config.from_environment()
print(cfg.catalog, cfg.schema, cfg.table)
print(cfg.dry_run, cfg.max_rows)
```

### Source-specific constructors

```python
from yggdrasil.environ import SystemParameters

@dataclass
class Config(SystemParameters):
    catalog: str = "main"
    schema: str = "default"

# From Databricks widgets (notebook context)
cfg = Config.from_dbutils()

# From environment variables (e.g. CATALOG=main SCHEMA=default)
cfg = Config.from_environ()

# From sys.argv (CLI: --catalog main --schema default)
cfg = Config.from_argv()

# From a dict
cfg = Config.from_({"catalog": "main", "schema": "default"})

# Auto-detect best source
cfg = Config.from_environment()
```

### Initialize Databricks widgets

Call `init_widgets()` once at notebook startup to register text/dropdown widgets so users can override values interactively:

```python
from yggdrasil.environ import SystemParameters

@dataclass
class RunConfig(SystemParameters):
    catalog: str = "main"
    schema: str = "staging"
    dry_run: bool = True

# Register widgets in the notebook (skip if already registered)
RunConfig.init_widgets()

# Read the current widget values
cfg = RunConfig.from_environment()
print(cfg)
```

### Access `dbutils`

```python
from yggdrasil.environ import SystemParameters

@dataclass
class Config(SystemParameters):
    catalog: str = "main"

cfg = Config.from_environment()
dbutils = cfg._dbutils   # dbutils handle when in Databricks
```

### `nice_label` — snake_case to title case

```python
from yggdrasil.environ import nice_label, LABEL_ACRONYMS

print(nice_label("dry_run"))     # "Dry Run"
print(nice_label("catalog_id"))  # "Catalog ID"
print(nice_label("aws_region"))  # "AWS Region"
```

---

## UserInfo — current user and host

```python
from yggdrasil.environ import UserInfo

info = UserInfo.current()

print(info.username)       # OS or Databricks login name
print(info.hostname)       # machine name
print(info.platform)       # "linux", "darwin", "windows"
print(info.python_version) # "3.12.3"
```

---

## End-to-end: isolated venv for a pipeline dependency

```python
from yggdrasil.environ import PyEnv
from pathlib import Path

# Isolate heavy dependencies from the base install
venv = PyEnv.current().get_or_create(venv_dir=Path("/tmp/pipeline-venv"))
venv.install(["pyspark==4.0.0", "delta-spark==3.3.0"])

# Run a pipeline script in that venv
venv.run_python_code(
    "from pyspark.sql import SparkSession; print(SparkSession.builder.getOrCreate().version)"
)
```
