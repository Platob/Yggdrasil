# yggdrasil.environ

Runtime environment helpers — Python interpreter introspection, dynamic import/install, user identity, and system parameter utilities. Used internally by the Databricks jobs layer and the optional-dependency pattern.

---

## One-liner

```python
from yggdrasil.environ import runtime_import_module

# Import a module, installing its pip package automatically if missing
json_module = runtime_import_module("orjson", install_if_missing=True)
```

---

## 1) `runtime_import_module` — import-or-install

```python
from yggdrasil.environ import runtime_import_module

# Import normally (raises ImportError if missing)
pa = runtime_import_module("pyarrow")

# Auto-install via uv/pip if the module is not present
pl = runtime_import_module("polars", install_if_missing=True)

# Install under a different pip name than the import name
bs4 = runtime_import_module("bs4", install_if_missing=True, pip_name="beautifulsoup4")
```

The install respects the current virtual environment. It uses `uv pip install` when `uv` is available, falling back to `pip`.

---

## 2) `PyEnv` — Python interpreter wrapper

Wraps a Python interpreter — useful for running code in a subprocess or bootstrapping a fresh venv.

```python
from yggdrasil.environ import PyEnv

# Current interpreter
env = PyEnv()
print(env.version)       # "3.12.3"
print(env.executable)    # "/usr/bin/python3"

# Run a code snippet
result = env.run("import sys; print(sys.version)")
print(result.stdout)

# Install packages into the environment
env.pip_install(["polars>=1.0", "pyarrow>=20"])
```

### From a specific path

```python
env = PyEnv(path="/opt/venv/bin/python")
env.pip_install(["ygg[databricks]"])
```

---

## 3) `UserInfo` — user and machine identity

`UserInfo` aggregates user/process identity — used by the Databricks workflow layer to stamp job metadata with author and project information.

```python
from yggdrasil.environ import UserInfo

info = UserInfo.current()

print(info.username)        # OS username
print(info.hostname)        # machine hostname
print(info.product)         # project name (from nearest pyproject.toml)
print(info.product_version) # project version
print(info.email)           # git user.email (if configured)
```

### In Databricks job context

```python
from yggdrasil.databricks import DatabricksClient
from yggdrasil.environ import UserInfo

client = DatabricksClient()
info   = UserInfo.current()

job_name = f"[{info.product}/{info.product_version}] daily-etl"
tags     = {"author": info.email or info.username, "host": info.hostname}
```

---

## 4) `SystemParameters` — runtime parameter introspection

Parses system-level parameters and provides human-readable labels.

```python
from yggdrasil.environ import SystemParameters

params = SystemParameters()
print(params.python_version)
print(params.platform)
```

### `nice_label` — convert camelCase / snake_case to display labels

```python
from yggdrasil.environ import nice_label

print(nice_label("catalog_name"))    # "Catalog Name"
print(nice_label("apiKey"))          # "Api Key"
print(nice_label("MaxRowCount"))     # "Max Row Count"
```

---

## 5) `cached_from_import` — lazy module attribute cache

```python
from yggdrasil.environ import cached_from_import

# Cache an attribute from an optional module
orjson_dumps = cached_from_import("orjson", "dumps", install_if_missing=False)

if orjson_dumps is not None:
    blob = orjson_dumps({"k": "v"})
```

---

## 6) Typical use: safe optional-dep guard

The `lib.py` pattern used throughout yggdrasil wraps `runtime_import_module` to give a helpful "install extra X" error:

```python
# yggdrasil/polars/lib.py  (pattern used across the codebase)
from yggdrasil.environ import runtime_import_module

def _get_polars():
    mod = runtime_import_module("polars", install_if_missing=False)
    if mod is None:
        raise ImportError(
            "polars is not installed. Run: pip install 'ygg[data]'"
        )
    return mod

polars = _get_polars()
```

Callers always go through the guard:

```python
from yggdrasil.polars.lib import polars   # correct — raises helpful error if missing
import polars                             # wrong — opaque ImportError on base installs
```
