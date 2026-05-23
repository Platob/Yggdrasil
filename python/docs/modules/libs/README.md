# Optional dependency guards (`lib.py` pattern)

Yggdrasil keeps most integrations optional. The only hard runtime dependency is `pyarrow >= 20`. Everything else — Polars, pandas, Spark, Databricks SDK, MongoDB, Kafka, blake3, xxhash, zstandard, cloudpickle — is guarded behind a `lib.py` module that raises a helpful "install extra X" error instead of an opaque `ImportError`.

---

## Why this matters

```python
import polars          # ❌ opaque ImportError on base installs
from yggdrasil.polars.lib import polars   # ✅ clear message: "install ygg[data]"
```

The guard keeps base installs lightweight and gives callers a single stable import path that works whether the extra is installed or not.

---

## How to use in your own code

Always go through the `lib.py` guard instead of bare `import`:

```python
# Engine libraries
from yggdrasil.polars.lib  import polars   # polars
from yggdrasil.pandas.lib  import pandas   # pandas
from yggdrasil.spark.lib   import pyspark  # pyspark

# Platform integrations
from yggdrasil.databricks.lib import databricks_sdk   # databricks-sdk
from yggdrasil.mongo.lib      import pymongo          # pymongo

# Hashing / compression extras
from yggdrasil.blake3.lib    import blake3      # blake3
from yggdrasil.xxhash.lib    import xxhash      # xxhash
from yggdrasil.pickle.lib    import zstandard   # zstandard
```

---

## Guard import semantics

Each guard does **at most one** import call and caches the module. The import is deferred to the first call site — not at module-import time — so a `from yggdrasil.polars.lib import polars` at the top of a file doesn't trigger the import until the name is actually used.

```python
from yggdrasil.polars.lib import polars

def transform(df):
    # polars is resolved here on first call, not when the module loads
    return polars.DataFrame(df).filter(polars.col("active"))
```

If the extra is absent, the guard raises:

```
ImportError: polars is not installed.
  Run: pip install "ygg[data]"   # or: pip install polars
```

---

## Extra ↔ package mapping

| Install extra | Installs | Guarded via |
|---|---|---|
| `ygg[data]` | polars, pandas, numpy | `yggdrasil.polars.lib`, `yggdrasil.pandas.lib` |
| `ygg[bigdata]` | pyspark, delta | `yggdrasil.spark.lib` |
| `ygg[databricks]` | databricks-sdk, databricks-connect | `yggdrasil.databricks.lib` |
| `ygg[mongo]` | pymongo, pymongoarrow | `yggdrasil.mongo.lib` |
| `ygg[http]` | urllib3, certifi | *(always available in http_)* |
| `ygg[pickle]` | cloudpickle, dill, zstandard | `yggdrasil.pickle.lib` |
| `ygg[blake3]` | blake3 | `yggdrasil.blake3.lib` |
| `ygg[xxhash]` | xxhash | `yggdrasil.xxhash.lib` |
| `ygg[kafka]` | confluent-kafka | `yggdrasil.kafka.lib` |
| `ygg[api]` | fastapi, uvicorn, pydantic | `yggdrasil.fastapi.lib` |
| `ygg[aws]` | boto3 | `yggdrasil.aws.lib` |

Full combinations:

```bash
pip install "ygg[data,databricks,http]"        # data pipeline
pip install "ygg[data,mongo,postgres]"         # multi-store
pip install "ygg[bigdata,databricks,pickle]"   # Spark + Databricks
pip install "ygg[api,databricks]"              # REST service + Databricks
```

---

## Write your own guard

```python
# my_package/vendor/lib.py
from yggdrasil.environ import runtime_import_module

def _get_vendor():
    mod = runtime_import_module("vendor_sdk", install_if_missing=False)
    if mod is None:
        raise ImportError(
            "vendor_sdk is not installed. Run: pip install vendor-sdk"
        )
    return mod

vendor_sdk = _get_vendor()
```

Callers import it the same way:

```python
from my_package.vendor.lib import vendor_sdk
```

---

## `lazy_imports` — PEP 562 lazy module attribute

`yggdrasil.lazy_imports` provides module-level `__getattr__` lazy loading. A module that exposes optional attributes via `__getattr__` defers the import until the first attribute access:

```python
# Some modules expose optional attrs lazily:
from yggdrasil import lazy_imports

polars = lazy_imports.polars   # triggers import only here
```

This is an internal mechanism used by `yggdrasil.__init__` — prefer the explicit `lib.py` guard in application code.
