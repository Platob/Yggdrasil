# Optional dependency guards (`lib.py` pattern)

Yggdrasil's only hard runtime dependency is `pyarrow >= 20`. Everything
else — Polars, pandas, Spark, Databricks SDK, blake3, xxhash, cloudpickle,
dill, zstandard — is optional and imported through a `lib.py` guard.

The guard does one import attempt; on failure it raises a
`ModuleNotFoundError` with an actionable "install extra X" message instead
of a bare `ImportError` at the bottom of a stack trace.

---

## Guarded imports — quick reference

```python
from yggdrasil.polars.lib     import polars          # raises if not installed
from yggdrasil.pandas.lib     import pandas
from yggdrasil.spark.lib      import spark           # pyspark
from yggdrasil.xxhash         import xxhash          # guarded re-export
from yggdrasil.pickle.lib     import cloudpickle, dill, zstandard
```

Or use the centralised `lazy_imports` shim:

```python
from yggdrasil.lazy_imports import (
    polars_module,
    pyarrow_dataset_module,
)
```

---

## 1) Polars guard

```python
from yggdrasil.polars.lib import polars

df = polars.DataFrame({"id": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
print(df.filter(polars.col("v") > 15))
```

Error when missing:

```
ModuleNotFoundError: polars is not installed.
  Install it with: pip install ygg[data]
```

---

## 2) pandas guard

```python
from yggdrasil.pandas.lib import pandas

df = pandas.DataFrame({"a": [1, 2], "b": [3, 4]})
print(df.describe())
```

---

## 3) Spark guard

```python
from yggdrasil.spark.lib import spark   # lazy-imports pyspark

# Use pyspark types without a bare `import pyspark`
print(spark.sql.types.StringType())
```

---

## 4) Databricks SDK guard

```python
from yggdrasil.databricks.lib import databricks_sdk

ws = databricks_sdk.WorkspaceClient()
```

---

## 5) Hashing guards (blake3, xxhash)

```python
from yggdrasil.xxhash import xxhash          # guard + re-export

hasher = xxhash.xxh64()
hasher.update(b"my payload")
print(hasher.hexdigest())
```

```python
from yggdrasil.blake3 import blake3          # guard + re-export

h = blake3.blake3(b"my payload")
print(h.hexdigest())
```

Install with: `pip install ygg[pickle]`

---

## 6) Serialization codec guards

```python
from yggdrasil.pickle.lib import cloudpickle, dill, zstandard

# cloudpickle — serialize lambdas and closures
blob = cloudpickle.dumps(lambda x: x * 2)
fn   = cloudpickle.loads(blob)
print(fn(21))   # 42

# dill — full Python object graph serialization
blob = dill.dumps({"key": some_object})
back = dill.loads(blob)

# zstandard — fast compression
cctx  = zstandard.ZstdCompressor()
dctx  = zstandard.ZstdDecompressor()
compressed = cctx.compress(b"long payload " * 1000)
original   = dctx.decompress(compressed)
```

Install with: `pip install ygg[pickle]`

---

## 7) Write your own `lib.py` guard

Follow the same pattern in your own extension module:

```python
# mypackage/lib.py
from __future__ import annotations

from yggdrasil.environ import cached_from_import

httpx = cached_from_import(
    "httpx",
    install_hint="pip install httpx",
)
```

Then in your feature code:

```python
# mypackage/client.py
from mypackage.lib import httpx   # safe import

resp = httpx.get("https://example.com")
```

---

## 8) Which extra installs which guard

| Extra | What it unlocks |
|---|---|
| `ygg[data]` | `polars`, `pandas`, `numpy` |
| `ygg[bigdata]` | `pyspark`, `delta-spark` |
| `ygg[databricks]` | `databricks-sdk` |
| `ygg[api]` | `fastapi`, `uvicorn`, `pydantic` |
| `ygg[pickle]` | `cloudpickle`, `dill`, `zstandard`, `xxhash`, `blake3` |
| `ygg[http]` | `urllib3` |
| `ygg[mongo]` | `mongoengine`, `pymongo` |
