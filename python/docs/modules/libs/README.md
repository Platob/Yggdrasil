# Optional dependency guards (`lib.py` pattern)

Yggdrasil keeps most integrations optional. The only hard runtime dependency is `pyarrow >= 20`. Every other engine or platform integration is guarded behind a `lib.py` module that raises a helpful install error on first use.

---

## Extras map

| Extra | Installs | Unlocks |
|---|---|---|
| `ygg[data]` | *(included in base)* | `yggdrasil.data`, `yggdrasil.arrow` — always available |
| `ygg[polars]` | `polars` | `yggdrasil.polars`, Polars converters, `PolarsTestCase` |
| `ygg[pandas]` | `pandas` | `yggdrasil.pandas`, pandas converters, `PandasTestCase` |
| `ygg[bigdata]` | `pyspark` | `yggdrasil.spark`, `SparkTestCase`, `Dataset`/`SparkTabular` |
| `ygg[databricks]` | `databricks-sdk` | `yggdrasil.databricks`, `DatabricksClient` |
| `ygg[api]` | `fastapi`, `uvicorn`, `pydantic` | `yggdrasil.fastapi`, `ygg-api` CLI |
| `ygg[pickle]` | `cloudpickle`, `dill`, `zstandard`, `xxhash`, `blake3` | `yggdrasil.pickle` full codec suite |
| `ygg[delta]` | `deltalake` | `DeltaFolder` write path (log reading needs no extra) |

Install multiple extras at once:

```bash
pip install "ygg[data,databricks,api]"
# or
uv pip install "ygg[polars,databricks,pickle]"
```

---

## Safe imports via the guard

Always import optional engines through their `lib.py` guard, not with a bare top-level import:

```python
# Correct — raises a helpful "install ygg[polars]" error if missing
from yggdrasil.polars.lib import polars as pl

# Wrong — raises ImportError with no guidance; breaks base installs
import polars as pl
```

Every optional subsystem exposes the same guard pattern:

```python
from yggdrasil.polars.lib import polars           # pl
from yggdrasil.pandas.lib import pandas           # pd
from yggdrasil.spark.lib import pyspark           # ps
from yggdrasil.databricks.lib import databricks   # databricks SDK
```

---

## Check availability at runtime

```python
from yggdrasil.polars.lib import polars

try:
    pl = polars()   # lazy-loaded once on first call
    print("Polars available:", pl.__version__)
except ImportError:
    print("Polars not installed — run: pip install 'ygg[polars]'")
```

---

## Lazy imports via `yggdrasil.lazy_imports`

Module-level helpers resolve optional packages lazily — they return `None` when the package is absent instead of raising at import time, which is useful for conditional feature code:

```python
from yggdrasil.lazy_imports import polars_module, pandas_module, spark_sql_module

if polars_module() is not None:
    import polars as pl
    df = pl.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
    print(df)
else:
    print("Polars not available — skipping polars branch")
```

---

## Writing a new `lib.py` guard

When adding a new optional subsystem, follow this pattern:

```python
# yggdrasil/myengine/lib.py
from __future__ import annotations

_module = None

def myengine():
    global _module
    if _module is None:
        try:
            import myengine as _myengine
            _module = _myengine
        except ImportError:
            raise ImportError(
                "myengine is required. Install it with: pip install 'ygg[myengine]'"
            ) from None
    return _module
```

Usage:

```python
from yggdrasil.myengine.lib import myengine

eng = myengine()   # raises ImportError with actionable message if missing
```

---

## `cached_from_import` — module-level singleton import

For a slightly richer pattern that caches the result and lets callers pass a factory:

```python
from yggdrasil.environ import cached_from_import

blake3 = cached_from_import("blake3", extra="pickle")
h = blake3.blake3(b"hello world").hexdigest()
```

---

## Converter registration is import-triggered

Engine converters register themselves into the global cast registry on import. If an expected cross-engine cast does not fire, verify the engine module has been imported at least once:

```python
# Trigger Polars converter registration
import yggdrasil.polars.cast   # noqa: F401

# Now polars.DataFrame → pa.Table works through convert()
from yggdrasil.data.cast.registry import convert
import polars as pl
import pyarrow as pa

df = pl.DataFrame({"id": [1, 2, 3]})
tbl = convert(df, pa.Table)
print(tbl)
```

---

## Extension helpers per engine

Engine-specific Arrow↔engine bridging helpers live in `<engine>/cast.py`:

```python
# Polars → Arrow
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table
import polars as pl

pl_df = pl.DataFrame({"x": [1, 2, 3]})
arrow_tbl = polars_dataframe_to_arrow_table(pl_df)

# pandas → Arrow
from yggdrasil.pandas.cast import pandas_dataframe_to_arrow_table
import pandas as pd

pd_df = pd.DataFrame({"x": [1, 2, 3]})
arrow_tbl = pandas_dataframe_to_arrow_table(pd_df)
```

See [engine cast helpers](extensions/README.md) for the full bridging reference.

---

## TestCase bases skip cleanly when extras are missing

Engine test bases detect missing dependencies and skip the whole class automatically — no `skipIf` decoration required:

```python
from yggdrasil.polars.tests import PolarsTestCase

class TestMyTransform(PolarsTestCase):
    def test_cast(self):
        df = self.df({"id": [1, 2], "val": [1.0, 2.0]})   # uses self.pl
        out = my_transform(df)
        self.assertFrameEqual(out, df)
        # Skipped automatically on base installs without polars
```
