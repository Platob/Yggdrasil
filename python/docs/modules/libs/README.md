# Optional engine libraries

Yggdrasil lazy-imports optional engines (Polars, pandas, PySpark) at runtime using `PyEnv.runtime_import_module`. If a library is not installed, an informative error is raised at first use — not at import time.

## Install extras

```bash
pip install "ygg[polars]"        # Polars
pip install "ygg[pandas]"        # pandas
pip install "ygg[spark]"         # PySpark
pip install "ygg[databricks]"    # Databricks SDK
```

---

## Bootstrap: Polars (safe import)

```python
from yggdrasil.polars.lib import polars   # raises if not installed

df = polars.DataFrame({"id": [1, 2], "value": [3.14, 2.71]})
```

---

## Bootstrap: pandas (safe import)

```python
from yggdrasil.pandas.lib import pandas   # raises if not installed

df = pandas.DataFrame({"id": [1, 2], "value": [3.14, 2.71]})
```

---

## Bootstrap: PySpark (safe import)

```python
from yggdrasil.spark.lib import pyspark, pyspark_sql   # raises if not installed

spark = pyspark_sql.SparkSession.getActiveSession()
```

---

## Bootstrap: guard at function boundary

```python
def cast_with_polars(df, schema):
    from yggdrasil.polars.lib import polars   # import guard here, not at module top
    from yggdrasil.polars.cast import cast_polars_dataframe
    from yggdrasil.data.cast import CastOptions

    return cast_polars_dataframe(df, CastOptions(target_field=schema))
```

---

## Bootstrap: `PyEnv` — dynamic module import

```python
from yggdrasil.environ import PyEnv

# Attempt to import; returns module or raises ImportError with hint
mod = PyEnv.runtime_import_module("duckdb", pip_name="duckdb")
conn = mod.connect()
```

---

## Engine availability check

```python
import importlib.util

def has_polars() -> bool:
    return importlib.util.find_spec("polars") is not None

def has_pyspark() -> bool:
    return importlib.util.find_spec("pyspark") is not None
```
