# Optional dependency guards (`lib.py` pattern)

Yggdrasil keeps most integrations optional. Import external libraries through package `lib.py` modules.

## Safe imports

```python
from yggdrasil.polars.lib import polars
from yggdrasil.pandas.lib import pandas
from yggdrasil.spark.lib import pyspark, pyspark_sql
```

These wrappers use runtime import/install guards so base installs stay lightweight.

## Example

```python
from yggdrasil.polars.lib import polars

df = polars.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
print(df)
```
