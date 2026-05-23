# Optional dependency guards (`lazy_imports` pattern)

Yggdrasil keeps most integrations optional. Import external libraries through `yggdrasil.lazy_imports`.

## Safe imports

```python
from yggdrasil.lazy_imports import polars
from yggdrasil.lazy_imports import pandas
```

These wrappers use runtime import/install guards so base installs stay lightweight.

## Example

```python
from yggdrasil.lazy_imports import polars

df = polars.DataFrame({"id": [1, 2], "value": [10.0, 20.0]})
print(df)
```
