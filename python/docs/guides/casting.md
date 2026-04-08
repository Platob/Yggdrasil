# Casting Guide

## Scalar conversion

```python
from yggdrasil.data.cast.registry import convert

assert convert("10", int) == 10
assert convert("false", bool) is False
```

## Schema-aware tabular casting

```python
import yggdrasil.arrow as pa
from yggdrasil.data.cast import CastOptions
from yggdrasil.arrow.cast import cast_arrow_tabular

source = pa.table({"id": ["1"], "price": ["9.99"]})
target = pa.schema([
    pa.field("id", pa.int64(), nullable=False),
    pa.field("price", pa.float64(), nullable=False),
])

out = cast_arrow_tabular(source, CastOptions(target_field=target, strict_match_names=True))
```

## Engine converters

- `yggdrasil.polars.cast`
- `yggdrasil.pandas.cast`
- `yggdrasil.spark.cast`

Each module registers converters when imported.
