# Engine cast extension helpers

Engine modules expose Arrow↔engine conversion helpers in their `cast.py` modules.

## Polars examples

```python
import pyarrow as pa
import polars as pl
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

arrow_table = pa.table({"id": [1, 2]})

# Arrow → Polars (use polars directly)
pl_df = pl.from_arrow(arrow_table)

# Polars → Arrow
roundtrip = polars_dataframe_to_arrow_table(pl_df)
```

## pandas and Spark

Equivalent conversion helpers exist in:
- `yggdrasil.pandas.cast`
- `yggdrasil.spark.cast`
