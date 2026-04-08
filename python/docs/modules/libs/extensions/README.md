# Engine cast extension helpers

Engine modules expose Arrow↔engine conversion helpers in their `cast.py` modules.

## Polars examples

```python
import yggdrasil.arrow as pa
from yggdrasil.polars.cast import (
    arrow_table_to_polars_dataframe,
    polars_dataframe_to_arrow_table,
)
from yggdrasil.polars.lib import polars

arrow_table = pa.table({"id": [1, 2]})
pl_df = arrow_table_to_polars_dataframe(arrow_table)
roundtrip = polars_dataframe_to_arrow_table(pl_df)
```

## pandas and Spark

Equivalent conversion helpers exist in:
- `yggdrasil.pandas.cast`
- `yggdrasil.spark.cast`
