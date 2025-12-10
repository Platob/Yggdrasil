# yggdrasil.types

Type conversion and Arrow interoperability utilities.

## Conversion registry

### `convert(value, target_hint, options=None, **kwargs)`
Central entry point to cast runtime values to annotated types. It understands optionals, enums, dataclasses, iterables, mappings, and Arrow/Polars/Pandas/Spark-specific converters registered in `yggdrasil.types.cast.registry`.

Examples:
```python
from yggdrasil.types.cast import convert
import datetime

convert("3.14", float)            # 3.14
convert(["1", "2", "3"], list[int])  # [1, 2, 3]
convert("2024-02-03T04:05:06Z", datetime.datetime)  # aware datetime

import polars as pl
import pandas as pd
import pyarrow as pa

pl_df = pl.DataFrame({"id": [1, 2], "value": ["a", "b"]})
pdf = convert(pl_df, pd.DataFrame)  # Polars -> pandas using the registered dataframe bridge

from pyspark.sql import SparkSession, types as T
from yggdrasil.types.cast import CastOptions

spark = SparkSession.builder.master("local[1]").getOrCreate()
spark_df = spark.createDataFrame([(1, "a"), (2, "b")], schema=T.StructType([
    T.StructField("id", T.LongType(), nullable=False),
    T.StructField("value", T.StringType(), nullable=True),
]))

schema_hint = CastOptions.check_arg(
    target_field=pa.field("row", pa.struct([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("value", pa.string()),
    ]))
)
spark_pdf = convert(spark_df, pd.DataFrame, options=schema_hint)  # Enforce schema during Spark -> pandas casting

# Arrow field <-> Spark schema for streamlined casting hints
arrow_field = pa.field(
    "payload",
    pa.struct([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("ts", pa.timestamp("us", tz="UTC")),
    ]),
)
spark_struct = convert(arrow_field, T.StructField)
spark_schema = T.StructType([spark_struct])
roundtripped_arrow = convert(spark_schema, pa.Schema)
```

### `register_converter(from_hint, to_hint)`
Decorator to register new converters. Functions accept `(value, options)` and should return the converted value.

```python
from yggdrasil.types.cast import register_converter

@register_converter(str, complex)
def parse_complex(value, options):
    return complex(value)
```

### `CastOptions`
Options object (validated via `CastOptions.check_arg`) passed into converters to control defaults or Arrow metadata.

## Arrow helpers

### `arrow_field_from_hint(hint, name=None, index=None)`
Builds a `pyarrow.Field` from a Python type hint, handling optionals, containers, dataclasses, tuples, and Annotated metadata for custom Arrow details.

## Defaults

### `default_scalar(hint)` / `default_python_scalar(hint)` / `default_arrow_scalar(dtype, nullable)`
Produce sensible default values for Python and Arrow types, including nested structs, lists, maps, and dataclasses.

## Notes
- Primitive string, numeric, boolean, datetime, and timedelta parsers are included out of the box.
- Iterable conversion supports lists, sets, tuples (including variadic tuples), and mappings with per-element casting.
- Arrow/Polars/Spark conversions rely on optional dependencies; install the relevant libraries to enable those paths.

## Navigation
- [Module overview](../../modules.md)
- [Dataclasses](../dataclasses/README.md)
- [Libs](../libs/README.md)
- [Requests](../requests/README.md)
- [Types](./README.md)
- [Databricks](../databricks/README.md)
- [Pyutils](../pyutils/README.md)
- [Ser](../ser/README.md)
