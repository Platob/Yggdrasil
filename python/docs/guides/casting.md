# Casting

Every cast in Yggdrasil — scalar, dataclass, Arrow, dataframe engine — runs through the same registry. This page shows the patterns you'll actually use.

## Scalar conversion

```python
from yggdrasil.data.cast.registry import convert

convert("10", int)              # 10
convert("false", bool)          # False
convert("3.14", float)          # 3.14
convert("2024-06-01", "date")   # datetime.date(2024, 6, 1)
```

## Dict → dataclass

```python
from dataclasses import dataclass
from yggdrasil.data.cast.registry import convert

@dataclass
class User:
    id: int
    email: str
    active: bool = True

convert({"id": "1", "email": "ada@example.com", "active": "yes"}, User)
```

## Register a custom converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import register_converter, convert

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

## Schema-aware tabular casting (Arrow)

```python
import pyarrow as pa
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.arrow.cast import cast_arrow_tabular

source = pa.table({"id": ["1"], "price": ["9.99"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("price", pa.float64(), nullable=False),
])
out = cast_arrow_tabular(source, CastOptions(target_field=target, strict_match_names=True))
```

Streaming readers:

```python
from yggdrasil.arrow.cast import cast_arrow_record_batch_reader

# reader: pyarrow.RecordBatchReader, opts: CastOptions
for batch in cast_arrow_record_batch_reader(reader, opts):
    process(batch)
```

## Dataclass → Arrow field

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Position:
    symbol: str
    quantity: float

print(dataclass_to_arrow_field(Position))
```

## Engine bridges

| Helper | Module |
|---|---|
| `cast_arrow_tabular`, `cast_arrow_record_batch_reader` | `yggdrasil.arrow.cast` |
| `cast_pandas_dataframe` | `yggdrasil.pandas.cast` |
| `cast_polars_dataframe`, `cast_polars_lazyframe` | `yggdrasil.polars.cast` |
| `cast_spark_dataframe` | `yggdrasil.spark.cast` |

Each module registers its converters **on import**. Always reach the optional engines via their `lib.py` guard so base installs stay functional:

```python
from yggdrasil.lazy_imports import polars
from yggdrasil.lazy_imports import pandas
```

### Polars

```python
import pyarrow as pa
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.polars.cast import cast_polars_dataframe
from yggdrasil.lazy_imports import polars

df = polars.DataFrame({"id": ["1"], "score": ["4.5"]})
target = pa.schema([pa.field("id", pa.int64()), pa.field("score", pa.float64())])
out = cast_polars_dataframe(df, CastOptions(target_field=target))
```

### Arrow ↔ Polars round-trip

```python
import polars as pl
from yggdrasil.polars.cast import polars_dataframe_to_arrow_table

# Arrow → Polars (use polars directly)
pl_df = pl.from_arrow(arrow_table)

# Polars → Arrow
roundtrip = polars_dataframe_to_arrow_table(pl_df)
```

### pandas / Spark

`yggdrasil.pandas.cast` and `yggdrasil.spark.cast` mirror the same shape:

```python
from yggdrasil.pandas.cast import cast_pandas_dataframe
from yggdrasil.spark.cast import cast_spark_dataframe
```

## Reusing `CastOptions` in custom helpers

```python
from yggdrasil.data.cast.options import CastOptions

def normalize_options(options=None, *, target_field=None) -> CastOptions:
    return CastOptions.check(options, target_field=target_field, strict_match_names=True)
```

## When the cast doesn't fire

1. Confirm the engine cast module is imported (`yggdrasil.polars.cast`, etc.). Engines register on import — `find_converter` also auto-triggers these imports when the source or target lives in the `polars` / `pandas` / `pyspark` / `pyarrow` namespace, so missing converters past that probe are real misses.
2. Check `CastOptions.target_field` — `cast_arrow_tabular` and friends need the target schema.
3. Inspect the dispatch order in [Architecture](architecture.md#the-cast-registry). Most "missing converter" cases are an MRO miss; register a converter or add an `Any`-wildcard fallback. The registry **does not** auto-compose two registered hops (`X → Y → int`) into a synthetic direct cast — that path was deliberately removed because the chosen intermediate depended on the order of unrelated registrations. Register the direct `X → int` if you need one.
