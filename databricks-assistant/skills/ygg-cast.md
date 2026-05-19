# Skill: cast / convert values, dicts, and frames with Yggdrasil

## When to use

The user asks to convert / cast / coerce / parse a value, dict, JSON
payload, dataclass, or Arrow / Polars / pandas frame to another type
or schema. Triggers include "convert", "cast", "parse", "to dict /
dataclass / Arrow / Polars / pandas / Spark", "normalize this JSON",
"enforce this schema".

## Primary surface

`yggdrasil.data.cast.convert(value, target, options=...)` plus
`CastOptions` for tabular casts.

```python
from yggdrasil.data.cast import convert
from yggdrasil.data.cast.options import CastOptions
```

Dispatch order: **exact match → identity → `Any` wildcard → MRO
fallback → one-hop composition.**

## Scalar / dict / dataclass conversion

```python
convert("42", int)              # 42
convert("true", bool)           # True
convert("2024-01-15", "date")   # datetime.date(2024, 1, 15)

from dataclasses import dataclass

@dataclass
class Order:
    id: int
    amount: float
    paid: bool = False

convert({"id": "7", "amount": "99.50", "paid": "yes"}, Order)
# Order(id=7, amount=99.5, paid=True)
```

`convert` is forgiving on input (strings, numbers, mixed dicts) and
strict on meaning (conflicting fields raise).

## Tabular casts — lock a target schema

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
target = pa.schema([
    pa.field("id",    pa.int64(),   nullable=False),
    pa.field("score", pa.float64(), nullable=False),
])
out = cast_arrow_tabular(raw, CastOptions(target_field=target))
```

For Polars / pandas / Spark frames, import the engine's `cast` module
(`yggdrasil.polars.cast`, `.pandas.cast`, `.spark.cast`); registration
happens on import.

## Cross-engine moves

Don't `to_pylist()` between engines. Use:

| From → To | Call |
| --- | --- |
| Arrow → pandas | `arr.to_pandas()` |
| Arrow → Polars | `pl.from_arrow(arr)` |
| Polars → Arrow | `series.to_arrow()` |
| pandas → Arrow | `pa.array(series, from_pandas=True, type=...)` |
| pandas → Polars | `pl.from_pandas(df)` |

## When to extend the registry

If you find yourself writing the same `convert` chain at two call
sites, add a converter:

```python
from yggdrasil.data.cast.registry import register_converter

@register_converter(MyInput, MyOutput)
def _my_input_to_output(value, options):
    ...
```

…and let `convert(value, MyOutput)` find it everywhere.

## Don'ts

- Don't loop / comprehension over `array.to_pylist()` for type
  coercion — vectorise via `pyarrow.compute` first, then `convert` on
  the resulting array, or build a `CastOptions(target_field=...)`.
- Don't invent parallel option objects. Extend `CastOptions` or pass
  it through.
- Don't hand-roll `pd.to_datetime` / `int(x)` / `pl.Int64.cast` if the
  registry handles the same coercion.
