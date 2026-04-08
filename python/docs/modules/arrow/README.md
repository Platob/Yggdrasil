# yggdrasil.arrow

Arrow utilities for:
- inferring Arrow fields from Python hints
- normalizing Arrow types across engines
- casting Arrow arrays/tables/readers via `yggdrasil.arrow.cast`

## Basic inference

```python
from yggdrasil.arrow import arrow_field_from_hint

print(arrow_field_from_hint(int, name="id"))
print(arrow_field_from_hint(list[str], name="tags"))
print(arrow_field_from_hint(dict[str, float], name="metrics"))
```

## Dataclass to Arrow struct field

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Event:
    id: int
    score: float

print(dataclass_to_arrow_field(Event))
```

## Tabular cast through Arrow

```python
import yggdrasil.arrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

raw = pa.table({"id": ["1", "2"], "ts": ["2024-01-01", "2024-01-02"]})

target = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("ts", pa.timestamp("us", tz="UTC")),
])

out = cast_arrow_tabular(raw, CastOptions(target_field=target))
print(out.schema)
```
