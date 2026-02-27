# yggdrasil.dataclasses

Arrow-aware dataclass helpers.

## API

```python
from yggdrasil.dataclasses import dataclass_to_arrow_field

dataclass_to_arrow_field(cls_or_instance) -> pa.Field
```

Returns a cached `pa.Field` of type `pa.StructType` describing the dataclass.
Accepts a class or an instance. Results are cached per class.

---

## Bootstrap: class → Arrow struct field

```python
from dataclasses import dataclass
from typing import Optional
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Event:
    id: int
    name: str
    score: float
    tag: Optional[str] = None

field = dataclass_to_arrow_field(Event)
print(field.type)
# struct<id: int64, name: string, score: double, tag: string>

schema = field.type.to_schema()
```

---

## Bootstrap: instance also works

```python
event = Event(id=1, name="click", score=0.9)
field = dataclass_to_arrow_field(event)
# same result as passing the class
```

---

## Bootstrap: use schema to cast a table

```python
import pyarrow as pa
from yggdrasil.dataclasses import dataclass_to_arrow_field
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast import CastOptions

field = dataclass_to_arrow_field(Event)

raw = pa.table({"id": ["1"], "name": ["click"], "score": ["0.9"]})
out = cast_arrow_tabular(raw, CastOptions(target_field=field))
```

---

## Bootstrap: nested dataclasses

```python
from dataclasses import dataclass
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Location:
    lat: float
    lon: float

@dataclass
class Reading:
    sensor_id: int
    value: float
    location: Location

field = dataclass_to_arrow_field(Reading)
# struct<sensor_id: int64, value: double, location: struct<lat: double, lon: double>>
```
