# yggdrasil.arrow

Arrow utilities for type inference, schema normalization, and casting. The hard runtime dependency of `ygg` is `pyarrow >= 20` — all other engine bridges are optional, but Arrow is always available.

## One-liner

```python
import yggdrasil.arrow as pa   # re-exports pyarrow + adds ygg helpers

raw = pa.table({"id": ["1", "2"], "score": ["9.1", "8.7"]})
```

## Infer Arrow fields from Python type hints

```python
from yggdrasil.arrow import arrow_field_from_hint
import pyarrow as pa

# Primitives
print(arrow_field_from_hint(int,   name="id"))        # field('id', int64)
print(arrow_field_from_hint(float, name="score"))     # field('score', double)
print(arrow_field_from_hint(str,   name="label"))     # field('label', string)
print(arrow_field_from_hint(bool,  name="active"))    # field('active', bool)

# Optional (nullable)
from typing import Optional
print(arrow_field_from_hint(Optional[int], name="ref"))  # field('ref', int64, nullable=True)

# Collections
print(arrow_field_from_hint(list[str],        name="tags"))     # list<string>
print(arrow_field_from_hint(dict[str, float], name="metrics"))  # map<string, double>

# Dataclass → struct
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

print(arrow_field_from_hint(Point, name="location"))
# field('location', struct<x: double, y: double>)
```

## Dataclass → Arrow struct field

```python
from dataclasses import dataclass
from typing import Optional
from yggdrasil.dataclasses import dataclass_to_arrow_field

@dataclass
class Order:
    id:      int
    amount:  float
    note:    Optional[str] = None

field = dataclass_to_arrow_field(Order)
# struct<id: int64, amount: double, note: string>
```

## Cast an Arrow array

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_array
from yggdrasil.data.cast.options import CastOptions

arr    = pa.array(["1", "2", "3"])
target = pa.field("id", pa.int64(), nullable=False)
out    = cast_arrow_array(arr, CastOptions(target_field=target))
# pa.array([1, 2, 3], type=int64)
```

## Cast an Arrow table / RecordBatch

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

raw = pa.table({
    "id":    ["1", "2"],
    "ts":    ["2024-01-01", "2024-01-02"],
    "score": ["9.1", "8.7"],
})
target = pa.schema([
    pa.field("id",    pa.int64(),                  nullable=False),
    pa.field("ts",    pa.timestamp("us", tz="UTC")),
    pa.field("score", pa.float64()),
])
out = cast_arrow_tabular(raw, CastOptions(target_field=target))
print(out.schema)
```

## Cast with strict name matching

```python
from yggdrasil.data.cast.options import CastOptions

# Raise if source has extra columns not in target schema
opts = CastOptions(target_field=target, strict_match_names=True)
out  = cast_arrow_tabular(raw, opts)
```

## Convert a RecordBatchReader (streaming)

```python
import pyarrow as pa
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

schema = pa.schema([pa.field("id", pa.int64()), pa.field("v", pa.float64())])

def produce_batches():
    for i in range(5):
        yield pa.record_batch({"id": [i], "v": [float(i)]})

reader = pa.RecordBatchReader.from_batches(schema, produce_batches())
out    = cast_arrow_tabular(reader, CastOptions(target_field=target))
```

## JSON → Arrow (vectorized)

```python
import pyarrow as pa
import pyarrow.json as paj

# Fast path: parse NDJSON bytes directly
ndjson = b'{"id":1,"v":2.5}\n{"id":2,"v":3.0}\n'
table  = paj.read_json(pa.py_buffer(ndjson))
```

## Type normalization

```python
from yggdrasil.arrow import normalize_arrow_type
import pyarrow as pa

# Normalizes timestamps, string variants, large_binary, etc.
t = normalize_arrow_type(pa.large_string())   # → pa.string()
t = normalize_arrow_type(pa.timestamp("ms"))  # → pa.timestamp("us", tz="UTC")
```

## Arrow TestCase base

```python
from yggdrasil.arrow.tests import ArrowTestCase

class MyTest(ArrowTestCase):
    def test_cast(self):
        tbl = self.table({"id": [1, 2], "v": [3.0, 4.0]})
        out = my_transform(tbl)
        self.assertSchemaEqual(out.schema, expected_schema)
        # helpers: self.pa, self.table(...), self.record_batch(...),
        #          self.write_parquet(...), self.tmp_path
```
