# yggdrasil.io.tabular

In-memory `Tabular` holders — wrap data you already have in memory and expose
it through the unified Arrow record-batch read/write contract.

## Surface map

| Class | What it holds |
|---|---|
| `ArrowTabular` | One or more Arrow `RecordBatch` objects |
| `Dataset` / `SparkTabular` | A (mutable) Spark `DataFrame` |
| `LazyTabular` | A lazy generator of Arrow batches (streaming) |
| `UnionTabular` | Concatenation view over multiple `Tabular` objects |
| `Tabular[O]` | Abstract base — implement `_read_arrow_batches` + `_write_arrow_batches` |
| `is_tabular_source(obj)` | True when `Tabular.from_()` can coerce `obj` |

---

## 1) One-liners

```python
import pyarrow as pa
from yggdrasil.io.tabular import ArrowTabular

# Wrap an Arrow table as a Tabular
t = ArrowTabular(pa.table({"id": [1, 2, 3], "score": [9.1, 8.7, 7.4]}))
print(t.schema)
print(t.to_arrow_table())
print(t.to_pandas())
```

---

## 2) `ArrowTabular` — Arrow data on the driver

```python
import pyarrow as pa
from yggdrasil.io.tabular import ArrowTabular

# From an Arrow Table
table = pa.table({"id": [1, 2], "name": ["alice", "bob"]})
t     = ArrowTabular(table)

# From individual RecordBatches
batches = [
    pa.record_batch({"x": [1, 2]}, schema=pa.schema([pa.field("x", pa.int64())])),
    pa.record_batch({"x": [3, 4]}, schema=pa.schema([pa.field("x", pa.int64())])),
]
t = ArrowTabular.from_batches(batches)

# Schema access
print(t.schema)        # yggdrasil.data.Schema
print(t.arrow_schema)  # pa.Schema

# Convert to other engines
arrow_tbl = t.to_arrow_table()
pandas_df  = t.to_pandas()
polars_df  = t.to_polars()

# Iterate raw batches (zero-copy when the source was a single table)
for batch in t.iter_arrow_batches():
    print(batch.num_rows)

# Iterate Python row dicts (use only for sinks that need Python objects)
for row in t.iter_pylist():
    print(row)
```

### Write back into an ArrowTabular

```python
from yggdrasil.io.tabular import ArrowTabular
from yggdrasil.data.enums import Mode
import pyarrow as pa

t = ArrowTabular(pa.table({"id": [1]}))

# Overwrite
t.write_arrow_table(pa.table({"id": [10, 20]}), mode=Mode.OVERWRITE)

# Append
t.write_arrow_table(pa.table({"id": [30]}), mode=Mode.APPEND)
print(t.to_arrow_table())   # id: [10, 20, 30]
```

---

## 3) `Dataset` / `SparkTabular` — Spark DataFrame wrapper

```python
from yggdrasil.io.tabular import Dataset   # SparkTabular is the same class

# Wrap an existing Spark DataFrame
sdf     = spark.createDataFrame([{"id": 1, "v": 2.5}, {"id": 2, "v": 3.0}])
dataset = Dataset(sdf)

# Convert
arrow_tbl = dataset.to_arrow_table()    # collect via Arrow
pandas_df  = dataset.to_pandas()

# Iterate Arrow batches (efficient: uses Arrow columnar exchange)
for batch in dataset.iter_arrow_batches(batch_size=10_000):
    process(batch)

# Write back (overwrite / append)
dataset.write_arrow_table(arrow_tbl, mode="overwrite")
```

---

## 4) `LazyTabular` — streaming Arrow batches

Use `LazyTabular` when you have a generator of Arrow batches and want to
slot it into the `Tabular` contract without materialising everything in RAM.

```python
import pyarrow as pa
from yggdrasil.io.tabular import LazyTabular

def generate_batches():
    for i in range(10):
        yield pa.record_batch(
            {"chunk": [i], "value": [float(i) * 1.5]},
            schema=pa.schema([pa.field("chunk", pa.int64()), pa.field("value", pa.float64())]),
        )

t = LazyTabular(generate_batches, schema=pa.schema([
    pa.field("chunk", pa.int64()),
    pa.field("value", pa.float64()),
]))

for batch in t.iter_arrow_batches():
    print(batch)
```

---

## 5) `UnionTabular` — concatenate multiple Tabulars

```python
import pyarrow as pa
from yggdrasil.io.tabular import ArrowTabular, UnionTabular

a = ArrowTabular(pa.table({"id": [1, 2]}))
b = ArrowTabular(pa.table({"id": [3, 4]}))
c = ArrowTabular(pa.table({"id": [5]}))

union = UnionTabular([a, b, c])
print(union.to_arrow_table())   # id: [1, 2, 3, 4, 5]
```

---

## 6) `is_tabular_source` — coercion probe

```python
from yggdrasil.io.tabular import is_tabular_source

print(is_tabular_source("data.parquet"))      # True  (path-shaped string)
print(is_tabular_source("s3://bucket/key"))   # True
print(is_tabular_source("hello world"))       # False (content string)
print(is_tabular_source(my_arrow_tabular))    # True
print(is_tabular_source(pathlib.Path("/x"))) # True
```

---

## 7) `Tabular.from_()` — universal coercion

`Tabular.from_()` dispatches to the right subclass based on the input
type and optional `media_type` hint.

```python
from yggdrasil.io.tabular import Tabular

# From a file path (auto-detects format from extension)
t = Tabular.from_("events.parquet")
t = Tabular.from_("/data/orders.csv")

# From an Arrow table
import pyarrow as pa
t = Tabular.from_(pa.table({"id": [1, 2]}))

# From a Spark DataFrame (when yggdrasil.spark is imported)
import yggdrasil.spark
t = Tabular.from_(spark_df)

# With explicit media type
t = Tabular.from_("response.bin", media_type="application/vnd.apache.parquet")
```

---

## 8) Full pipeline: fetch → transform → write

```python
import pyarrow as pa
import pyarrow.compute as pc
from yggdrasil.io.tabular import ArrowTabular, UnionTabular

# Simulate reading from two sources
raw_a = ArrowTabular(pa.table({"ts": ["2026-01-01"], "value": [10.0]}))
raw_b = ArrowTabular(pa.table({"ts": ["2026-01-02"], "value": [20.0]}))

combined = UnionTabular([raw_a, raw_b]).to_arrow_table()

# Compute with pyarrow.compute (no Python loop)
result = combined.append_column(
    "value_doubled",
    pc.multiply(combined.column("value"), 2.0),
)

out = ArrowTabular(result)
print(out.to_pandas())
```
