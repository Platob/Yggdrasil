# Skill: common pitfalls when generating Yggdrasil code

## When to use

The user pastes generated code that already uses `yggdrasil` /
`ygg`, but the Assistant should sanity-check it before delivering;
or the user asks "is this idiomatic?" / "why is this slow?". Use as
a checklist after generating, before returning.

## Red flags to catch

### 1. Row-by-row Python loops over data

```python
# bad
for row in df.to_pylist():
    out.append(do_something(row))

# good
out = pc.if_else(df["col"].is_valid(), ..., ...)   # pyarrow.compute
# or
out = df.with_columns(pl.col("col").map_elements(...))  # polars expression
```

`.to_pylist()` / `.to_list()` / `.tolist()` are the same trap with
one less line. Three exemptions only: documented per-row fallback
for shapes vectorised engines can't emit, genuine row endpoints
(ndjson / xlsx writers, kafka producers, JSON HTTP responses), and
diagnostics.

### 2. Re-importing engines instead of going through `lib.py`

```python
# bad
import polars as pl

# good
from yggdrasil.polars.lib import polars
```

Same for `pandas`, `pyspark`, `databricks-sdk`, `blake3`, `xxhash`.
The guard raises a helpful "install extra X" error on missing deps;
a bare import breaks base installs.

### 3. Pre-checking remote existence

```python
# bad
if path.exists():
    path.read_bytes()

# good
try:
    path.read_bytes()
except FileNotFoundError:
    ...
```

Probes double the round trip and race concurrent writers. Catch
`NotFound` / `FileNotFoundError` from the real call.

### 4. Bypassing resource singletons

```python
# bad
dbc.workspace_client.volumes.create(catalog="main", schema="default", name="v")

# good
dbc.volume("main.default.v").ensure_created()
```

The singleton method wraps retries, cache warm-up, defaults
(managed-volume-type, owner / comment normalization), and
`if_not_exists` / `missing_ok` ergonomics.

### 5. Hand-rolled JSON

```python
# bad
import json
payload = json.dumps({"ts": datetime.utcnow(), "id": UUID(...), "path": Path(...)})

# good
from yggdrasil.pickle.json import dumps
payload = dumps({"ts": datetime.utcnow(), "id": UUID(...), "path": Path(...)})
```

`yggdrasil.pickle.json` wraps `orjson` with datetime / UUID / Path
/ Enum / dataclass / Decimal coverage.

### 6. New options object instead of extending `CastOptions`

`CastOptions` is the single normalised carrier for source hints,
target field / schema, safety / memory / nullability flags,
strictness. Add a flag to it; don't invent a parallel dataclass.

### 7. Eager f-string logging

```python
# bad
logger.debug(f"Deleting {self!r} (age={age:.0f}s)")

# good
logger.debug("Deleting %r (age=%.0fs)", self, age)
```

Lazy `%r` skips `repr()` when the level is disabled. Follow the
`<Verb> <ResourceNoun> %r (key=value, …)` shape.

### 8. New file when an existing function would do

Prefer editing the function / class that already does most of the
job over a new module / helper. Three nearly-identical lines is
better than a premature abstraction.

### 9. `if logger.isEnabledFor(...)` around a single lazy call

Drop the guard — `logger.debug("…", obj)` already skips the format
args when the level is disabled. Keep the guard only when there's
real pre-computation (materialising a generator, building a summary
dict) outside the call.

### 10. Row-shape data between pipeline stages

```python
# bad — both stages know the schema; the dict carrier re-infers per row
def fetch_pages(url) -> Iterable[dict]:
    for page in paginate(url):
        for row in page["items"]:
            yield row

def write_rows(rows: Iterable[dict], target_table):
    for row in rows:
        dbc.sql.execute(f"INSERT INTO {target_table} VALUES (...)", row)

# good — one Arrow batch crosses the boundary, vectorised cast + insert
def fetch_pages(url, schema: Schema) -> Iterator[pa.RecordBatch]:
    for page in paginate(url):
        yield pa.RecordBatch.from_pylist(page["items"], schema=schema.arrow_schema)

def write_batches(batches: Iterator[pa.RecordBatch], target_table):
    dbc.table(target_table).insert(pa.Table.from_batches(batches), mode="APPEND")
```

Same trap with `.collect()` / `.iter_rows(named=True)` / `.to_pylist()`
between a Spark / Polars producer and a Python consumer that both
know the schema. Pass Arrow batches (or a Spark DataFrame on the
Spark path); per-row dicts are for genuine row endpoints (ndjson,
kafka producers, JSON HTTP responses), not for inter-stage transport.

### 11. Logging cache hits at debug

Hits are the steady-state success path; they drown the rare
interesting events. Log misses / expiries / invalidations instead:

```python
if cached is not None:
    return cached                                         # silent
logger.debug("Cache expired for %r (age=%.0fs) — refreshing", key, age)
```

## When in doubt

- Search `python/src/yggdrasil/` for the verb the user wants ("merge",
  "insert", "upload", "cast", "execute") — chances are the function
  already exists.
- Read [`AGENTS.md`](https://github.com/Platob/Yggdrasil/blob/main/AGENTS.md)
  for the full house style; this skill is the condensed call-site
  version.
