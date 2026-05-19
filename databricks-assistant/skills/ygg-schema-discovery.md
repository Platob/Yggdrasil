# Skill: discover a schema from an unknown endpoint or payload

## When to use

The user has an API / file / queue / vendor feed with **no published
schema** (or a published one that doesn't match reality) and asks to
"figure out the columns", "infer the schema from a sample", "build a
Schema from this JSON", "what types should I use?", or "validate that
my schema matches the live data".

Pair with [`ygg-ingestion-pipeline`](ygg-ingestion-pipeline.md) (it
calls into this skill at step 2) and
[`ygg-schema-fields`](ygg-schema-fields.md) (Field / Schema / DataType
primitives) for the final hand-tuning.

## The discovery loop

1. **Probe the source.** Hit the endpoint / read a file. Cap the
   sample at 100–500 rows — enough to surface optionality and the
   long-tail of numeric/string mixtures, not so big that the probe
   itself becomes the workload.
2. **Let an engine infer.** Hand the sample to pyarrow / polars /
   pandas — they already know how to walk JSON / CSV / NDJSON / Avro /
   Parquet and emit a typed schema.
3. **Lift into a `Schema`.** Use `Field.from_arrow_schema(...)` /
   `Field.from_polars_schema(...)` / `Field.from_pandas(...)` to get a
   yggdrasil `Schema` with `Field` / `DataType` carriers.
4. **Tighten by hand.** Inferred types are *suggestions*. Fix:
   - nullability (the engine sees `null` once → `nullable=True`; real
     contract says required → flip to `False`),
   - decimal precision/scale (`float64` from JSON should usually be
     `decimal(18, 2)` for money),
   - timezone intent (`timestamp[us]` from a naive ISO string is
     almost always `timestamp("UTC")` for vendor APIs),
   - string enums (status / currency / country → consider an
     `enum`-like check, even if Unity Catalog stores `string`).
5. **Test the schema against fresh data.** Pull a *different* page,
   cast it through the schema, assert it round-trips without raising
   in `STRICT` mode.

## Probe an HTTP endpoint

```python
from yggdrasil.io.http_ import HTTPSession, HTTPRequest

session = HTTPSession.from_url("https://api.vendor.example.com")
resp = session.send(HTTPRequest(method="GET", url="/v1/orders?limit=200"))
sample = resp.json()["data"]            # list[dict]
```

For NDJSON / chunked responses:

```python
import pyarrow as pa
import pyarrow.json as paj

raw = resp.content                       # bytes
buf = pa.BufferReader(raw)
sample = paj.read_json(buf)              # pyarrow.Table, vectorised
```

`pyarrow.json` is the fastest, type-strictest reader — prefer it when
the payload is newline-delimited or already a JSON array.

## Lift the engine schema into a yggdrasil `Schema`

```python
import pyarrow as pa
from yggdrasil.data import Field, Schema

# pyarrow path — cleanest when the source is JSON / CSV / Parquet / Arrow.
arrow_tbl = pa.Table.from_pylist(sample)        # or paj.read_json(...)
schema = Field.from_arrow_schema(arrow_tbl.schema)

# polars path — when the sample is messy and you want polars's
# stronger string / null heuristics.
from yggdrasil.polars.lib import polars
df = polars.from_dicts(sample)
schema = Field.from_polars_schema(df.schema)

# pandas path — only when you genuinely have a pandas DataFrame.
schema = Field.from_pandas(pandas_df)
```

`Field.from_*` returns a single root `Field` whose `dtype` is a
nested `StructType` — call `.fields` (or treat it as a `Schema`) to
get the column list.

## Tighten the inferred schema

```python
from yggdrasil.data import Field, DataType, Schema

tightened = Schema.from_fields([
    Field("order_id",   DataType.string(),         nullable=False),  # required
    Field("amount",     DataType.decimal(18, 2),    nullable=False),  # was float64
    Field("paid_at",    DataType.timestamp("UTC"),  nullable=False),  # was timestamp[us]
    Field("note",       DataType.string(),         nullable=True),   # really optional
])
```

The tightened schema is the **commit-to-source** version. Drop the
inferred one — it was scaffolding for the discovery phase, not the
runtime contract.

## Validate against fresh data (round-trip test)

```python
from yggdrasil.data.cast.options import CastOptions
from yggdrasil.data.cast import convert

# Pull a *different* page than the sample.
fresh_resp = session.send(HTTPRequest(method="GET", url="/v1/orders?since=2026-01-01"))
fresh_rows = fresh_resp.json()["data"]

arrow_fresh = pa.Table.from_pylist(
    fresh_rows,
    schema=tightened.to_arrow(),
)
# If this raises, the schema is wrong — tighten or relax and retry.

# Or via the registry, with strict mode:
casted = convert(
    fresh_rows,
    tightened,
    options=CastOptions(target_field=tightened, strict=True),
)
```

`CastOptions(strict=True)` turns silent demotions (decimal precision
loss, integer overflow, timestamp truncation) into typed exceptions —
catch them, log the offending row, then either fix the schema or fix
the source.

## OpenAPI / Swagger as the seed

When the user provides an OpenAPI URL, the schema is already in
machine-readable form. Pull it, extract the relevant response schema,
*then still sample the live endpoint* — vendor specs lie. The spec
gets you 80 % of the columns; the sample catches the optional fields
and `nullable: true` discrepancies.

```python
spec = session.send(HTTPRequest(method="GET", url="/openapi.json")).json()
response_schema = spec["paths"]["/v1/orders"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
# Translate to DataType primitives manually — OpenAPI's "type": "integer", "format": "int64"
# maps to DataType.int64(); "format": "date-time" → DataType.timestamp("UTC"); etc.
```

## Don'ts

- Don't infer at runtime in the production job — discover once,
  commit the `Schema(...)` literal to source, let `Table.insert`
  reconcile against it.
- Don't trust a single sample — pull at least two pages from
  different time windows so optionality surfaces.
- Don't infer with `pandas.read_json` if the payload is large —
  `pyarrow.json.read_json` is vectorised and 10–100× faster.
- Don't accept the engine's default `nullable=True` for fields the
  contract says are required; the wrong default lets bad data write
  silently.
- Don't store money in `float64`; lift to `decimal(precision, scale)`
  and let the cast registry refuse precision-losing inputs.
- Don't store naive timestamps; pick the source's timezone (UTC for
  most vendor APIs) and pin it via `DataType.timestamp("UTC")`.
- Don't walk the sample row-by-row to "find the columns" — the
  vectorised reader already produced a typed schema.
