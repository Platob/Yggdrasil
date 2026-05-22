# Skill: manage and write Databricks tables with `Table` / `Tables`

## When to use

The user asks to create / drop / describe / insert into / MERGE into
a Unity Catalog table, stage rows via Parquet on a Volume, run an
async insert, or reconcile a frame against a target Delta schema.

## Primary surface

```python
from yggdrasil.databricks import DatabricksClient

dbc = DatabricksClient()
tbl = dbc.table("main.default.orders")      # Table singleton
tbls = dbc.tables                            # Tables service (find, list, …)
```

`Table` is a singleton resource. Reuse the instance across calls; it
caches schema info and reconciles updates.

## Read / inspect

```python
tbl.exists
tbl.read_info()        # fresh metadata fetch
tbl.schema             # yggdrasil Schema
tbl.arrow_schema       # pyarrow.Schema
tbl.columns            # list[Field]
```

## Create / ensure

```python
from yggdrasil.data import Schema, Field, DataType

schema = Schema.from_fields([
    Field("id", DataType.int64(), nullable=False),
    Field("amount", DataType.decimal(18, 2)),
    Field("paid_at", DataType.timestamp("UTC")),
])

tbl.ensure_created(schema=schema, comment="orders v1")
# idempotent: creates if missing, no-op if it already matches.
```

Route through `tbl.create(...)` / `tbl.delete(...)` /
`tbl.ensure_created(...)`, **not** `ws.tables.create(...)` — the
singleton method handles retries, cache warm-up, and
`missing_ok` ergonomics.

## Insert frames

Synchronous insert (small frames, ad-hoc):

```python
import pyarrow as pa

batch = pa.table({"id": [1, 2], "amount": [9.99, 12.00], "paid_at": [...]})
tbl.insert(batch)
```

`insert` reconciles the input against the target schema (names,
order, nullability, decimal precision/scale, timezone) before writing.

Async insert (large frames, staged through a Volume):

```python
job = tbl.async_insert(batch, staging_volume="main.default.staging")
job.wait()
```

`AsyncInsert` stages Parquet on a Volume, then runs a Databricks SQL
`COPY INTO` (or Delta `MERGE`, when keys are supplied). The
`parquet_paths` / `metadata_paths` fields hold live `VolumePath`
objects internally and serialise as URL strings — pickle-safe across
job tasks.

## MERGE / upsert

```python
tbl.merge(
    batch,
    keys=["id"],
    prune_by=["paid_at"],      # narrows the target scan window
    update_columns=["amount", "paid_at"],
)
```

The merge path builds the Delta `MERGE … USING …` statement, applies
optional `prune_by` predicates to narrow the scanned partitions, and
returns a `StatementResult`. For very large frames pair `merge` with
the async / staged-volume path.

## DELETE / DML

```python
tbl.delete_where("paid_at < '2024-01-01'")
```

…or `dbc.sql.execute("DELETE FROM main.default.orders WHERE …")` for
one-off DML.

## Don'ts

- Don't loop row-by-row to insert; build an Arrow / Polars frame and
  pass it to `insert` / `async_insert` / `merge`.
- Don't pre-`exists()` before inserting — `ensure_created` is
  idempotent and `insert` raises a typed error if the table is gone.
- Don't build the `COPY INTO` SQL by hand; `async_insert` already
  does it with the right retries and schema reconciliation.
- Don't pickle a live `VolumePath` into a Spark task without going
  through `AsyncInsert.to_dict()` — let the existing serialisation
  emit URL strings for you.
