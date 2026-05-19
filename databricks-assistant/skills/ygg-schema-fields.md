# Skill: describe columns and schemas with `Field` / `Schema` / `DataType`

## When to use

The user asks to build / inspect / merge / compare a schema, define a
column, set nullability or metadata, map between engine dtypes, or
ensure schema intent (names, order, nullability, nested structure,
precision/scale, timezone) is preserved across Arrow / Polars / pandas
/ Spark / Databricks.

## Primary surface

```python
from yggdrasil.data import Field, Schema, DataType, DataTypeId
```

- `Field` / `Field` — a column descriptor with name, type,
  nullability, metadata, tags, nested structure, engine-side dtype
  intent.
- `Schema` — ordered collection of fields.
- `DataType` / `DataTypeId` — a single type description that emits
  pyarrow / polars / pandas / Spark dtypes on demand.

## Build a field

```python
from yggdrasil.data import Field, DataType

f = Field(name="id", type=DataType.int64(), nullable=False)
f.to_arrow()    # pyarrow.Field
f.to_polars()   # polars dtype
f.to_pandas()   # pandas dtype
```

`Field.from_arrow(pa_field)`, `Field.from_polars(pl_dtype, name)`,
`Field.from_pandas(pd_series)` go the other direction — use these
instead of duplicating per-engine constructors.

## Build a schema

```python
from yggdrasil.data import Schema, Field, DataType

schema = Schema.from_fields([
    Field("id", DataType.int64(), nullable=False),
    Field("amount", DataType.decimal(18, 2)),
    Field("paid", DataType.boolean()),
])

schema.to_arrow()    # pyarrow.Schema
schema.to_polars()   # polars schema dict
schema.to_pandas()   # pandas dtype mapping
schema.to_spark()    # pyspark.sql.types.StructType
```

## Nested / list / map / struct

`DataType` carries nested structure intact. Build with
`DataType.list_of(...)`, `DataType.struct({...})`,
`DataType.map(key=..., value=...)`. The engine projections preserve
the nesting.

## Preserve schema intent

Field names, order, nullability, metadata, nested structure,
precision/scale, and timezone intent are part of the user contract.
Don't drop them when converting — pass a `Schema` or `Field` through
`CastOptions(target_field=...)`:

```python
from yggdrasil.arrow.cast import cast_arrow_tabular
from yggdrasil.data.cast.options import CastOptions

out = cast_arrow_tabular(raw_table, CastOptions(target_field=schema.to_arrow()))
```

## Don'ts

- Don't hand-roll `pa.int64()` / `pl.Int64` / `"bigint"` strings when
  a `DataType` can emit all of them.
- Don't drop metadata / nullability by rebuilding a `pa.schema(...)`
  from name + dtype tuples — round-trip via `Schema` instead.
- Don't write parallel per-engine field classes; extend `Field`.
