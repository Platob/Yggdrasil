# Architecture

Yggdrasil is built around a **single conversion registry** that every engine plugs into.

## The cast registry

Source: [`python/src/yggdrasil/data/cast/registry.py`](https://github.com/Platob/Yggdrasil/blob/main/python/src/yggdrasil/data/cast/registry.py).

Register converters with `@register_converter(from_hint, to_hint)`; dispatch them via `convert(value, target)`. Dispatch order:

1. **Exact match** — registered `(from, to)` pair (cache hit, ~170 ns).
2. **Identity** — value already matches the target type (or target is `Any` / `object`).
3. **`Any` wildcards** — fall back to converters registered with `Any → to_hint`.
4. **Namespace late-import** — when `from_type` or `to_hint` lives in a `polars` / `pandas` / `pyspark` / `pyarrow` namespace, import the matching `yggdrasil.<engine>.cast` module to trigger its registrations, then retry.
5. **MRO fallback** — walk the source type's MRO to find a registered ancestor.
6. **`issubclass` scan** — final pass against every registered pair for odd typing-protocol keys.

No auto-composition: the registry refuses to chain `X → Y` plus `Y → int` into a synthetic `X → int` cast. Register the direct `X → int` converter explicitly so the intermediate is intentional rather than an artifact of registration order.

Engine modules register their converters **on import**:

```python
import yggdrasil.arrow.cast      # noqa: F401
import yggdrasil.polars.cast     # noqa: F401  (needs polars installed)
import yggdrasil.pandas.cast     # noqa: F401  (needs pandas installed)
import yggdrasil.spark.cast      # noqa: F401  (needs pyspark installed)
```

If a conversion you expect isn't firing, check whether the engine module has actually been imported.

### Register your own

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import convert, register_converter

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

## `CastOptions`

Source: [`python/src/yggdrasil/data/options.py`](https://github.com/Platob/Yggdrasil/blob/main/python/src/yggdrasil/data/options.py).

`CastOptions` is the **single normalized options carrier**. It threads through every cast helper and holds source hints, target field/schema, safety/memory/nullability behavior, and strictness flags.

```python
import pyarrow as pa
from yggdrasil.data.options import CastOptions

opts = CastOptions(
    target=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
)
```

In your own helpers, normalize input through `CastOptions.check`:

```python
def normalize_options(options=None, *, target_field=None) -> CastOptions:
    return CastOptions.check(options, target=target_field)
```

Don't invent parallel per-call option objects — extend `CastOptions` or pass it through.

## `yggdrasil.data` is the canonical surface

Reach for `yggdrasil.data` before raw engine APIs:

- `Field` / `Schema` for describing columns (names, nullability, metadata, nested structure).
- `DataType` / `DataTypeId` for type hints (don't hand-roll `pa.int64()` / `pl.Int64` / `"bigint"` strings).
- `DataTable` / `StatementResult` for "execute a query, then move rows somewhere".
- `convert(value, target, options=...)` for value conversion.
- `yggdrasil.data.enums` for normalized currency / geozone / timezone values.

Only drop down to `polars` / `pandas` / `pyspark` / `pyarrow` when you actually need something the abstraction doesn't cover. When you do, register the new behavior back into `yggdrasil.data` so the next caller gets it for free.

## Optional dependencies — the `lazy_imports` pattern

Subsystems that depend on optional packages are imported through `yggdrasil.lazy_imports`, which does the import once and raises a helpful "install extra X" error on failure.

```python
from yggdrasil.lazy_imports import polars   # correct
import polars                               # wrong — breaks base installs
```

Same applies to pandas, spark, and Databricks-related modules — always import via `yggdrasil.lazy_imports`.

The only **hard** runtime deps are `pyarrow>=20`, `polars>=1.3`, `xxhash`, and `orjson>=3.10`. Base installs must keep working without anything else.

## Schema intent across boundaries

Names, order, nullability, metadata, nested structure, precision/scale, and timezone intent are **part of the user contract**. Don't drop them unless the API documents the loss. The cast registry preserves them by default; engine bridges round-trip through Arrow rather than each engine's native parser to avoid silent drift.
