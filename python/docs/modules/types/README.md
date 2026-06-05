# `yggdrasil.data` — DataType, Field, Schema, convert

Single entry point for value / type / column / row description. Pair this page with the longer [Data Model guide](../../guides/data-model.md) and the [Casting guide](../../guides/casting.md).

## Surface map

| Symbol | Module | Use for |
| --- | --- | --- |
| `DataType` | `yggdrasil.data.types.base` | Type descriptor (int / str / list / dataclass / …) |
| `Field` | `yggdrasil.data.data_field` | Named column with dtype + nullability + metadata |
| `Schema` | `yggdrasil.data.schema` | Ordered field list (= row shape) |
| `UnionType` | `yggdrasil.data.types.union` | `Union[T, U]` / `Optional[T]` at the DataType layer |
| `convert(value, target, options=…)` | `yggdrasil.data.cast.registry` | Single dispatch surface for casts |
| `register_converter(from, to)` | `yggdrasil.data.cast.registry` | Decorator to teach the registry |
| `find_converter(from_type, to_hint)` | `yggdrasil.data.cast.registry` | Lookup-only path |
| `CastOptions` | `yggdrasil.data.cast.options` | Normalised options carrier — target schema, safety, chunk size |

## DataType — type hint round-trip

```python
import datetime as dt
from yggdrasil.data.types.base import DataType

DataType.from_pytype(int).to_pyhint()                  # int
DataType.from_pytype(list[int]).to_pyhint()            # list[int]
DataType.from_pytype(dict[str, float]).to_pyhint()     # dict[str, float]
DataType.from_pytype(dt.date).to_arrow()               # pa.date32()

# User dataclass / Enum survives intact via the _pyhint_cache stamp.
from dataclasses import dataclass

@dataclass
class Row:
    id: int
    name: str

DataType.from_pytype(Row).to_pyhint()                  # <class 'Row'>
```

`from_pytype` stamps the original parsed hint on the resulting instance so the canonical reconstruction doesn't lose user-defined types. `to_pyhint()` reads the cache first, falls back to per-subclass `_default_pyhint()` reconstruction.

## DataType — centralised typing utilities

Every typing-resolution call site in yggdrasil (`safe_function`, `cast.registry`, `data_field`, `arrow.python_defaults`) routes through the classmethods below. One place to fix when typing semantics change.

```python
DataType.PYHINT_ALIASES               # {'pa.': 'pyarrow.', 'pl.': 'polars.', ...}
DataType.expand_alias("pa.Table")     # "pyarrow.Table"

DataType.strip_annotated(...)         # strip Annotated[T, ...] to T
DataType.unwrap_newtype(...)          # unwrap NewType chain
DataType.normalize_hint(...)          # strip_annotated + unwrap_newtype
DataType.unwrap_optional(hint)        # (is_optional, inner)
DataType.unwrap_nullable_hint(hint)   # (inner, has_null) — Field-flavoured
DataType.is_runtime_value(x)          # True for 42, False for int / list[int]

DataType.resolve_str_annotation("Optional[int]")  # typing.Optional[int]
DataType.resolve_str_annotation("pa.Table")       # pyarrow.Table (alias expansion)

def f(t: "pa.Table") -> "pa.RecordBatch": ...
DataType.resolve_function_annotations(f)
# {'t': pyarrow.Table, 'return': pyarrow.RecordBatch}
```

## Field — column descriptor

```python
from yggdrasil.data import field
from yggdrasil.data.types.primitive import IntegerType

f = field(
    name="id",
    dtype=IntegerType(byte_size=8),
    nullable=False,
    metadata={"unit": "rows"},
)

f.to_arrow_field()        # pa.Field("id", pa.int64(), nullable=False, metadata=...)
f.to_polars_field()
f.to_pyspark_field()
```

Builders for the polymorphic entry points:

```python
from yggdrasil.data import Field

Field.from_pytype("id", int, nullable=False)
Field.from_pytype("score", float | None)         # nullable=True from the Optional
Field.from_arrow_field(pa.field("price", pa.float64()))
Field.from_str("id:int64?")
Field.from_dict({"name": "id", "dtype": {...}, "nullable": False})
Field.from_any(...)                              # accepts every shape above
```

Frozen dataclass — use `with_name` / `with_dtype` / `with_nullable` / `with_metadata` for non-destructive edits.

## UnionType — Union / Optional at the DataType layer

```python
from yggdrasil.data.types import UnionType, IntegerType, NullType

u = UnionType(members=(IntegerType(), NullType()))
u.nullable                  # True
u.to_pyhint()               # Optional[int]
u.to_arrow()                # pa.int64()    (delegates to single non-null member)
```

`to_field()` flattens the union into a Field — drops `NullType` into `nullable=True` and un-nests when only one non-null arm remains:

| Union | `to_field()` |
| --- | --- |
| `UnionType(Int, Null)` | `Field(dtype=Int, nullable=True)` |
| `UnionType(Int, Str, Null)` | `Field(dtype=UnionType(Int, Str), nullable=True)` |
| `UnionType(Int, Str)` | `Field(dtype=UnionType(Int, Str), nullable=…)` |
| `UnionType(Null,)` / `UnionType()` | `Field(dtype=NullType, nullable=True)` |

`UnionType` activates via explicit construction today; `from_pytype(Optional[int])` still collapses to `IntegerType()` for backward compatibility.

## Schema — ordered field list

```python
from yggdrasil.data import schema, field
from yggdrasil.data.types.primitive import IntegerType, StringType

s = schema([
    field("id",   IntegerType(), nullable=False),
    field("name", StringType()),
])

s.to_arrow_schema()
s.to_polars_schema()
s.to_spark_schema()

s.merge_with(other_schema)             # union of fields, widen mismatched types
```

`Schema.from_pytype(MyDataclass)`, `Schema.from_arrow_schema(...)`, `Schema.from_str("id:int64, name:string")` and friends all return a `Schema`.

## convert — the single dispatch surface

```python
from yggdrasil.data.cast.registry import convert

convert(42, int)                    # identity (~140 ns)
convert("42", int)                  # 42         (registered str->int)
convert("2024-06-01", "date")       # datetime.date(2024, 6, 1)
convert({"id": "1"}, MyRow)         # dict -> dataclass
convert([1, 2, 3], list[str])       # generic container coercion
```

The dispatch order (full version in the [Data Model guide](../../guides/data-model.md#convert-and-the-cast-registry)):

1. `Any` / `object` target → identity passthrough.
2. Plain-type identity → `isinstance(value, target)` short-circuit.
3. `Optional[T]` unwrap (generic-alias targets only).
4. `None` → `None` if optional, else `default_scalar(target)`.
5. Registry lookup (exact → wildcard → namespace late-import → MRO → issubclass scan).
6. Enum / dataclass.
7. Container generics (`list[T]` / `dict[K, V]` / `tuple[A, B]` / …).
8. `TypeError` — no path found.

No auto-composition: register the direct `from → to` converter explicitly instead of relying on `X → Y → int` chaining.

## Register a custom converter

```python
from decimal import Decimal
from yggdrasil.data.cast.registry import register_converter, convert

@register_converter(str, Decimal)
def _str_to_decimal(value: str, options=None) -> Decimal:
    return Decimal(value.replace(",", "."))

convert("19,95", Decimal)   # Decimal('19.95')
```

Tips:

- The converter takes `(value, options)`. `options` is a `CastOptions` instance or `None`.
- Stay idempotent — `convert(value, X)` on a value already of type X short-circuits before your function runs (the identity check at convert-entry).
- Register cross-engine paths next to the existing engine module (`yggdrasil.polars.cast`, etc.) so a single `import yggdrasil.<engine>.cast` lights up every related conversion.

## `CastOptions` — normalised options carrier

```python
import pyarrow as pa
from yggdrasil.data.cast.options import CastOptions

opts = CastOptions(
    target=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
    safe=True,
    byte_size=128 * 1024 * 1024,
    strict_match_names=True,
)
```

`CastOptions.check(options, **kwargs)` accepts `None`, a bare `pa.Schema` / `pa.Field` / `pa.DataType` (lifts into a `CastOptions` with `target=` set), or an existing `CastOptions`:

```python
def my_helper(source, options=None, *, target=None):
    options = CastOptions.check(options, target=target)
    ...
```

Don't fork a per-call options object — extend `CastOptions` instead.

## Engine bridges

Engine modules register their converters **on import**. Always import via `yggdrasil.lazy_imports` so base installs stay functional.

```python
from yggdrasil.lazy_imports import polars
from yggdrasil.lazy_imports import pandas
```

| Engine | Cast helpers |
| --- | --- |
| Arrow | `yggdrasil.arrow.cast` — `any_to_arrow_table`, `cast_arrow_tabular`, `cast_arrow_record_batch_reader` |
| Polars | `yggdrasil.polars.cast` — `cast_polars_dataframe`, `cast_polars_lazyframe` |
| pandas | `yggdrasil.pandas.cast` — `cast_pandas_dataframe` |
| Spark | `yggdrasil.spark.cast` — `cast_spark_dataframe`, `any_to_spark_dataframe`, `spark_dataframe_to_arrow` |

See [engine cast helpers](cast/README.md) for the per-engine surface.
